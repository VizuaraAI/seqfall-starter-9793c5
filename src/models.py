"""Model backends.

Three of them:

  ``scripted``  No model at all. Emits batched or serialised calls with a
                configurable mix of the five fallback modes, so you can test the
                whole pipeline (classifier, metrics, plots) in two seconds
                before you touch a GPU. Its outputs are synthetic by design and
                must never appear in a results table.
  ``hf``        Local transformers. Uses the tokenizer's chat template with
                ``tools=`` so each model speaks its own native tool protocol.
  ``openai``    Any OpenAI-compatible endpoint, which is how you talk to a
                vLLM or llama.cpp server. Fastest path for the 7B and 32B runs.

Every backend returns a raw assistant string. Parsing lives in ``harness.py``
on purpose: how a model fails to format a parallel block is part of what we are
measuring, so we do not want the backend hiding it.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Backend(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict]) -> Completion: ...


# ---------------------------------------------------------------------------
# Scripted backend (pipeline testing only)
# ---------------------------------------------------------------------------

#: Probability mass over behaviours, indexed loosely by "model size" so the
#: demo run produces a plausible-looking scale trend. Replace with real runs.
SCRIPTED_PROFILES = {
    "tiny": dict(batched=0.05, complete=0.45, truncated=0.25, drift=0.15, collapsed=0.10),
    "small": dict(batched=0.20, complete=0.60, truncated=0.10, drift=0.07, collapsed=0.03),
    "mid": dict(batched=0.45, complete=0.46, truncated=0.05, drift=0.03, collapsed=0.01),
}


class ScriptedBackend:
    """Deterministic stand-in that exercises every branch of the pipeline."""

    def __init__(self, profile: str = "small", seed: int = 7):
        if profile not in SCRIPTED_PROFILES:
            raise ValueError(f"profile must be one of {sorted(SCRIPTED_PROFILES)}")
        self.name = f"scripted:{profile}"
        self.profile = SCRIPTED_PROFILES[profile]
        self.rng = random.Random(seed)
        self._plan: list[dict] = []
        self._mode: str | None = None
        self._cursor = 0

    def _choose_mode(self) -> str:
        roll = self.rng.random()
        acc = 0.0
        for mode, mass in self.profile.items():
            acc += mass
            if roll <= acc:
                return mode
        return "complete"

    def prime(self, reference_calls: list[dict]) -> None:
        """The harness calls this once per task so the stand-in knows the plan."""
        self._plan = [dict(c) for c in reference_calls]
        self._mode = self._choose_mode()
        self._cursor = 0

    def chat(self, messages: list[dict], tools: list[dict]) -> Completion:
        if not self._plan:
            return Completion("I have everything I need.", 50, 8)
        calls: list[dict]
        if self._mode == "batched":
            calls, self._cursor = self._plan, len(self._plan)
        elif self._mode == "collapsed":
            merged = dict(self._plan[0])
            merged["args"] = dict(merged.get("args", {}))
            calls, self._cursor = [merged], len(self._plan)
        elif self._mode == "truncated":
            keep = max(1, len(self._plan) - 1 - self.rng.randint(0, 1))
            if self._cursor >= keep:
                return Completion("Done, that covers it.", 60, 9)
            calls = [self._plan[self._cursor]]
            self._cursor += 1
        else:  # complete or drift, one call per turn
            if self._cursor >= len(self._plan):
                return Completion("All steps are complete.", 60, 9)
            call = dict(self._plan[self._cursor])
            call["args"] = dict(call.get("args", {}))
            if self._mode == "drift" and self._cursor > 0 and call["args"]:
                key = sorted(call["args"])[0]
                call["args"][key] = f"{call['args'][key]}_x"
            calls = [call]
            self._cursor += 1
        payload = json.dumps([{"name": c["name"], "arguments": c.get("args", {})}
                              for c in calls])
        return Completion(f"<tool_call>{payload}</tool_call>", 180, 30 * len(calls))


# ---------------------------------------------------------------------------
# Hugging Face transformers backend
# ---------------------------------------------------------------------------


class HFBackend:
    """Local inference through the model's own chat template."""

    def __init__(self, model_id: str, max_new_tokens: int = 384,
                 temperature: float = 0.0, device: str = "auto",
                 enable_thinking: bool | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()

    def chat(self, messages: list[dict], tools: list[dict]) -> Completion:
        import torch

        kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tools=tools, **kwargs)
        except TypeError:
            # Older templates do not accept tools; fall back to an inline list.
            inlined = [{"role": "system",
                        "content": "Available tools:\n" + json.dumps(tools)}] + messages
            prompt = self.tokenizer.apply_chat_template(inlined, **kwargs)
        batch = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][batch["input_ids"].shape[1]:]
        return Completion(
            self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            prompt_tokens=int(batch["input_ids"].shape[1]),
            completion_tokens=int(new_tokens.shape[0]),
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (vLLM, llama.cpp, hosted)
# ---------------------------------------------------------------------------


class OpenAIBackend:
    def __init__(self, model_id: str, base_url: str | None = None,
                 temperature: float = 0.0, api_key_env: str = "OPENAI_API_KEY"):
        from openai import OpenAI

        self.name = model_id
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get(api_key_env, "not-needed-for-local"),
        )

    def chat(self, messages: list[dict], tools: list[dict]) -> Completion:
        resp = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            payload = [{"name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments or "{}")}
                       for tc in msg.tool_calls]
            text = f"<tool_call>{json.dumps(payload)}</tool_call>"
        else:
            text = msg.content or ""
        usage = resp.usage
        return Completion(text,
                          prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                          completion_tokens=getattr(usage, "completion_tokens", 0) or 0)


def build_backend(cfg: dict) -> Backend:
    kind = cfg.get("backend", "scripted")
    if kind == "scripted":
        return ScriptedBackend(profile=cfg.get("profile", "small"),
                               seed=int(cfg.get("seed", 7)))
    if kind == "hf":
        return HFBackend(cfg["model_id"],
                         max_new_tokens=int(cfg.get("max_new_tokens", 384)),
                         temperature=float(cfg.get("temperature", 0.0)),
                         enable_thinking=cfg.get("enable_thinking"))
    if kind == "openai":
        return OpenAIBackend(cfg["model_id"],
                             base_url=cfg.get("base_url"),
                             temperature=float(cfg.get("temperature", 0.0)))
    raise ValueError(f"unknown backend '{kind}'")
