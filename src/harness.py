"""The four harness conditions and the rollout loop.

H1 native             the model's own template, parallel calls permitted
H2 parallel_forced    explicit instruction plus an exemplar demanding one block
H3 serial_sanctioned  one call per turn is declared acceptable and the
                      outstanding sub-goals are re-pinned every turn
H4 serial_repeat      H3 with the request duplicated in the prompt

H3 is the deliberate encouragement the project is about. H4 adds the cheapest
reliability lever in the literature: duplicating the input costs no generated
tokens.

Tool-call parsing is intentionally permissive. A model that emits a malformed
parallel block is data, not a crash.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .sandbox import FaultSpec, WorldState, execute, tool_specs
from .tasks import Task

SYSTEM_BASE = (
    "You are a tool-using assistant. You have access to the listed tools. "
    "Emit tool calls as JSON. Do not invent tools or arguments."
)

HARNESS_PROMPTS = {
    "h1_native": "",
    "h2_parallel_forced": (
        "\n\nIMPORTANT: when a request needs several independent tool calls, you "
        "must emit ALL of them together in a single assistant turn as one list. "
        "Do not call one tool and wait. Example of the required form:\n"
        '<tool_call>[{"name": "get_temperature", "arguments": {"city": "delhi"}}, '
        '{"name": "get_temperature", "arguments": {"city": "pune"}}]</tool_call>'
    ),
    "h3_serial_sanctioned": (
        "\n\nYou may call one tool per turn if you prefer. That is fully "
        "acceptable. What matters is that every requested item is eventually "
        "handled, with its arguments unchanged from the original request. "
        "Work through the outstanding items one at a time and do not stop early."
    ),
    "h4_serial_repeat": (
        "\n\nYou may call one tool per turn if you prefer. That is fully "
        "acceptable. What matters is that every requested item is eventually "
        "handled, with its arguments unchanged from the original request. "
        "Work through the outstanding items one at a time and do not stop early."
    ),
}

HARNESSES = tuple(HARNESS_PROMPTS)

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Pull every tool call out of an assistant turn, however it was wrapped."""
    candidates: list[str] = _TOOL_CALL_BLOCK.findall(text)
    if not candidates:
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        candidates = fenced or ([text] if ("{" in text and '"name"' in text) else [])

    calls: list[dict] = []
    for blob in candidates:
        parsed = None
        try:
            parsed = json.loads(blob.strip())
        except json.JSONDecodeError:
            parsed = []
            for match in _JSON_OBJECT.findall(blob):
                try:
                    parsed.append(json.loads(match))
                except json.JSONDecodeError:
                    continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool") or item.get("function")
            if isinstance(name, dict):
                name = name.get("name")
            if not name:
                continue
            args = item.get("arguments", item.get("args", item.get("parameters", {})))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({"name": str(name), "args": args if isinstance(args, dict) else {}})
    return calls


@dataclass
class Turn:
    text: str
    calls: list[dict]
    observations: list[dict]
    completion_tokens: int = 0


@dataclass
class Trace:
    task_id: str
    regime: str
    k: int
    model: str
    harness: str
    turns: list[Turn] = field(default_factory=list)
    final_digest: str = ""
    reference_digest: str = ""
    wall_seconds: float = 0.0
    model_seconds: float = 0.0
    tool_seconds: float = 0.0

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def emitted_calls(self) -> list[dict]:
        return [c for t in self.turns for c in t.calls]

    @property
    def max_calls_in_one_turn(self) -> int:
        return max((len(t.calls) for t in self.turns), default=0)

    @property
    def completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self.turns)

    @property
    def tool_errors(self) -> int:
        return sum(1 for t in self.turns for o in t.observations if "error" in o)

    def async_wall_seconds(self, per_call_latency: float = 0.25) -> float:
        """Latency if the runtime had executed each turn's calls concurrently.

        A batched turn of k calls costs one call's latency; a serialised trace
        pays per turn. This is the AsyncLM-style accounting that separates the
        model's overhead from the harness's.
        """
        tool_cost = per_call_latency * self.n_turns
        return self.model_seconds + tool_cost


def _outstanding(task: Task, done: list[dict]) -> str:
    remaining = []
    for call in task.reference_calls:
        if not any(d["name"] == call["name"] and d.get("args") == call.get("args")
                   for d in done):
            remaining.append(f"{call['name']}({json.dumps(call.get('args', {}))})")
    if not remaining:
        return "Nothing outstanding. Summarise the results."
    return "Still outstanding: " + "; ".join(remaining)


def rollout(backend, task: Task, harness: str, max_turns: int | None = None,
            per_call_latency: float = 0.0, pin_outstanding: bool = True) -> Trace:
    """Run one task against one model under one harness condition."""
    if harness not in HARNESS_PROMPTS:
        raise ValueError(f"unknown harness '{harness}'")
    if hasattr(backend, "prime"):
        backend.prime(task.reference_calls)

    max_turns = max_turns or (len(task.reference_calls) + 2)
    system = SYSTEM_BASE + HARNESS_PROMPTS[harness]
    user = task.prompt
    if harness == "h4_serial_repeat":
        user = f"{task.prompt}\n\nRepeating the request verbatim:\n{task.prompt}"

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    specs = tool_specs(task.allowed_tools)
    state = WorldState()
    fault = FaultSpec(task.fault_tool) if task.fault_tool else None

    trace = Trace(task.id, task.regime, task.k, getattr(backend, "name", "model"),
                  harness, reference_digest=task.reference_digest)
    started = time.perf_counter()

    for _ in range(max_turns):
        t0 = time.perf_counter()
        completion = backend.chat(messages, specs)
        trace.model_seconds += time.perf_counter() - t0

        calls = parse_tool_calls(completion.text)
        observations: list[dict] = []
        for call in calls:
            t1 = time.perf_counter()
            obs, _ = execute(state, call["name"], call.get("args", {}), fault)
            trace.tool_seconds += time.perf_counter() - t1
            if per_call_latency:
                trace.tool_seconds += per_call_latency
            observations.append(obs if isinstance(obs, dict) else {"result": obs})

        trace.turns.append(Turn(completion.text, calls, observations,
                                completion.completion_tokens))

        if not calls:
            break

        messages.append({"role": "assistant", "content": completion.text})
        obs_text = json.dumps(observations, default=str)[:2000]
        followup = f"Tool results: {obs_text}"
        if pin_outstanding and harness in ("h3_serial_sanctioned", "h4_serial_repeat"):
            followup += "\n" + _outstanding(task, trace.emitted_calls)
        messages.append({"role": "user", "content": followup})

    trace.wall_seconds = time.perf_counter() - started
    trace.final_digest = state.digest()
    return trace
