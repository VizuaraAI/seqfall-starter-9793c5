"""Optional training entry point: teach a small model to serialise *well*.

The paper's claim is that sequential fallback is fine as long as it is complete
and drift-free. That makes an obvious intervention available: mine preference
pairs from your own traces, preferring M2 (serialised and complete) over M3
(truncated), M4 (drift) and M5 (collapsed), then run DPO. Prior work has shown
DPO moves small-model call behaviour, so this is a credible week-7 extension
rather than a long shot.

    # 1. produce traces first
    python -m src.evaluate --config configs/default.yaml
    # 2. mine pairs and train
    python -m src.train --traces results/traces_qwen2.5-1.5b.jsonl \
        --model Qwen/Qwen2.5-1.5B-Instruct --out checkpoints/serial-dpo

This is deliberately the smallest thing that can work: LoRA, one epoch, short
sequences. Do not start here. Get the measurement right first.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

GOOD_MODE = "M2_serialised_complete"
BAD_MODES = ("M3_serialised_truncated", "M4_serialised_drift", "M5_collapsed")


def load_traces(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mine_pairs(records: list[dict], max_pairs: int = 2000) -> list[dict]:
    """Pair a good trace against a bad one for the same task."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_task[rec["task_id"]].append(rec)

    pairs: list[dict] = []
    for task_id, recs in by_task.items():
        good = [r for r in recs if r["mode"] == GOOD_MODE]
        bad = [r for r in recs if r["mode"] in BAD_MODES]
        if not good or not bad:
            continue
        prompt = (
            "Handle every item in this request using the available tools. "
            "One call per turn is acceptable; do not drop items and do not "
            "change their arguments.\n\n"
            f"Reference plan: {json.dumps(recs[0]['reference_calls'])}"
        )
        for g in good[:2]:
            for b in bad[:2]:
                pairs.append({
                    "prompt": prompt,
                    "chosen": json.dumps(g["emitted_calls"]),
                    "rejected": json.dumps(b["emitted_calls"]),
                    "task_id": task_id,
                })
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def train(pairs: list[dict], model_id: str, out_dir: str, epochs: float = 1.0,
          lr: float = 5e-6, beta: float = 0.1) -> None:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(
        [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
         for p in pairs]
    )

    trainer = DPOTrainer(
        model=model_id,
        args=DPOConfig(
            output_dir=out_dir,
            num_train_epochs=epochs,
            learning_rate=lr,
            beta=beta,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            max_length=1024,
            max_prompt_length=512,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            report_to=[],
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                               task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"saved adapter to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO on serialisation quality.")
    parser.add_argument("--traces", required=True, help="JSONL from src.evaluate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="checkpoints/serial-dpo")
    parser.add_argument("--max-pairs", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true",
                        help="mine and report pairs without loading a model")
    args = parser.parse_args()

    pairs = mine_pairs(load_traces(args.traces), args.max_pairs)
    print(f"mined {len(pairs)} preference pairs over "
          f"{len({p['task_id'] for p in pairs})} tasks")
    if args.dry_run:
        for p in pairs[:3]:
            print(json.dumps(p, indent=2)[:600])
        return
    if not pairs:
        raise SystemExit("no pairs mined; run more harness conditions first")
    train(pairs, args.model, args.out)


if __name__ == "__main__":
    main()
