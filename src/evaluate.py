"""Evaluation entry point.

    python -m src.evaluate --config configs/default.yaml

Writes one JSONL of raw traces (every turn, every observation, every label) and
one CSV of summary rows whose columns line up with the paper's results table.
Keep the JSONL: the qualitative appendix comes out of it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import yaml

from .classify import classify, drift_detail
from .harness import HARNESSES, rollout
from .metrics import format_table, summarise
from .models import build_backend
from .tasks import build_suite


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def trace_record(trace, mode: str, reference_calls: list[dict]) -> dict:
    return {
        "task_id": trace.task_id,
        "regime": trace.regime,
        "k": trace.k,
        "model": trace.model,
        "harness": trace.harness,
        "mode": mode,
        "n_turns": trace.n_turns,
        "max_calls_in_one_turn": trace.max_calls_in_one_turn,
        "emitted_calls": trace.emitted_calls,
        "reference_calls": reference_calls,
        "final_digest": trace.final_digest,
        "reference_digest": trace.reference_digest,
        "outcome_equivalent": trace.final_digest == trace.reference_digest,
        "completion_tokens": trace.completion_tokens,
        "tool_errors": trace.tool_errors,
        "wall_seconds": round(trace.wall_seconds, 4),
        "model_seconds": round(trace.model_seconds, 4),
        "drift": drift_detail(trace, reference_calls),
        "turns": [{"text": t.text[:1500], "calls": t.calls,
                   "observations": t.observations} for t in trace.turns],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SeqFall evaluation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of tasks, for a quick smoke run")
    parser.add_argument("--harnesses", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    suite_cfg = cfg.get("suite", {})
    tasks = build_suite(
        n_per_cell=int(suite_cfg.get("n_per_cell", 15)),
        ks=tuple(suite_cfg.get("ks", [2, 3, 4, 6])),
        regimes=tuple(suite_cfg.get("regimes", ["independent", "commutative_write",
                                                "order_sensitive", "dependent"])),
        seed=int(suite_cfg.get("seed", 20261012)),
        inject_faults=bool(suite_cfg.get("inject_faults", False)),
    )
    if args.limit:
        tasks = tasks[: args.limit]
    references = {t.id: t.reference_calls for t in tasks}

    harnesses = args.harnesses or cfg.get("harnesses", list(HARNESSES))
    per_call_latency = float(cfg.get("per_call_latency", 0.25))

    all_rows = []
    for model_cfg in cfg["models"]:
        backend = build_backend({**cfg.get("backend_defaults", {}), **model_cfg})
        tag = model_cfg.get("tag") or getattr(backend, "name", "model")
        jsonl_path = os.path.join(args.out_dir, f"traces_{tag.replace('/', '_')}.jsonl")

        cells = defaultdict(list)
        with open(jsonl_path, "w") as fh:
            for harness in harnesses:
                for task in tasks:
                    trace = rollout(backend, task, harness,
                                    max_turns=int(cfg.get("max_turns", 0)) or None)
                    trace.model = tag
                    mode = classify(trace, task.reference_calls)
                    fh.write(json.dumps(trace_record(trace, mode, task.reference_calls),
                                        default=str) + "\n")
                    cells[(harness, task.regime)].append(trace)
                    cells[(harness, "ALL")].append(trace)

        for (harness, regime), traces in sorted(cells.items()):
            row = summarise(traces, references,
                            alpha=float(cfg.get("alpha", 0.25)),
                            beta=float(cfg.get("beta", 0.25)),
                            per_call_latency=per_call_latency)
            row.model, row.harness, row.regime = tag, harness, regime
            all_rows.append(row)
        print(f"wrote {jsonl_path}")

    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].to_dict()))
        writer.writeheader()
        for row in all_rows:
            record = row.to_dict()
            record["mode_shares"] = json.dumps(record["mode_shares"])
            writer.writerow(record)

    print()
    print(format_table([r for r in all_rows if r.regime == "ALL"]))
    print()
    print(f"summary -> {csv_path}")


if __name__ == "__main__":
    main()
