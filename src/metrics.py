"""Metrics. These are the columns of the paper's results table.

  Par. AST  structural parallel accuracy, reproducing the status quo
  FBR       fallback rate, the share of items with no batched block
  OER       outcome equivalence rate among fallbacks  <-- the headline
  GDS       OER discounted by turn and latency inflation
  pass^k    consistency over repeated trials
  Recovery  share of fault-injected runs that still reach the reference state
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict

from .classify import MODES, ast_match, classify, is_fallback, outcome_equivalent
from .harness import Trace


@dataclass
class Row:
    model: str
    harness: str
    regime: str
    n: int
    par_ast: float
    fallback_rate: float
    oer: float
    gds: float
    mean_turns: float
    mean_tokens: float
    naive_latency: float
    async_latency: float
    recovery_rate: float
    mode_shares: dict

    def to_dict(self) -> dict:
        return asdict(self)


def graceful_degradation_score(oer: float, turn_inflation: float,
                               latency_inflation: float,
                               alpha: float = 0.25, beta: float = 0.25) -> float:
    """OER discounted by the overhead the fallback actually imposes.

    Report this over a sweep of (alpha, beta) rather than a single pair, so a
    reader with a different cost model can read off their own answer.
    """
    penalty = 1.0 + alpha * max(0.0, turn_inflation) + beta * max(0.0, latency_inflation)
    return oer / penalty


def summarise(traces: list[Trace], references: dict[str, list[dict]],
              alpha: float = 0.25, beta: float = 0.25,
              per_call_latency: float = 0.25) -> Row:
    """Collapse a set of traces from one (model, harness, regime) cell into a row."""
    if not traces:
        raise ValueError("no traces to summarise")

    modes = [classify(t, references[t.task_id]) for t in traces]
    n = len(traces)
    counts = Counter(modes)

    par_ast = sum(ast_match(t, references[t.task_id]) for t in traces) / n
    fallbacks = [(t, m) for t, m in zip(traces, modes) if is_fallback(m)]
    fbr = len(fallbacks) / n

    if fallbacks:
        oer = sum(1 for _, m in fallbacks if m == "M2_serialised_complete") / len(fallbacks)
        mean_fb_turns = sum(t.n_turns for t, _ in fallbacks) / len(fallbacks)
    else:
        oer, mean_fb_turns = 0.0, 1.0

    batched = [t for t, m in zip(traces, modes) if m == "M1_batched_correct"]
    base_latency = (sum(t.async_wall_seconds(per_call_latency) for t in batched)
                    / len(batched)) if batched else None

    naive = sum(t.wall_seconds + per_call_latency * len(t.emitted_calls)
                for t in traces) / n
    asynced = sum(t.async_wall_seconds(per_call_latency) for t in traces) / n

    turn_inflation = max(0.0, mean_fb_turns - 1.0)
    latency_inflation = ((asynced / base_latency) - 1.0) if base_latency else 0.0
    gds = graceful_degradation_score(oer, turn_inflation, latency_inflation, alpha, beta)

    injected = [t for t in traces if t.tool_errors > 0]
    recovery = (sum(outcome_equivalent(t) for t in injected) / len(injected)
                if injected else float("nan"))

    return Row(
        model=traces[0].model,
        harness=traces[0].harness,
        regime=traces[0].regime,
        n=n,
        par_ast=100 * par_ast,
        fallback_rate=100 * fbr,
        oer=100 * oer,
        gds=100 * gds,
        mean_turns=sum(t.n_turns for t in traces) / n,
        mean_tokens=sum(t.completion_tokens for t in traces) / n,
        naive_latency=naive,
        async_latency=asynced,
        recovery_rate=100 * recovery if recovery == recovery else float("nan"),
        mode_shares={m: round(100 * counts.get(m, 0) / n, 1) for m in MODES},
    )


def pass_at_k(repeat_traces: list[list[Trace]]) -> float:
    """pass^k following tau-bench: every one of the k repeats must succeed.

    ``repeat_traces`` is a list of per-task lists, each holding that task's k
    independent samples.
    """
    if not repeat_traces:
        return 0.0
    hits = sum(1 for reps in repeat_traces
               if reps and all(outcome_equivalent(t) for t in reps))
    return 100 * hits / len(repeat_traces)


def format_table(rows: list[Row]) -> str:
    header = (f"{'model':28} {'harness':22} {'regime':18} "
              f"{'AST':>6} {'FBR':>6} {'OER':>6} {'GDS':>6} {'turns':>6} {'lat_s':>7}")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.model[:28]:28} {r.harness[:22]:22} {r.regime[:18]:18} "
            f"{r.par_ast:6.1f} {r.fallback_rate:6.1f} {r.oer:6.1f} {r.gds:6.1f} "
            f"{r.mean_turns:6.2f} {r.async_latency:7.3f}"
        )
    return "\n".join(lines)
