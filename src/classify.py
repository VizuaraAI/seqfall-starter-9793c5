"""The five-mode fallback taxonomy.

This file is the scientific heart of the project, so keep it rule-based and
auditable. No judge model: every label must be reproducible from the trace and
the state digest alone, otherwise reviewers will reasonably ask whether the
taxonomy is doing the work or the judge is.

  M1 batched_correct      parallel block emitted as requested, state matches
  M2 serialised_complete  all k calls across multiple turns, state matches
  M3 serialised_truncated fewer than k calls emitted, model answered anyway
  M4 serialised_drift     all k calls emitted but arguments mutated, state differs
  M5 collapsed            calls merged, malformed, or a wrong tool chosen

Only M3, M4 and M5 are failures. Current benchmarks score M2 identically to M5,
which is the mis-scoring the paper measures.
"""

from __future__ import annotations

from .harness import Trace

MODES = (
    "M1_batched_correct",
    "M2_serialised_complete",
    "M3_serialised_truncated",
    "M4_serialised_drift",
    "M5_collapsed",
)

HARMFUL_MODES = ("M3_serialised_truncated", "M4_serialised_drift", "M5_collapsed")


def _normalise(args: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in (args or {}).items()))


def call_multiset(calls: list[dict]) -> list[tuple]:
    return sorted((c["name"], _normalise(c.get("args", {}))) for c in calls)


def ast_match(trace: Trace, reference_calls: list[dict]) -> bool:
    """BFCL-style structural score: the reference set, emitted in one turn."""
    if trace.max_calls_in_one_turn < len(reference_calls):
        return False
    for turn in trace.turns:
        if call_multiset(turn.calls) == call_multiset(reference_calls):
            return True
    return False


def outcome_equivalent(trace: Trace) -> bool:
    return bool(trace.final_digest) and trace.final_digest == trace.reference_digest


def classify(trace: Trace, reference_calls: list[dict]) -> str:
    """Assign exactly one mode. Order of the checks matters."""
    emitted = trace.emitted_calls
    ref_names = {c["name"] for c in reference_calls}
    k = len(reference_calls)

    # Wrong tools, or everything crammed into one under-specified call.
    unknown_tool = any(c["name"] not in ref_names for c in emitted)
    if unknown_tool:
        return "M5_collapsed"
    if emitted and len(emitted) < k and trace.max_calls_in_one_turn == len(emitted) == 1:
        # One single call standing in for k requested ones, with no further turns.
        if trace.n_turns <= 2:
            return "M5_collapsed"
    if not emitted:
        return "M5_collapsed"

    batched = trace.max_calls_in_one_turn >= k

    if batched and outcome_equivalent(trace):
        return "M1_batched_correct"
    if batched:
        # Emitted the requested shape but landed in the wrong state.
        return "M4_serialised_drift" if len(emitted) >= k else "M5_collapsed"

    if len(emitted) < k:
        return "M3_serialised_truncated"
    if outcome_equivalent(trace):
        return "M2_serialised_complete"
    return "M4_serialised_drift"


def is_fallback(mode: str) -> bool:
    return mode != "M1_batched_correct"


def drift_detail(trace: Trace, reference_calls: list[dict]) -> list[str]:
    """Which arguments moved. Useful for the qualitative appendix."""
    ref = {}
    for call in reference_calls:
        ref.setdefault(call["name"], []).append(_normalise(call.get("args", {})))
    notes: list[str] = []
    for call in trace.emitted_calls:
        want = ref.get(call["name"], [])
        got = _normalise(call.get("args", {}))
        if want and got not in want:
            notes.append(f"{call['name']}: emitted {dict(got)}, expected one of "
                         f"{[dict(w) for w in want]}")
    return notes
