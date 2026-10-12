"""SeqFall task generation.

Each task is a request that *should* be answerable with k tool calls, together
with the reference plan and a ground-truth dependency regime. The regime label
is what lets us say something precise rather than vague about when sequential
fallback is safe:

  independent        read-only calls that commute and change nothing
  commutative_write  writes to disjoint keys, so order cannot matter
  order_sensitive    writes to the same key, so order changes the end state
  dependent          call j consumes call i's output, so no correct parallel
                     plan exists at all (the control that should reward
                     serialisation and punish batching)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Any

from .sandbox import CATALOG, CITY_WEATHER, execute_plan

REGIMES = ("independent", "commutative_write", "order_sensitive", "dependent")

CITIES = sorted(CITY_WEATHER)
SKUS = sorted(CATALOG)


@dataclass
class Task:
    id: str
    regime: str
    k: int
    prompt: str
    reference_calls: list[dict[str, Any]]
    allowed_tools: list[str]
    reference_digest: str = ""
    fault_tool: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _finalize(task: Task) -> Task:
    task.reference_digest = execute_plan(task.reference_calls).digest()
    return task


# ---------------------------------------------------------------------------
# Regime builders
# ---------------------------------------------------------------------------


def _independent(rng: random.Random, k: int, idx: int) -> Task:
    cities = rng.sample(CITIES, k)
    field_tool = rng.choice(["get_temperature", "get_humidity", "get_condition"])
    noun = {"get_temperature": "temperature", "get_humidity": "humidity",
            "get_condition": "sky condition"}[field_tool]
    calls = [{"name": field_tool, "args": {"city": c}} for c in cities]
    prompt = (
        f"I am planning a trip. What is the current {noun} in "
        + ", ".join(c.title() for c in cities[:-1])
        + f" and {cities[-1].title()}? Report all {k} of them."
    )
    return _finalize(Task(f"ind-{idx:04d}", "independent", k, prompt, calls,
                          [field_tool]))


def _commutative_write(rng: random.Random, k: int, idx: int) -> Task:
    keys = rng.sample(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"], k)
    vals = [str(rng.randint(10, 99)) for _ in keys]
    calls = [{"name": "kv_set", "args": {"key": a, "value": b}} for a, b in zip(keys, vals)]
    pairs = ", ".join(f"{a}={b}" for a, b in zip(keys, vals))
    prompt = (
        f"Store these {k} settings in the key-value store, all of them: {pairs}. "
        "The keys are unrelated to each other."
    )
    return _finalize(Task(f"cwr-{idx:04d}", "commutative_write", k, prompt, calls,
                          ["kv_set", "kv_get", "kv_keys"]))


def _order_sensitive(rng: random.Random, k: int, idx: int) -> Task:
    sku = rng.choice(SKUS)
    qtys = [rng.randint(1, 5) for _ in range(k)]
    calls = [{"name": "cart_set_qty", "args": {"sku": sku, "qty": q}} for q in qtys]
    steps = " then ".join(f"set it to {q}" for q in qtys)
    prompt = (
        f"For {sku} in my cart: {steps}. Apply these {k} updates in the order given; "
        "the final quantity must be the last one."
    )
    return _finalize(Task(f"ord-{idx:04d}", "order_sensitive", k, prompt, calls,
                          ["cart_set_qty", "cart_add", "cart_total"]))


def _dependent(rng: random.Random, k: int, idx: int) -> Task:
    city = rng.choice(CITIES)
    calls = [
        {"name": "get_temperature", "args": {"city": city}},
        {"name": "c_to_f", "args": {"celsius": CITY_WEATHER[city]["temp_c"]}},
    ]
    if k >= 3:
        calls.append({"name": "kv_set", "args": {"key": f"{city}_f", "value": "pending"}})
    if k >= 4:
        calls.append({"name": "log_note", "args": {"text": f"converted {city}"}})
    prompt = (
        f"Get the current temperature in {city.title()}, convert that exact number to "
        "Fahrenheit, and record the result. You cannot convert before you have looked "
        "it up."
    )
    return _finalize(Task(f"dep-{idx:04d}", "dependent", min(k, len(calls)), prompt,
                          calls, ["get_temperature", "c_to_f", "kv_set", "log_note"]))


BUILDERS = {
    "independent": _independent,
    "commutative_write": _commutative_write,
    "order_sensitive": _order_sensitive,
    "dependent": _dependent,
}


def build_suite(
    n_per_cell: int = 15,
    ks: tuple[int, ...] = (2, 3, 4, 6),
    regimes: tuple[str, ...] = REGIMES,
    seed: int = 20261012,
    inject_faults: bool = False,
) -> list[Task]:
    """Build the suite. Default shape is 4 regimes x 4 k values x 15 = 240 items.

    Scale ``n_per_cell`` to 56 to reach the 900-item suite described in the
    paper; start small while you are debugging the harness.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []
    counter = 0
    for regime in regimes:
        for k in ks:
            if regime == "independent" and k > len(CITIES):
                continue
            for _ in range(n_per_cell):
                counter += 1
                task = BUILDERS[regime](rng, k, counter)
                if inject_faults:
                    task.fault_tool = task.reference_calls[0]["name"]
                tasks.append(task)
    return tasks


if __name__ == "__main__":
    import json

    suite = build_suite(n_per_cell=2)
    print(f"{len(suite)} tasks")
    for t in suite[:4]:
        print(json.dumps(t.to_dict(), indent=2)[:400])
