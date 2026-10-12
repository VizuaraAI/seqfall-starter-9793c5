"""Deterministic tool sandbox for SeqFall.

Forty tools across five domains. Every tool is a pure function of an explicit,
serialisable ``WorldState``, which is the whole point: if two execution orders
(one batched parallel block, one serialised across turns) reach the same state,
we can prove it by comparing hashes instead of asking a judge model.

Tools are registered with a tiny JSON-schema-like spec so the same registry can
be handed to a chat template as an OpenAI-style tool list.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# World state
# ---------------------------------------------------------------------------

#: Read-only reference tables. Deterministic by construction, which removes the
#: API nondeterminism that StableToolBench showed inflates apparent failure.
CITY_WEATHER = {
    "delhi": {"temp_c": 34, "humidity": 41, "condition": "haze"},
    "mumbai": {"temp_c": 31, "humidity": 78, "condition": "showers"},
    "chennai": {"temp_c": 33, "humidity": 71, "condition": "cloudy"},
    "pune": {"temp_c": 29, "humidity": 55, "condition": "clear"},
    "kolkata": {"temp_c": 32, "humidity": 69, "condition": "cloudy"},
    "bengaluru": {"temp_c": 26, "humidity": 60, "condition": "clear"},
    "jaipur": {"temp_c": 36, "humidity": 28, "condition": "sunny"},
    "kochi": {"temp_c": 30, "humidity": 82, "condition": "rain"},
}

CITY_META = {
    "delhi": {"iata": "DEL", "tz": "Asia/Kolkata", "population_m": 32.9},
    "mumbai": {"iata": "BOM", "tz": "Asia/Kolkata", "population_m": 21.3},
    "chennai": {"iata": "MAA", "tz": "Asia/Kolkata", "population_m": 11.5},
    "pune": {"iata": "PNQ", "tz": "Asia/Kolkata", "population_m": 7.0},
    "kolkata": {"iata": "CCU", "tz": "Asia/Kolkata", "population_m": 15.1},
    "bengaluru": {"iata": "BLR", "tz": "Asia/Kolkata", "population_m": 13.6},
    "jaipur": {"iata": "JAI", "tz": "Asia/Kolkata", "population_m": 4.1},
    "kochi": {"iata": "COK", "tz": "Asia/Kolkata", "population_m": 2.3},
}

CATALOG = {
    "sku-101": {"name": "mechanical keyboard", "price": 4999, "stock": 12},
    "sku-102": {"name": "usb-c hub", "price": 1899, "stock": 40},
    "sku-103": {"name": "noise cancelling headphones", "price": 8999, "stock": 5},
    "sku-104": {"name": "laptop stand", "price": 1299, "stock": 23},
    "sku-105": {"name": "27 inch monitor", "price": 18999, "stock": 3},
}

FX_RATES = {"usd": 1.0, "inr": 83.2, "eur": 0.92, "gbp": 0.79, "jpy": 151.4}


@dataclass
class WorldState:
    """Everything a tool is allowed to touch."""

    kv: dict[str, Any] = field(default_factory=dict)
    cart: dict[str, int] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def clone(self) -> "WorldState":
        return copy.deepcopy(self)

    def canonical(self) -> str:
        """Order-insensitive canonical serialisation of the mutable state."""
        return json.dumps(
            {
                "kv": dict(sorted(self.kv.items())),
                "cart": dict(sorted((k, v) for k, v in self.cart.items() if v != 0)),
                "files": dict(sorted(self.files.items())),
                "notes": sorted(self.notes),
            },
            sort_keys=True,
            default=str,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]
    required: list[str]
    fn: Callable[..., Any]
    writes: bool

    def openai_spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: {"type": v} for k, v in self.params.items()
                    },
                    "required": self.required,
                },
            },
        }


REGISTRY: dict[str, Tool] = {}


def tool(description: str, params: dict[str, str], required: list[str], writes: bool):
    def deco(fn):
        REGISTRY[fn.__name__] = Tool(
            name=fn.__name__,
            description=description,
            params=params,
            required=required,
            fn=fn,
            writes=writes,
        )
        return fn

    return deco


class ToolError(Exception):
    """Raised for a bad call. Surfaced to the model as a structured error."""


def _city(name: str) -> str:
    key = str(name).strip().lower()
    if key not in CITY_WEATHER:
        raise ToolError(f"unknown city '{name}'; known: {sorted(CITY_WEATHER)}")
    return key


# --- Domain 1: weather and travel lookup (read-only, commutative) ----------


@tool("Current temperature in Celsius for a city.", {"city": "string"}, ["city"], False)
def get_temperature(state: WorldState, city: str):
    return {"city": _city(city), "temp_c": CITY_WEATHER[_city(city)]["temp_c"]}


@tool("Current humidity percentage for a city.", {"city": "string"}, ["city"], False)
def get_humidity(state: WorldState, city: str):
    return {"city": _city(city), "humidity": CITY_WEATHER[_city(city)]["humidity"]}


@tool("Short text weather condition for a city.", {"city": "string"}, ["city"], False)
def get_condition(state: WorldState, city: str):
    return {"city": _city(city), "condition": CITY_WEATHER[_city(city)]["condition"]}


@tool("Full weather record for a city.", {"city": "string"}, ["city"], False)
def get_weather(state: WorldState, city: str):
    return {"city": _city(city), **CITY_WEATHER[_city(city)]}


@tool("IATA airport code for a city.", {"city": "string"}, ["city"], False)
def get_airport_code(state: WorldState, city: str):
    return {"city": _city(city), "iata": CITY_META[_city(city)]["iata"]}


@tool("IANA timezone for a city.", {"city": "string"}, ["city"], False)
def get_timezone(state: WorldState, city: str):
    return {"city": _city(city), "tz": CITY_META[_city(city)]["tz"]}


@tool("Population in millions for a city.", {"city": "string"}, ["city"], False)
def get_population(state: WorldState, city: str):
    return {"city": _city(city), "population_m": CITY_META[_city(city)]["population_m"]}


@tool(
    "Great-circle flight time estimate in hours between two cities.",
    {"origin": "string", "destination": "string"},
    ["origin", "destination"],
    False,
)
def get_flight_time(state: WorldState, origin: str, destination: str):
    a, b = _city(origin), _city(destination)
    span = abs(CITY_META[a]["population_m"] - CITY_META[b]["population_m"])
    return {"origin": a, "destination": b, "hours": round(1.2 + span / 12.0, 2)}


# --- Domain 2: key-value store (writes, disjoint keys commute) -------------


@tool("Write a value under a key.", {"key": "string", "value": "string"}, ["key", "value"], True)
def kv_set(state: WorldState, key: str, value: str):
    state.kv[str(key)] = value
    return {"key": key, "written": True}


@tool("Read a value by key.", {"key": "string"}, ["key"], False)
def kv_get(state: WorldState, key: str):
    if str(key) not in state.kv:
        raise ToolError(f"key '{key}' not found")
    return {"key": key, "value": state.kv[str(key)]}


@tool("Delete a key.", {"key": "string"}, ["key"], True)
def kv_delete(state: WorldState, key: str):
    state.kv.pop(str(key), None)
    return {"key": key, "deleted": True}


@tool("List all stored keys.", {}, [], False)
def kv_keys(state: WorldState):
    return {"keys": sorted(state.kv)}


@tool("Add an integer amount to a numeric key, creating it at zero if absent.",
      {"key": "string", "amount": "integer"}, ["key", "amount"], True)
def kv_increment(state: WorldState, key: str, amount: int):
    current = int(state.kv.get(str(key), 0))
    state.kv[str(key)] = current + int(amount)
    return {"key": key, "value": state.kv[str(key)]}


@tool("Append a string to an existing or new string key.",
      {"key": "string", "suffix": "string"}, ["key", "suffix"], True)
def kv_append(state: WorldState, key: str, suffix: str):
    state.kv[str(key)] = str(state.kv.get(str(key), "")) + str(suffix)
    return {"key": key, "value": state.kv[str(key)]}


@tool("Copy one key to another.", {"src": "string", "dst": "string"}, ["src", "dst"], True)
def kv_copy(state: WorldState, src: str, dst: str):
    if str(src) not in state.kv:
        raise ToolError(f"key '{src}' not found")
    state.kv[str(dst)] = state.kv[str(src)]
    return {"src": src, "dst": dst, "copied": True}


@tool("Remove every key from the store.", {}, [], True)
def kv_clear(state: WorldState):
    n = len(state.kv)
    state.kv.clear()
    return {"cleared": n}


# --- Domain 3: shopping cart (writes, order-sensitive on same sku) ---------


@tool("Catalog record for a sku.", {"sku": "string"}, ["sku"], False)
def catalog_lookup(state: WorldState, sku: str):
    key = str(sku).strip().lower()
    if key not in CATALOG:
        raise ToolError(f"unknown sku '{sku}'")
    return {"sku": key, **CATALOG[key]}


@tool("Price of a sku in paise-free rupees.", {"sku": "string"}, ["sku"], False)
def catalog_price(state: WorldState, sku: str):
    return {"sku": str(sku).lower(), "price": catalog_lookup(state, sku)["price"]}


@tool("Remaining stock for a sku.", {"sku": "string"}, ["sku"], False)
def catalog_stock(state: WorldState, sku: str):
    return {"sku": str(sku).lower(), "stock": catalog_lookup(state, sku)["stock"]}


@tool("Add a quantity of a sku to the cart.",
      {"sku": "string", "qty": "integer"}, ["sku", "qty"], True)
def cart_add(state: WorldState, sku: str, qty: int):
    key = catalog_lookup(state, sku)["sku"]
    state.cart[key] = state.cart.get(key, 0) + int(qty)
    return {"sku": key, "qty": state.cart[key]}


@tool("Remove a quantity of a sku from the cart.",
      {"sku": "string", "qty": "integer"}, ["sku", "qty"], True)
def cart_remove(state: WorldState, sku: str, qty: int):
    key = str(sku).strip().lower()
    state.cart[key] = max(0, state.cart.get(key, 0) - int(qty))
    return {"sku": key, "qty": state.cart[key]}


@tool("Overwrite the quantity of a sku in the cart.",
      {"sku": "string", "qty": "integer"}, ["sku", "qty"], True)
def cart_set_qty(state: WorldState, sku: str, qty: int):
    key = catalog_lookup(state, sku)["sku"]
    state.cart[key] = int(qty)
    return {"sku": key, "qty": state.cart[key]}


@tool("Total cart value in rupees.", {}, [], False)
def cart_total(state: WorldState):
    total = sum(CATALOG[s]["price"] * q for s, q in state.cart.items() if s in CATALOG)
    return {"total": total, "lines": len(state.cart)}


@tool("Empty the cart.", {}, [], True)
def cart_clear(state: WorldState):
    n = len(state.cart)
    state.cart.clear()
    return {"cleared": n}


# --- Domain 4: file system (writes, path-scoped) ---------------------------


@tool("Create or overwrite a file.",
      {"path": "string", "content": "string"}, ["path", "content"], True)
def fs_write(state: WorldState, path: str, content: str):
    state.files[str(path)] = str(content)
    return {"path": path, "bytes": len(str(content))}


@tool("Read a file.", {"path": "string"}, ["path"], False)
def fs_read(state: WorldState, path: str):
    if str(path) not in state.files:
        raise ToolError(f"no such file '{path}'")
    return {"path": path, "content": state.files[str(path)]}


@tool("Append text to a file, creating it if absent.",
      {"path": "string", "content": "string"}, ["path", "content"], True)
def fs_append(state: WorldState, path: str, content: str):
    state.files[str(path)] = state.files.get(str(path), "") + str(content)
    return {"path": path, "bytes": len(state.files[str(path)])}


@tool("Delete a file.", {"path": "string"}, ["path"], True)
def fs_delete(state: WorldState, path: str):
    state.files.pop(str(path), None)
    return {"path": path, "deleted": True}


@tool("List all file paths.", {}, [], False)
def fs_list(state: WorldState):
    return {"paths": sorted(state.files)}


@tool("Byte length of a file.", {"path": "string"}, ["path"], False)
def fs_size(state: WorldState, path: str):
    return {"path": path, "bytes": len(fs_read(state, path)["content"])}


@tool("Copy a file.", {"src": "string", "dst": "string"}, ["src", "dst"], True)
def fs_copy(state: WorldState, src: str, dst: str):
    state.files[str(dst)] = fs_read(state, src)["content"]
    return {"src": src, "dst": dst}


@tool("Count occurrences of a substring in a file.",
      {"path": "string", "needle": "string"}, ["path", "needle"], False)
def fs_grep_count(state: WorldState, path: str, needle: str):
    return {"path": path, "count": fs_read(state, path)["content"].count(str(needle))}


# --- Domain 5: converters and arithmetic (read-only, pure) ----------------


@tool("Celsius to Fahrenheit.", {"celsius": "number"}, ["celsius"], False)
def c_to_f(state: WorldState, celsius: float):
    return {"fahrenheit": round(float(celsius) * 9 / 5 + 32, 2)}


@tool("Kilometres to miles.", {"km": "number"}, ["km"], False)
def km_to_miles(state: WorldState, km: float):
    return {"miles": round(float(km) * 0.621371, 3)}


@tool("Kilograms to pounds.", {"kg": "number"}, ["kg"], False)
def kg_to_pounds(state: WorldState, kg: float):
    return {"pounds": round(float(kg) * 2.20462, 3)}


@tool("Convert an amount between two currencies.",
      {"amount": "number", "src": "string", "dst": "string"},
      ["amount", "src", "dst"], False)
def convert_currency(state: WorldState, amount: float, src: str, dst: str):
    a, b = str(src).lower(), str(dst).lower()
    for code in (a, b):
        if code not in FX_RATES:
            raise ToolError(f"unknown currency '{code}'")
    return {"amount": round(float(amount) / FX_RATES[a] * FX_RATES[b], 2), "currency": b}


@tool("Percentage of a value.", {"value": "number", "percent": "number"},
      ["value", "percent"], False)
def percent_of(state: WorldState, value: float, percent: float):
    return {"result": round(float(value) * float(percent) / 100.0, 4)}


@tool("Arithmetic mean of a list of numbers.", {"values": "array"}, ["values"], False)
def mean(state: WorldState, values: list):
    nums = [float(v) for v in values]
    if not nums:
        raise ToolError("values must be non-empty")
    return {"mean": round(sum(nums) / len(nums), 4)}


@tool("Round a number to n decimal places.", {"value": "number", "places": "integer"},
      ["value", "places"], False)
def round_to(state: WorldState, value: float, places: int):
    return {"result": round(float(value), int(places))}


@tool("Append a short note to the run log.", {"text": "string"}, ["text"], True)
def log_note(state: WorldState, text: str):
    state.notes.append(str(text))
    return {"notes": len(state.notes)}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class FaultSpec:
    """Make one tool fail on its first invocation (A3 in the paper)."""

    tool_name: str
    kind: str = "bad_argument"  # bad_argument | rate_limit | empty_result
    fired: bool = False

    def maybe_fire(self, name: str):
        if self.fired or name != self.tool_name:
            return None
        self.fired = True
        if self.kind == "rate_limit":
            return {"error": "rate_limit", "message": "429: retry after 1s"}
        if self.kind == "empty_result":
            return {"result": None, "message": "no records matched"}
        return {"error": "bad_argument", "message": f"invalid arguments for {name}"}


def execute(state: WorldState, name: str, args: dict, fault: FaultSpec | None = None):
    """Run one call against the state. Returns (observation, ok)."""
    if fault is not None:
        injected = fault.maybe_fire(name)
        if injected is not None:
            return injected, False
    if name not in REGISTRY:
        return {"error": "unknown_tool", "message": f"no tool named '{name}'"}, False
    try:
        return REGISTRY[name].fn(state, **(args or {})), True
    except ToolError as exc:
        return {"error": "tool_error", "message": str(exc)}, False
    except TypeError as exc:
        return {"error": "bad_signature", "message": str(exc)}, False
    except Exception as exc:
        # A model that emits a malformed argument (drift mode M4) is data, not a
        # crash. Surface it to the model the way a real API would and keep going.
        return {"error": "bad_argument",
                "message": f"{type(exc).__name__}: {exc}"}, False


def execute_plan(calls: list[dict], fault: FaultSpec | None = None) -> WorldState:
    """Run a whole plan from a fresh state. Used to build reference states."""
    state = WorldState()
    for call in calls:
        execute(state, call["name"], call.get("args", {}), fault)
    return state


def tool_specs(names: list[str] | None = None) -> list[dict]:
    chosen = names or sorted(REGISTRY)
    return [REGISTRY[n].openai_spec() for n in chosen if n in REGISTRY]


assert len(REGISTRY) == 40, f"expected 40 tools, registered {len(REGISTRY)}"
