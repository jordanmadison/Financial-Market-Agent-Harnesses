"""Source-backed harness for generate-implementation mode."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, replace
from types import MethodType
from typing import Any, Callable

from .domain import (
    EnvironmentObservation,
    Episode,
    Experience,
    LOBState,
    MarketEvent,
    OrderOutcome,
    ParentOrder,
    PlacementAction,
)

INTERFACE_VERSION = "implementation-interface-v1"
IMPLEMENTATION_COMPILER_VERSION = "implementation-compiler-v1"
FUNCTION_SIGNATURE_VERSION = "v1(self,value)"
FIXED_FUNCTIONS = ("observe", "augment", "constrain", "evaluate", "feedback")


@dataclass(frozen=True)
class FunctionImplementationSpec:
    function_name: str
    signature_version: str
    source: str
    description: str


@dataclass(frozen=True)
class HarnessImplementationSpec:
    interface_version: str
    functions: dict[str, FunctionImplementationSpec]


def _function(name: str, source: str, description: str) -> FunctionImplementationSpec:
    return FunctionImplementationSpec(name, FUNCTION_SIGNATURE_VERSION, source, description)


H0_IMPLEMENTATION = HarnessImplementationSpec(
    INTERFACE_VERSION,
    {
        "observe": _function(
            "observe",
            "def observe(self, state):\n"
            "    observation = state\n"
            "    book = observation.state\n"
            "    return {'parent_order': asdict(observation.episode.parent_order), 'book': "
            "{'bid_prices': book.bid_prices, 'bid_sizes': book.bid_sizes, "
            "'ask_prices': book.ask_prices, 'ask_sizes': book.ask_sizes}, "
            "'recent_events': [asdict(event) for event in observation.recent_events[-10:]]}\n",
            "Expose the causal top-of-book snapshot, parent order, and recent events.",
        ),
        "augment": _function(
            "augment",
            "def augment(self, observation):\n"
            "    book = observation['book']\n"
            "    bid = book['bid_prices'][0]\n"
            "    ask = book['ask_prices'][0]\n"
            "    context = dict(observation)\n"
            "    context['spread_ticks'] = (ask - bid) / self.tick_size\n"
            "    return context\n",
            "Add the causal level-one spread to the observation.",
        ),
        "constrain": _function(
            "constrain",
            "def constrain(self, context):\n"
            "    return {'allowed_offsets': [0, 1, 2], 'fallback_offset': 1}\n",
            "Expose the environment's three legal passive offsets and fixed fallback.",
        ),
        "evaluate": _function(
            "evaluate",
            "def evaluate(self, context):\n"
            "    return 'Choose exactly one legal passive offset after comparing fill and adverse-selection trade-offs.'\n",
            "Provide a direct, fixed action-comparison procedure.",
        ),
        "feedback": _function(
            "feedback",
            "def feedback(self, completed_experiences):\n"
            "    return {}\n",
            "Provide no historical feedback in the baseline harness.",
        ),
    },
)


def implementation_spec_dict(spec: HarnessImplementationSpec) -> dict[str, Any]:
    return {
        "interface_version": spec.interface_version,
        "functions": {name: asdict(spec.functions[name]) for name in sorted(spec.functions)},
    }


def canonical_implementation_spec(spec: HarnessImplementationSpec) -> str:
    return json.dumps(implementation_spec_dict(spec), sort_keys=True, separators=(",", ":"))


def implementation_harness_id(spec: HarnessImplementationSpec) -> str:
    material = f"{IMPLEMENTATION_COMPILER_VERSION}:{spec.interface_version}:{canonical_implementation_spec(spec)}"
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def load_generated_function(source: str, function_name: str) -> Callable[..., Any]:
    namespace: dict[str, Any] = {
        "asdict": asdict,
        "EnvironmentObservation": EnvironmentObservation,
        "Episode": Episode,
        "Experience": Experience,
        "LOBState": LOBState,
        "MarketEvent": MarketEvent,
        "OrderOutcome": OrderOutcome,
        "ParentOrder": ParentOrder,
        "PlacementAction": PlacementAction,
    }
    compiled = compile(source, f"<factory-generated:{function_name}>", "exec")
    exec(compiled, namespace)
    function = namespace.get(function_name)
    if not callable(function):
        raise ValueError(f"Generated source did not define callable {function_name}")
    parameters = tuple(inspect.signature(function).parameters)
    if parameters != ("self", _value_parameter(function_name)):
        raise ValueError(f"{function_name} must have fixed signature ({'self, ' + _value_parameter(function_name)})")
    return function


def _value_parameter(function_name: str) -> str:
    return {
        "observe": "state",
        "augment": "observation",
        "constrain": "context",
        "evaluate": "context",
        "feedback": "completed_experiences",
    }[function_name]


def replace_function_implementation(
    current_spec: HarnessImplementationSpec,
    implementation: FunctionImplementationSpec,
) -> HarnessImplementationSpec:
    if implementation.function_name not in FIXED_FUNCTIONS:
        raise ValueError(f"unknown fixed harness function: {implementation.function_name}")
    if implementation.signature_version != FUNCTION_SIGNATURE_VERSION:
        raise ValueError("implementation signature version does not match the fixed interface")
    functions = dict(current_spec.functions)
    functions[implementation.function_name] = implementation
    return replace(current_spec, functions=functions)


class Harness:
    """Runtime object whose five fixed methods are rebound from stored source."""

    def __init__(self, implementation_spec: HarnessImplementationSpec, tick_size: int = 100):
        self.implementation_spec = implementation_spec
        self.harness_id = implementation_harness_id(implementation_spec)
        self.tick_size = tick_size
        self._constraints: dict[str, Any] = {"allowed_offsets": [0, 1, 2], "fallback_offset": 1}

    def observe(self, state: EnvironmentObservation) -> dict[str, Any]: ...
    def augment(self, observation: dict[str, Any]) -> dict[str, Any]: ...
    def constrain(self, context: dict[str, Any]) -> dict[str, Any]: ...
    def evaluate(self, context: dict[str, Any]) -> str | dict[str, Any]: ...
    def feedback(self, completed_experiences: tuple[Experience, ...]) -> Any: ...

    def build_messages(self, state: EnvironmentObservation, history: tuple[Experience, ...] = (), tick_size: int = 100) -> tuple[list[dict], dict]:
        """Build this mode's prompt using the runner's fixed placement contract.

        ``evaluation.run_harness`` is shared by both evolution modes and expects
        a harness with ``build_messages`` and ``enforce`` methods.  This method is
        the generate-implementation implementation of that runner protocol; the
        baseline DSL continues to use ``CompiledHarness.build_messages``.

        Passive offset selection is not an evolvable harness behavior.  The
        repository's fixed decision task requires exactly one ``PlacementAction``,
        so the system instruction asks for one offset from the constraints returned
        by the active generated ``constrain`` function.  ``enforce`` remains the
        authoritative local check of the model response.
        """
        self.tick_size = tick_size
        observation = self.observe(state)
        context = self.augment(observation)
        feedback = self.feedback(history)
        constraints = self.constrain(context)
        guidance = self.evaluate(context)
        if not isinstance(context, dict) or not isinstance(constraints, dict):
            raise TypeError("augment and constrain must return dictionaries")
        allowed = constraints.get("allowed_offsets")
        fallback = constraints.get("fallback_offset")
        if not isinstance(allowed, (list, tuple)) or not allowed or any(value not in (0, 1, 2) for value in allowed):
            raise ValueError("constrain returned invalid allowed_offsets")
        if fallback not in allowed:
            raise ValueError("constrain fallback_offset must be allowed")
        self._constraints = {"allowed_offsets": tuple(allowed), "fallback_offset": fallback}
        prompt_context = {
            "harness_id": self.harness_id,
            "market_context": context,
            "feedback": feedback,
            "allowed_offsets": list(allowed),
            "evaluation_guidance": guidance,
        }
        instruction = f"Choose exactly one passive offset from {list(allowed)}. Return JSON {{\"offset_ticks\": integer}}"
        return ([{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(prompt_context, separators=(",", ":"))}], prompt_context)

    def enforce(self, proposed: PlacementAction | None) -> tuple[PlacementAction, tuple[str, ...]]:
        allowed = self._constraints["allowed_offsets"]
        if proposed is not None and proposed.offset_ticks in allowed:
            return proposed, ()
        return PlacementAction(self._constraints["fallback_offset"]), ("invalid action; fixed fallback invoked",)


def compile_implementation_harness(spec: HarnessImplementationSpec, tick_size: int = 100) -> Harness:
    if spec.interface_version != INTERFACE_VERSION or set(spec.functions) != set(FIXED_FUNCTIONS):
        raise ValueError("implementation specification must contain the complete fixed interface")
    runtime = Harness(spec, tick_size)
    for name in FIXED_FUNCTIONS:
        item = spec.functions[name]
        if item.function_name != name or item.signature_version != FUNCTION_SIGNATURE_VERSION:
            raise ValueError(f"invalid implementation metadata for {name}")
        setattr(runtime, name, MethodType(load_generated_function(item.source, name), runtime))
    return runtime


# Natural names for callers using implementation synthesis exclusively.
compile_harness = compile_implementation_harness
harness_id = implementation_harness_id
H0 = H0_IMPLEMENTATION
