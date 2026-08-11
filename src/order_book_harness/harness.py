from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .domain import EnvironmentObservation, Experience, PlacementAction
from .registry import (
    FEEDBACK_METRICS,
    FEATURE_REGISTRY,
    FIELD_REGISTRY,
    FeatureName,
    MetricName,
    MutationOperation,
    compute_feature,
    experience_metric,
    harness_language_prompt,
    replace_component,
)

BookLevels = Literal[1, 3, 5, 10]
EventLookback = Literal[10, 25, 50, 100]
HistoryWindow = Literal[0, 10, 25, 50]
DecisionProcedure = Literal["direct", "analyze_then_choose", "compare_all_actions", "propose_critic_revise"]
Offset = Literal[0, 1, 2]


@dataclass(frozen=True)
class ObservationSpec:
    book_levels: BookLevels
    event_lookback: EventLookback
    include_raw_events: bool


@dataclass(frozen=True)
class AugmentationSpec:
    features: tuple[FeatureName, ...]


@dataclass(frozen=True)
class DecisionSpec:
    procedure: DecisionProcedure
    require_action_rationale: bool


@dataclass(frozen=True)
class ConstraintSpec:
    allowed_offsets: tuple[Offset, ...]
    reject_invalid_action: bool
    fallback_offset: Offset


@dataclass(frozen=True)
class FeedbackSpec:
    history_window: HistoryWindow
    metrics_shown: tuple[MetricName, ...]


@dataclass(frozen=True)
class HarnessSpec:
    observation: ObservationSpec
    augmentation: AugmentationSpec
    decision: DecisionSpec
    constraints: ConstraintSpec
    feedback: FeedbackSpec


H0 = HarnessSpec(
    ObservationSpec(1, 10, True),
    AugmentationSpec((FeatureName.SPREAD_TICKS,)),
    DecisionSpec("direct", False),
    ConstraintSpec((0, 1, 2), True, 1),
    FeedbackSpec(0, ()),
)
FEATURES = frozenset(FEATURE_REGISTRY)
METRICS = frozenset(FEEDBACK_METRICS)
COMPILER_VERSION = "1"
LANGUAGE_VERSION = "1"
HARNESS_LANGUAGE = FIELD_REGISTRY


@dataclass(frozen=True)
class HarnessMutation:
    operation: MutationOperation
    target_path: str
    value: object


def apply_harness_mutation(spec: HarnessSpec, mutation: HarnessMutation) -> HarnessSpec:
    path, operation, value = mutation.target_path, mutation.operation, mutation.value
    field_spec = FIELD_REGISTRY.get(path)
    if field_spec is None:
        raise ValueError(f"target path outside harness language: {path}")
    if field_spec.kind == "scalar":
        if operation not in field_spec.operations or type(value) not in {type(item) for item in field_spec.values} or value not in field_spec.values:
            raise ValueError(f"illegal value or operation for {path}")
        component, name = path.split(".")
        return replace_component(spec, component, name, value)
    if operation not in field_spec.operations:
        raise ValueError(f"illegal collection operation for {path}")
    allowed = set(field_spec.values)
    component, name = path.split(".")
    current = tuple(getattr(getattr(spec, component), name))
    if operation == "REPLACE":
        if not isinstance(value, dict) or set(value) != {"old", "new"} or value["old"] not in allowed or value["new"] not in allowed:
            raise ValueError("REPLACE value must contain legal old and new values")
        old_value = _coerce_collection_value(path, value["old"])
        new_value = _coerce_collection_value(path, value["new"])
        if old_value not in current or new_value in current:
            raise ValueError("REPLACE is invalid or a no-op")
        updated = tuple(new_value if item == old_value else item for item in current)
    else:
        if value not in allowed and str(value) not in {str(item) for item in allowed}:
            raise ValueError(f"illegal value for {path}")
        coerced = _coerce_collection_value(path, value)
        if operation == "ADD":
            if coerced in current:
                raise ValueError("ADD mutation is a no-op")
            updated = current + (coerced,)
        else:
            if coerced not in current:
                raise ValueError("REMOVE mutation is a no-op")
            updated = tuple(item for item in current if item != coerced)
    return replace_component(spec, component, name, updated)


def _coerce_collection_value(path: str, value: object) -> object:
    if path == "augmentation.features":
        return FeatureName(str(value))
    if path == "feedback.metrics_shown":
        return MetricName(str(value))
    return value


def canonical_spec(spec: HarnessSpec) -> str:
    return json.dumps(_canonical_dict(spec), sort_keys=True, separators=(",", ":"))


def _canonical_dict(spec: HarnessSpec) -> dict[str, Any]:
    raw = asdict(spec)
    raw["augmentation"]["features"] = [str(feature) for feature in spec.augmentation.features]
    raw["feedback"]["metrics_shown"] = [str(metric) for metric in spec.feedback.metrics_shown]
    return raw


def harness_id(spec: HarnessSpec) -> str:
    return hashlib.sha256(f"{COMPILER_VERSION}:{LANGUAGE_VERSION}:{canonical_spec(spec)}".encode()).hexdigest()[:24]


def validate_spec(spec: HarnessSpec) -> None:
    book_levels = FIELD_REGISTRY["observation.book_levels"].values
    event_lookback = FIELD_REGISTRY["observation.event_lookback"].values
    if spec.observation.book_levels not in book_levels:
        raise ValueError("unsupported observation bounds")
    if spec.observation.event_lookback not in event_lookback:
        raise ValueError("unsupported observation bounds")
    if not spec.constraints.allowed_offsets or any(offset not in (0, 1, 2) for offset in spec.constraints.allowed_offsets):
        raise ValueError("allowed_offsets must be non-empty environment actions")
    if len(set(spec.constraints.allowed_offsets)) != len(spec.constraints.allowed_offsets):
        raise ValueError("duplicate offsets")
    if spec.constraints.fallback_offset not in spec.constraints.allowed_offsets:
        raise ValueError("fallback_offset must be allowed")
    if any(feature not in FEATURES for feature in spec.augmentation.features) or len(set(spec.augmentation.features)) != len(spec.augmentation.features):
        raise ValueError("unknown or duplicate feature")
    if any(metric not in METRICS for metric in spec.feedback.metrics_shown) or len(set(spec.feedback.metrics_shown)) != len(spec.feedback.metrics_shown):
        raise ValueError("unknown or duplicate metric")
    if spec.feedback.history_window == 0 and spec.feedback.metrics_shown:
        raise ValueError("feedback metrics require a non-zero window")


class CompiledHarness:
    def __init__(self, spec: HarnessSpec):
        self.spec = spec
        self.harness_id = harness_id(spec)

    def build_messages(self, observation: EnvironmentObservation, history: tuple[Experience, ...] = (), tick_size: int = 100) -> tuple[list[dict], dict]:
        spec, state = self.spec, observation.state
        levels = min(spec.observation.book_levels, len(state.bid_prices), len(state.ask_prices))
        context: dict[str, Any] = {
            "parent_order": asdict(observation.episode.parent_order),
            "book": {
                "bid_prices": state.bid_prices[:levels],
                "bid_sizes": state.bid_sizes[:levels],
                "ask_prices": state.ask_prices[:levels],
                "ask_sizes": state.ask_sizes[:levels],
            },
            "features": {str(name): compute_feature(name, observation, levels, tick_size) for name in spec.augmentation.features},
            "allowed_offsets": spec.constraints.allowed_offsets,
        }
        if spec.observation.include_raw_events:
            context["recent_events"] = [asdict(event) for event in observation.recent_events[-spec.observation.event_lookback:]]
        if spec.feedback.history_window:
            chosen = history[-spec.feedback.history_window:]
            context["feedback"] = [{str(metric): experience_metric(experience, metric) for metric in spec.feedback.metrics_shown} for experience in chosen]
        instruction = f"Choose exactly one passive offset from {list(spec.constraints.allowed_offsets)}. Procedure: {spec.decision.procedure}. Return JSON {{\"offset_ticks\": integer}}"
        if spec.decision.require_action_rationale:
            instruction += " and optionally include a short rationale field"
        return [{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(context, separators=(",", ":"))}], context

    def enforce(self, proposed: PlacementAction | None) -> tuple[PlacementAction, tuple[str, ...]]:
        if proposed is not None and proposed.offset_ticks in self.spec.constraints.allowed_offsets:
            return proposed, ()
        if self.spec.constraints.reject_invalid_action:
            return PlacementAction(self.spec.constraints.fallback_offset), ("invalid action; fixed fallback invoked",)
        return PlacementAction(self.spec.constraints.fallback_offset), ("action coerced to fallback",)


def compile_harness(spec: HarnessSpec) -> CompiledHarness:
    validate_spec(spec)
    return CompiledHarness(spec)


# Re-export for existing imports that expected these names on the harness module.
__all__ = [
    "H0",
    "FEATURES",
    "METRICS",
    "COMPILER_VERSION",
    "LANGUAGE_VERSION",
    "HARNESS_LANGUAGE",
    "FeatureName",
    "MetricName",
    "MutationOperation",
    "ObservationSpec",
    "AugmentationSpec",
    "DecisionSpec",
    "ConstraintSpec",
    "FeedbackSpec",
    "HarnessSpec",
    "HarnessMutation",
    "CompiledHarness",
    "apply_harness_mutation",
    "canonical_spec",
    "compile_harness",
    "compute_feature",
    "harness_id",
    "harness_language_prompt",
    "validate_spec",
]
