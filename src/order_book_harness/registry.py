from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Callable, Literal

from .domain import EnvironmentObservation, Experience, OrderOutcome

MutationOperation = Literal["ADD", "REMOVE", "REPLACE", "SET"]


class FeatureName(StrEnum):
    SPREAD_TICKS = "spread_ticks"
    LEVEL1_IMBALANCE = "level1_imbalance"
    MULTILEVEL_IMBALANCE = "multilevel_imbalance"
    MICROPRICE = "microprice"
    RECENT_TRADE_IMBALANCE = "recent_trade_imbalance"
    CANCEL_RATE = "cancel_rate"
    SHORT_TERM_VOLATILITY = "short_term_volatility"


class MetricName(StrEnum):
    REWARD = "reward"
    FILL_RATIO = "fill_ratio"
    TIME_TO_FILL = "time_to_fill"
    SPREAD_CAPTURE = "spread_capture"
    SIGNED_MARKOUT = "signed_markout"
    FEES = "fees"
    UNFILLED_PENALTY = "unfilled_penalty"
    EXECUTION_VALUE = "execution_value"
    CONSTRAINT_FALLBACK_RATE = "constraint_fallback_rate"


def _imbalance(bid: int, ask: int) -> float:
    return (bid - ask) / (bid + ask) if bid + ask else 0.0


def _compute_spread_ticks(obs: EnvironmentObservation, levels: int, tick_size: int) -> float:
    return (obs.state.ask_prices[0] - obs.state.bid_prices[0]) / tick_size


def _compute_level1_imbalance(obs: EnvironmentObservation, levels: int, tick_size: int) -> float:
    return _imbalance(obs.state.bid_sizes[0], obs.state.ask_sizes[0])


def _compute_multilevel_imbalance(obs: EnvironmentObservation, levels: int, tick_size: int) -> float:
    return _imbalance(sum(obs.state.bid_sizes[:levels]), sum(obs.state.ask_sizes[:levels]))


def _compute_microprice(obs: EnvironmentObservation, levels: int, tick_size: int) -> float:
    total = obs.state.bid_sizes[0] + obs.state.ask_sizes[0]
    if not total:
        return 0.0
    return (obs.state.ask_prices[0] * obs.state.bid_sizes[0] + obs.state.bid_prices[0] * obs.state.ask_sizes[0]) / total


def _compute_recent_trade_imbalance(obs: EnvironmentObservation, levels: int, tick_size: int) -> float:
    trades = [event for event in obs.recent_events if event.event_type in (4, 5)]
    if not trades:
        return 0.0
    return sum(-event.direction * event.size for event in trades) / sum(event.size for event in trades)


def _compute_cancel_rate(obs: EnvironmentObservation, levels: int, tick_size: int) -> float:
    return sum(event.size for event in obs.recent_events if event.event_type in (2, 3)) / max(
        1, sum(event.size for event in obs.recent_events)
    )


def _compute_short_term_volatility(obs: EnvironmentObservation, levels: int, tick_size: int) -> float:
    prices = [event.price for event in obs.recent_events if event.event_type in (4, 5) and event.price > 0]
    if len(prices) < 2:
        return 0.0
    returns = [(b - a) / a for a, b in zip(prices, prices[1:])]
    mean = sum(returns) / len(returns)
    return math.sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))


@dataclass(frozen=True)
class FeatureDef:
    name: FeatureName
    compute: Callable[[EnvironmentObservation, int, int], float]


FEATURE_REGISTRY: dict[FeatureName, FeatureDef] = {
    FeatureName.SPREAD_TICKS: FeatureDef(FeatureName.SPREAD_TICKS, _compute_spread_ticks),
    FeatureName.LEVEL1_IMBALANCE: FeatureDef(FeatureName.LEVEL1_IMBALANCE, _compute_level1_imbalance),
    FeatureName.MULTILEVEL_IMBALANCE: FeatureDef(FeatureName.MULTILEVEL_IMBALANCE, _compute_multilevel_imbalance),
    FeatureName.MICROPRICE: FeatureDef(FeatureName.MICROPRICE, _compute_microprice),
    FeatureName.RECENT_TRADE_IMBALANCE: FeatureDef(FeatureName.RECENT_TRADE_IMBALANCE, _compute_recent_trade_imbalance),
    FeatureName.CANCEL_RATE: FeatureDef(FeatureName.CANCEL_RATE, _compute_cancel_rate),
    FeatureName.SHORT_TERM_VOLATILITY: FeatureDef(FeatureName.SHORT_TERM_VOLATILITY, _compute_short_term_volatility),
}

# Heuristic factory preference order for adding features.
FEATURE_ADD_ORDER: tuple[FeatureName, ...] = (
    FeatureName.LEVEL1_IMBALANCE,
    FeatureName.MULTILEVEL_IMBALANCE,
    FeatureName.RECENT_TRADE_IMBALANCE,
    FeatureName.CANCEL_RATE,
    FeatureName.SHORT_TERM_VOLATILITY,
    FeatureName.MICROPRICE,
)


@dataclass(frozen=True)
class MetricDef:
    name: MetricName
    description: str
    from_experience: Callable[[Experience], float | None]
    from_outcome: Callable[[OrderOutcome], float | None] | None = None
    feedback_eligible: bool = False
    evidence_eligible: bool = False


def _metric_reward(experience: Experience) -> float:
    return experience.outcome.reward


def _metric_fill_ratio(experience: Experience) -> float:
    return experience.outcome.fill_ratio


def _metric_time_to_fill(experience: Experience) -> float | None:
    return experience.outcome.time_to_full_fill


def _metric_spread_capture(experience: Experience) -> float:
    return experience.outcome.spread_capture


def _metric_signed_markout(experience: Experience) -> float:
    return experience.outcome.signed_markout


def _metric_fees(experience: Experience) -> float:
    return experience.outcome.fees


def _metric_unfilled_penalty(experience: Experience) -> float:
    return experience.outcome.unfilled_penalty


def _metric_execution_value(experience: Experience) -> float:
    return experience.outcome.execution_value


def _metric_constraint_fallback_rate(experience: Experience) -> float:
    return 1.0 if experience.trajectory.parse_errors else 0.0


METRIC_REGISTRY: dict[MetricName, MetricDef] = {
    MetricName.REWARD: MetricDef(
        MetricName.REWARD,
        "environment reward for this episode; higher is better",
        _metric_reward,
        lambda outcome: outcome.reward,
        feedback_eligible=True,
        evidence_eligible=True,
    ),
    MetricName.FILL_RATIO: MetricDef(
        MetricName.FILL_RATIO,
        "filled/requested quantity",
        _metric_fill_ratio,
        lambda outcome: outcome.fill_ratio,
        feedback_eligible=True,
        evidence_eligible=True,
    ),
    MetricName.TIME_TO_FILL: MetricDef(
        MetricName.TIME_TO_FILL,
        "seconds from decision to full fill when filled",
        _metric_time_to_fill,
        lambda outcome: outcome.time_to_full_fill,
        feedback_eligible=True,
    ),
    MetricName.SPREAD_CAPTURE: MetricDef(
        MetricName.SPREAD_CAPTURE,
        "side-adjusted arrival midpoint minus fill price",
        _metric_spread_capture,
        lambda outcome: outcome.spread_capture,
        feedback_eligible=True,
        evidence_eligible=True,
    ),
    MetricName.SIGNED_MARKOUT: MetricDef(
        MetricName.SIGNED_MARKOUT,
        "side-adjusted terminal midpoint minus fill price",
        _metric_signed_markout,
        lambda outcome: outcome.signed_markout,
        feedback_eligible=True,
        evidence_eligible=True,
    ),
    MetricName.FEES: MetricDef(
        MetricName.FEES,
        "fees charged on filled quantity",
        _metric_fees,
        lambda outcome: outcome.fees,
        evidence_eligible=True,
    ),
    MetricName.UNFILLED_PENALTY: MetricDef(
        MetricName.UNFILLED_PENALTY,
        "penalty on unfilled quantity",
        _metric_unfilled_penalty,
        lambda outcome: outcome.unfilled_penalty,
        evidence_eligible=True,
    ),
    MetricName.EXECUTION_VALUE: MetricDef(
        MetricName.EXECUTION_VALUE,
        "side-adjusted markout value of filled quantity",
        _metric_execution_value,
        lambda outcome: outcome.execution_value,
        evidence_eligible=True,
    ),
    MetricName.CONSTRAINT_FALLBACK_RATE: MetricDef(
        MetricName.CONSTRAINT_FALLBACK_RATE,
        "1 when constraint fallback was used else 0",
        _metric_constraint_fallback_rate,
        evidence_eligible=True,
    ),
}

FEEDBACK_METRICS: tuple[MetricName, ...] = tuple(
    name for name, definition in METRIC_REGISTRY.items() if definition.feedback_eligible
)
EVIDENCE_METRICS: tuple[MetricName, ...] = tuple(
    name for name, definition in METRIC_REGISTRY.items() if definition.evidence_eligible
)
AGGREGATE_METRICS: tuple[MetricName, ...] = (
    MetricName.REWARD,
    MetricName.FILL_RATIO,
    MetricName.SIGNED_MARKOUT,
    MetricName.SPREAD_CAPTURE,
    MetricName.FEES,
    MetricName.UNFILLED_PENALTY,
    MetricName.EXECUTION_VALUE,
    MetricName.CONSTRAINT_FALLBACK_RATE,
)


@dataclass(frozen=True)
class FieldDef:
    path: str
    kind: Literal["scalar", "collection"]
    operations: tuple[MutationOperation, ...]
    values: tuple[Any, ...]


FIELD_REGISTRY: dict[str, FieldDef] = {
    "observation.book_levels": FieldDef("observation.book_levels", "scalar", ("SET",), (1, 3, 5, 10)),
    "observation.event_lookback": FieldDef("observation.event_lookback", "scalar", ("SET",), (10, 25, 50, 100)),
    "observation.include_raw_events": FieldDef("observation.include_raw_events", "scalar", ("SET",), (True, False)),
    "augmentation.features": FieldDef(
        "augmentation.features",
        "collection",
        ("ADD", "REMOVE", "REPLACE"),
        tuple(sorted(FEATURE_REGISTRY)),
    ),
    "decision.procedure": FieldDef(
        "decision.procedure",
        "scalar",
        ("SET",),
        ("direct", "analyze_then_choose", "compare_all_actions", "propose_critic_revise"),
    ),
    "decision.require_action_rationale": FieldDef(
        "decision.require_action_rationale",
        "scalar",
        ("SET",),
        (True, False),
    ),
    "constraints.allowed_offsets": FieldDef(
        "constraints.allowed_offsets",
        "collection",
        ("ADD", "REMOVE", "REPLACE"),
        (0, 1, 2),
    ),
    "constraints.reject_invalid_action": FieldDef(
        "constraints.reject_invalid_action",
        "scalar",
        ("SET",),
        (True, False),
    ),
    "constraints.fallback_offset": FieldDef("constraints.fallback_offset", "scalar", ("SET",), (0, 1, 2)),
    "feedback.history_window": FieldDef("feedback.history_window", "scalar", ("SET",), (0, 10, 25, 50)),
    "feedback.metrics_shown": FieldDef(
        "feedback.metrics_shown",
        "collection",
        ("ADD", "REMOVE", "REPLACE"),
        tuple(sorted(FEEDBACK_METRICS)),
    ),
}


def harness_language_prompt() -> dict[str, dict[str, object]]:
    language: dict[str, dict[str, object]] = {}
    for path, field in FIELD_REGISTRY.items():
        values: list[Any] = [str(value) if isinstance(value, StrEnum) else value for value in field.values]
        if field.kind == "collection" and values and isinstance(values[0], str):
            values = sorted(values)
        language[path] = {"operations": list(field.operations), "values": values}
    return language


def metric_definitions_prompt() -> dict[str, str]:
    return {
        name.value: METRIC_REGISTRY[name].description
        for name in (MetricName.REWARD, MetricName.FILL_RATIO, MetricName.SIGNED_MARKOUT)
    }


def compute_feature(name: str | FeatureName, obs: EnvironmentObservation, levels: int, tick_size: int = 100) -> float:
    feature = FeatureName(name)
    return FEATURE_REGISTRY[feature].compute(obs, levels, tick_size)


def experience_metric(experience: Experience, metric: str | MetricName) -> float | None:
    return METRIC_REGISTRY[MetricName(metric)].from_experience(experience)


def parse_feature_name(value: object) -> FeatureName:
    if not isinstance(value, str):
        raise ValueError("feature name must be a string")
    try:
        return FeatureName(value)
    except ValueError as exc:
        raise ValueError(f"unknown feature: {value}") from exc


def parse_metric_name(value: object) -> MetricName:
    if not isinstance(value, str):
        raise ValueError("metric name must be a string")
    try:
        return MetricName(value)
    except ValueError as exc:
        raise ValueError(f"unknown metric: {value}") from exc


def replace_component(spec: Any, component: str, field: str, value: object) -> Any:
    return replace(spec, **{component: replace(getattr(spec, component), **{field: value})})
