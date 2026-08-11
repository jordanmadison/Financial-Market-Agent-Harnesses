from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class LOBState:
    timestamp: float
    bid_prices: tuple[int, ...]
    bid_sizes: tuple[int, ...]
    ask_prices: tuple[int, ...]
    ask_sizes: tuple[int, ...]


@dataclass(frozen=True)
class MarketEvent:
    timestamp: float
    event_type: int
    order_id: int
    size: int
    price: int
    direction: int


@dataclass(frozen=True)
class ParentOrder:
    side: Side
    quantity: int
    decision_time: float
    expiration_time: float


@dataclass(frozen=True)
class PlacementAction:
    offset_ticks: Literal[0, 1, 2]

    def __post_init__(self) -> None:
        if self.offset_ticks not in (0, 1, 2):
            raise ValueError("offset_ticks must be 0, 1, or 2")


@dataclass(frozen=True)
class OrderOutcome:
    requested_quantity: int
    filled_quantity: int
    fill_ratio: float
    time_to_first_fill: float | None
    time_to_full_fill: float | None
    volume_weighted_fill_price: float | None
    arrival_mid_price: float
    terminal_mid_price: float
    spread_capture: float
    signed_markout: float
    fees: float
    unfilled_penalty: float
    reward: float
    execution_value: float = 0.0


@dataclass(frozen=True)
class Episode:
    episode_id: str
    ticker: str
    trading_date: str
    decision_event_index: int
    parent_order: ParentOrder
    replay_horizon_ms: int
    markout_horizon_ms: int
    dataset_id: str = ""


@dataclass(frozen=True)
class EnvironmentObservation:
    episode: Episode
    state: LOBState
    recent_events: tuple[MarketEvent, ...]


@dataclass(frozen=True)
class InferenceAttempt:
    attempt_index: int
    request_id: str | None
    returned_model: str | None
    upstream_provider: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    finish_reason: str | None
    response_length_chars: int
    latency_ms: int
    error: str | None
    api_error: dict[str, Any] | None = None


@dataclass(frozen=True)
class InferenceRecord:
    request_id: str | None
    requested_model: str
    returned_model: str | None
    upstream_provider: str | None
    agent_config_hash: str
    harness_id: str
    prompt_hash: str
    raw_response: str | None
    parsed_action: PlacementAction | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    retry_count: int
    error: str | None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    response_length_chars: int = 0
    attempts: tuple[InferenceAttempt, ...] = ()
    api_error: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentTrajectory:
    agent_observation: list[dict]
    augmented_context: dict[str, Any]
    proposed_actions: tuple[PlacementAction, ...]
    pre_action_evaluations: tuple[dict[str, Any], ...]
    final_action: PlacementAction
    inference_record: InferenceRecord
    parse_errors: tuple[str, ...]
    latency_ms: int
    token_usage: dict[str, int]


@dataclass(frozen=True)
class Experience:
    experience_id: str
    episode_id: str
    harness_id: str
    trajectory: AgentTrajectory
    outcome: OrderOutcome


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_dict(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_dict(v) for v in value]
    return value
