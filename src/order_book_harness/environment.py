"""Replay hypothetical passive orders against historical market events."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .domain import (
    EnvironmentObservation,
    Episode,
    LOBState,
    OrderOutcome,
    PlacementAction,
)
from .episodes import LoadedDataset


@dataclass(frozen=True)
class EnvironmentConfig:
    tick_size: int = 100
    queue_model: str = "conservative"
    fee_per_share: float = 0.0
    unfilled_penalty_per_share: float = 100.0

    def __post_init__(self) -> None:
        if self.tick_size <= 0 or self.unfilled_penalty_per_share < 0:
            raise ValueError("invalid environment configuration")
        if self.queue_model not in ("conservative", "l3_fifo"):
            raise ValueError("queue_model must be conservative or l3_fifo")

    @property
    def cancellation_model(self) -> str:
        """Backward-compatible name used by older readers and reports."""
        return self.queue_model


@dataclass
class _DisplayedOrder:
    order_id: int
    price: int
    direction: int
    remaining: int
    arrival_order: int


@dataclass(frozen=True)
class QueueReplayDiagnostics:
    queue_model: str
    action_price: int
    snapshot_depth: int
    reconstructed_depth: int
    anonymous_file_boundary_depth: int
    reconstruction_excess: int
    initial_queue_ahead: int
    final_queue_ahead: int
    minimum_queue_ahead: int
    ahead_cancelled_quantity: int
    behind_removed_quantity: int
    visible_execution_quantity: int


@dataclass(frozen=True)
class ActionOutcomeSummary:
    offset_ticks: int
    fill_ratio: float
    filled_quantity: int
    time_to_first_fill: float | None
    time_to_full_fill: float | None
    reward: float


@dataclass(frozen=True)
class ActionCounterfactualReport:
    action_outcomes: tuple[ActionOutcomeSummary, ...]
    constant_outcome: bool
    outcome_differentiated: bool
    best_action_reward: float
    chosen_action_reward: float | None
    decision_regret: float | None
    available: bool
    fillable_offsets: tuple[int, ...]
    selected_action_fillable: bool
    captured: bool | None
    unavoidable_no_fill: bool
    best_available_reward: float
    best_available_fill_ratio: float


class _L3ReplayIndex:
    """Lazily reconstruct displayed L3 orders at requested event indices.

    Only type-1 submissions observed in the file create identifiable orders.
    Snapshot depth not represented by those orders is anonymous inventory: it
    may predate the file or have entered the recorded depth from outside the
    depth-limited feed. It is ahead of a hypothetical order and is never reduced
    by a cancellation that cannot be tied to an identifiable observed order.
    """

    def __init__(self, data: LoadedDataset):
        self.data = data
        self._orders: dict[int, _DisplayedOrder] = {}
        self._cursor = -1
        self._cache: dict[int, dict[int, _DisplayedOrder]] = {}

    @staticmethod
    def _copy_orders(orders: dict[int, _DisplayedOrder]) -> dict[int, _DisplayedOrder]:
        return {order_id: replace(order) for order_id, order in orders.items()}

    @staticmethod
    def apply_event(orders: dict[int, _DisplayedOrder], event, index: int) -> None:
        if event.event_type == 1 and event.order_id:
            orders[event.order_id] = _DisplayedOrder(
                event.order_id, event.price, event.direction, event.size, index
            )
            return
        if event.event_type not in (2, 3, 4) or not event.order_id:
            return
        order = orders.get(event.order_id)
        if order is None:
            return
        if event.event_type == 3 or event.size >= order.remaining:
            orders.pop(event.order_id, None)
        else:
            order.remaining -= event.size

    def orders_after(self, event_index: int) -> dict[int, _DisplayedOrder]:
        cached = self._cache.get(event_index)
        if cached is not None:
            return self._copy_orders(cached)
        if event_index < self._cursor:
            self._orders = _L3ReplayIndex(self.data)._orders
            self._cursor = -1
        for index in range(self._cursor + 1, event_index + 1):
            self.apply_event(self._orders, self.data.events[index], index)
        self._cursor = event_index
        self._cache[event_index] = self._copy_orders(self._orders)
        return self._copy_orders(self._orders)


def action_to_price(
    state: LOBState, side: str, action: PlacementAction, tick_size: int
) -> int:
    if side == "BUY":
        return state.bid_prices[0] - action.offset_ticks * tick_size
    return state.ask_prices[0] + action.offset_ticks * tick_size


class OrderBookEnvironment:
    def __init__(
        self,
        datasets: dict[str, LoadedDataset],
        config: EnvironmentConfig = EnvironmentConfig(),
        max_lookback: int = 100,
    ):
        self.datasets = datasets
        self.config = config
        self.max_lookback = max_lookback
        self._episode: Episode | None = None
        self._data: LoadedDataset | None = None
        self._l3_indices: dict[str, _L3ReplayIndex] = {}
        self.last_queue_diagnostics: QueueReplayDiagnostics | None = None

    def reset(self, episode: Episode) -> EnvironmentObservation:
        data = self.datasets[episode.dataset_id]
        i = episode.decision_event_index
        if i < self.max_lookback or i >= len(data.events):
            raise ValueError("episode index cannot support causal lookback")
        self._episode, self._data = episode, data
        return EnvironmentObservation(
            episode, data.states[i], data.events[i - self.max_lookback + 1 : i + 1]
        )

    def valid_actions(self) -> tuple[PlacementAction, ...]:
        return tuple(PlacementAction(x) for x in (0, 1, 2))

    def step(self, action: PlacementAction) -> OrderOutcome:
        if self._episode is None or self._data is None:
            raise RuntimeError("reset must be called before step")
        if self.config.queue_model == "l3_fifo":
            return self._step_l3_fifo(action)
        self.last_queue_diagnostics = None
        return self._step_conservative(action)

    def _step_conservative(self, action: PlacementAction) -> OrderOutcome:
        ep, data = self._episode, self._data
        i = ep.decision_event_index
        state = data.states[i]
        price = action_to_price(
            state, ep.parent_order.side, action, self.config.tick_size
        )
        levels = dict(
            zip(
                state.bid_prices if ep.parent_order.side == "BUY" else state.ask_prices,
                state.bid_sizes if ep.parent_order.side == "BUY" else state.ask_sizes,
            )
        )
        queue_ahead = max(0, levels.get(price, 0))
        remaining = ep.parent_order.quantity
        fills: list[tuple[float, int, int]] = []
        terminal_state = state
        halted = False
        terminal_time = ep.parent_order.decision_time + ep.markout_horizon_ms / 1000
        for j in range(i + 1, len(data.events)):
            event, after = data.events[j], data.states[j]
            if event.timestamp <= terminal_time:
                terminal_state = after
            if event.timestamp > max(ep.parent_order.expiration_time, terminal_time):
                break
            if event.event_type == 7:
                halted = True
                break
            if event.timestamp > ep.parent_order.expiration_time or remaining == 0:
                continue
            # Executions against resting buys have direction +1; against resting sells -1.
            compatible_direction = 1 if ep.parent_order.side == "BUY" else -1
            if (
                event.event_type == 4
                and event.direction == compatible_direction
                and event.price == price
            ):
                executable = event.size
                consumed = min(queue_ahead, executable)
                queue_ahead -= consumed
                executable -= consumed
                filled = min(remaining, executable)
                if filled:
                    fills.append((event.timestamp, filled, price))
                    remaining -= filled
        filled_qty = ep.parent_order.quantity - remaining
        arrival_mid = (state.bid_prices[0] + state.ask_prices[0]) / 2
        terminal_mid = (
            (terminal_state.bid_prices[0] + terminal_state.ask_prices[0]) / 2
            if terminal_state.bid_prices and terminal_state.ask_prices
            else arrival_mid
        )
        vwap = sum(q * p for _, q, p in fills) / filled_qty if filled_qty else None
        sign = 1 if ep.parent_order.side == "BUY" else -1
        execution_value = (
            filled_qty * sign * (terminal_mid - vwap) if vwap is not None else 0.0
        )
        fees = filled_qty * self.config.fee_per_share
        penalty = remaining * self.config.unfilled_penalty_per_share
        reward = execution_value - fees - penalty
        spread_capture = sign * (arrival_mid - vwap) if vwap is not None else 0.0
        markout = sign * (terminal_mid - vwap) if vwap is not None else 0.0
        first = fills[0][0] - ep.parent_order.decision_time if fills else None
        full = (
            fills[-1][0] - ep.parent_order.decision_time
            if remaining == 0 and fills
            else None
        )
        # Halts deterministically stop replay; fills before the halt remain valid.
        _ = halted
        return OrderOutcome(
            ep.parent_order.quantity,
            filled_qty,
            filled_qty / ep.parent_order.quantity,
            first,
            full,
            vwap,
            arrival_mid,
            terminal_mid,
            spread_capture,
            markout,
            fees,
            penalty,
            reward,
            execution_value,
        )

    @staticmethod
    def _consume_ahead(
        quantity: int,
        ahead_by_id: dict[int, int],
        ahead_order: list[int],
        anonymous_ahead: int,
    ) -> tuple[int, int]:
        remaining = quantity
        # Anonymous file-boundary inventory predates every observed submission.
        consumed = min(remaining, anonymous_ahead)
        anonymous_ahead -= consumed
        remaining -= consumed
        for order_id in ahead_order:
            if not remaining:
                break
            available = ahead_by_id.get(order_id, 0)
            consumed = min(remaining, available)
            ahead_by_id[order_id] = available - consumed
            remaining -= consumed
        return quantity - remaining, anonymous_ahead

    def _step_l3_fifo(self, action: PlacementAction) -> OrderOutcome:
        assert self._episode is not None and self._data is not None
        ep, data = self._episode, self._data
        i = ep.decision_event_index
        state = data.states[i]
        price = action_to_price(
            state, ep.parent_order.side, action, self.config.tick_size
        )
        direction = 1 if ep.parent_order.side == "BUY" else -1
        levels = dict(
            zip(
                state.bid_prices if ep.parent_order.side == "BUY" else state.ask_prices,
                state.bid_sizes if ep.parent_order.side == "BUY" else state.ask_sizes,
            )
        )
        snapshot_depth = max(0, levels.get(price, 0))
        if ep.dataset_id not in self._l3_indices:
            self._l3_indices[ep.dataset_id] = _L3ReplayIndex(data)
        index = self._l3_indices[ep.dataset_id]
        active = index.orders_after(i)
        at_price = sorted(
            (
                order
                for order in active.values()
                if order.price == price and order.direction == direction
            ),
            key=lambda order: (order.arrival_order, order.order_id),
        )
        reconstructed_depth = sum(order.remaining for order in at_price)
        reconstruction_excess = max(0, reconstructed_depth - snapshot_depth)
        capacity = snapshot_depth
        ahead_by_id: dict[int, int] = {}
        ahead_order: list[int] = []
        for order in at_price:
            allocated = min(capacity, order.remaining)
            if allocated:
                ahead_by_id[order.order_id] = allocated
                ahead_order.append(order.order_id)
                capacity -= allocated
        anonymous_ahead = capacity
        queue_ahead = snapshot_depth
        minimum_queue_ahead = queue_ahead
        remaining = ep.parent_order.quantity
        fills: list[tuple[float, int, int]] = []
        terminal_state = state
        terminal_time = ep.parent_order.decision_time + ep.markout_horizon_ms / 1000
        ahead_cancelled = behind_removed = visible_quantity = 0
        for j in range(i + 1, len(data.events)):
            event, after = data.events[j], data.states[j]
            if event.timestamp <= terminal_time:
                terminal_state = after
            if event.timestamp > max(ep.parent_order.expiration_time, terminal_time):
                break
            if event.event_type == 7:
                break
            if event.timestamp > ep.parent_order.expiration_time or remaining == 0:
                continue
            compatible = event.direction == direction and event.price == price
            prior = active.get(event.order_id)
            if compatible and event.event_type in (2, 3) and prior is not None:
                removable = min(event.size, prior.remaining)
                if event.order_id in ahead_by_id:
                    reduction = min(removable, ahead_by_id[event.order_id])
                    ahead_by_id[event.order_id] -= reduction
                    queue_ahead = max(0, queue_ahead - reduction)
                    ahead_cancelled += reduction
                else:
                    behind_removed += removable
            if compatible and event.event_type == 4:
                visible_quantity += event.size
                consumed, anonymous_ahead = self._consume_ahead(
                    min(queue_ahead, event.size),
                    ahead_by_id,
                    ahead_order,
                    anonymous_ahead,
                )
                queue_ahead = max(0, queue_ahead - consumed)
                executable = event.size - consumed
                filled = min(remaining, executable)
                if filled:
                    fills.append((event.timestamp, filled, price))
                    remaining -= filled
            _L3ReplayIndex.apply_event(active, event, j)
            minimum_queue_ahead = min(minimum_queue_ahead, queue_ahead)
        filled_qty = ep.parent_order.quantity - remaining
        arrival_mid = (state.bid_prices[0] + state.ask_prices[0]) / 2
        terminal_mid = (
            (terminal_state.bid_prices[0] + terminal_state.ask_prices[0]) / 2
            if terminal_state.bid_prices and terminal_state.ask_prices
            else arrival_mid
        )
        vwap = sum(q * p for _, q, p in fills) / filled_qty if filled_qty else None
        sign = 1 if ep.parent_order.side == "BUY" else -1
        execution_value = (
            filled_qty * sign * (terminal_mid - vwap) if vwap is not None else 0.0
        )
        fees = filled_qty * self.config.fee_per_share
        penalty = remaining * self.config.unfilled_penalty_per_share
        reward = execution_value - fees - penalty
        spread_capture = sign * (arrival_mid - vwap) if vwap is not None else 0.0
        markout = sign * (terminal_mid - vwap) if vwap is not None else 0.0
        first = fills[0][0] - ep.parent_order.decision_time if fills else None
        full = (
            fills[-1][0] - ep.parent_order.decision_time
            if remaining == 0 and fills
            else None
        )
        self.last_queue_diagnostics = QueueReplayDiagnostics(
            "l3_fifo",
            price,
            snapshot_depth,
            reconstructed_depth,
            snapshot_depth - min(snapshot_depth, reconstructed_depth),
            reconstruction_excess,
            snapshot_depth,
            queue_ahead,
            minimum_queue_ahead,
            ahead_cancelled,
            behind_removed,
            visible_quantity,
        )
        return OrderOutcome(
            ep.parent_order.quantity,
            filled_qty,
            filled_qty / ep.parent_order.quantity,
            first,
            full,
            vwap,
            arrival_mid,
            terminal_mid,
            spread_capture,
            markout,
            fees,
            penalty,
            reward,
            execution_value,
        )


def evaluate_action_counterfactuals(
    env: OrderBookEnvironment,
    episode: Episode,
    chosen_action: PlacementAction,
    observed_outcome: OrderOutcome | None = None,
) -> ActionCounterfactualReport:
    """Replay every legal action once and derive opportunity plus reward diagnostics."""
    summaries: list[ActionOutcomeSummary] = []
    for action in env.valid_actions():
        if (
            observed_outcome is not None
            and action.offset_ticks == chosen_action.offset_ticks
        ):
            outcome = observed_outcome
        else:
            env.reset(episode)
            outcome = env.step(action)
        summaries.append(
            ActionOutcomeSummary(
                action.offset_ticks,
                outcome.fill_ratio,
                outcome.filled_quantity,
                outcome.time_to_first_fill,
                outcome.time_to_full_fill,
                outcome.reward,
            )
        )
    action_outcomes = tuple(summaries)
    rewards = [item.reward for item in action_outcomes]
    fillable = tuple(
        item.offset_ticks for item in action_outcomes if item.fill_ratio > 0
    )
    best_reward = max(rewards)
    best_fill_ratio = max(item.fill_ratio for item in action_outcomes)
    chosen = next(
        item
        for item in action_outcomes
        if item.offset_ticks == chosen_action.offset_ticks
    )
    selected_fillable = chosen.fill_ratio > 0
    available = bool(fillable)
    return ActionCounterfactualReport(
        action_outcomes=action_outcomes,
        constant_outcome=len(set(rewards)) == 1,
        outcome_differentiated=len(set(rewards)) > 1,
        best_action_reward=best_reward,
        chosen_action_reward=chosen.reward,
        decision_regret=best_reward - chosen.reward,
        available=available,
        fillable_offsets=fillable,
        selected_action_fillable=selected_fillable,
        captured=selected_fillable if available else None,
        unavoidable_no_fill=not available and not selected_fillable,
        best_available_reward=best_reward,
        best_available_fill_ratio=best_fill_ratio,
    )
