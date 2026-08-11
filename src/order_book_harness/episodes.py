from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .data import DatasetPair, iter_aligned
from .domain import Episode, LOBState, MarketEvent, ParentOrder


@dataclass(frozen=True)
class EpisodeConfig:
    quantity: int = 100
    replay_horizon_ms: int = 30_000
    markout_horizon_ms: int = 30_000
    max_lookback: int = 100
    spacing_ms: int = 60_000
    max_episodes_per_dataset: int = 100


@dataclass(frozen=True)
class LoadedDataset:
    pair: DatasetPair
    events: tuple[MarketEvent, ...]
    states: tuple[LOBState, ...]


def load_dataset(pair: DatasetPair) -> LoadedDataset:
    rows = list(iter_aligned(pair))
    return LoadedDataset(pair, tuple(x[1] for x in rows), tuple(x[2] for x in rows))


def generate_episodes(data: LoadedDataset, config: EpisodeConfig = EpisodeConfig()) -> list[Episode]:
    events, states = data.events, data.states
    result: list[Episode] = []
    next_time = float("-inf")
    required_seconds = max(config.replay_horizon_ms, config.markout_horizon_ms) / 1000
    last_time = events[-1].timestamp if events else 0
    for i in range(config.max_lookback, len(events)):
        state, event = states[i], events[i]
        if event.timestamp < next_time or event.timestamp + required_seconds > last_time:
            continue
        if not state.bid_prices or not state.ask_prices or state.bid_prices[0] >= state.ask_prices[0]:
            continue
        # Side is stable and independent of future outcomes.
        digest = hashlib.sha256(f"{data.pair.dataset_id}:{i}:side".encode()).digest()
        side = "BUY" if digest[0] % 2 == 0 else "SELL"
        expiration = event.timestamp + config.replay_horizon_ms / 1000
        episode_id = hashlib.sha256(f"{data.pair.dataset_id}:{i}:{side}:{config.quantity}:{config.replay_horizon_ms}:{config.markout_horizon_ms}".encode()).hexdigest()[:24]
        result.append(Episode(episode_id, data.pair.ticker, data.pair.trading_date, i, ParentOrder(side, config.quantity, event.timestamp, expiration), config.replay_horizon_ms, config.markout_horizon_ms, data.pair.dataset_id))
        next_time = event.timestamp + config.spacing_ms / 1000
        if len(result) >= config.max_episodes_per_dataset:
            break
    return result


def chronological_split(episodes: list[Episode], train_fraction: float = .6, validation_fraction: float = .2) -> dict[str, list[Episode]]:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave non-empty positive test fraction")
    ordered = sorted(episodes, key=lambda e: (e.trading_date, e.parent_order.decision_time, e.ticker, e.episode_id))
    n = len(ordered); train_end = int(n * train_fraction); validation_end = train_end + int(n * validation_fraction)
    return {"train": ordered[:train_end], "validation": ordered[train_end:validation_end], "test": ordered[validation_end:]}


def chronological_split_by_ticker(episodes: list[Episode], train_fraction: float = .6, validation_fraction: float = .2) -> dict[str, list[Episode]]:
    """Split each instrument independently, then pool like partitions.

    This prevents a high-activity instrument from occupying an entire global split
    and preserves early/middle/late chronology within every instrument.
    """
    tickers = sorted({episode.ticker for episode in episodes})
    pooled: dict[str, list[Episode]] = {"train": [], "validation": [], "test": []}
    for ticker in tickers:
        ticker_splits = chronological_split(
            [episode for episode in episodes if episode.ticker == ticker],
            train_fraction,
            validation_fraction,
        )
        if not all(ticker_splits.values()):
            raise ValueError(f"not enough episodes to split ticker {ticker}")
        for name, values in ticker_splits.items():
            pooled[name].extend(values)
    for values in pooled.values():
        values.sort(key=lambda episode: (episode.parent_order.decision_time, episode.ticker, episode.episode_id))
    return pooled
