from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .domain import LOBState, MarketEvent

SENTINEL_ASK = 9_999_999_999
SENTINEL_BID = -9_999_999_999
NAME_RE = re.compile(r"(?P<ticker>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_.*_(?P<kind>message|orderbook)_(?P<levels>\d+)\.csv$")


@dataclass(frozen=True)
class DatasetPair:
    dataset_id: str
    ticker: str
    trading_date: str
    levels: int
    message_path: Path
    orderbook_path: Path


def discover_datasets(root: str | Path) -> list[DatasetPair]:
    root = Path(root)
    pairs: list[DatasetPair] = []
    for message in sorted(root.rglob("*message_*.csv")):
        match = NAME_RE.match(message.name)
        if not match:
            continue
        orderbook = message.with_name(message.name.replace("_message_", "_orderbook_"))
        if not orderbook.exists():
            continue
        rel = message.relative_to(root).as_posix()
        pairs.append(DatasetPair(hashlib.sha256(rel.encode()).hexdigest()[:16], match["ticker"], match["date"], int(match["levels"]), message, orderbook))
    return pairs


def parse_event(row: list[str]) -> MarketEvent:
    if len(row) != 6:
        raise ValueError(f"message row has {len(row)} columns, expected 6")
    return MarketEvent(float(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))


def parse_state(row: list[str], timestamp: float) -> LOBState:
    if not row or len(row) % 4:
        raise ValueError("order-book row must have 4 columns per level")
    asks_p, asks_s, bids_p, bids_s = [], [], [], []
    for i in range(0, len(row), 4):
        ap, av, bp, bv = map(int, row[i:i + 4])
        if ap != SENTINEL_ASK and av > 0:
            asks_p.append(ap); asks_s.append(av)
        if bp != SENTINEL_BID and bv > 0:
            bids_p.append(bp); bids_s.append(bv)
    return LOBState(timestamp, tuple(bids_p), tuple(bids_s), tuple(asks_p), tuple(asks_s))


def iter_aligned(pair: DatasetPair) -> Iterator[tuple[int, MarketEvent, LOBState]]:
    with pair.message_path.open(newline="") as mf, pair.orderbook_path.open(newline="") as bf:
        messages, books = csv.reader(mf), csv.reader(bf)
        index = 0
        while True:
            mr, br = next(messages, None), next(books, None)
            if mr is None or br is None:
                if mr is not None or br is not None:
                    raise ValueError(f"row-count mismatch in {pair.dataset_id}")
                return
            event = parse_event(mr)
            yield index, event, parse_state(br, event.timestamp)
            index += 1


def validate_dataset(pair: DatasetPair, full: bool = False, sample_rows: int = 10_000) -> dict:
    count, previous = 0, float("-inf")
    event_types: set[int] = set()
    for _, event, state in iter_aligned(pair):
        if event.timestamp < previous:
            raise ValueError("timestamps are not non-decreasing")
        if event.event_type not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError(f"unexpected event type {event.event_type}")
        if event.event_type != 7 and event.direction not in (-1, 1):
            raise ValueError(f"unexpected direction {event.direction}")
        if state.bid_prices and state.ask_prices and state.bid_prices[0] >= state.ask_prices[0]:
            raise ValueError(f"crossed book at row {count}")
        previous = event.timestamp
        event_types.add(event.event_type); count += 1
        if not full and count >= sample_rows:
            break
    return {"dataset_id": pair.dataset_id, "ticker": pair.ticker, "date": pair.trading_date, "levels": pair.levels, "rows_checked": count, "event_types": sorted(event_types)}
