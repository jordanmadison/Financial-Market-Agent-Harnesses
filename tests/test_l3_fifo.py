from __future__ import annotations

import unittest
from pathlib import Path

from order_book_harness.data import DatasetPair
from order_book_harness.domain import Episode, LOBState, MarketEvent, ParentOrder, PlacementAction
from order_book_harness.environment import EnvironmentConfig, OrderBookEnvironment, evaluate_action_counterfactuals
from order_book_harness.episodes import LoadedDataset


PRICE = 9_900


def state(timestamp: float, bid_size: int = 100) -> LOBState:
    return LOBState(timestamp, (PRICE,), (bid_size,), (10_100,), (100,))


def environment(events: tuple[MarketEvent, ...], decision_index: int, quantity: int = 100,
                expiration: float = 100, queue_model: str = "l3_fifo",
                states: tuple[LOBState, ...] | None = None) -> tuple[OrderBookEnvironment, Episode]:
    states = states or tuple(state(event.timestamp) for event in events)
    pair = DatasetPair("d", "X", "2020-01-01", 10, Path("m"), Path("b"))
    data = LoadedDataset(pair, events, states)
    decision_time = events[decision_index].timestamp
    episode = Episode("e", "X", "2020-01-01", decision_index,
                      ParentOrder("BUY", quantity, decision_time, expiration),
                      int((expiration - decision_time) * 1000),
                      int((expiration - decision_time) * 1000), "d")
    env = OrderBookEnvironment({"d": data}, EnvironmentConfig(queue_model=queue_model), max_lookback=0)
    return env, episode


def run(env: OrderBookEnvironment, episode: Episode):
    env.reset(episode)
    outcome = env.step(PlacementAction(0))
    diagnostics = env.last_queue_diagnostics
    if diagnostics is not None:
        assert diagnostics.minimum_queue_ahead >= 0
        assert diagnostics.final_queue_ahead >= 0
    assert outcome.filled_quantity <= outcome.requested_quantity
    return outcome, diagnostics


class L3FIFOTests(unittest.TestCase):
    def test_ahead_order_partial_cancellation_reduces_queue(self):
        events = (
            MarketEvent(1, 1, 1, 60, PRICE, 1),
            MarketEvent(2, 1, 2, 40, PRICE, 1),
            MarketEvent(3, 2, 1, 30, PRICE, 1),
            MarketEvent(4, 4, 1, 30, PRICE, 1),
            MarketEvent(5, 4, 2, 50, PRICE, 1),
        )
        env, episode = environment(events, 1, quantity=10)
        outcome, diagnostics = run(env, episode)
        self.assertEqual(outcome.filled_quantity, 10)
        self.assertEqual(diagnostics.ahead_cancelled_quantity, 30)

    def test_ahead_order_deletion_reduces_queue(self):
        events = (
            MarketEvent(1, 1, 1, 60, PRICE, 1),
            MarketEvent(2, 1, 2, 40, PRICE, 1),
            MarketEvent(3, 3, 1, 60, PRICE, 1),
            MarketEvent(4, 4, 2, 50, PRICE, 1),
        )
        env, episode = environment(events, 1, quantity=10)
        outcome, diagnostics = run(env, episode)
        self.assertEqual(outcome.filled_quantity, 10)
        self.assertEqual(diagnostics.ahead_cancelled_quantity, 60)

    def test_behind_order_cancellation_never_reduces_queue_ahead(self):
        events = (
            MarketEvent(1, 1, 1, 100, PRICE, 1),
            MarketEvent(2, 1, 9, 1, 10_100, -1),
            MarketEvent(3, 1, 2, 50, PRICE, 1),
            MarketEvent(4, 3, 2, 50, PRICE, 1),
            MarketEvent(5, 4, 1, 80, PRICE, 1),
        )
        env, episode = environment(events, 1, quantity=10)
        outcome, diagnostics = run(env, episode)
        self.assertEqual(outcome.filled_quantity, 0)
        self.assertEqual(diagnostics.behind_removed_quantity, 50)
        self.assertEqual(diagnostics.ahead_cancelled_quantity, 0)
        self.assertEqual(diagnostics.final_queue_ahead, 20)

    def test_visible_executions_cross_multiple_ahead_orders_then_partial_fill(self):
        events = (
            MarketEvent(1, 1, 1, 60, PRICE, 1),
            MarketEvent(2, 1, 2, 40, PRICE, 1),
            MarketEvent(3, 4, 1, 60, PRICE, 1),
            MarketEvent(4, 4, 2, 50, PRICE, 1),
        )
        env, episode = environment(events, 1, quantity=25)
        outcome, diagnostics = run(env, episode)
        self.assertEqual(outcome.filled_quantity, 10)
        self.assertEqual(diagnostics.final_queue_ahead, 0)

    def test_price_level_disappearance_and_reappearance_preserves_priority(self):
        events = (
            MarketEvent(1, 1, 1, 50, PRICE, 1),
            MarketEvent(2, 1, 9, 1, 10_100, -1),
            MarketEvent(3, 3, 1, 50, PRICE, 1),
            MarketEvent(4, 1, 2, 50, PRICE, 1),
            MarketEvent(5, 4, 2, 30, PRICE, 1),
        )
        states = (state(1, 50), state(2, 50), state(3, 1), state(4, 50), state(5, 20))
        env, episode = environment(events, 1, quantity=30, states=states)
        outcome, diagnostics = run(env, episode)
        self.assertEqual(outcome.filled_quantity, 30)
        self.assertEqual(diagnostics.ahead_cancelled_quantity, 50)

    def test_unknown_file_boundary_inventory_remains_conservatively_ahead(self):
        events = (
            MarketEvent(1, 1, 9, 1, 10_100, -1),
            MarketEvent(2, 2, 77, 40, PRICE, 1),
            MarketEvent(3, 5, 0, 100, PRICE, 1),
            MarketEvent(4, 4, 0, 40, PRICE, 1),
        )
        states = tuple(state(event.timestamp, 50) for event in events)
        env, episode = environment(events, 0, quantity=10, states=states)
        outcome, diagnostics = run(env, episode)
        self.assertEqual(outcome.filled_quantity, 0)
        self.assertEqual(diagnostics.anonymous_file_boundary_depth, 50)
        self.assertEqual(diagnostics.final_queue_ahead, 10)

    def test_snapshot_depth_is_authoritative_when_reconstruction_is_excessive(self):
        events = (
            MarketEvent(1, 1, 1, 100, PRICE, 1),
            MarketEvent(2, 1, 2, 50, PRICE, 1),
            MarketEvent(3, 2, 2, 50, PRICE, 1),
            MarketEvent(4, 4, 1, 80, PRICE, 1),
        )
        states = tuple(state(event.timestamp, 100) for event in events)
        env, episode = environment(events, 1, quantity=10, states=states)
        outcome, diagnostics = run(env, episode)
        self.assertEqual(diagnostics.reconstructed_depth, 150)
        self.assertEqual(diagnostics.reconstruction_excess, 50)
        self.assertEqual(diagnostics.behind_removed_quantity, 50)
        self.assertEqual(diagnostics.final_queue_ahead, 20)
        self.assertEqual(outcome.filled_quantity, 0)

    def test_hidden_execution_does_not_consume_displayed_queue(self):
        events = (
            MarketEvent(1, 1, 1, 100, PRICE, 1),
            MarketEvent(2, 1, 9, 1, 10_100, -1),
            MarketEvent(3, 5, 0, 200, PRICE, 1),
        )
        env, episode = environment(events, 1, quantity=10)
        outcome, diagnostics = run(env, episode)
        self.assertEqual(outcome.filled_quantity, 0)
        self.assertEqual(diagnostics.final_queue_ahead, 100)

    def test_halt_and_expiration_stop_fills(self):
        halt_events = (
            MarketEvent(1, 1, 1, 10, PRICE, 1),
            MarketEvent(2, 3, 1, 10, PRICE, 1),
            MarketEvent(3, 7, 0, 0, 0, 0),
            MarketEvent(4, 4, 2, 20, PRICE, 1),
        )
        env, episode = environment(halt_events, 0, quantity=10)
        self.assertEqual(run(env, episode)[0].filled_quantity, 0)

        expiration_events = (
            MarketEvent(1, 1, 1, 10, PRICE, 1),
            MarketEvent(2, 3, 1, 10, PRICE, 1),
            MarketEvent(4, 4, 2, 20, PRICE, 1),
        )
        env, episode = environment(expiration_events, 0, quantity=10, expiration=3)
        self.assertEqual(run(env, episode)[0].filled_quantity, 0)

    def test_replay_is_deterministic(self):
        events = (
            MarketEvent(1, 1, 1, 50, PRICE, 1),
            MarketEvent(2, 2, 1, 25, PRICE, 1),
            MarketEvent(3, 4, 1, 35, PRICE, 1),
        )
        env, episode = environment(events, 0, quantity=10)
        first = run(env, episode)
        second = run(env, episode)
        self.assertEqual(first, second)

    def test_conservative_model_keeps_prior_cancellation_behavior(self):
        events = (
            MarketEvent(1, 1, 1, 100, PRICE, 1),
            MarketEvent(2, 3, 1, 100, PRICE, 1),
            MarketEvent(3, 4, 2, 50, PRICE, 1),
        )
        env, episode = environment(events, 0, quantity=10, queue_model="conservative")
        outcome, diagnostics = run(env, episode)
        self.assertEqual(outcome.filled_quantity, 0)
        self.assertIsNone(diagnostics)

    def test_counterfactual_reporting_does_not_change_rewards(self):
        events = (
            MarketEvent(1, 1, 1, 10, PRICE, 1),
            MarketEvent(2, 4, 1, 20, PRICE, 1),
        )
        env, episode = environment(events, 0, quantity=10)
        report = evaluate_action_counterfactuals(env, episode, PlacementAction(0))
        self.assertTrue(hasattr(report, "constant_outcome"))
        self.assertTrue(hasattr(report, "best_action_reward"))
        self.assertEqual(report.decision_regret, report.best_action_reward - report.chosen_action_reward)


if __name__ == "__main__":
    unittest.main()
