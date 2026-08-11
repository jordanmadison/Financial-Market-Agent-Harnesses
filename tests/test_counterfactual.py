import unittest
from pathlib import Path
from unittest.mock import patch

from order_book_harness.data import DatasetPair
from order_book_harness.domain import Episode, LOBState, MarketEvent, ParentOrder, PlacementAction
from order_book_harness.environment import EnvironmentConfig, OrderBookEnvironment, evaluate_action_counterfactuals
from order_book_harness.episodes import LoadedDataset


class CounterfactualTests(unittest.TestCase):
    def _env(self):
        states = (LOBState(1, (9900,), (10,), (10100,), (10,)),) * 3
        events = (
            MarketEvent(1, 1, 1, 10, 9900, 1),
            MarketEvent(2, 4, 1, 20, 9900, 1),
            MarketEvent(3, 1, 2, 1, 10100, -1),
        )
        loaded = LoadedDataset(DatasetPair("d", "X", "2020-01-01", 1, Path("m"), Path("b")), events, states)
        episode = Episode("e", "X", "2020-01-01", 0, ParentOrder("BUY", 10, 1, 4), 3000, 3000, "d")
        env = OrderBookEnvironment({"d": loaded}, EnvironmentConfig(unfilled_penalty_per_share=1), max_lookback=0)
        return env, episode

    def test_single_pass_reuses_observed_outcome(self):
        env, episode = self._env()
        env.reset(episode)
        observed = env.step(PlacementAction(0))
        with patch.object(env, "step", wraps=env.step) as stepped:
            report = evaluate_action_counterfactuals(env, episode, PlacementAction(0), observed)
        self.assertEqual(stepped.call_count, 2)
        self.assertEqual(len(report.action_outcomes), 3)
        self.assertEqual(report.chosen_action_reward, observed.reward)
        self.assertEqual(report.decision_regret, report.best_action_reward - observed.reward)

    def test_without_observed_outcome_steps_every_action_once(self):
        env, episode = self._env()
        with patch.object(env, "step", wraps=env.step) as stepped:
            report = evaluate_action_counterfactuals(env, episode, PlacementAction(1))
        self.assertEqual(stepped.call_count, 3)
        self.assertEqual(len(report.action_outcomes), 3)
        self.assertIsNotNone(report.chosen_action_reward)


if __name__ == "__main__":
    unittest.main()
