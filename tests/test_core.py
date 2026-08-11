import json
import tempfile
import unittest
from pathlib import Path

from order_book_harness.agent import parse_placement_action
from order_book_harness.data import DatasetPair, iter_aligned
from order_book_harness.domain import Episode, LOBState, MarketEvent, ParentOrder, PlacementAction
from order_book_harness.environment import EnvironmentConfig, OrderBookEnvironment, action_to_price
from order_book_harness.episodes import LoadedDataset
from order_book_harness.harness import H0, compile_harness, harness_id, validate_spec


class CoreTests(unittest.TestCase):
    def test_action_mapping_symmetry(self):
        s = LOBState(1, (9900,), (5,), (10100,), (6,))
        self.assertEqual(action_to_price(s, "BUY", PlacementAction(2), 100), 9700)
        self.assertEqual(action_to_price(s, "SELL", PlacementAction(2), 100), 10300)

    def test_queue_fill_and_reward(self):
        states = (LOBState(1, (9900,), (10,), (10100,), (10,)),) * 3
        events = (MarketEvent(1, 1, 1, 10, 9900, 1), MarketEvent(2, 4, 1, 12, 9900, 1), MarketEvent(3, 4, 2, 8, 9900, 1))
        data = LoadedDataset(DatasetPair("d", "X", "2020-01-01", 1, Path("m"), Path("b")), events, states)
        ep = Episode("e", "X", "2020-01-01", 0, ParentOrder("BUY", 10, 1, 4), 3000, 3000, "d")
        env = OrderBookEnvironment({"d": data}, EnvironmentConfig(tick_size=100, unfilled_penalty_per_share=1), max_lookback=0)
        env.reset(ep); out = env.step(PlacementAction(0))
        self.assertEqual(out.filled_quantity, 10); self.assertEqual(out.reward, 1000)

    def test_compiler_stable_and_parser_closed(self):
        self.assertEqual(harness_id(H0), compile_harness(H0).harness_id)
        self.assertEqual(parse_placement_action('{"offset_ticks":2}'), PlacementAction(2))
        with self.assertRaises(ValueError): parse_placement_action('{"offset_ticks":true}')

    def test_features_use_the_configured_tick_size(self):
        states = (LOBState(1, (9900,), (10,), (10100,), (10,)),) * 3
        events = (MarketEvent(1, 1, 1, 10, 9900, 1),) * 3
        data = LoadedDataset(DatasetPair("d", "X", "2020-01-01", 1, Path("m"), Path("b")), events, states)
        ep = Episode("e", "X", "2020-01-01", 0, ParentOrder("BUY", 10, 1, 4), 3000, 3000, "d")
        env = OrderBookEnvironment({"d": data}, EnvironmentConfig(tick_size=50), max_lookback=0)
        harness = compile_harness(H0)
        observation = env.reset(ep)
        # spread is 200 price units; ticks depend on the environment's tick size.
        _, default_context = harness.build_messages(observation)
        self.assertEqual(default_context["features"]["spread_ticks"], 2.0)
        _, context = harness.build_messages(observation, (), env.config.tick_size)
        self.assertEqual(context["features"]["spread_ticks"], 4.0)

    def test_aligned_row_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            m, b = Path(td) / "m.csv", Path(td) / "b.csv"
            m.write_text("1,1,1,1,9900,1\n2,1,2,1,9900,1\n"); b.write_text("10100,1,9900,1\n")
            pair = DatasetPair("d", "X", "x", 1, m, b)
            with self.assertRaises(ValueError): list(iter_aligned(pair))


if __name__ == "__main__": unittest.main()
