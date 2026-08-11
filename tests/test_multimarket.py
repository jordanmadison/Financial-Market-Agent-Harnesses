import unittest

from order_book_harness.domain import Episode, ParentOrder
from order_book_harness.episodes import chronological_split_by_ticker
from order_book_harness.experiment import ExperimentConfig


def episode(ticker: str, index: int) -> Episode:
    timestamp = float(index)
    return Episode(f"{ticker}-{index}", ticker, "2020-01-01", index, ParentOrder("BUY", 1, timestamp, timestamp + 1), 1000, 1000, ticker)


class MultiMarketTests(unittest.TestCase):
    def test_default_execution_horizons_are_thirty_seconds(self):
        config = ExperimentConfig()
        self.assertEqual(config.replay_horizon_ms, 30_000)
        self.assertEqual(config.markout_horizon_ms, 30_000)

    def test_each_ticker_is_chronologically_represented(self):
        episodes = [episode(ticker, index) for ticker in ("A", "B") for index in range(5)]
        splits = chronological_split_by_ticker(episodes)
        for ticker in ("A", "B"):
            train = [e for e in splits["train"] if e.ticker == ticker]
            validation = [e for e in splits["validation"] if e.ticker == ticker]
            test = [e for e in splits["test"] if e.ticker == ticker]
            self.assertEqual((len(train), len(validation), len(test)), (3, 1, 1))
            self.assertLess(train[-1].parent_order.decision_time, validation[0].parent_order.decision_time)
            self.assertLess(validation[-1].parent_order.decision_time, test[0].parent_order.decision_time)

    def test_openrouter_requires_pinned_provider(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(agent="openrouter")
        config = ExperimentConfig(agent="openrouter", upstream_provider="pinned-provider")
        self.assertEqual(config.upstream_provider, "pinned-provider")

    def test_one_to_five_unique_tickers(self):
        config = ExperimentConfig(tickers=("AAPL",))
        self.assertEqual(config.tickers, ("AAPL",))
        with self.assertRaises(ValueError):
            ExperimentConfig(tickers=("A", "B", "C", "D", "D"))
        with self.assertRaises(ValueError):
            ExperimentConfig(tickers=())
        with self.assertRaises(ValueError):
            ExperimentConfig(tickers=("A", "B", "C", "D", "E", "F"))

    def test_single_ticker_split(self):
        splits = chronological_split_by_ticker([episode("AAPL", index) for index in range(5)])
        self.assertEqual({name: len(values) for name, values in splits.items()}, {"train": 3, "validation": 1, "test": 1})
        self.assertTrue(all(e.ticker == "AAPL" for values in splits.values() for e in values))


if __name__ == "__main__":
    unittest.main()
