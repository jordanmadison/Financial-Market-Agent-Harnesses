import tempfile
import unittest
from pathlib import Path

from order_book_harness.agent import HeuristicAgent
from order_book_harness.data import DatasetPair
from order_book_harness.domain import Episode, LOBState, MarketEvent, ParentOrder
from order_book_harness.environment import EnvironmentConfig, OrderBookEnvironment
from order_book_harness.episodes import LoadedDataset
from order_book_harness.factory import (
    Diagnosis,
    FactoryCallRecord,
    FactoryProposal,
    HarnessInspection,
    HarnessMutation,
    Improvement,
)
from order_book_harness.generation import GenerationInput, GenerationServices, RunAccumulator, execute_generation
from order_book_harness.harness import H0, harness_id


class RecordingFactory:
    backend_name = "fake"

    def __init__(self, mutation):
        self.mutation = mutation

    def propose(self, current_spec, evidence, prior_attempts, trajectory=None, outcome=None):
        proposed = FactoryProposal(
            Diagnosis("d", "fill_ratio", (evidence.evidence_id,), "augmentation", "cause", 0.5),
            HarnessInspection("augmentation", "current", "limit"),
            Improvement("augmentation", "rationale", "effect"),
            self.mutation,
        )
        return proposed, FactoryCallRecord("fake", "factory-v4", None, None, None, None, "", (), None, proposed, None, None, 0, 0, (), None)


class GenerationTransactionTests(unittest.TestCase):
    def _services(self, directory, mutation):
        states = (LOBState(1, (9900,), (10,), (10100,), (10,)),) * 3
        events = (MarketEvent(1, 1, 1, 10, 9900, 1), MarketEvent(2, 4, 1, 20, 9900, 1), MarketEvent(3, 1, 2, 1, 10100, -1))
        loaded = LoadedDataset(DatasetPair("AAPL", "AAPL", "2020-01-01", 1, Path("m"), Path("b")), events, states)
        episode = Episode("episode-0", "AAPL", "2020-01-01", 0, ParentOrder("BUY", 10, 1, 2.5), 1500, 1500, "AAPL")
        env = OrderBookEnvironment({"AAPL": loaded}, EnvironmentConfig(unfilled_penalty_per_share=1), max_lookback=0)
        return GenerationServices(Path(directory), HeuristicAgent(), env, RecordingFactory(mutation)), episode

    def test_promoted_generation_is_committed_only_via_accumulator(self):
        mutation = HarnessMutation("ADD", "augmentation.features", "level1_imbalance")
        with tempfile.TemporaryDirectory() as directory:
            services, episode = self._services(directory, mutation)
            accumulator = RunAccumulator(H0, {harness_id(H0): H0}, lineage_ids=(harness_id(H0),))
            before = accumulator
            result = execute_generation(services, GenerationInput(0, H0, episode, (), ()))
            self.assertEqual(before.experiences, ())
            self.assertTrue(result.promoted)
            self.assertTrue(result.record.successor_measured is False)
            committed = accumulator.apply(result)
            self.assertEqual(len(committed.experiences), 1)
            self.assertIn(result.successor_id, committed.harness_specs)
            self.assertEqual(committed.lineage_ids[-1], result.successor_id)

    def test_invalid_generation_keeps_parent_identity(self):
        mutation = HarnessMutation("SET", "observation.book_levels", 1)
        with tempfile.TemporaryDirectory() as directory:
            services, episode = self._services(directory, mutation)
            result = execute_generation(services, GenerationInput(0, H0, episode, (), ()))
        self.assertFalse(result.promoted)
        self.assertEqual(result.successor_id, harness_id(H0))
        self.assertIsNone(result.successor_attempt.successor_id)
        self.assertEqual(result.record.transition_status, "INVALID")


if __name__ == "__main__":
    unittest.main()
