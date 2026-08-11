import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from order_book_harness.agent import HeuristicAgent
from order_book_harness.data import DatasetPair
from order_book_harness.domain import Episode, LOBState, MarketEvent, ParentOrder
from order_book_harness.environment import (
    ActionCounterfactualReport,
    ActionOutcomeSummary,
    EnvironmentConfig,
    OrderBookEnvironment,
)
from order_book_harness.episodes import LoadedDataset
from order_book_harness.experiment import ExperimentConfig, RunContext, _render_plots
from order_book_harness.factory import (
    Diagnosis,
    Evidence,
    FactoryCallRecord,
    FactoryProposal,
    HarnessInspection,
    HarnessMutation,
    Improvement,
    SuccessorAttempt,
    apply_mutation,
    build_factory_messages,
)
from order_book_harness.generation import (
    GenerationInput,
    GenerationRecord,
    GenerationResult,
    GenerationServices,
    RunAccumulator,
    apply_promotion_rule,
    execute_generation,
)
from order_book_harness.harness import H0, harness_id


def proposal(evidence_id, mutation):
    return FactoryProposal(
        Diagnosis("diagnosis", "fill_ratio", (evidence_id,), mutation.target_path, "episode outcome", 0.5),
        HarnessInspection(mutation.target_path, "current", "limited"),
        Improvement(mutation.target_path, "operational change", "observe next episode"),
        mutation,
    )


def call_record(proposed):
    return FactoryCallRecord("fake", "factory-v5", None, None, None, "local", "prompt", (), None, proposed, 11, 13, 7, 0, (), None)


class CountingAgent:
    def __init__(self):
        self.delegate = HeuristicAgent()
        self.calls = 0

    def act(self, messages, harness_id):
        self.calls += 1
        return self.delegate.act(messages, harness_id)


class CountingFactory:
    backend_name = "fake"

    def __init__(self, mutations):
        self.mutations = list(mutations)
        self.calls = []

    def propose(self, current_spec, evidence, prior_attempts, trajectory=None, outcome=None):
        self.calls.append((current_spec, evidence, trajectory, outcome))
        proposed = proposal(evidence.evidence_id, self.mutations[len(self.calls) - 1])
        return proposed, call_record(proposed)


def context(directory, factory, agent, generation_count=3):
    states = (LOBState(1, (9900,), (10,), (10100,), (10,)),) * 6
    events = tuple(
        MarketEvent(float(index + 1), 1 if index % 2 == 0 else 4, index + 1, 20, 9900, 1)
        for index in range(6)
    )
    pair = DatasetPair("AAPL", "AAPL", "2020-01-01", 1, Path("m"), Path("b"))
    loaded = LoadedDataset(pair, events, states)
    episodes = [
        Episode(f"episode-{g}", "AAPL", "2020-01-01", g * 2, ParentOrder("BUY", 10, g * 2 + 1, g * 2 + 2.5), 1500, 1500, "AAPL")
        for g in range(generation_count)
    ]
    config = ExperimentConfig(tickers=("AAPL",), max_generations=generation_count)
    return RunContext(
        config, Path(directory), "run", [pair], episodes, episodes,
        OrderBookEnvironment({"AAPL": loaded}, EnvironmentConfig(unfilled_penalty_per_share=1), max_lookback=0),
        EnvironmentConfig(unfilled_penalty_per_share=1), agent, {}, factory, {}, {e.episode_id: "AAPL" for e in episodes},
    )


def opportunity_report(differentiated):
    rewards = (-50.0, 10.0, 100.0) if differentiated else (-10.0, -10.0, -10.0)
    return ActionCounterfactualReport(
        action_outcomes=tuple(ActionOutcomeSummary(offset, 0.0, 0, None, None, reward) for offset, reward in enumerate(rewards)),
        constant_outcome=not differentiated,
        outcome_differentiated=differentiated,
        best_action_reward=max(rewards),
        chosen_action_reward=rewards[0],
        decision_regret=max(rewards) - rewards[0],
        available=False,
        fillable_offsets=(),
        selected_action_fillable=False,
        captured=None,
        unavoidable_no_fill=True,
        best_available_reward=max(rewards),
        best_available_fill_ratio=0.0,
    )


class DirectSuccessorTests(unittest.TestCase):
    def _record(self, generation, active_harness_id, reward, status, successor, differentiated=True):
        return GenerationRecord(
            generation, "fake", "factory-v5", None, active_harness_id, f"episode-{generation}", 1, 0.0, None, None,
            0.0, 0.0, 0.0, reward, opportunity_report(differentiated), True, False, {"input": 0, "output": 0}, 1,
            {"input": 0, "output": 0}, 1, 0, status, successor, status != "PROMOTED",
            Evidence("evidence", ("exp",), {}, {}), None, None, {},
        )

    def _result(self, generation, active_harness_id, reward, differentiated):
        record = self._record(generation, active_harness_id, reward, "PROMOTED", "next-harness", differentiated)
        attempt = SuccessorAttempt("next-harness", active_harness_id, H0, (), None, {}, "PROMOTED", call_record(None))
        return GenerationResult(record, attempt, None, {}, H0, H0)

    def test_three_generations_are_one_episode_one_agent_one_factory_call(self):
        mutations = [
            HarnessMutation("ADD", "augmentation.features", "level1_imbalance"),
            HarnessMutation("SET", "decision.procedure", "compare_all_actions"),
            HarnessMutation("SET", "observation.book_levels", 3),
        ]
        agent, factory = CountingAgent(), CountingFactory(mutations)
        with tempfile.TemporaryDirectory() as directory:
            ctx = context(directory, factory, agent)
            services = GenerationServices(ctx.root, ctx.agent, ctx.env, ctx.factory_backend)
            accumulator = RunAccumulator(H0, {harness_id(H0): H0}, lineage_ids=(harness_id(H0),))
            for g, episode in enumerate(ctx.evolution_episodes):
                result = execute_generation(
                    services,
                    GenerationInput(g, accumulator.incumbent_spec, episode, accumulator.experiences, accumulator.successors),
                )
                accumulator = accumulator.apply(result)

            generations = [record.as_telemetry() for record in accumulator.generations]
            self.assertEqual(agent.calls, 3)
            self.assertEqual(len(factory.calls), 3)
            self.assertEqual(len(accumulator.experiences), 3)
            self.assertEqual([g["episode_id"] for g in generations], [e.episode_id for e in ctx.evolution_episodes])
            # Evidence aggregates a trailing window that grows until it reaches evidence_window.
            self.assertEqual([evidence.sample_count for _, evidence, _, _ in factory.calls], [1, 2, 3])
            self.assertTrue(all(len(evidence.experience_ids) == evidence.sample_count for _, evidence, _, _ in factory.calls))
            self.assertTrue(all("latest_episode" in evidence.metric_slices for _, evidence, _, _ in factory.calls))
            self.assertTrue(all(trajectory is not None and outcome is not None for _, _, trajectory, outcome in factory.calls))
            self.assertEqual(generations[0]["active_harness_id"], harness_id(H0))
            self.assertEqual(generations[1]["active_harness_id"], generations[0]["successor_harness_id"])
            self.assertTrue(all(g["transition_status"] == "PROMOTED" for g in generations))
            self.assertTrue(all("action_outcomes" in g["opportunity"] for g in generations))
            self.assertTrue(all(len(g["opportunity"]["action_outcomes"]) == 3 for g in generations))
            self.assertTrue(all("opportunity" not in evidence.aggregate_metrics for _, evidence, _, _ in factory.calls))
            self.assertNotIn(generations[-1]["successor_harness_id"], [g["active_harness_id"] for g in generations])
            progress = json.loads((ctx.root / "progress.json").read_text())
            self.assertEqual(progress["status"], "factory_complete")
            self.assertNotIn("OPENROUTER_API_KEY", json.dumps(progress))

            calls_before_plot = (agent.calls, len(factory.calls))
            with patch.object(ctx.env, "step", side_effect=AssertionError("plot must not simulate")):
                _render_plots(ctx.root, accumulator.generations)
            self.assertEqual((agent.calls, len(factory.calls)), calls_before_plot)

    def test_invalid_noop_retains_parent_and_creates_no_successor_id(self):
        agent = CountingAgent()
        factory = CountingFactory([HarnessMutation("SET", "observation.book_levels", 1)])
        with tempfile.TemporaryDirectory() as directory:
            ctx = context(directory, factory, agent, 1)
            services = GenerationServices(ctx.root, ctx.agent, ctx.env, ctx.factory_backend)
            result = execute_generation(
                services,
                GenerationInput(0, H0, ctx.evolution_episodes[0], (), ()),
            )
        self.assertFalse(result.promoted)
        self.assertEqual(result.successor_id, harness_id(H0))
        self.assertIsNone(result.successor_attempt.successor_id)
        self.assertEqual(result.successor_attempt.decision, "INVALID")

    def test_gated_rule_reverts_a_regressing_first_measurement(self):
        parent_id, successor_id = "parent-harness", "successor-harness"
        previous = self._record(0, parent_id, reward=100.0, status="PROMOTED", successor=successor_id)
        current = self._result(1, successor_id, reward=-50.0, differentiated=True)
        specs = {parent_id: H0, successor_id: apply_mutation(H0, HarnessMutation("SET", "observation.book_levels", 3))}
        gated = apply_promotion_rule(current, previous, specs, "gated")
        self.assertEqual(gated.record.transition_status, "REVERTED")
        self.assertEqual(gated.record.successor_harness_id, parent_id)
        self.assertEqual(gated.next_incumbent_spec, H0)
        self.assertIsNone(gated.new_harness_spec)
        self.assertIn("noncausal", gated.record.decision_reason)
        # A valid proposal blocked by the lineage rollback is recorded as reverted.
        self.assertEqual(gated.successor_attempt.decision, gated.record.transition_status)
        self.assertIn("lineage_rule", gated.successor_attempt.validation_results)

    def test_gated_rollback_preserves_invalid_proposal_status(self):
        parent_id, successor_id = "parent-harness", "successor-harness"
        previous = self._record(0, parent_id, reward=100.0, status="PROMOTED", successor=successor_id)
        current = self._result(1, successor_id, reward=-50.0, differentiated=True)
        current = replace(
            current,
            record=replace(current.record, transition_status="INVALID", successor_harness_id=successor_id),
            successor_attempt=replace(
                current.successor_attempt,
                successor_id=None,
                successor_spec=None,
                decision="INVALID",
                validation_results={"error": "factory output invalid"},
            ),
            next_incumbent_spec=apply_mutation(H0, HarnessMutation("SET", "observation.book_levels", 3)),
            new_harness_spec=None,
        )
        specs = {parent_id: H0, successor_id: current.next_incumbent_spec}
        gated = apply_promotion_rule(current, previous, specs, "gated")
        self.assertEqual(gated.record.transition_status, "REVERTED")
        self.assertEqual(gated.successor_attempt.decision, "INVALID")
        self.assertEqual(gated.successor_attempt.validation_results["error"], "factory output invalid")
        self.assertIn("lineage_rule", gated.successor_attempt.validation_results)

    def test_gated_rule_keeps_optimal_decisions_and_constant_outcome_episodes(self):
        parent_id, successor_id = "parent-harness", "successor-harness"
        previous = self._record(0, parent_id, reward=100.0, status="PROMOTED", successor=successor_id)
        specs = {parent_id: H0, successor_id: H0}
        optimal = self._result(1, successor_id, reward=-50.0, differentiated=True)
        optimal = replace(
            optimal,
            record=replace(
                optimal.record,
                opportunity=replace(
                    optimal.record.opportunity,
                    chosen_action_reward=optimal.record.opportunity.best_action_reward,
                    decision_regret=0.0,
                ),
            ),
        )
        kept = apply_promotion_rule(optimal, previous, specs, "gated")
        self.assertEqual(kept.record.transition_status, "PROMOTED")
        # A constant episode carries no decision signal for the rollback rule.
        constant = apply_promotion_rule(self._result(1, successor_id, reward=-50.0, differentiated=False), previous, specs, "gated")
        self.assertEqual(constant.record.transition_status, "PROMOTED")

    def test_unconditional_rule_preserves_v0_behavior(self):
        parent_id, successor_id = "parent-harness", "successor-harness"
        previous = self._record(0, parent_id, reward=100.0, status="PROMOTED", successor=successor_id)
        regressing = self._result(1, successor_id, reward=-50.0, differentiated=True)
        unchanged = apply_promotion_rule(regressing, previous, {parent_id: H0, successor_id: H0}, "unconditional")
        self.assertIs(unchanged, regressing)
        self.assertEqual(unchanged.record.transition_status, "PROMOTED")

    def test_factory_message_contains_typed_completed_experience(self):
        agent = CountingAgent()
        factory = CountingFactory([HarnessMutation("ADD", "augmentation.features", "level1_imbalance")])
        with tempfile.TemporaryDirectory() as directory:
            ctx = context(directory, factory, agent, 1)
            services = GenerationServices(ctx.root, ctx.agent, ctx.env, ctx.factory_backend)
            execute_generation(services, GenerationInput(0, H0, ctx.evolution_episodes[0], (), ()))
        _, evidence, trajectory, outcome = factory.calls[0]
        messages = build_factory_messages(H0, evidence, (), trajectory, outcome)
        self.assertIn('"completed_experience"', messages[1]["content"])
        self.assertIn('"selected_action"', messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
