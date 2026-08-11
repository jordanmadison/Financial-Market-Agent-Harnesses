import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error
from dataclasses import asdict
from pathlib import Path

from order_book_harness.agent import HeuristicAgent
from order_book_harness.data import DatasetPair
from order_book_harness.domain import Episode, LOBState, MarketEvent, ParentOrder
from order_book_harness.environment import EnvironmentConfig, OrderBookEnvironment
from order_book_harness.episodes import LoadedDataset
from order_book_harness.evaluation import run_harness
from order_book_harness.experiment import ExperimentConfig, build_lineage, run_experiment
from order_book_harness.factory import (
    FACTORY_PROMPT_VERSION,
    Diagnosis,
    Evidence,
    FactoryCallRecord,
    FactoryModelConfig,
    FactoryProposal,
    HarnessInspection,
    HarnessMutation,
    HeuristicFactory,
    Improvement,
    LLMFactory,
    SuccessorAttempt,
    allowed_evidence_refs,
    apply_mutation,
    build_factory_messages,
    validate_factory_proposal,
)
from order_book_harness.factory.evidence import canonical_mutation, inverse_mutation
from order_book_harness.factory.heuristic import LADDER_EXHAUSTED
from order_book_harness.factory.validation import OSCILLATION_COOLDOWN_ATTEMPTS
from order_book_harness.harness import H0, canonical_spec, compile_harness, harness_id
from order_book_harness.registry import FEATURE_ADD_ORDER


class FakeClient:
    def __init__(self, results):
        self.results = list(results); self.calls = []

    def complete(self, payload, timeout):
        self.calls.append((payload, timeout)); value = self.results.pop(0)
        if isinstance(value, BaseException): raise value
        return value


def proposal_dict(path="augmentation.features", value="level1_imbalance", operation="ADD", evidence_ref="evidence-g0"):
    return {
        "diagnosis": {"affected_metric": "fill_ratio", "evidence_refs": [evidence_ref], "suspected_harness_component": "augmentation", "suspected_cause": "limited queue context", "confidence": .8},
        "inspection": {"component": "augmentation", "current_behavior": "spread only", "limitation": "no imbalance"},
        "improvement": {"component": "augmentation", "rationale": "add causal pressure", "expected_metric_effect": "increase fill_ratio"},
        "mutation": {"operation": operation, "target_path": path, "value": value},
    }


def response(value):
    return {"id": "factory-request", "model": "deepseek/deepseek-v4-flash-0731", "provider": "DeepInfra", "choices": [{"message": {"content": value}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 200}}


def evidence():
    return Evidence("evidence-g0", ("experience-1",), {"reward": -1000, "fill_ratio": 0.0, "signed_markout": 0.0}, {"AAPL": {"fill_ratio": 0.0}})


class FactoryTests(unittest.TestCase):
    def make_factory(self, content):
        client = FakeClient([response(content)])
        return LLMFactory(FactoryModelConfig(upstream_provider="DeepInfra"), client), client

    def test_valid_evidence_linked_mutation(self):
        factory, _ = self.make_factory(json.dumps(proposal_dict()))
        proposal, record = factory.propose(H0, evidence(), ())
        self.assertIsNotNone(proposal); self.assertIsNone(record.error)
        self.assertEqual(proposal.diagnosis.evidence_refs, ("evidence-g0",))
        self.assertEqual(apply_mutation(H0, proposal.mutation).augmentation.features, ("spread_ticks", "level1_imbalance"))

    def test_unknown_evidence_reference(self):
        factory, _ = self.make_factory(json.dumps(proposal_dict(evidence_ref="invented")))
        proposal, record = factory.propose(H0, evidence(), ())
        self.assertIsNone(proposal); self.assertIn("unknown evidence", record.error)

    def test_explicit_aggregate_and_slice_evidence_references(self):
        refs = allowed_evidence_refs(evidence())
        self.assertIn("evidence-g0:aggregate:fill_ratio", refs)
        self.assertIn("evidence-g0:slice:AAPL:fill_ratio", refs)
        for reference in ("evidence-g0:aggregate:fill_ratio", "evidence-g0:slice:AAPL:fill_ratio"):
            with self.subTest(reference=reference):
                factory, _ = self.make_factory(json.dumps(proposal_dict(evidence_ref=reference)))
                proposal, record = factory.propose(H0, evidence(), ())
                self.assertIsNotNone(proposal); self.assertIsNone(record.error)
        messages = build_factory_messages(H0, evidence(), ())
        supplied = json.loads(messages[1]["content"])
        self.assertEqual(supplied["allowed_evidence_refs"], sorted(refs))
        self.assertIn("exact strings copied verbatim", messages[0]["content"])

    def test_model_constructed_metric_expression_is_not_a_reference(self):
        factory, _ = self.make_factory(json.dumps(proposal_dict(evidence_ref="metric_slices.AAPL.fill_ratio=0.0")))
        proposal, record = factory.propose(H0, evidence(), ())
        self.assertIsNone(proposal); self.assertIn("unknown evidence", record.error)

    def test_unknown_and_fixed_target_paths(self):
        for path in ("unknown.field", "model.model_id", "reward.lambda", "environment.tick_size"):
            with self.subTest(path=path):
                factory, _ = self.make_factory(json.dumps(proposal_dict(path=path, value="x", operation="SET")))
                proposal, record = factory.propose(H0, evidence(), ())
                self.assertIsNone(proposal); self.assertIn("outside harness language", record.error)

    def test_illegal_field_value(self):
        factory, _ = self.make_factory(json.dumps(proposal_dict(path="observation.book_levels", value=7, operation="SET")))
        proposal, record = factory.propose(H0, evidence(), ())
        self.assertIsNone(proposal); self.assertIn("illegal value", record.error)

    def test_noop_mutation(self):
        factory, _ = self.make_factory(json.dumps(proposal_dict(path="observation.book_levels", value=1, operation="SET")))
        proposal, record = factory.propose(H0, evidence(), ())
        self.assertIsNone(proposal); self.assertIn("no-op", record.error)

    def test_malformed_json_and_missing_fields(self):
        for content in ("not-json", json.dumps({"mutation": {}})):
            with self.subTest(content=content):
                factory, client = self.make_factory(content)
                proposal, record = factory.propose(H0, evidence(), ())
                self.assertIsNone(proposal); self.assertIsNotNone(record.error)
                self.assertEqual(len(client.calls), 1)

    def test_exhausted_transient_retries_no_heuristic_fallback(self):
        client = FakeClient([urllib.error.URLError("down")] * 3)
        factory = LLMFactory(FactoryModelConfig(upstream_provider="DeepInfra", max_retries=2), client)
        proposal, record = factory.propose(H0, evidence(), ())
        self.assertIsNone(proposal); self.assertEqual(len(client.calls), 3)
        self.assertEqual(record.retry_count, 2); self.assertEqual(len(record.attempts), 3)
        self.assertNotIsInstance(factory, HeuristicFactory)

    def test_llm_emits_at_most_one_candidate(self):
        factory, client = self.make_factory(json.dumps(proposal_dict()))
        proposal, _ = factory.propose(H0, evidence(), ())
        self.assertIsNotNone(proposal); self.assertEqual(len(client.calls), 1)

    def test_returned_model_or_provider_substitution_fails_closed(self):
        for field, value in (("model", "other/model"), ("provider", "OtherProvider")):
            with self.subTest(field=field):
                reply = response(json.dumps(proposal_dict())); reply[field] = value
                client = FakeClient([reply]); factory = LLMFactory(FactoryModelConfig(upstream_provider="DeepInfra"), client)
                proposal, record = factory.propose(H0, evidence(), ())
                self.assertIsNone(proposal); self.assertIn("unexpected", record.error)

    def test_deterministic_application_and_compilation(self):
        mutation = HarnessMutation("SET", "decision.procedure", "compare_all_actions")
        first, second = apply_mutation(H0, mutation), apply_mutation(H0, mutation)
        self.assertEqual(first, second); self.assertEqual(compile_harness(first).spec, first)

    def test_confidence_bounds_are_enforced(self):
        for value in (80, "0.8", 1.5, -0.1):
            with self.subTest(value=value):
                body = proposal_dict()
                body["diagnosis"]["confidence"] = value
                factory, _ = self.make_factory(json.dumps(body))
                proposal, record = factory.propose(H0, evidence(), ())
                self.assertIsNone(proposal)
                self.assertIn("confidence", record.error)

    def test_factory_prompt_states_confidence_format(self):
        prompt = build_factory_messages(H0, evidence(), ())[0]["content"]
        self.assertIn("between 0 and 1", prompt)

    def test_backend_selection_and_api_key_exclusion(self):
        self.assertEqual(ExperimentConfig().factory_backend, "heuristic")
        config = ExperimentConfig(factory_backend="llm", factory_upstream_provider="DeepInfra")
        self.assertEqual(config.factory_backend, "llm")
        factory_config = FactoryModelConfig(upstream_provider="DeepInfra")
        self.assertEqual(factory_config.max_output_tokens, 16_384)
        self.assertNotIn("api_key", json.dumps(asdict(factory_config)).lower())
        messages = LLMFactory(factory_config, FakeClient([]))
        self.assertFalse(hasattr(messages, "api_key"))

    def test_run_experiment_consumes_generation_state_mapping(self):
        config = ExperimentConfig(max_generations=1)
        ctx = SimpleNamespace(
            root=Path("mock-run"),
            run_id="mock-run",
            agent_manifest={},
            factory_manifest={},
            evolution_episodes=[SimpleNamespace(episode_id="e0")],
            config=config,
            env_config=SimpleNamespace(),
            pairs=[],
            episodes=[],
            agent=object(),
            env=object(),
            factory_backend=SimpleNamespace(backend_name="heuristic"),
        )
        from order_book_harness.generation import GenerationResult, GenerationRecord
        from order_book_harness.environment import ActionCounterfactualReport, ActionOutcomeSummary
        from order_book_harness.factory import Evidence, SuccessorAttempt, FactoryCallRecord
        from order_book_harness.domain import Experience, AgentTrajectory, InferenceRecord, OrderOutcome, PlacementAction

        opportunity = ActionCounterfactualReport(
            action_outcomes=(ActionOutcomeSummary(0, 0.0, 0, None, None, 0.0),),
            constant_outcome=True,
            outcome_differentiated=False,
            best_action_reward=0.0,
            chosen_action_reward=0.0,
            decision_regret=0.0,
            available=False,
            fillable_offsets=(),
            selected_action_fillable=False,
            captured=None,
            unavoidable_no_fill=True,
            best_available_reward=0.0,
            best_available_fill_ratio=0.0,
        )
        evidence = Evidence("evidence-g0", ("exp",), {"reward": 0.0}, {})
        factory_call = FactoryCallRecord("heuristic", "factory-v4", None, None, None, None, "", (), None, None, None, None, 0, 0, (), None)
        attempt = SuccessorAttempt(None, harness_id(H0), None, (), None, {}, "INVALID", factory_call)
        outcome = OrderOutcome(1, 0, 0.0, None, None, None, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        trajectory = AgentTrajectory([], {}, (), (), PlacementAction(1), InferenceRecord(None, "m", "m", "local", "h", "h", "p", None, PlacementAction(1), None, None, 0, 0, None), (), 0, {})
        experience = Experience("exp", "e0", harness_id(H0), trajectory, outcome)
        record = GenerationRecord(
            0, "heuristic", "factory-v4", None, harness_id(H0), "e0", 1, 0.0, None, None, 0.0, 0.0, 0.0, 0.0,
            opportunity, True, False, {}, 0, {}, 0, 0, "INVALID", harness_id(H0), True, evidence, None, None, {},
        )
        generation_result = GenerationResult(record, attempt, experience, {"generation": 0}, H0, None)
        with (
            patch("order_book_harness.experiment._setup_run", return_value=ctx),
            patch("order_book_harness.experiment.write_json"),
            patch("order_book_harness.experiment.execute_generation", return_value=generation_result),
            patch("order_book_harness.experiment.persist_run_artifacts"),
        ):
            self.assertEqual(run_experiment(config), ctx.root)

    def test_lineage_promoted_and_invalid_transitions(self):
        h1 = "h1"
        generations = [
            {"generation": 0, "active_harness_id": harness_id(H0), "successor_harness_id": h1, "transition_status": "PROMOTED"},
            {"generation": 1, "active_harness_id": h1, "successor_harness_id": h1, "transition_status": "INVALID"},
        ]
        lineage = build_lineage([harness_id(H0), h1], generations)
        self.assertEqual([t["status"] for t in lineage["transitions"]], ["PROMOTED", "INVALID"])
        self.assertEqual(lineage["final_harness_id"], h1)
        self.assertEqual(lineage["promoted_harness_ids"], [h1])

    def test_promoted_mutation_persists_in_lineage(self):
        states = (LOBState(1, (9900,), (10,), (10100,), (1,)),) * 3
        events = (MarketEvent(1, 1, 1, 10, 9900, 1), MarketEvent(2, 4, 1, 20, 9900, 1), MarketEvent(3, 1, 2, 1, 10100, -1))
        loaded = LoadedDataset(DatasetPair("AAPL", "AAPL", "2020-01-01", 1, Path("m"), Path("b")), events, states)
        episode = Episode("episode", "AAPL", "2020-01-01", 0, ParentOrder("BUY", 10, 1, 4), 3000, 3000, "AAPL")
        env = OrderBookEnvironment({"AAPL": loaded}, EnvironmentConfig(unfilled_penalty_per_share=1), max_lookback=0)
        agent = HeuristicAgent(); incumbent = run_harness(agent, compile_harness(H0), env, [episode])
        objective = Evidence("evidence-g0", tuple(e.experience_id for e in incumbent.experiences), incumbent.aggregate_metrics, {"AAPL": incumbent.aggregate_metrics})
        factory, _ = self.make_factory(json.dumps(proposal_dict()))
        proposal, call = factory.propose(H0, objective, ())
        successor_spec = apply_mutation(H0, proposal.mutation)
        successor = run_harness(agent, compile_harness(successor_spec), env, [episode])
        self.assertEqual(incumbent.episode_ids, successor.episode_ids)
        lineage = build_lineage(
            [harness_id(H0), harness_id(successor_spec)],
            [{"generation": 0, "active_harness_id": harness_id(H0), "successor_harness_id": harness_id(successor_spec), "transition_status": "PROMOTED"}],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lineage.json"; path.write_text(json.dumps(lineage)); persisted = json.loads(path.read_text())
        self.assertEqual(persisted["final_harness_id"], harness_id(successor_spec))
        self.assertIsNotNone(call)


class HeuristicLadderTests(unittest.TestCase):
    """The deterministic baseline must exhaust cleanly instead of repeating no-ops."""

    LEGACY_RUNGS = 10  # features, decision.procedure, feedback.history_window, book_levels 3 then 5

    def walk(self, limit=200):
        """Apply heuristic proposals until exhaustion; return the mutations and the final record."""
        factory, spec, mutations = HeuristicFactory(), H0, []
        for generation in range(limit):
            objective = Evidence(f"evidence-g{generation}", ("experience-1",), {"reward": -1000, "fill_ratio": 0.0, "signed_markout": 0.0}, {})
            proposal, record = factory.propose(spec, objective, ())
            if proposal is None:
                return spec, mutations, record
            candidate = apply_mutation(spec, proposal.mutation)
            self.assertNotEqual(canonical_spec(candidate), canonical_spec(spec))
            compile_harness(candidate)
            spec, _ = candidate, mutations.append(proposal.mutation)
        self.fail("heuristic ladder did not terminate")

    def test_ladder_continues_past_the_legacy_rungs(self):
        _, mutations, _ = self.walk()
        self.assertGreater(len(mutations), self.LEGACY_RUNGS)
        self.assertEqual(
            [canonical_mutation(mutation) for mutation in mutations[: self.LEGACY_RUNGS]],
            [
                canonical_mutation(HarnessMutation("ADD", "augmentation.features", feature.value))
                for feature in FEATURE_ADD_ORDER
            ]
            + [
                canonical_mutation(HarnessMutation("SET", "decision.procedure", "compare_all_actions")),
                canonical_mutation(HarnessMutation("SET", "feedback.history_window", 10)),
                canonical_mutation(HarnessMutation("SET", "observation.book_levels", 3)),
                canonical_mutation(HarnessMutation("SET", "observation.book_levels", 5)),
            ],
        )

    def test_ladder_never_repeats_a_mutation_or_a_spec(self):
        _, mutations, _ = self.walk()
        identities = [canonical_mutation(mutation) for mutation in mutations]
        self.assertEqual(len(identities), len(set(identities)))
        specs = [canonical_spec(H0)]
        spec = H0
        for mutation in mutations:
            spec = apply_mutation(spec, mutation)
            specs.append(canonical_spec(spec))
        self.assertEqual(len(specs), len(set(specs)))

    def test_exhaustion_is_a_distinct_terminal_condition(self):
        exhausted, _, record = self.walk()
        self.assertIsNone(record.parsed_proposal)
        self.assertTrue(record.error.startswith(LADDER_EXHAUSTED), record.error)
        self.assertNotIn("mutation is a no-op", record.error)
        for generation in range(3):
            objective = Evidence(f"evidence-g{generation}", ("experience-1",), {"reward": -1000, "fill_ratio": 0.0, "signed_markout": 0.0}, {})
            proposal, repeated = HeuristicFactory().propose(exhausted, objective, ())
            self.assertIsNone(proposal)
            self.assertTrue(repeated.error.startswith(LADDER_EXHAUSTED), repeated.error)

    def test_proposals_stay_deterministic(self):
        first_spec, first, _ = self.walk()
        second_spec, second, _ = self.walk()
        self.assertEqual([canonical_mutation(m) for m in first], [canonical_mutation(m) for m in second])
        self.assertEqual(canonical_spec(first_spec), canonical_spec(second_spec))


class OscillationGuardTests(unittest.TestCase):
    def _promoted(self, mutation):
        call = FactoryCallRecord("fake", FACTORY_PROMPT_VERSION, None, None, None, None, "", (), None, None, None, None, 0, 0, (), None)
        return SuccessorAttempt("successor", "parent", None, ("evidence-g0",), mutation, {}, "PROMOTED", call)

    def _undo_feature(self):
        return FactoryProposal(
            Diagnosis("d", "fill_ratio", ("evidence-g0",), "augmentation", "feature unhelpful", 0.5),
            HarnessInspection("augmentation", "has feature", "noisy"),
            Improvement("augmentation", "drop it", "improve fill_ratio"),
            HarnessMutation("REMOVE", "augmentation.features", "level1_imbalance"),
        )

    def test_inverse_of_recently_promoted_mutation_is_rejected(self):
        added = HarnessMutation("ADD", "augmentation.features", "level1_imbalance")
        spec = apply_mutation(H0, added)
        with self.assertRaises(ValueError) as raised:
            validate_factory_proposal(self._undo_feature(), spec, evidence(), (self._promoted(added),))
        self.assertIn("reverses a recently promoted mutation", str(raised.exception))

    def test_inverse_is_allowed_after_the_cooldown(self):
        added = HarnessMutation("ADD", "augmentation.features", "level1_imbalance")
        spec = apply_mutation(H0, added)
        filler = self._promoted(HarnessMutation("SET", "observation.book_levels", 3))
        history = (self._promoted(added),) + (filler,) * OSCILLATION_COOLDOWN_ATTEMPTS
        validate_factory_proposal(self._undo_feature(), spec, evidence(), history)

    def test_set_mutations_have_no_derivable_inverse(self):
        self.assertIsNone(inverse_mutation(HarnessMutation("SET", "observation.book_levels", 3)))
        self.assertEqual(
            inverse_mutation(HarnessMutation("ADD", "constraints.allowed_offsets", 0)),
            HarnessMutation("REMOVE", "constraints.allowed_offsets", 0),
        )
        self.assertEqual(
            inverse_mutation(HarnessMutation("REPLACE", "augmentation.features", {"old": "a", "new": "b"})),
            HarnessMutation("REPLACE", "augmentation.features", {"old": "b", "new": "a"}),
        )


if __name__ == "__main__":
    unittest.main()
