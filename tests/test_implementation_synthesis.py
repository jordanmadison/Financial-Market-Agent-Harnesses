from __future__ import annotations

import json
import unittest
from dataclasses import asdict

from order_book_harness.domain import EnvironmentObservation, Episode, InferenceRecord, LOBState, OrderOutcome, ParentOrder, PlacementAction
from order_book_harness.factory.implementation import (
    IMPLEMENTATION_FACTORY_SYSTEM_PROMPT,
    ImplementationFactoryCall,
    parse_implementation_proposal,
)
from order_book_harness.implementation import (
    FIXED_FUNCTIONS,
    FUNCTION_SIGNATURE_VERSION,
    H0_IMPLEMENTATION,
    FunctionImplementationSpec,
    canonical_implementation_spec,
    compile_implementation_harness,
    implementation_harness_id,
    replace_function_implementation,
)
from order_book_harness.implementation_generation import construct_successor, execute_implementation_generation


def observation() -> EnvironmentObservation:
    episode = Episode("e0", "TEST", "2026-01-01", 0, ParentOrder("BUY", 100, 1.0, 31.0), 30_000, 30_000)
    return EnvironmentObservation(episode, LOBState(1.0, (10000,), (10,), (10100,), (20,)), ())


def proposal(source: str = "def augment(self, observation):\n    return {'generated': 0.63, **observation}\n"):
    return parse_implementation_proposal(json.dumps({
        "diagnosis": {"affected_metric": "fill_ratio", "evidence_refs": ["evidence-g0:aggregate:fill_ratio"],
                      "suspected_harness_function": "augment", "suspected_cause": "missing pressure", "confidence": .8},
        "inspection": {"function_name": "augment", "current_behavior": "spread only", "limitation": "no pressure"},
        "improvement": {"function_name": "augment", "rationale": "add pressure", "expected_metric_effect": "better context"},
        "implementation": {"function_name": "augment", "description": "generated marker", "source": source},
    }))


class ImplementationSynthesisTests(unittest.TestCase):
    def test_factory_role_is_precise_without_execution_trust_language(self):
        self.assertTrue(IMPLEMENTATION_FACTORY_SYSTEM_PROMPT.startswith("You are an Agent Harness Implementation Factory."))
        self.assertNotIn("trusted development-time code", IMPLEMENTATION_FACTORY_SYSTEM_PROMPT)
        self.assertIn("trajectory and outcome from the just-completed episode", IMPLEMENTATION_FACTORY_SYSTEM_PROMPT)
        self.assertIn("Do not assume later market states", IMPLEMENTATION_FACTORY_SYSTEM_PROMPT)
        self.assertIn("Promotion checks only whether", IMPLEMENTATION_FACTORY_SYSTEM_PROMPT)
        self.assertIn("performance is measured only when it becomes active", IMPLEMENTATION_FACTORY_SYSTEM_PROMPT)

    def test_h0_complete_concrete_and_deterministic(self):
        self.assertEqual(set(H0_IMPLEMENTATION.functions), set(FIXED_FUNCTIONS))
        self.assertTrue(all("def " in item.source and "\n    pass" not in item.source for item in H0_IMPLEMENTATION.functions.values()))
        self.assertEqual(implementation_harness_id(H0_IMPLEMENTATION), implementation_harness_id(H0_IMPLEMENTATION))
        self.assertEqual(canonical_implementation_spec(H0_IMPLEMENTATION), canonical_implementation_spec(H0_IMPLEMENTATION))

    def test_exactly_one_replacement_and_generated_execution_reaches_prompt(self):
        parsed = proposal()
        successor = replace_function_implementation(H0_IMPLEMENTATION, parsed.implementation)
        for name in FIXED_FUNCTIONS:
            if name != "augment":
                self.assertIs(successor.functions[name], H0_IMPLEMENTATION.functions[name])
        self.assertEqual(successor.functions["augment"].source, parsed.implementation.source)
        self.assertNotEqual(implementation_harness_id(successor), implementation_harness_id(H0_IMPLEMENTATION))
        harness = compile_implementation_harness(successor)
        messages, context = harness.build_messages(observation())
        self.assertEqual(context["market_context"]["generated"], .63)
        self.assertEqual(json.loads(messages[-1]["content"])["market_context"]["generated"], .63)

    def test_constraints_authoritative_and_one_action(self):
        harness = compile_implementation_harness(H0_IMPLEMENTATION)
        messages, _ = harness.build_messages(observation())
        self.assertIn("Choose exactly one passive offset", messages[0]["content"])
        self.assertIn("generate-implementation implementation", harness.build_messages.__doc__)
        action, errors = harness.enforce(PlacementAction(2))
        self.assertIsInstance(action, PlacementAction)
        self.assertFalse(errors)
        fallback, errors = harness.enforce(None)
        self.assertEqual(fallback, PlacementAction(1))
        self.assertTrue(errors)

    def test_invalid_source_or_missing_callable_retains_parent_without_id(self):
        for source in ("def augment(:\n", "def something_else(self, observation):\n    return observation\n"):
            successor, successor_id, error = construct_successor(H0_IMPLEMENTATION, proposal(source))
            self.assertIsNone(successor)
            self.assertIsNone(successor_id)
            self.assertTrue(error)

    def test_valid_is_promoted_without_measurement(self):
        successor, successor_id, error = construct_successor(H0_IMPLEMENTATION, proposal())
        self.assertIsNotNone(successor)
        self.assertEqual(successor_id, implementation_harness_id(successor))
        self.assertIsNone(error)

    def test_parser_rejects_multiple_or_mismatched_implementations(self):
        raw = asdict(proposal())
        raw["implementations"] = [raw.pop("implementation")]
        with self.assertRaises(ValueError):
            parse_implementation_proposal(json.dumps(raw))
        bad = asdict(proposal())
        bad["inspection"]["function_name"] = "observe"
        with self.assertRaises(ValueError):
            parse_implementation_proposal(json.dumps(bad))

    def test_signature_is_fixed(self):
        invalid = FunctionImplementationSpec("augment", FUNCTION_SIGNATURE_VERSION,
                                             "def augment(observation):\n    return observation\n", "bad")
        with self.assertRaises(ValueError):
            compile_implementation_harness(replace_function_implementation(H0_IMPLEMENTATION, invalid))

    def test_mocked_end_to_end_h0_to_h1_lineage(self):
        class Agent:
            contexts = []

            def act(self, messages, harness_id):
                self.contexts.append(json.loads(messages[-1]["content"]))
                action = PlacementAction(1)
                return action, InferenceRecord(None, "fixed", "fixed", "local", "hash", harness_id,
                    "prompt", json.dumps({"offset_ticks": 1}), action, 1, 1, 0, 0, None)

        class Env:
            config = type("Config", (), {"tick_size": 100})()

            def reset(self, episode):
                return EnvironmentObservation(episode, LOBState(1.0, (10000,), (10,), (10100,), (20,)), ())

            def step(self, action):
                return OrderOutcome(100, 100, 1.0, 1.0, 2.0, 10050.0, 10050.0, 10060.0,
                                    0.0, 0.0, 0.0, 0.0, 1.0)

        class Factory:
            backend_name = "mock"

            def propose(self, current_spec, evidence, prior_attempts, trajectory, outcome):
                parsed = proposal()
                call = ImplementationFactoryCall("mock", "implementation-factory-v1", "config", "prompt", (),
                                                 json.dumps(asdict(parsed)), parsed, 1, 1, 0, 0, (), None)
                return parsed, call

        agent, env = Agent(), Env()
        episode0 = observation().episode
        result0 = execute_implementation_generation(0, H0_IMPLEMENTATION, episode0, (), (), agent, env, Factory())
        self.assertEqual(result0.transition.status, "PROMOTED")
        self.assertEqual(result0.transition.parent_harness_id, implementation_harness_id(H0_IMPLEMENTATION))
        episode1 = Episode("e1", "TEST", "2026-01-01", 1, episode0.parent_order, 30_000, 30_000)
        result1 = execute_implementation_generation(1, result0.next_spec, episode1, (result0.experience,), (), agent, env, Factory())
        self.assertEqual(agent.contexts[-1]["market_context"]["generated"], .63)
        lineage = [result0.transition.parent_harness_id, result0.transition.successor_harness_id]
        self.assertEqual(lineage, [implementation_harness_id(H0_IMPLEMENTATION), implementation_harness_id(result0.next_spec)])


if __name__ == "__main__":
    unittest.main()
