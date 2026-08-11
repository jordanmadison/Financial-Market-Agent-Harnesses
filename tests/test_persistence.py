import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from order_book_harness.environment import ActionCounterfactualReport, ActionOutcomeSummary, EnvironmentConfig
from order_book_harness.factory import Evidence, FactoryCallRecord, SuccessorAttempt
from order_book_harness.generation import GenerationRecord
from order_book_harness.harness import H0, harness_id
from order_book_harness.persistence import RunArtifacts, mark_run_failed, persist_run_artifacts, write_json


class PersistenceTests(unittest.TestCase):
    def _artifacts(self, run_id="run-1"):
        opportunity = ActionCounterfactualReport(
            action_outcomes=(ActionOutcomeSummary(0, 0.0, 0, None, None, -1.0),),
            constant_outcome=True,
            outcome_differentiated=False,
            best_action_reward=-1.0,
            chosen_action_reward=-1.0,
            decision_regret=0.0,
            available=False,
            fillable_offsets=(),
            selected_action_fillable=False,
            captured=None,
            unavoidable_no_fill=True,
            best_available_reward=-1.0,
            best_available_fill_ratio=0.0,
        )
        evidence = Evidence("evidence-g0", ("exp",), {"reward": -1.0, "fill_ratio": 0.0, "signed_markout": 0.0}, {})
        factory_call = FactoryCallRecord("heuristic", "factory-v4", None, None, None, None, "", (), None, None, None, None, 0, 0, (), None)
        attempt = SuccessorAttempt(None, harness_id(H0), None, (), None, {"factory_output": "failed"}, "INVALID", factory_call)
        record = GenerationRecord(
            0,
            "heuristic",
            "factory-v4",
            None,
            harness_id(H0),
            "episode-0",
            1,
            0.0,
            None,
            None,
            0.0,
            0.0,
            1.0,
            -1.0,
            opportunity,
            True,
            False,
            {"input": 0, "output": 0},
            1,
            {"input": 0, "output": 0},
            1,
            0,
            "INVALID",
            harness_id(H0),
            True,
            evidence,
            None,
            None,
            attempt.validation_results,
        )
        config = type(
            "Config",
            (),
            {
                "tickers": ("AAPL",),
                "depth": 10,
                "__dataclass_fields__": {},
            },
        )()
        # Use a real ExperimentConfig-like object via simple namespace with asdict support.
        from order_book_harness.experiment import ExperimentConfig

        return RunArtifacts(
            run_id=run_id,
            config=ExperimentConfig(tickers=("AAPL",), max_generations=1),
            agent_manifest={"kind": "offline"},
            factory_manifest={"kind": "heuristic"},
            env_config=EnvironmentConfig(),
            pairs=[],
            episodes=[],
            evolution_episodes=[],
            max_generations=1,
            harness_specs={harness_id(H0): H0},
            generations=(record,),
            successors=(attempt,),
            factory_calls=({"generation": 0, "evidence": evidence, "call": factory_call},),
            experiences=(),
            lineage_ids=(harness_id(H0),),
        )

    def test_complete_marker_is_written_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            write_order = []
            original = write_json

            def tracking_write(path, value):
                write_order.append((str(path.name), value.get("status") if isinstance(value, dict) else None))
                return original(path, value)

            with patch("order_book_harness.persistence.write_json", side_effect=tracking_write):
                persist_run_artifacts(root, self._artifacts())
            complete_writes = [item for item in write_order if item[0] == "run.json" and item[1] == "complete"]
            self.assertEqual(len(complete_writes), 1)
            self.assertEqual(write_order[-1], ("run.json", "complete"))
            self.assertTrue((root / "generations.jsonl").exists())
            self.assertTrue((root / "lineage.json").exists())
            self.assertFalse((root / ".staging").exists())
            run = json.loads((root / "run.json").read_text())
            self.assertEqual(run["status"], "complete")

    def test_failed_marker_never_reports_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            mark_run_failed(root, "run-fail", "boom", config=None, agent_manifest={}, factory_manifest={})
            run = json.loads((root / "run.json").read_text())
            self.assertEqual(run["status"], "failed")
            self.assertIn("boom", run["error"])

    def test_partial_staging_failure_does_not_advertise_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            write_json(root / "run.json", {"run_id": "run-1", "status": "in_progress"})

            def boom(path, value):
                if path.name == "report.json":
                    raise OSError("disk full")
                return write_json.__wrapped__(path, value) if hasattr(write_json, "__wrapped__") else None

            # Inject failure after some staging writes by patching write_jsonl for report path via write_json.
            original = write_json

            def failing_write(path, value):
                if path.name == "report.json":
                    raise OSError("disk full")
                return original(path, value)

            with patch("order_book_harness.persistence.write_json", side_effect=failing_write):
                with self.assertRaises(OSError):
                    persist_run_artifacts(root, self._artifacts())
            run = json.loads((root / "run.json").read_text())
            self.assertEqual(run["status"], "in_progress")
            self.assertFalse((root / "generations.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
