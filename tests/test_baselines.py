import json
import tempfile
import unittest
from pathlib import Path

from order_book_harness.cli import main
from order_book_harness.persistence import _performance_summary, analyze_run, baseline_policy_rewards, write_json


def telemetry_record(generation, chosen_offset, rewards, transition_status="PROMOTED"):
    """One generation record with the fields the baseline and summary code reads."""
    return {
        "generation": generation,
        "reward": rewards[chosen_offset],
        "fill_ratio": 0.0,
        "signed_markout": 0.0,
        "unfilled_penalty": 0.0,
        "transition_status": transition_status,
        "opportunity": {
            "action_outcomes": [
                {"offset_ticks": offset, "reward": reward, "fill_ratio": 0.0}
                for offset, reward in sorted(rewards.items())
            ],
            "best_action_reward": max(rewards.values()),
            "decision_regret": max(rewards.values()) - rewards[chosen_offset],
            "constant_outcome": len(set(rewards.values())) == 1,
            "outcome_differentiated": len(set(rewards.values())) > 1,
            "captured": None,
            "available": False,
            "fillable_offsets": [],
            "selected_action_fillable": False,
            "unavoidable_no_fill": True,
            "best_available_reward": max(rewards.values()),
            "best_available_fill_ratio": 0.0,
        },
    }


class BaselineSummaryTests(unittest.TestCase):
    def test_constant_offset_series_come_from_counterfactuals(self):
        telemetry = [
            telemetry_record(0, 1, {0: -10.0, 1: -10.0, 2: -10.0}),
            telemetry_record(1, 0, {0: -30.0, 1: 10.0, 2: 100.0}),
        ]
        series = baseline_policy_rewards(telemetry)
        self.assertEqual(series["always_offset_0"], [-10.0, -30.0])
        self.assertEqual(series["always_offset_2"], [-10.0, 100.0])
        self.assertEqual(series["agent"], [-10.0, -30.0])
        self.assertEqual(series["oracle"], [-10.0, 100.0])

    def test_summary_identifies_best_constant_baseline_and_agent_delta(self):
        telemetry = [
            telemetry_record(0, 1, {0: -10.0, 1: -10.0, 2: -10.0}),
            telemetry_record(1, 0, {0: -30.0, 1: 10.0, 2: 100.0}),
        ]
        summary = _performance_summary(telemetry)["baseline_summary"]
        self.assertEqual(summary["best_constant_baseline"], "always_offset_2")
        self.assertEqual(summary["policies"]["always_offset_2"]["mean_reward"], 45.0)
        self.assertEqual(summary["policies"]["agent"]["mean_reward"], -20.0)
        self.assertEqual(summary["agent_minus_best_constant_mean_reward"], -65.0)
        # Only generation 1 is outcome-differentiated.
        self.assertEqual(summary["policies"]["always_offset_2"]["mean_reward_outcome_differentiated"], 100.0)

    def test_agent_beating_baselines_produces_positive_delta(self):
        telemetry = [telemetry_record(0, 2, {0: -30.0, 1: 10.0, 2: 100.0})]
        summary = _performance_summary(telemetry)["baseline_summary"]
        self.assertEqual(summary["agent_minus_best_constant_mean_reward"], 0.0)
        self.assertEqual(summary["policies"]["agent"]["mean_reward"], summary["policies"]["oracle"]["mean_reward"])

    def test_offsets_absent_from_some_generations_are_excluded(self):
        telemetry = [
            telemetry_record(0, 1, {0: -10.0, 1: -10.0, 2: -10.0}),
            telemetry_record(1, 1, {1: 5.0, 2: 7.0}),
        ]
        series = baseline_policy_rewards(telemetry)
        self.assertNotIn("always_offset_0", series)
        self.assertIn("always_offset_1", series)


class AnalyzeRunTests(unittest.TestCase):
    def _run_dir(self, directory):
        root = Path(directory) / "run-x"
        (root / "plots").mkdir(parents=True)
        telemetry = [
            telemetry_record(0, 1, {0: -10.0, 1: -10.0, 2: -10.0}),
            telemetry_record(1, 0, {0: -30.0, 1: 10.0, 2: 100.0}),
        ]
        (root / "generations.jsonl").write_text("".join(json.dumps(item) + "\n" for item in telemetry))
        write_json(root / "report.json", {"scope": "unchanged", "performance_summary": {"mean_reward": 999.0}})
        return root

    def test_analyze_recomputes_summary_and_renders_baseline_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._run_dir(directory)
            original_report = (root / "report.json").read_bytes()
            summary = analyze_run(root)
            self.assertEqual(summary["mean_reward"], -20.0)
            self.assertEqual((root / "report.json").read_bytes(), original_report)
            report = json.loads((root / "analysis" / "report.json").read_text())
            self.assertEqual(report["kind"], "derived_post_run_analysis")
            self.assertEqual(report["performance_summary"]["mean_reward"], -20.0)
            self.assertEqual(report["performance_summary"]["baseline_summary"]["best_constant_baseline"], "always_offset_2")
            self.assertTrue((root / "analysis" / "plots" / "baseline_comparison.svg").exists())
            self.assertIn("always_offset_2", (root / "analysis" / "plots" / "baseline_comparison.svg").read_text())

    def test_analyze_cli_returns_zero_and_requires_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._run_dir(directory)
            self.assertEqual(main(["analyze", str(root)]), 0)
            missing = Path(directory) / "empty"
            missing.mkdir()
            with self.assertRaises(FileNotFoundError):
                analyze_run(missing)


if __name__ == "__main__":
    unittest.main()
