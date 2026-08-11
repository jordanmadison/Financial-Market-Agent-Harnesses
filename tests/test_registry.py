import unittest

from order_book_harness.harness import H0, FEATURES, METRICS, HARNESS_LANGUAGE, canonical_spec, harness_id
from order_book_harness.registry import (
    FEATURE_REGISTRY,
    FIELD_REGISTRY,
    FEEDBACK_METRICS,
    METRIC_REGISTRY,
    FeatureName,
    MetricName,
)


class RegistryDriftTests(unittest.TestCase):
    def test_feature_registry_covers_language_and_enum(self):
        self.assertEqual(set(FEATURE_REGISTRY), set(FeatureName))
        self.assertEqual(FEATURES, frozenset(FEATURE_REGISTRY))
        self.assertEqual(set(FIELD_REGISTRY["augmentation.features"].values), set(FEATURE_REGISTRY))

    def test_metric_registry_covers_feedback_language(self):
        self.assertEqual(set(FEEDBACK_METRICS), set(METRICS))
        self.assertTrue(set(FEEDBACK_METRICS).issubset(METRIC_REGISTRY))
        self.assertEqual(set(FIELD_REGISTRY["feedback.metrics_shown"].values), set(FEEDBACK_METRICS))

    def test_field_registry_is_single_source(self):
        self.assertEqual(HARNESS_LANGUAGE, FIELD_REGISTRY)
        self.assertEqual(
            set(FIELD_REGISTRY),
            {
                "observation.book_levels",
                "observation.event_lookback",
                "observation.include_raw_events",
                "augmentation.features",
                "decision.procedure",
                "decision.require_action_rationale",
                "constraints.allowed_offsets",
                "constraints.reject_invalid_action",
                "constraints.fallback_offset",
                "feedback.history_window",
                "feedback.metrics_shown",
            },
        )

    def test_h0_identity_is_stable(self):
        self.assertEqual(harness_id(H0), "e41aadfa6cfc3df994526f43")
        self.assertEqual(
            canonical_spec(H0),
            '{"augmentation":{"features":["spread_ticks"]},"constraints":{"allowed_offsets":[0,1,2],"fallback_offset":1,"reject_invalid_action":true},"decision":{"procedure":"direct","require_action_rationale":false},"feedback":{"history_window":0,"metrics_shown":[]},"observation":{"book_levels":1,"event_lookback":10,"include_raw_events":true}}',
        )


if __name__ == "__main__":
    unittest.main()
