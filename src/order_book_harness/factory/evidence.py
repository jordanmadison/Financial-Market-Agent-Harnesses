from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from ..domain import Experience
from ..harness import HarnessMutation, HarnessSpec, MutationOperation, apply_harness_mutation, canonical_spec
from ..registry import EVIDENCE_METRICS, METRIC_REGISTRY, MetricName
from .types import Evidence


def derive_evidence(experiences: tuple[Experience, ...], generation: int, metric_slices: dict[str, dict[str, float]] | None = None) -> Evidence:
    n = len(experiences)
    metrics = {
        MetricName.REWARD.value: sum(experience.outcome.reward for experience in experiences) / n,
        MetricName.FILL_RATIO.value: sum(experience.outcome.fill_ratio for experience in experiences) / n,
        MetricName.SIGNED_MARKOUT.value: sum(experience.outcome.signed_markout for experience in experiences) / n,
    } if n else {}
    return Evidence(f"evidence-g{generation}", tuple(experience.experience_id for experience in experiences), metrics, metric_slices or {}, n)


def derive_episode_evidence(experience: Experience, generation: int, ticker: str) -> Evidence:
    """Construct objective evidence from exactly one completed generation episode."""
    metrics = _evidence_metrics((experience,))
    return Evidence(
        f"evidence-g{generation}",
        (experience.experience_id,),
        metrics,
        {ticker: dict(metrics)},
        1,
    )


def derive_window_evidence(experiences: tuple[Experience, ...], generation: int, window: int) -> Evidence:
    """Construct objective evidence from a trailing window of completed episodes.

    The aggregate covers up to ``window`` most recent experiences (always including
    the just-completed one); the ``latest_episode`` slice isolates the newest episode
    so a single-episode diagnosis stays possible. Windowed experiences may span
    several harness versions; per-harness attribution lives in generations.jsonl.
    """
    if not experiences:
        raise ValueError("window evidence requires at least one completed experience")
    chosen = experiences[-max(1, window):]
    return Evidence(
        f"evidence-g{generation}",
        tuple(experience.experience_id for experience in chosen),
        _evidence_metrics(chosen),
        {"latest_episode": _evidence_metrics(chosen[-1:])},
        len(chosen),
    )


def _evidence_metrics(experiences: tuple[Experience, ...]) -> dict[str, float]:
    metrics = {}
    for metric in EVIDENCE_METRICS:
        definition = METRIC_REGISTRY[metric]
        values = [definition.from_experience(experience) for experience in experiences]
        metrics[metric.value] = sum(0.0 if value is None else float(value) for value in values) / len(values)
    return metrics


def allowed_evidence_refs(evidence: Evidence) -> set[str]:
    refs = {evidence.evidence_id}
    refs.update(f"{evidence.evidence_id}:aggregate:{metric}" for metric in evidence.aggregate_metrics)
    for slice_name, metrics in evidence.metric_slices.items():
        refs.add(f"{evidence.evidence_id}:slice:{slice_name}")
        refs.update(f"{evidence.evidence_id}:slice:{slice_name}:{metric}" for metric in metrics)
    return refs


def apply_mutation(spec: HarnessSpec, mutation: HarnessMutation) -> HarnessSpec:
    return apply_harness_mutation(spec, mutation)


def canonical_mutation(mutation: HarnessMutation) -> str:
    return json.dumps(asdict(mutation), sort_keys=True, separators=(",", ":"))


def inverse_mutation(mutation: HarnessMutation) -> HarnessMutation | None:
    """Return the mutation that exactly undoes ``mutation``, when one is derivable.

    SET has no derivable inverse because the prior value is not part of the mutation.
    """
    if mutation.operation == "ADD":
        return HarnessMutation("REMOVE", mutation.target_path, mutation.value)
    if mutation.operation == "REMOVE":
        return HarnessMutation("ADD", mutation.target_path, mutation.value)
    if mutation.operation == "REPLACE" and isinstance(mutation.value, dict) and set(mutation.value) == {"old", "new"}:
        return HarnessMutation("REPLACE", mutation.target_path, {"old": mutation.value["new"], "new": mutation.value["old"]})
    return None


def spec_diff(parent: HarnessSpec, candidate: HarnessSpec) -> dict:
    parent_dict = json.loads(canonical_spec(parent))
    candidate_dict = json.loads(canonical_spec(candidate))
    result: dict = {}
    for component in parent_dict:
        for field in parent_dict[component]:
            if parent_dict[component][field] != candidate_dict[component][field]:
                result[f"{component}.{field}"] = {"from": parent_dict[component][field], "to": candidate_dict[component][field]}
    return result


def artifact_id(prefix: str, *values: str) -> str:
    return f"{prefix}-" + hashlib.sha256(":".join(values).encode()).hexdigest()[:16]


def parse_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def parse_confidence(value: object) -> float:
    if type(value) not in (int, float) or not 0 <= float(value) <= 1:
        raise ValueError("confidence must be between zero and one")
    return float(value)


def parse_operation(value: object) -> MutationOperation:
    if value not in ("ADD", "REMOVE", "REPLACE", "SET"):
        raise ValueError("unknown mutation operation")
    return value  # type: ignore[return-value]
