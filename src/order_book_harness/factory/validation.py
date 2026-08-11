from __future__ import annotations

import json

from ..harness import METRICS, HarnessMutation, HarnessSpec, compile_harness, canonical_spec
from ..registry import FEEDBACK_METRICS
from .evidence import (
    allowed_evidence_refs,
    apply_mutation,
    artifact_id,
    canonical_mutation,
    inverse_mutation,
    parse_confidence,
    parse_operation,
    parse_string,
)

# A promoted mutation cannot be undone by its exact inverse within this many
# subsequent factory attempts; prevents unbounded add/remove oscillation.
OSCILLATION_COOLDOWN_ATTEMPTS = 3
from .types import Diagnosis, Evidence, FactoryProposal, HarnessInspection, Improvement, SuccessorAttempt


def parse_factory_proposal(raw: str, evidence: Evidence) -> FactoryProposal:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"diagnosis", "inspection", "improvement", "mutation"}:
        raise ValueError("factory output must contain exactly diagnosis, inspection, improvement, and mutation")
    required = {
        "diagnosis": {"affected_metric", "evidence_refs", "suspected_harness_component", "suspected_cause", "confidence"},
        "inspection": {"component", "current_behavior", "limitation"},
        "improvement": {"component", "rationale", "expected_metric_effect"},
        "mutation": {"operation", "target_path", "value"},
    }
    for name, fields in required.items():
        if not isinstance(value[name], dict) or set(value[name]) != fields:
            raise ValueError(f"{name} fields must be exactly {sorted(fields)}")
    diagnosis_value, inspection_value, improvement_value, mutation_value = (
        value["diagnosis"],
        value["inspection"],
        value["improvement"],
        value["mutation"],
    )
    refs = (
        tuple(diagnosis_value["evidence_refs"])
        if isinstance(diagnosis_value["evidence_refs"], list) and all(isinstance(item, str) for item in diagnosis_value["evidence_refs"])
        else ()
    )
    diagnosis = Diagnosis(
        artifact_id("diagnosis", evidence.evidence_id, json.dumps(diagnosis_value, sort_keys=True)),
        parse_string(diagnosis_value["affected_metric"], "affected_metric"),
        refs,
        parse_string(diagnosis_value["suspected_harness_component"], "suspected_harness_component"),
        parse_string(diagnosis_value["suspected_cause"], "suspected_cause"),
        parse_confidence(diagnosis_value["confidence"]),
    )
    inspection = HarnessInspection(
        parse_string(inspection_value["component"], "inspection.component"),
        parse_string(inspection_value["current_behavior"], "inspection.current_behavior"),
        parse_string(inspection_value["limitation"], "inspection.limitation"),
    )
    mutation = HarnessMutation(
        parse_operation(mutation_value["operation"]),
        parse_string(mutation_value["target_path"], "mutation.target_path"),
        mutation_value["value"],
    )
    improvement = Improvement(
        parse_string(improvement_value["component"], "improvement.component"),
        parse_string(improvement_value["rationale"], "improvement.rationale"),
        parse_string(improvement_value["expected_metric_effect"], "improvement.expected_metric_effect"),
    )
    return FactoryProposal(diagnosis, inspection, improvement, mutation)


def validate_factory_proposal(
    proposal: FactoryProposal,
    current_spec: HarnessSpec,
    evidence: Evidence,
    prior_attempts: tuple[SuccessorAttempt, ...],
) -> None:
    valid_refs = allowed_evidence_refs(evidence)
    if not proposal.diagnosis.evidence_refs or any(ref not in valid_refs for ref in proposal.diagnosis.evidence_refs):
        raise ValueError("diagnosis contains an unknown evidence reference")
    if (
        proposal.diagnosis.affected_metric not in evidence.aggregate_metrics
        and proposal.diagnosis.affected_metric not in {metric.value for metric in FEEDBACK_METRICS}
        and proposal.diagnosis.affected_metric not in METRICS
    ):
        raise ValueError("affected metric is not supplied objective evidence")
    candidate = apply_mutation(current_spec, proposal.mutation)
    if canonical_spec(candidate) == canonical_spec(current_spec):
        raise ValueError("mutation is a no-op")
    compile_harness(candidate)
    identity = canonical_mutation(proposal.mutation)
    for prior in prior_attempts:
        if (
            prior.decision == "INVALID"
            and prior.mutation
            and canonical_mutation(prior.mutation) == identity
            and set(proposal.diagnosis.evidence_refs).issubset(prior.diagnosis_refs)
        ):
            raise ValueError("previously rejected mutation lacks new supporting evidence")
    inverse = inverse_mutation(proposal.mutation)
    if inverse is not None:
        inverse_identity = canonical_mutation(inverse)
        for prior in prior_attempts[-OSCILLATION_COOLDOWN_ATTEMPTS:]:
            if prior.decision == "PROMOTED" and prior.mutation and canonical_mutation(prior.mutation) == inverse_identity:
                raise ValueError(
                    "mutation reverses a recently promoted mutation; "
                    f"undoing it requires at least {OSCILLATION_COOLDOWN_ATTEMPTS} intervening attempts"
                )
