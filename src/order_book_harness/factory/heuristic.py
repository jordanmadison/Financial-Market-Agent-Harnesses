from __future__ import annotations

import time
from enum import StrEnum
from typing import Iterator

from ..domain import AgentTrajectory, OrderOutcome
from ..harness import HarnessMutation, HarnessSpec, canonical_spec
from ..registry import FEATURE_ADD_ORDER, FIELD_REGISTRY
from .evidence import artifact_id
from .types import (
    Diagnosis,
    Evidence,
    FACTORY_PROMPT_VERSION,
    FactoryCallRecord,
    FactoryProposal,
    HarnessInspection,
    Improvement,
    SuccessorAttempt,
)
from .validation import validate_factory_proposal

# Terminal condition for the deterministic baseline: every rung of its finite,
# monotone ladder is already present in the active specification, so no legal
# non-no-op mutation remains to propose. Reported as this distinct error rather
# than by re-emitting a mutation validation would reject as a no-op.
LADDER_EXHAUSTED = "heuristic ladder exhausted: no remaining legal non-no-op mutation for this spec"


class HeuristicFactory:
    """Deterministic baseline retaining the existing evidence/feature heuristic.

    Candidates are enumerated in a fixed order: first the preferred rungs (features
    in ``FEATURE_ADD_ORDER``, then ``decision.procedure``, ``feedback.history_window``,
    ``observation.book_levels``), then a generic walk over ``FIELD_REGISTRY``. The
    first candidate that passes factory validation against the current spec is
    proposed. The walk is monotone — scalars only advance forward through their
    declared value order and collections only grow — so the ladder is finite and
    terminates in ``LADDER_EXHAUSTED`` instead of cycling.
    """

    backend_name = "heuristic"

    def propose(
        self,
        current_spec: HarnessSpec,
        evidence: Evidence,
        prior_attempts: tuple[SuccessorAttempt, ...],
        trajectory: AgentTrajectory | None = None,
        outcome: OrderOutcome | None = None,
    ) -> tuple[FactoryProposal | None, FactoryCallRecord]:
        start = time.monotonic()
        if evidence.aggregate_metrics.get("fill_ratio", 0) < 0.5:
            metric, component, cause, confidence = "fill_ratio", "augmentation", "insufficient queue-pressure context", 0.8
        else:
            metric, component, cause, confidence = "signed_markout", "decision", "decision procedure does not compare execution trade-offs", 0.65
        diagnosis = Diagnosis(
            artifact_id("diagnosis", evidence.evidence_id, metric, component),
            metric,
            (evidence.evidence_id,),
            component,
            cause,
            confidence,
        )
        inspection = HarnessInspection(component, canonical_spec(current_spec), cause)
        proposal: FactoryProposal | None = None
        error = last_rejection = None
        for mutation, rationale in _candidate_mutations(current_spec, metric):
            candidate = FactoryProposal(diagnosis, inspection, Improvement(component, rationale, f"improve {metric}"), mutation)
            try:
                validate_factory_proposal(candidate, current_spec, evidence, prior_attempts)
            except ValueError as exc:
                last_rejection = str(exc)
                continue
            proposal = candidate
            break
        if proposal is None:
            error = LADDER_EXHAUSTED if last_rejection is None else f"{LADDER_EXHAUSTED}; last rejection: {last_rejection}"
        record = FactoryCallRecord(
            self.backend_name,
            FACTORY_PROMPT_VERSION,
            None,
            None,
            None,
            None,
            "",
            (),
            None,
            proposal,
            None,
            None,
            round((time.monotonic() - start) * 1000),
            0,
            (),
            error,
        )
        return proposal, record


def _candidate_mutations(spec: HarnessSpec, metric: str) -> Iterator[tuple[HarnessMutation, str]]:
    yield from _preferred_candidates(spec, metric)
    yield from _registry_candidates(spec)


def _preferred_candidates(spec: HarnessSpec, metric: str) -> Iterator[tuple[HarnessMutation, str]]:
    """The original hand-ordered ladder, preserved as the leading preference."""
    for feature in FEATURE_ADD_ORDER:
        if feature not in spec.augmentation.features:
            yield HarnessMutation("ADD", "augmentation.features", feature.value), f"add {feature.value} to address {metric}"
    if spec.decision.procedure == "direct":
        yield HarnessMutation("SET", "decision.procedure", "compare_all_actions"), "compare all permitted placements"
    if spec.feedback.history_window == 0:
        yield HarnessMutation("SET", "feedback.history_window", 10), "show bounded objective history"
    for levels in (3, 5):
        if spec.observation.book_levels < levels:
            yield HarnessMutation("SET", "observation.book_levels", levels), "broaden causal book context"


def _registry_candidates(spec: HarnessSpec) -> Iterator[tuple[HarnessMutation, str]]:
    """Every remaining forward move in the harness language, in registry order.

    Scalars only advance to values declared after the current one and collections
    only gain members, which bounds the ladder and rules out add/remove or
    value-toggling cycles between generations.
    """
    for path, field in FIELD_REGISTRY.items():
        component, name = path.split(".")
        current = getattr(getattr(spec, component), name)
        if field.kind == "scalar":
            if "SET" not in field.operations or current not in field.values:
                continue
            for value in field.values[field.values.index(current) + 1:]:
                yield HarnessMutation("SET", path, value), f"advance {path} to {value}"
        elif "ADD" in field.operations:
            present = {str(item) for item in current}
            for value in field.values:
                if str(value) not in present:
                    yield HarnessMutation("ADD", path, str(value) if isinstance(value, StrEnum) else value), f"extend {path} with {value}"
