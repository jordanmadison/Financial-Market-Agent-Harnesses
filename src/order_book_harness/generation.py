from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .agent import parse_placement_action
from .domain import Experience, to_dict
from .environment import ActionCounterfactualReport, OrderBookEnvironment, evaluate_action_counterfactuals
from .episodes import Episode
from .evaluation import ActingAgent, run_harness
from .factory import (
    Evidence,
    FactoryBackend,
    FactoryCallRecord,
    FactoryProposal,
    SuccessorAttempt,
    apply_mutation,
    derive_window_evidence,
    spec_diff,
    validate_factory_proposal,
)
from .harness import HarnessSpec, compile_harness, harness_id


@dataclass(frozen=True)
class GenerationInput:
    generation: int
    incumbent_spec: HarnessSpec
    episode: Episode
    prior_experiences: tuple[Experience, ...]
    prior_successors: tuple[SuccessorAttempt, ...]


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    factory_backend: str
    factory_prompt_version: str
    factory_model_config_hash: str | None
    active_harness_id: str
    episode_id: str
    selected_action: int
    fill_ratio: float
    time_to_fill: float | None
    time_to_first_fill: float | None
    spread_capture: float
    signed_markout: float
    unfilled_penalty: float
    reward: float
    opportunity: ActionCounterfactualReport
    parse_success: bool
    fallback_used: bool
    decision_agent_token_usage: dict[str, int]
    decision_agent_latency_ms: int
    factory_token_usage: dict[str, int]
    factory_latency_ms: int
    factory_retry_count: int
    transition_status: Literal["PROMOTED", "INVALID", "REVERTED"]
    successor_harness_id: str
    successor_measured: bool
    evidence: Evidence
    diagnosis: dict[str, Any] | None
    mutation: dict[str, Any] | None
    validation: dict[str, Any]
    episode_timestamp: float = 0.0
    active_harness_version: int = 0
    candidate_harness_id: str | None = None
    factory_decision: str = ""
    decision_reason: str = ""
    decision_agent_error: str | None = None
    factory_error: str | None = None

    def as_telemetry(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = asdict(self.evidence)
        payload["opportunity"] = asdict(self.opportunity)
        return payload


@dataclass(frozen=True)
class GenerationResult:
    record: GenerationRecord
    successor_attempt: SuccessorAttempt
    experience: Experience
    factory_call_entry: dict[str, Any]
    next_incumbent_spec: HarnessSpec
    new_harness_spec: HarnessSpec | None

    @property
    def promoted(self) -> bool:
        return self.new_harness_spec is not None

    @property
    def successor_id(self) -> str:
        return self.record.successor_harness_id


@dataclass(frozen=True)
class RunAccumulator:
    incumbent_spec: HarnessSpec
    harness_specs: dict[str, HarnessSpec]
    experiences: tuple[Experience, ...] = ()
    generations: tuple[GenerationRecord, ...] = ()
    successors: tuple[SuccessorAttempt, ...] = ()
    factory_calls: tuple[dict[str, Any], ...] = ()
    lineage_ids: tuple[str, ...] = field(default_factory=tuple)

    def apply(self, result: GenerationResult) -> RunAccumulator:
        harness_specs = dict(self.harness_specs)
        lineage_ids = self.lineage_ids
        if result.new_harness_spec is not None:
            harness_specs[result.successor_id] = result.new_harness_spec
            lineage_ids = lineage_ids + (result.successor_id,)
        return RunAccumulator(
            incumbent_spec=result.next_incumbent_spec,
            harness_specs=harness_specs,
            experiences=self.experiences + (result.experience,),
            generations=self.generations + (result.record,),
            successors=self.successors + (result.successor_attempt,),
            factory_calls=self.factory_calls + (result.factory_call_entry,),
            lineage_ids=lineage_ids,
        )


@dataclass(frozen=True)
class GenerationServices:
    root: Path
    agent: ActingAgent
    env: OrderBookEnvironment
    factory_backend: FactoryBackend
    evidence_window: int = 5


def execute_generation(services: GenerationServices, generation_input: GenerationInput) -> GenerationResult:
    """Run one generation without mutating caller-owned collections."""
    current = compile_harness(generation_input.incumbent_spec)
    history = generation_input.prior_experiences
    evaluation = run_harness(services.agent, current, services.env, [generation_input.episode], history)
    experience = evaluation.experiences[0]
    evidence = derive_window_evidence(history + (experience,), generation_input.generation, services.evidence_window)
    _write_progress(
        services.root,
        {
            "status": "waiting_for_factory",
            "generation": generation_input.generation,
            "active_harness_id": current.harness_id,
            "episode_id": generation_input.episode.episode_id,
            "experience": experience,
            "evidence": evidence,
        },
    )
    proposal, factory_call = services.factory_backend.propose(
        generation_input.incumbent_spec,
        evidence,
        generation_input.prior_successors,
        experience.trajectory,
        experience.outcome,
    )
    factory_call_entry = {
        "generation": generation_input.generation,
        "evidence": evidence,
        "call": factory_call,
    }
    _write_progress(
        services.root,
        {
            "status": "factory_complete",
            "generation": generation_input.generation,
            "active_harness_id": current.harness_id,
            "episode_id": generation_input.episode.episode_id,
            "experience": experience,
            "evidence": evidence,
            "factory_call": factory_call,
        },
    )
    history_with_current = history + (experience,)
    attempt, successor_spec = validate_successor(
        proposal,
        factory_call,
        generation_input.incumbent_spec,
        current.harness_id,
        services.env,
        generation_input.episode,
        history_with_current,
        evidence,
        generation_input.prior_successors,
    )
    opportunity = evaluate_action_counterfactuals(
        services.env,
        generation_input.episode,
        experience.trajectory.final_action,
        experience.outcome,
    )
    promoted = successor_spec is not None
    successor_id = harness_id(successor_spec) if successor_spec is not None else current.harness_id
    next_incumbent = successor_spec if successor_spec is not None else generation_input.incumbent_spec
    outcome = experience.outcome
    trajectory = experience.trajectory
    record = GenerationRecord(
        generation=generation_input.generation,
        factory_backend=services.factory_backend.backend_name,
        factory_prompt_version=factory_call.prompt_version,
        factory_model_config_hash=factory_call.factory_config_hash,
        active_harness_id=current.harness_id,
        episode_id=generation_input.episode.episode_id,
        selected_action=trajectory.final_action.offset_ticks,
        fill_ratio=outcome.fill_ratio,
        time_to_fill=outcome.time_to_full_fill,
        time_to_first_fill=outcome.time_to_first_fill,
        spread_capture=outcome.spread_capture,
        signed_markout=outcome.signed_markout,
        unfilled_penalty=outcome.unfilled_penalty,
        reward=outcome.reward,
        opportunity=opportunity,
        parse_success=not trajectory.parse_errors,
        fallback_used=bool(trajectory.parse_errors),
        decision_agent_token_usage=trajectory.token_usage,
        decision_agent_latency_ms=trajectory.latency_ms,
        factory_token_usage={"input": factory_call.input_tokens or 0, "output": factory_call.output_tokens or 0},
        factory_latency_ms=factory_call.latency_ms,
        factory_retry_count=factory_call.retry_count,
        transition_status=attempt.decision,
        successor_harness_id=successor_id,
        successor_measured=not promoted,
        evidence=evidence,
        diagnosis=asdict(proposal.diagnosis) if proposal else None,
        mutation=asdict(proposal.mutation) if proposal else None,
        validation=attempt.validation_results,
        episode_timestamp=generation_input.episode.parent_order.decision_time,
        active_harness_version=sum(item.decision == "PROMOTED" for item in generation_input.prior_successors),
        candidate_harness_id=attempt.successor_id,
        factory_decision=attempt.decision,
        decision_reason=(
            "operational_validation_passed"
            if attempt.decision == "PROMOTED"
            else str(attempt.validation_results.get("error", "factory proposal invalid"))
        ),
        decision_agent_error=trajectory.inference_record.error,
        factory_error=factory_call.error,
    )
    return GenerationResult(
        record=record,
        successor_attempt=attempt,
        experience=experience,
        factory_call_entry=factory_call_entry,
        next_incumbent_spec=next_incumbent,
        new_harness_spec=successor_spec,
    )


def apply_promotion_rule(
    result: GenerationResult,
    previous: GenerationRecord | None,
    harness_specs: dict[str, HarnessSpec],
    promotion_rule: str,
) -> GenerationResult:
    """Apply the configured V0.3 lineage rule to a completed generation result.

    ``unconditional`` preserves V0 behavior. ``gated`` is retained as the
    backward-compatible name for a first-measurement rollback heuristic:
    when the active harness is a successor promoted by the immediately preceding
    generation and has positive decision regret on its first outcome-differentiated
    episode, the lineage reverts to the parent spec and records a ``REVERTED``
    transition. Regret is computed within the current episode, so no rewards from
    different market episodes are compared. This remains a rollback heuristic rather
    than causal evidence that the mutation degraded performance.

    The factory proposal made during the rollback generation remains independently
    auditable. An invalid proposal stays ``INVALID``; an operationally valid proposal
    that the rollback prevents from entering the lineage becomes ``REVERTED``.
    """
    if promotion_rule != "gated" or previous is None or previous.transition_status != "PROMOTED":
        return result
    record = result.record
    if not record.opportunity.outcome_differentiated or record.opportunity.decision_regret <= 0:
        return result
    parent_spec = harness_specs[previous.active_harness_id]
    reverted = replace(
        record,
        transition_status="REVERTED",
        successor_harness_id=previous.active_harness_id,
        successor_measured=True,
        decision_reason="first_measurement_positive_decision_regret_rollback_to_parent_noncausal",
    )
    proposal_was_valid = result.successor_attempt.decision == "PROMOTED"
    attempt = replace(
        result.successor_attempt,
        decision="REVERTED" if proposal_was_valid else result.successor_attempt.decision,
        validation_results={
            **result.successor_attempt.validation_results,
            "lineage_rule": "first_measurement_positive_decision_regret_rollback_noncausal",
        },
    )
    return replace(
        result,
        record=reverted,
        successor_attempt=attempt,
        next_incumbent_spec=parent_spec,
        new_harness_spec=None,
    )


def validate_successor(
    proposal: FactoryProposal | None,
    factory_call: FactoryCallRecord,
    current_spec: HarnessSpec,
    parent_harness_id: str,
    env: OrderBookEnvironment,
    episode: Episode,
    history: tuple[Experience, ...],
    evidence: Evidence,
    prior_attempts: tuple[SuccessorAttempt, ...],
) -> tuple[SuccessorAttempt, HarnessSpec | None]:
    checks: dict[str, Any] = {
        "factory_output": "passed" if proposal else "failed",
        "schema_semantics": "not_run",
        "compile": "not_run",
        "interface": "not_run",
        "causality": "not_run",
        "offline_smoke_test": "not_run",
    }
    successor_spec = None
    successor_id = None
    try:
        if proposal is None:
            raise ValueError(factory_call.error or "factory returned no valid proposal")
        validate_factory_proposal(proposal, current_spec, evidence, prior_attempts)
        successor_spec = apply_mutation(current_spec, proposal.mutation)
        checks["spec_diff"] = spec_diff(current_spec, successor_spec)
        checks["schema_semantics"] = "passed"
        compiled = compile_harness(successor_spec)
        checks["compile"] = "passed"
        if not all(hasattr(compiled, name) for name in ("build_messages", "enforce")):
            raise ValueError("compiled harness interface validation failed")
        checks["interface"] = "passed"
        observation = env.reset(episode)
        messages, _ = compiled.build_messages(observation, history, env.config.tick_size)
        checks["causality"] = "passed"
        local_action = parse_placement_action('{"offset_ticks": %d}' % compiled.spec.constraints.allowed_offsets[0])
        final_action, errors = compiled.enforce(local_action)
        if errors or final_action.offset_ticks not in compiled.spec.constraints.allowed_offsets or len(messages) != 2:
            raise ValueError("offline compiled-harness smoke test failed")
        fallback_action, fallback_errors = compiled.enforce(None)
        if not fallback_errors or fallback_action.offset_ticks != compiled.spec.constraints.fallback_offset:
            raise ValueError("offline fallback smoke test failed")
        checks["offline_smoke_test"] = "passed"
        successor_id = compiled.harness_id
    except (KeyError, TypeError, ValueError) as exc:
        checks["error"] = f"{type(exc).__name__}: {exc}"
        successor_spec = None
    status: Literal["PROMOTED", "INVALID"] = "PROMOTED" if successor_spec is not None else "INVALID"
    record = SuccessorAttempt(
        successor_id,
        parent_harness_id,
        successor_spec,
        proposal.diagnosis.evidence_refs if proposal else (),
        proposal.mutation if proposal else None,
        checks,
        status,
        factory_call,
    )
    return record, successor_spec


def _write_progress(root: Path, value: Any) -> None:
    (root / "progress.json").write_text(json.dumps(to_dict(value), indent=2, sort_keys=True) + "\n")
