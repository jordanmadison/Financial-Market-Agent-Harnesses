"""One-active-lineage generation loop for generate-implementation mode."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .domain import Experience
from .environment import OrderBookEnvironment, evaluate_action_counterfactuals
from .episodes import Episode
from .evaluation import ActingAgent, run_harness
from .factory.evidence import derive_window_evidence
from .factory.implementation import ImplementationFactoryBackend, ImplementationFactoryCall, ImplementationProposal
from .factory.types import Evidence
from .implementation import (
    FunctionImplementationSpec,
    HarnessImplementationSpec,
    compile_implementation_harness,
    replace_function_implementation,
)


@dataclass(frozen=True)
class ImplementationTransition:
    generation: int
    episode_id: str
    parent_harness_id: str
    active_harness_id: str
    parent_implementation_spec: HarnessImplementationSpec
    decision_trajectory: dict[str, Any]
    selected_action: dict[str, Any]
    objective_outcome: dict[str, Any]
    evidence: Evidence
    factory_model_config_hash: str | None
    factory_prompt_version: str
    factory_prompt_hash: str
    raw_factory_response: str | None
    diagnosis: dict[str, Any] | None
    inspection: dict[str, Any] | None
    improvement: dict[str, Any] | None
    target_function_name: str | None
    proposed_implementation_description: str | None
    proposed_implementation_source: str | None
    implementation_source_hash: str | None
    successor_specification: HarnessImplementationSpec | None
    successor_harness_id: str | None
    status: Literal["PROMOTED", "INVALID"]
    error: str | None
    token_usage: dict[str, int]
    latency_ms: int
    retries: int
    api_errors: tuple[dict[str, Any], ...]
    successor_measured: bool = False


@dataclass(frozen=True)
class ImplementationGenerationResult:
    transition: ImplementationTransition
    experience: Experience
    next_spec: HarnessImplementationSpec


def construct_successor(current_spec: HarnessImplementationSpec, proposal: ImplementationProposal | None,
                        factory_error: str | None = None) -> tuple[HarnessImplementationSpec | None, str | None, str | None]:
    """Compile one replacement; invalid output has neither a spec nor an ID."""
    try:
        if proposal is None:
            raise ValueError(factory_error or "factory returned no parsed implementation")
        successor = replace_function_implementation(current_spec, proposal.implementation)
        compiled = compile_implementation_harness(successor)
        return successor, compiled.harness_id, None
    except Exception as exc:  # malformed trusted development code must not end the run
        return None, None, f"{type(exc).__name__}: {exc}"


def execute_implementation_generation(generation: int, current_spec: HarnessImplementationSpec,
                                      episode: Episode, history: tuple[Experience, ...],
                                      prior_attempts: tuple[dict[str, Any], ...], agent: ActingAgent,
                                      env: OrderBookEnvironment, factory: ImplementationFactoryBackend,
                                      evidence_window: int = 5) -> ImplementationGenerationResult:
    current = compile_implementation_harness(current_spec, env.config.tick_size)
    evaluation = run_harness(agent, current, env, [episode], history)
    experience = evaluation.experiences[0]
    evidence = derive_window_evidence(history + (experience,), generation, evidence_window)
    proposal, call = factory.propose(current_spec, evidence, prior_attempts, experience.trajectory, experience.outcome)
    successor, successor_id, error = construct_successor(current_spec, proposal, call.error)
    status: Literal["PROMOTED", "INVALID"] = "PROMOTED" if successor is not None else "INVALID"
    implementation: FunctionImplementationSpec | None = proposal.implementation if proposal else None
    api_errors = tuple(attempt.api_error for attempt in call.attempts if attempt.api_error is not None)
    transition = ImplementationTransition(
        generation, episode.episode_id, current.harness_id, current.harness_id, current_spec,
        asdict(experience.trajectory), asdict(experience.trajectory.final_action), asdict(experience.outcome), evidence,
        call.factory_config_hash, call.prompt_version, call.prompt_hash, call.raw_response,
        asdict(proposal.diagnosis) if proposal else None, asdict(proposal.inspection) if proposal else None,
        asdict(proposal.improvement) if proposal else None, implementation.function_name if implementation else None,
        implementation.description if implementation else None, implementation.source if implementation else None,
        hashlib.sha256(implementation.source.encode()).hexdigest() if implementation else None,
        successor, successor_id, status, error,
        {"input": call.input_tokens or 0, "output": call.output_tokens or 0}, call.latency_ms,
        call.retry_count, api_errors, False,
    )
    # Operational validity alone promotes; no reward comparison or successor inference occurs here.
    return ImplementationGenerationResult(transition, experience, successor or current_spec)
