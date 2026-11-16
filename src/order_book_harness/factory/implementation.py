"""Factory schemas and prompting for source implementation synthesis."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from ..completion import CompletionClient, complete_with_retries, response_usage
from ..domain import AgentTrajectory, OrderOutcome
from ..implementation import (
    FIXED_FUNCTIONS,
    FUNCTION_SIGNATURE_VERSION,
    HarnessImplementationSpec,
    FunctionImplementationSpec,
    implementation_spec_dict,
)
from .evidence import allowed_evidence_refs
from .types import Evidence, FactoryAttempt, FactoryModelConfig

IMPLEMENTATION_FACTORY_PROMPT_VERSION = "implementation-factory-v1"


@dataclass(frozen=True)
class ImplementationDiagnosis:
    affected_metric: str
    evidence_refs: tuple[str, ...]
    suspected_harness_function: str
    suspected_cause: str
    confidence: float


@dataclass(frozen=True)
class FunctionInspection:
    function_name: str
    current_behavior: str
    limitation: str


@dataclass(frozen=True)
class ImplementationImprovement:
    function_name: str
    rationale: str
    expected_metric_effect: str


@dataclass(frozen=True)
class ImplementationProposal:
    diagnosis: ImplementationDiagnosis
    inspection: FunctionInspection
    improvement: ImplementationImprovement
    implementation: FunctionImplementationSpec


@dataclass(frozen=True)
class ImplementationFactoryCall:
    backend: str
    prompt_version: str
    factory_config_hash: str | None
    prompt_hash: str
    messages: tuple[dict, ...]
    raw_response: str | None
    parsed_proposal: ImplementationProposal | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    retry_count: int
    attempts: tuple[FactoryAttempt, ...]
    error: str | None


class ImplementationFactoryBackend(Protocol):
    backend_name: str

    def propose(self, current_spec: HarnessImplementationSpec, evidence: Evidence,
                prior_attempts: tuple[dict[str, Any], ...], trajectory: AgentTrajectory,
                outcome: OrderOutcome) -> tuple[ImplementationProposal | None, ImplementationFactoryCall]: ...


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def parse_implementation_proposal(raw: str) -> ImplementationProposal:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"diagnosis", "inspection", "improvement", "implementation"}:
        raise ValueError("factory output must contain exactly diagnosis, inspection, improvement, and implementation")
    fields = {
        "diagnosis": {"affected_metric", "evidence_refs", "suspected_harness_function", "suspected_cause", "confidence"},
        "inspection": {"function_name", "current_behavior", "limitation"},
        "improvement": {"function_name", "rationale", "expected_metric_effect"},
        "implementation": {"function_name", "description", "source"},
    }
    for key, expected in fields.items():
        if not isinstance(value[key], dict) or set(value[key]) != expected:
            raise ValueError(f"{key} fields must be exactly {sorted(expected)}")
    diagnosis, inspection, improvement, implementation = (value[name] for name in fields)
    refs = diagnosis["evidence_refs"]
    confidence = diagnosis["confidence"]
    if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) for ref in refs):
        raise ValueError("diagnosis.evidence_refs must be a non-empty string list")
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        raise ValueError("diagnosis.confidence must be between zero and one")
    target = _string(implementation["function_name"], "implementation.function_name")
    named_targets = (diagnosis["suspected_harness_function"], inspection["function_name"], improvement["function_name"])
    if target not in FIXED_FUNCTIONS or any(item != target for item in named_targets):
        raise ValueError("all proposal sections must name one fixed harness function")
    return ImplementationProposal(
        ImplementationDiagnosis(_string(diagnosis["affected_metric"], "affected_metric"), tuple(refs), target,
                                _string(diagnosis["suspected_cause"], "suspected_cause"), float(confidence)),
        FunctionInspection(target, _string(inspection["current_behavior"], "current_behavior"),
                           _string(inspection["limitation"], "limitation")),
        ImplementationImprovement(target, _string(improvement["rationale"], "rationale"),
                                  _string(improvement["expected_metric_effect"], "expected_metric_effect")),
        FunctionImplementationSpec(target, FUNCTION_SIGNATURE_VERSION,
                                   _string(implementation["source"], "implementation.source"),
                                   _string(implementation["description"], "implementation.description")),
    )


IMPLEMENTATION_FACTORY_SYSTEM_PROMPT = """You are an Agent Harness Implementation Factory. Return exactly one JSON object containing diagnosis, inspection, improvement, and implementation. Generate one complete Python function implementation for exactly one of observe(self, state), augment(self, observation), constrain(self, context), evaluate(self, context), or feedback(self, completed_experiences). Do not return candidates, a patch, markdown, or configuration mutations. Environment replay, queue semantics, rewards, episodes, evidence, orchestration, promotion, persistence, analysis, model/provider configuration, and credentials are immutable. Base the diagnosis only on the supplied trajectory and outcome from the just-completed episode, derived evidence from that episode and earlier completed episodes, and bounded records of earlier implementation attempts. The replacement function must compute only from its arguments: observe may use the decision-time state and recent events passed to it, while feedback may use only completed historical experiences. Do not assume later market states, later episode outcomes, or any information that was unavailable when the placement action was selected. Promotion checks only whether the single proposed implementation parses, compiles, loads, and constructs the fixed harness interface. Do not request or invent a predicted successor reward, compare the proposal with the parent's reward, or recommend retaining the parent based on expected performance; the successor's performance is measured only when it becomes active on the next chronological episode."""


def build_implementation_factory_messages(current_spec: HarnessImplementationSpec, evidence: Evidence,
                                          prior_attempts: tuple[dict[str, Any], ...], trajectory: AgentTrajectory,
                                          outcome: OrderOutcome) -> list[dict]:
    prior = []
    for item in prior_attempts[-10:]:
        prior.append({key: item.get(key) for key in ("status", "target_function_name", "implementation_source_hash", "error")})
    supplied = {
        "current_implementation_spec": implementation_spec_dict(current_spec),
        "fixed_interface": {name: f"{name}(self, {_parameter(name)})" for name in FIXED_FUNCTIONS},
        "signature_version": FUNCTION_SIGNATURE_VERSION,
        "domain_types": {
            "state": "EnvironmentObservation with episode, LOBState, and causal recent_events",
            "observation_and_context": "JSON-serializable dictionaries",
            "completed_experiences": "tuple[Experience, ...] containing completed historical episodes only",
            "constraints": "dict with nonempty allowed_offsets subset of [0,1,2] and included fallback_offset",
        },
        # Deliberately omit inference request payloads/raw envelopes. They are not
        # causal market evidence and keeping them out makes the credential boundary
        # explicit (the CompletionClient's private API key is never accepted here).
        "completed_trajectory": {
            "augmented_context": trajectory.augmented_context,
            "proposed_actions": [asdict(action) for action in trajectory.proposed_actions],
            "pre_action_evaluations": list(trajectory.pre_action_evaluations),
            "final_action": asdict(trajectory.final_action),
            "parse_errors": list(trajectory.parse_errors),
            "latency_ms": trajectory.latency_ms,
            "token_usage": trajectory.token_usage,
        },
        "parsing_telemetry": {"parse_success": not trajectory.parse_errors, "parse_errors": list(trajectory.parse_errors), "fallback_used": bool(trajectory.parse_errors)},
        "selected_action": asdict(trajectory.final_action),
        "objective_outcome": asdict(outcome),
        "evidence": asdict(evidence),
        "allowed_evidence_refs": sorted(allowed_evidence_refs(evidence)),
        "prior_implementation_attempts": prior,
        "immutable_constraints": ["fixed interface and decision model", "environment/replay/reward/evidence are immutable", "one implementation only", "no future episode information", "no API credentials"],
    }
    return [{"role": "system", "content": IMPLEMENTATION_FACTORY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(supplied, sort_keys=True, separators=(",", ":"))}]


def _parameter(name: str) -> str:
    return {"observe": "state", "augment": "observation", "constrain": "context", "evaluate": "context", "feedback": "completed_experiences"}[name]


class LLMImplementationFactory:
    backend_name = "llm-generate-implementation"

    def __init__(self, config: FactoryModelConfig, client: CompletionClient):
        self.config, self.client = config, client

    def propose(self, current_spec: HarnessImplementationSpec, evidence: Evidence,
                prior_attempts: tuple[dict[str, Any], ...], trajectory: AgentTrajectory,
                outcome: OrderOutcome) -> tuple[ImplementationProposal | None, ImplementationFactoryCall]:
        messages = build_implementation_factory_messages(current_spec, evidence, prior_attempts, trajectory, outcome)
        prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        payload = {"model": self.config.model_id, "messages": messages, "temperature": self.config.temperature,
                   "top_p": self.config.top_p, "max_tokens": self.config.max_output_tokens, "stream": False,
                   "provider": {"only": [self.config.upstream_provider], "allow_fallbacks": False}}

        def parse(response: dict) -> ImplementationProposal:
            if response.get("model") != self.config.model_id or response.get("provider") != self.config.upstream_provider:
                raise ValueError("factory returned an unpinned model or provider")
            return parse_implementation_proposal(response["choices"][0]["message"].get("content"))

        run = complete_with_retries(self.client, payload, self.config.timeout_seconds, self.config.max_retries,
                                    parse, _attempt)
        usage = response_usage(run.response)
        return run.parsed, ImplementationFactoryCall(self.backend_name, IMPLEMENTATION_FACTORY_PROMPT_VERSION,
            self.config.config_hash, prompt_hash, tuple(messages), run.raw, run.parsed,
            usage.get("prompt_tokens"), usage.get("completion_tokens"), run.latency_ms, run.retry_count,
            tuple(run.attempts), run.error)


def _attempt(index: int, response: dict | None, started: float, error: str | None,
             api_error: dict | None = None) -> FactoryAttempt:
    usage = response_usage(response)
    return FactoryAttempt(index, response.get("id") if response else None,
        response.get("model") if response else None, response.get("provider") if response else None,
        usage.get("prompt_tokens"), usage.get("completion_tokens"), round((time.monotonic() - started) * 1000),
        error, api_error)
