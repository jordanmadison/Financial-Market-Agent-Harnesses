from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from ..agent import MODEL_ID
from ..domain import AgentTrajectory, OrderOutcome
from ..harness import HarnessMutation, HarnessSpec

FACTORY_PROMPT_VERSION = "factory-v5"
SuccessorDecision = Literal["PROMOTED", "INVALID", "REVERTED"]


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    experience_ids: tuple[str, ...]
    aggregate_metrics: dict[str, float]
    metric_slices: dict[str, dict[str, float]]
    sample_count: int = 1


@dataclass(frozen=True)
class Diagnosis:
    diagnosis_id: str
    affected_metric: str
    evidence_refs: tuple[str, ...]
    suspected_harness_component: str
    suspected_cause: str
    confidence: float


@dataclass(frozen=True)
class HarnessInspection:
    component: str
    current_behavior: str
    limitation: str


@dataclass(frozen=True)
class Improvement:
    component: str
    rationale: str
    expected_metric_effect: str


@dataclass(frozen=True)
class FactoryProposal:
    diagnosis: Diagnosis
    inspection: HarnessInspection
    improvement: Improvement
    mutation: HarnessMutation


@dataclass(frozen=True)
class FactoryAttempt:
    attempt_index: int
    request_id: str | None
    returned_model: str | None
    upstream_provider: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    error: str | None
    api_error: dict[str, Any] | None


@dataclass(frozen=True)
class FactoryCallRecord:
    backend: str
    prompt_version: str
    factory_config_hash: str | None
    requested_model: str | None
    returned_model: str | None
    upstream_provider: str | None
    prompt_hash: str
    messages: tuple[dict, ...]
    raw_response: str | None
    parsed_proposal: FactoryProposal | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    retry_count: int
    attempts: tuple[FactoryAttempt, ...]
    error: str | None


@dataclass(frozen=True)
class SuccessorAttempt:
    successor_id: str | None
    parent_harness_id: str
    successor_spec: HarnessSpec | None
    diagnosis_refs: tuple[str, ...]
    mutation: HarnessMutation | None
    validation_results: dict[str, Any]
    decision: SuccessorDecision
    factory_call: FactoryCallRecord


class FactoryBackend(Protocol):
    backend_name: str

    def propose(
        self,
        current_spec: HarnessSpec,
        evidence: Evidence,
        prior_attempts: tuple[SuccessorAttempt, ...],
        trajectory: AgentTrajectory | None = None,
        outcome: OrderOutcome | None = None,
    ) -> tuple[FactoryProposal | None, FactoryCallRecord]: ...


@dataclass(frozen=True)
class FactoryModelConfig:
    provider: Literal["openrouter"] = "openrouter"
    model_id: Literal["deepseek/deepseek-v4-flash-0731"] = MODEL_ID
    upstream_provider: str = ""
    allow_fallbacks: Literal[False] = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 16_384
    timeout_seconds: int = 180
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID or self.allow_fallbacks is not False:
            raise ValueError("factory model and fallback policy are fixed")
        if not self.upstream_provider:
            raise ValueError("factory upstream provider must be pinned")
        if self.max_output_tokens <= 0 or self.timeout_seconds <= 0 or self.max_retries < 0:
            raise ValueError("invalid factory inference bounds")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
