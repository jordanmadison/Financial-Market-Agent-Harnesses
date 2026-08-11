from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

from ..completion import CompletionClient, complete_with_retries, response_usage
from ..domain import AgentTrajectory, OrderOutcome
from ..harness import HarnessSpec, canonical_spec, harness_language_prompt
from ..registry import metric_definitions_prompt
from .evidence import allowed_evidence_refs
from .types import (
    Evidence,
    FACTORY_PROMPT_VERSION,
    FactoryAttempt,
    FactoryCallRecord,
    FactoryModelConfig,
    FactoryProposal,
    SuccessorAttempt,
)
from .validation import parse_factory_proposal, validate_factory_proposal

FACTORY_SYSTEM_PROMPT = """You are a constrained harness factory. Diagnose the supplied current harness using only the supplied completed-experience telemetry and objective evidence, then choose exactly one legal atomic harness mutation. Evidence aggregate_metrics average a trailing window of completed episodes (sample_count episodes); the latest_episode metric slice isolates the newest episode. Never invent metrics or evidence references. diagnosis.evidence_refs must contain one or more exact strings copied verbatim from allowed_evidence_refs; do not construct paths, expressions, or references containing metric values. diagnosis.confidence must be a JSON number between 0 and 1 inclusive — not a percentage, not a string. Never modify the environment, reward, evaluator, episode ordering, compiler, model, provider, or inference configuration. Never generate code. Never propose the exact inverse of a recently promoted mutation; reversals within the cooldown are rejected. You propose; deterministic local checks decide only whether the successor is operationally valid, not whether it performs better. Return only one JSON object with exactly these objects: diagnosis {affected_metric,evidence_refs,suspected_harness_component,suspected_cause,confidence}, inspection {component,current_behavior,limitation}, improvement {component,rationale,expected_metric_effect}, mutation {operation,target_path,value}."""


class LLMFactory:
    backend_name = "llm"

    def __init__(self, config: FactoryModelConfig, client: CompletionClient):
        self.config = config
        self.client = client

    def propose(
        self,
        current_spec: HarnessSpec,
        evidence: Evidence,
        prior_attempts: tuple[SuccessorAttempt, ...],
        trajectory: AgentTrajectory | None = None,
        outcome: OrderOutcome | None = None,
    ) -> tuple[FactoryProposal | None, FactoryCallRecord]:
        messages = build_factory_messages(current_spec, evidence, prior_attempts, trajectory, outcome)
        prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_output_tokens,
            "stream": False,
            "provider": {"only": [self.config.upstream_provider], "allow_fallbacks": False},
        }

        def parse_response(response: dict) -> FactoryProposal:
            if response.get("model") != self.config.model_id:
                raise ValueError(f"factory returned unexpected model {response.get('model')!r}")
            if response.get("provider") != self.config.upstream_provider:
                raise ValueError(f"factory returned unexpected provider {response.get('provider')!r}")
            raw = response["choices"][0]["message"].get("content")
            if not raw:
                raise ValueError("empty factory response")
            proposal = parse_factory_proposal(raw, evidence)
            validate_factory_proposal(proposal, current_spec, evidence, prior_attempts)
            return proposal

        run = complete_with_retries(
            self.client,
            payload,
            self.config.timeout_seconds,
            self.config.max_retries,
            parse_response,
            _factory_attempt,
        )
        usage = response_usage(run.response)
        record = FactoryCallRecord(
            self.backend_name,
            FACTORY_PROMPT_VERSION,
            self.config.config_hash,
            self.config.model_id,
            run.response.get("model") if run.response else None,
            run.response.get("provider") if run.response else None,
            prompt_hash,
            tuple(messages),
            run.raw,
            run.parsed,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            run.latency_ms,
            run.retry_count,
            tuple(run.attempts),
            run.error,
        )
        return run.parsed, record


def build_factory_messages(
    current_spec: HarnessSpec,
    evidence: Evidence,
    prior_attempts: tuple[SuccessorAttempt, ...],
    trajectory: AgentTrajectory | None = None,
    outcome: OrderOutcome | None = None,
) -> list[dict]:
    bounded = prior_attempts[-10:]
    prior = [
        {
            "successor_harness_id": item.successor_id,
            "status": item.decision,
            "mutation": asdict(item.mutation) if item.mutation else None,
            "diagnosis_refs": item.diagnosis_refs,
        }
        for item in bounded
    ]
    supplied = {
        "current_spec": json.loads(canonical_spec(current_spec)),
        "evidence": asdict(evidence),
        "allowed_evidence_refs": sorted(allowed_evidence_refs(evidence)),
        "metric_definitions": metric_definitions_prompt(),
        "harness_language": harness_language_prompt(),
        "invariants": [
            "one atomic mutation",
            "fallback_offset must remain allowed",
            "feedback metrics require nonzero history",
            "candidate must differ from incumbent",
        ],
        "prior_attempts": prior,
        "successor_limit": 1,
    }
    if trajectory is not None and outcome is not None:
        supplied["completed_experience"] = {
            "selected_action": asdict(trajectory.final_action),
            "parse_success": not trajectory.parse_errors,
            "parse_errors": list(trajectory.parse_errors),
            "fallback_used": bool(trajectory.parse_errors),
            "features_available": sorted(trajectory.augmented_context.get("features", {})),
            "outcome": {
                "filled_quantity": outcome.filled_quantity,
                "fill_ratio": outcome.fill_ratio,
                "time_to_first_fill": outcome.time_to_first_fill,
                "time_to_full_fill": outcome.time_to_full_fill,
                "spread_capture": outcome.spread_capture,
                "signed_markout": outcome.signed_markout,
                "unfilled_penalty": outcome.unfilled_penalty,
                "reward": outcome.reward,
            },
        }
    return [
        {"role": "system", "content": FACTORY_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(supplied, sort_keys=True, separators=(",", ":"))},
    ]


def _factory_attempt(index: int, response: dict | None, started: float, error: str | None, api_error: dict | None) -> FactoryAttempt:
    usage = response_usage(response)
    return FactoryAttempt(
        index,
        response.get("id") if response else None,
        response.get("model") if response else None,
        response.get("provider") if response else None,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        round((time.monotonic() - started) * 1000),
        error,
        api_error,
    )
