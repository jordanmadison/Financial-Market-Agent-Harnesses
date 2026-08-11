from __future__ import annotations

from .evidence import (
    allowed_evidence_refs,
    apply_mutation,
    canonical_mutation,
    derive_episode_evidence,
    derive_evidence,
    derive_window_evidence,
    spec_diff,
)
from .heuristic import HeuristicFactory
from .llm import LLMFactory, build_factory_messages
from .types import (
    FACTORY_PROMPT_VERSION,
    Diagnosis,
    Evidence,
    FactoryBackend,
    FactoryCallRecord,
    FactoryModelConfig,
    FactoryProposal,
    HarnessInspection,
    HarnessMutation,
    Improvement,
    SuccessorAttempt,
)
from .validation import parse_factory_proposal, validate_factory_proposal

__all__ = [
    "FACTORY_PROMPT_VERSION",
    "Diagnosis",
    "Evidence",
    "FactoryBackend",
    "FactoryCallRecord",
    "FactoryModelConfig",
    "FactoryProposal",
    "HarnessInspection",
    "HarnessMutation",
    "HeuristicFactory",
    "Improvement",
    "LLMFactory",
    "SuccessorAttempt",
    "allowed_evidence_refs",
    "apply_mutation",
    "build_factory_messages",
    "canonical_mutation",
    "derive_evidence",
    "derive_episode_evidence",
    "derive_window_evidence",
    "parse_factory_proposal",
    "spec_diff",
    "validate_factory_proposal",
]
