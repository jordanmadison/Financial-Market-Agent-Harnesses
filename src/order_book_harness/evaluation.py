from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .domain import AgentTrajectory, Experience, InferenceRecord, PlacementAction
from .environment import OrderBookEnvironment
from .harness import CompiledHarness
from .registry import AGGREGATE_METRICS, METRIC_REGISTRY


class ActingAgent(Protocol):
    def act(self, messages: list[dict], harness_id: str) -> tuple[PlacementAction | None, InferenceRecord]: ...


@dataclass(frozen=True)
class EvaluationResult:
    harness_id: str
    episode_ids: tuple[str, ...]
    aggregate_metrics: dict[str, float]
    standard_errors: dict[str, float]
    experiences: tuple[Experience, ...]


def run_harness(agent: ActingAgent, harness: CompiledHarness, environment: OrderBookEnvironment, episodes: list, history: tuple[Experience, ...] = ()) -> EvaluationResult:
    experiences = []
    for episode in episodes:
        obs = environment.reset(episode); messages, context = harness.build_messages(obs, history + tuple(experiences), environment.config.tick_size); start = time.monotonic()
        proposed, record = agent.act(messages, harness.harness_id); final, errors = harness.enforce(proposed)
        outcome = environment.step(final); latency = round((time.monotonic() - start) * 1000)
        trajectory = AgentTrajectory(messages, context, (() if proposed is None else (proposed,)), (), final, record, errors, latency, {"input": record.input_tokens or 0, "output": record.output_tokens or 0})
        exp_id = hashlib.sha256(f"{episode.episode_id}:{harness.harness_id}:{final.offset_ticks}".encode()).hexdigest()[:24]
        experiences.append(Experience(exp_id, episode.episode_id, harness.harness_id, trajectory, outcome))
    return aggregate(harness.harness_id, experiences)


def aggregate(harness_id: str, experiences: list[Experience]) -> EvaluationResult:
    if not experiences:
        return EvaluationResult(harness_id, (), {}, {}, ())
    fields = {
        metric.value: [METRIC_REGISTRY[metric].from_experience(experience) or 0.0 for experience in experiences]
        for metric in AGGREGATE_METRICS
    }
    means = {key: sum(values) / len(values) for key, values in fields.items()}
    errors = {
        key: (
            math.sqrt(sum((value - means[key]) ** 2 for value in values) / (len(values) - 1)) / math.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
        for key, values in fields.items()
    }
    return EvaluationResult(harness_id, tuple(experience.episode_id for experience in experiences), means, errors, tuple(experiences))


def metric_slices(result: EvaluationResult, episode_tickers: dict[str, str]) -> dict[str, dict[str, float]]:
    """Return objective aggregates by instrument without changing selection metrics."""
    slices: dict[str, dict[str, float]] = {}
    for ticker in sorted({episode_tickers[e.episode_id] for e in result.experiences}):
        subset = [e for e in result.experiences if episode_tickers[e.episode_id] == ticker]
        slices[ticker] = aggregate(result.harness_id, subset).aggregate_metrics
    return slices


class ConstantAgent:
    def __init__(self, offset: int): self.offset = offset
    def act(self, messages: list[dict], harness_id: str):
        action = PlacementAction(self.offset)
        return action, InferenceRecord(None, f"constant-{self.offset}", f"constant-{self.offset}", "local", f"constant-{self.offset}", harness_id, "", None, action, None, None, 0, 0, None)


class RandomAgent:
    def __init__(self, seed: int = 0): self.rng = random.Random(seed)
    def act(self, messages: list[dict], harness_id: str):
        allowed = tuple(json.loads(messages[-1]["content"])["allowed_offsets"])
        action = PlacementAction(self.rng.choice(allowed))
        return action, InferenceRecord(None, "random", "random", "local", "random-fixed-seed", harness_id, "", None, action, None, None, 0, 0, None)
