from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import AgentConfig, FixedAgent, HeuristicAgent, OpenRouterClient
from .data import DatasetPair, discover_datasets
from .environment import EnvironmentConfig, OrderBookEnvironment
from .episodes import Episode, EpisodeConfig, generate_episodes, load_dataset
from .evaluation import ActingAgent
from .factory import FACTORY_PROMPT_VERSION, FactoryBackend, FactoryModelConfig, HeuristicFactory, LLMFactory
from .generation import GenerationInput, GenerationServices, RunAccumulator, apply_promotion_rule, execute_generation
from .harness import H0, harness_id
from .persistence import RunArtifacts, build_lineage, mark_run_failed, persist_run_artifacts, render_plots, write_json

# Re-export for tests and callers that historically imported from experiment.
__all__ = [
    "ExperimentConfig",
    "RunContext",
    "build_lineage",
    "run_experiment",
]


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: str = "data"
    output_dir: str = "runs"
    tick_size: int = 100
    quantity: int = 100
    replay_horizon_ms: int = 30_000
    markout_horizon_ms: int = 30_000
    spacing_ms: int = 900_000
    queue_model: str = "conservative"
    max_episodes_per_dataset: int = 30
    depth: int = 10
    tickers: tuple[str, ...] = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")
    agent: str = "offline"
    upstream_provider: str = ""
    max_generations: int = 3
    evolution_mode: str = "direct_successor"
    factory_backend: str = "heuristic"
    factory_upstream_provider: str = ""
    factory_max_output_tokens: int = 16_384
    promotion_rule: str = "unconditional"
    evidence_window: int = 5
    seed: int = 7

    def __post_init__(self) -> None:
        if self.depth not in (1, 3, 5, 10):
            raise ValueError("depth must be supported by the harness language: 1, 3, 5, or 10")
        if self.queue_model not in ("conservative", "l3_fifo"):
            raise ValueError("queue_model must be conservative or l3_fifo")
        if not 1 <= len(self.tickers) <= 5 or len(set(self.tickers)) != len(self.tickers):
            raise ValueError("tickers must contain one to five unique instruments")
        if self.agent not in ("offline", "openrouter"):
            raise ValueError("agent must be offline or openrouter")
        if self.agent == "openrouter" and not self.upstream_provider:
            raise ValueError("--upstream-provider is required for OpenRouter runs")
        if self.max_generations <= 0 or self.max_episodes_per_dataset <= 0:
            raise ValueError("generation and episode limits must be positive")
        if self.evolution_mode != "direct_successor":
            raise ValueError("only direct_successor evolution is supported by the primary runner")
        if self.factory_backend not in ("heuristic", "llm"):
            raise ValueError("factory_backend must be heuristic or llm")
        if self.factory_backend == "llm" and not self.factory_upstream_provider:
            raise ValueError("--factory-upstream-provider is required for LLM factory runs")
        if self.promotion_rule not in ("unconditional", "gated"):
            raise ValueError("promotion_rule must be unconditional or gated")
        if self.evidence_window < 1:
            raise ValueError("evidence_window must be at least one episode")


@dataclass
class RunContext:
    config: ExperimentConfig
    root: Path
    run_id: str
    pairs: list[DatasetPair]
    episodes: list[Episode]
    evolution_episodes: list[Episode]
    env: OrderBookEnvironment
    env_config: EnvironmentConfig
    agent: ActingAgent
    agent_manifest: dict[str, Any]
    factory_backend: FactoryBackend
    factory_manifest: dict[str, Any]
    episode_tickers: dict[str, str]


def run_experiment(config: ExperimentConfig = ExperimentConfig(), factory_backend_override: FactoryBackend | None = None) -> Path:
    ctx = _setup_run(config, factory_backend_override)
    write_json(
        ctx.root / "run.json",
        {
            "run_id": ctx.run_id,
            "status": "in_progress",
            "config": asdict(config),
            "fixed_agent": ctx.agent_manifest,
            "factory": ctx.factory_manifest,
        },
    )
    accumulator = RunAccumulator(
        incumbent_spec=H0,
        harness_specs={harness_id(H0): H0},
        lineage_ids=(harness_id(H0),),
    )
    services = GenerationServices(ctx.root, ctx.agent, ctx.env, ctx.factory_backend, config.evidence_window)
    try:
        for generation, episode in enumerate(ctx.evolution_episodes[: config.max_generations]):
            result = execute_generation(
                services,
                GenerationInput(
                    generation=generation,
                    incumbent_spec=accumulator.incumbent_spec,
                    episode=episode,
                    prior_experiences=accumulator.experiences,
                    prior_successors=accumulator.successors,
                ),
            )
            result = apply_promotion_rule(
                result,
                accumulator.generations[-1] if accumulator.generations else None,
                accumulator.harness_specs,
                config.promotion_rule,
            )
            accumulator = accumulator.apply(result)
        persist_run_artifacts(
            ctx.root,
            RunArtifacts(
                run_id=ctx.run_id,
                config=ctx.config,
                agent_manifest=ctx.agent_manifest,
                factory_manifest=ctx.factory_manifest,
                env_config=ctx.env_config,
                pairs=ctx.pairs,
                episodes=ctx.episodes,
                evolution_episodes=ctx.evolution_episodes,
                max_generations=ctx.config.max_generations,
                harness_specs=accumulator.harness_specs,
                generations=accumulator.generations,
                successors=accumulator.successors,
                factory_calls=accumulator.factory_calls,
                experiences=accumulator.experiences,
                lineage_ids=accumulator.lineage_ids,
            ),
        )
    except Exception as exc:
        mark_run_failed(
            ctx.root,
            ctx.run_id,
            f"{type(exc).__name__}: {exc}",
            config=ctx.config,
            agent_manifest=ctx.agent_manifest,
            factory_manifest=ctx.factory_manifest,
        )
        raise
    return ctx.root


def _setup_run(config: ExperimentConfig, factory_backend_override: FactoryBackend | None) -> RunContext:
    available = discover_datasets(config.data_dir)
    selected = {(pair.ticker, pair.levels): pair for pair in available}
    missing = [ticker for ticker in config.tickers if (ticker, config.depth) not in selected]
    if missing:
        raise ValueError(f"depth {config.depth} is unavailable for: {', '.join(missing)}")
    pairs = [selected[(ticker, config.depth)] for ticker in config.tickers]
    loaded = {pair.dataset_id: load_dataset(pair) for pair in pairs}
    ep_config = EpisodeConfig(
        config.quantity,
        config.replay_horizon_ms,
        config.markout_horizon_ms,
        100,
        config.spacing_ms,
        config.max_episodes_per_dataset,
    )
    episodes = [episode for data in loaded.values() for episode in generate_episodes(data, ep_config)]
    episodes.sort(key=lambda episode: (episode.parent_order.decision_time, episode.ticker, episode.episode_id))
    if len(episodes) < config.max_generations:
        raise ValueError(f"only {len(episodes)} eligible evolution episodes for {config.max_generations} generations")
    env_config = EnvironmentConfig(config.tick_size, config.queue_model)
    env = OrderBookEnvironment(loaded, env_config)
    if config.agent == "openrouter":
        agent_config = AgentConfig(upstream_provider=config.upstream_provider)
        agent: ActingAgent = FixedAgent(agent_config, OpenRouterClient())
        agent_manifest = {"kind": "openrouter", "config": asdict(agent_config), "config_hash": agent_config.config_hash}
    else:
        agent = HeuristicAgent()
        agent_manifest = {"kind": "offline", "model": "offline-heuristic-v1", "config_hash": agent.config_hash}
    if factory_backend_override is not None:
        factory_backend = factory_backend_override
        factory_manifest = {"kind": factory_backend.backend_name}
    elif config.factory_backend == "llm":
        factory_config = FactoryModelConfig(
            upstream_provider=config.factory_upstream_provider,
            max_output_tokens=config.factory_max_output_tokens,
        )
        factory_backend = LLMFactory(factory_config, OpenRouterClient())
        factory_manifest = {
            "kind": "llm",
            "prompt_version": FACTORY_PROMPT_VERSION,
            "config": asdict(factory_config),
            "config_hash": factory_config.config_hash,
        }
    else:
        factory_backend = HeuristicFactory()
        factory_manifest = {"kind": "heuristic", "prompt_version": FACTORY_PROMPT_VERSION}
    run_material = json.dumps({"config": asdict(config), "episodes": [episode.episode_id for episode in episodes]}, sort_keys=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + hashlib.sha256(run_material.encode()).hexdigest()[:8]
    root = Path(config.output_dir) / run_id
    (root / "plots").mkdir(parents=True)
    episode_tickers = {episode.episode_id: episode.ticker for episode in episodes}
    return RunContext(
        config,
        root,
        run_id,
        pairs,
        episodes,
        episodes,
        env,
        env_config,
        agent,
        agent_manifest,
        factory_backend,
        factory_manifest,
        episode_tickers,
    )


# Compatibility aliases used by existing tests during the generation refactor.
_render_plots = render_plots
_write_json = write_json
