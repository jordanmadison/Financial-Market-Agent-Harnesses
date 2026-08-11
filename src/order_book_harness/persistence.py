from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import DatasetPair, validate_dataset
from .domain import Experience, to_dict
from .environment import EnvironmentConfig
from .episodes import Episode
from .factory import SuccessorAttempt
from .generation import GenerationRecord
from .harness import H0, HarnessSpec, canonical_spec, harness_id


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    config: Any
    agent_manifest: dict[str, Any]
    factory_manifest: dict[str, Any]
    env_config: EnvironmentConfig
    pairs: list[DatasetPair]
    episodes: list[Episode]
    evolution_episodes: list[Episode]
    max_generations: int
    harness_specs: dict[str, HarnessSpec]
    generations: tuple[GenerationRecord, ...]
    successors: tuple[SuccessorAttempt, ...]
    factory_calls: tuple[dict[str, Any], ...]
    experiences: tuple[Experience, ...]
    lineage_ids: tuple[str, ...]


def build_lineage(accepted_ids: list[str] | tuple[str, ...], generations: list[dict] | tuple[GenerationRecord, ...]) -> dict:
    telemetry = [item.as_telemetry() if isinstance(item, GenerationRecord) else item for item in generations]
    return {
        "initial_harness_id": accepted_ids[0],
        "promoted_harness_ids": list(accepted_ids[1:]),
        "final_harness_id": telemetry[-1]["successor_harness_id"] if telemetry else accepted_ids[0],
        "transitions": [
            {
                "generation": item["generation"],
                "parent_harness_id": item["active_harness_id"],
                "successor_harness_id": item["successor_harness_id"],
                "status": item["transition_status"],
            }
            for item in telemetry
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(value), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            handle.write(json.dumps(to_dict(value), sort_keys=True) + "\n")


def render_plots(root: Path, generations: list[dict] | tuple[GenerationRecord, ...]) -> None:
    telemetry = [item.as_telemetry() if isinstance(item, GenerationRecord) else item for item in generations]
    (root / "plots").mkdir(parents=True, exist_ok=True)
    points = lambda field: [(f"g{entry['generation']}", entry[field]) for entry in telemetry]
    _svg_plot(root / "plots" / "reward.svg", "Observed reward by generation", points("reward"))
    _svg_plot(root / "plots" / "fill_ratio.svg", "Observed fill ratio by generation", points("fill_ratio"))
    _svg_plot(root / "plots" / "signed_markout.svg", "Observed signed markout by generation", points("signed_markout"))
    _svg_plot(root / "plots" / "unfilled_penalty.svg", "Observed unfilled penalty by generation", points("unfilled_penalty"))
    _svg_plot(
        root / "plots" / "best_available_reward.svg",
        "Best available reward by generation",
        [(f"g{entry['generation']}", entry["opportunity"]["best_available_reward"]) for entry in telemetry],
    )
    _svg_plot(
        root / "plots" / "decision_regret.svg",
        "Decision regret by generation",
        [(f"g{entry['generation']}", entry["opportunity"]["decision_regret"]) for entry in telemetry],
    )
    _svg_plot(
        root / "plots" / "opportunity_capture.svg",
        "Opportunity capture when available",
        [
            (f"g{entry['generation']}", 1.0 if entry["opportunity"]["captured"] else 0.0)
            for entry in telemetry
            if entry["opportunity"]["captured"] is not None
        ],
    )
    cumulative_reward = []
    reward_total = 0.0
    differentiated_regret = []
    regret_total = 0.0
    regret_count = 0
    for entry in telemetry:
        reward_total += entry["reward"]
        cumulative_reward.append((f"g{entry['generation']}", reward_total / (entry["generation"] + 1)))
        if entry["opportunity"]["outcome_differentiated"]:
            regret_total += entry["opportunity"]["decision_regret"] or 0.0
            regret_count += 1
            differentiated_regret.append((f"g{entry['generation']}", regret_total / regret_count))
    _svg_plot(root / "plots" / "expanding_mean_reward.svg", "Expanding mean reward", cumulative_reward)
    _svg_plot(
        root / "plots" / "expanding_mean_differentiated_regret.svg",
        "Expanding mean regret on outcome-differentiated episodes",
        differentiated_regret,
    )
    baseline_series = baseline_policy_rewards(telemetry)
    if baseline_series:
        labels = [f"g{entry['generation']}" for entry in telemetry]
        cumulative = {}
        for name, values in baseline_series.items():
            total = 0.0
            points = []
            for label, value in zip(labels, values):
                total += value
                points.append((label, total))
            cumulative[name] = points
        _svg_multi_plot(
            root / "plots" / "baseline_comparison.svg",
            "Cumulative reward: agent vs constant-offset baselines vs oracle",
            cumulative,
        )


def persist_run_artifacts(root: Path, artifacts: RunArtifacts) -> None:
    staging = root / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / "plots").mkdir(parents=True, exist_ok=True)
    telemetry = [record.as_telemetry() for record in artifacts.generations]
    write_json(staging / "environment_config.json", artifacts.env_config)
    write_json(staging / "dataset_manifest.json", [validate_dataset(pair) for pair in artifacts.pairs])
    write_json(
        staging / "evolution_manifest.json",
        {"episode_ids": [episode.episode_id for episode in artifacts.evolution_episodes[: artifacts.max_generations]]},
    )
    write_jsonl(staging / "episodes.jsonl", artifacts.episodes)
    write_jsonl(
        staging / "harnesses.jsonl",
        [
            {
                "harness_id": hid,
                "compiler_version": "1",
                "language_version": "1",
                "canonical_spec": canonical_spec(spec),
                "spec": json.loads(canonical_spec(spec)),
            }
            for hid, spec in artifacts.harness_specs.items()
        ],
    )
    write_jsonl(staging / "factory_calls.jsonl", artifacts.factory_calls)
    write_jsonl(staging / "successors.jsonl", artifacts.successors)
    write_jsonl(
        staging / "transitions.jsonl",
        [
            {
                "generation": item["generation"],
                "parent_harness_id": item["active_harness_id"],
                "successor_harness_id": item["successor_harness_id"],
                "status": item["transition_status"],
                "mutation": item["mutation"],
                "validation": item["validation"],
            }
            for item in telemetry
        ],
    )
    write_jsonl(staging / "generations.jsonl", telemetry)
    write_jsonl(staging / "experiences.jsonl", artifacts.experiences)
    write_json(staging / "lineage.json", build_lineage(artifacts.lineage_ids, artifacts.generations))
    write_json(
        staging / "report.json",
        {
            "evolution_mode": "direct_successor",
            "promoted_lineage": list(artifacts.lineage_ids),
            "generation_telemetry": telemetry,
            "final_successor": {
                "harness_id": telemetry[-1]["successor_harness_id"] if telemetry else harness_id(H0),
                "measured": False if telemetry and telemetry[-1]["transition_status"] == "PROMOTED" else bool(telemetry),
            },
            "limitations": [
                "no-impact historical replay",
                f"{artifacts.env_config.queue_model} queue model",
                "sample data covers a single trading date",
            ],
            "scope": (
                f"sequential one-episode evolution across {len(artifacts.config.tickers)} "
                f"instruments at common depth {artifacts.config.depth}"
            ),
            "evaluation_definitions": {
                "chosen_reward": "reward of the action selected during the normal generation call",
                "best_action_reward": "maximum deterministic reward across legal offsets 0, 1, and 2",
                "decision_regret": "best_action_reward minus chosen_reward",
                "constant_outcome": "all three legal offsets have equal scalar reward",
                "outcome_differentiated": "at least two legal offsets have different scalar rewards",
            },
            "performance_summary": _performance_summary(telemetry),
        },
    )
    render_plots(staging, artifacts.generations)
    # Complete marker is written last inside staging, then promoted with every other artifact.
    write_json(
        staging / "run.json",
        {
            "run_id": artifacts.run_id,
            "status": "complete",
            "config": asdict(artifacts.config),
            "fixed_agent": artifacts.agent_manifest,
            "factory": artifacts.factory_manifest,
        },
    )
    _promote_staging(staging, root)


def _performance_summary(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [item["reward"] for item in telemetry]
    differentiated = [item for item in telemetry if item["opportunity"]["outcome_differentiated"]]
    regrets = [item["opportunity"]["decision_regret"] or 0.0 for item in differentiated]
    return {
        "generation_count": len(telemetry),
        "constant_outcome_episodes": sum(item["opportunity"]["constant_outcome"] for item in telemetry),
        "outcome_differentiated_episodes": len(differentiated),
        "promoted_successors": sum(item["transition_status"] == "PROMOTED" for item in telemetry),
        "invalid_successors": sum(item["transition_status"] == "INVALID" for item in telemetry),
        "reverted_successors": sum(item["transition_status"] == "REVERTED" for item in telemetry),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "final_expanding_mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "mean_regret_on_outcome_differentiated_episodes": sum(regrets) / len(regrets) if regrets else None,
        "baseline_summary": _baseline_summary(telemetry),
    }


def baseline_policy_rewards(telemetry: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Per-generation reward series for the agent, each constant-offset policy, and the oracle.

    Constant-offset policies come from the recorded counterfactuals, so they cost no
    additional replay or inference. Only offsets present in every generation's
    counterfactual report are included.
    """
    if not telemetry:
        return {}
    per_generation = [
        {outcome["offset_ticks"]: outcome["reward"] for outcome in item["opportunity"]["action_outcomes"]}
        for item in telemetry
    ]
    common_offsets = set(per_generation[0])
    for outcomes in per_generation[1:]:
        common_offsets &= set(outcomes)
    series: dict[str, list[float]] = {
        f"always_offset_{offset}": [outcomes[offset] for outcomes in per_generation]
        for offset in sorted(common_offsets)
    }
    series["agent"] = [item["reward"] for item in telemetry]
    series["oracle"] = [item["opportunity"]["best_action_reward"] for item in telemetry]
    return series


def _baseline_summary(telemetry: list[dict[str, Any]]) -> dict[str, Any] | None:
    series = baseline_policy_rewards(telemetry)
    if not series:
        return None
    differentiated_indices = [
        index for index, item in enumerate(telemetry) if item["opportunity"]["outcome_differentiated"]
    ]
    policies = {
        name: {
            "mean_reward": sum(values) / len(values),
            "cumulative_reward": sum(values),
            "mean_reward_outcome_differentiated": (
                sum(values[index] for index in differentiated_indices) / len(differentiated_indices)
                if differentiated_indices
                else None
            ),
        }
        for name, values in series.items()
    }
    constants = {name: stats for name, stats in policies.items() if name.startswith("always_offset_")}
    best_constant = max(constants, key=lambda name: constants[name]["mean_reward"]) if constants else None
    return {
        "definition": (
            "constant-offset baselines are computed from the recorded per-generation "
            "counterfactual rewards; no additional replay or inference was performed"
        ),
        "policies": policies,
        "best_constant_baseline": best_constant,
        "agent_minus_best_constant_mean_reward": (
            policies["agent"]["mean_reward"] - constants[best_constant]["mean_reward"] if best_constant else None
        ),
    }


def analyze_run(root: Path) -> dict[str, Any]:
    """Create derived performance analysis without modifying completed run artifacts.

    Reads persisted generation telemetry only; never reruns harnesses or calls any API.
    Writes ``analysis/report.json`` and ``analysis/plots/`` beneath the run directory;
    the original report, plots, and completion marker remain unchanged.
    """
    generations_path = root / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(f"{generations_path} does not exist; analyze requires a persisted run")
    telemetry = [json.loads(line) for line in generations_path.read_text().splitlines() if line.strip()]
    if not telemetry:
        raise ValueError(f"{generations_path} contains no generation records")
    summary = _performance_summary(telemetry)
    analysis_root = root / "analysis"
    write_json(
        analysis_root / "report.json",
        {
            "kind": "derived_post_run_analysis",
            "source": "../generations.jsonl",
            "performance_summary": summary,
        },
    )
    render_plots(analysis_root, telemetry)
    return summary


def mark_run_failed(
    root: Path,
    run_id: str,
    error: str,
    *,
    config: Any | None = None,
    agent_manifest: dict[str, Any] | None = None,
    factory_manifest: dict[str, Any] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "status": "failed",
        "error": error[:4000],
        "config": asdict(config) if config is not None else None,
        "fixed_agent": agent_manifest,
        "factory": factory_manifest,
    }
    write_json(root / "run.json", payload)


def _promote_staging(staging: Path, root: Path) -> None:
    for path in sorted(staging.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(staging)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    shutil.rmtree(staging, ignore_errors=True)


_MULTI_PLOT_COLORS = ("#2563eb", "#16a34a", "#d97706", "#7c3aed", "#dc2626", "#0891b2")


def _svg_multi_plot(path: Path, title: str, series_by_name: dict[str, list[tuple[str, float]]]) -> None:
    width, height, margin = 800, 420, 55
    values = [value for series in series_by_name.values() for _, value in series] or [0.0]
    lo, hi = min(values), max(values)
    if lo == hi:
        lo -= 1
        hi += 1
    length = max(len(series) for series in series_by_name.values())
    x_for = lambda index: margin + index * (width - 2 * margin) / max(1, length - 1)
    y_for = lambda value: height - margin - (value - lo) * (height - 2 * margin) / (hi - lo)
    polylines = []
    legend = []
    for slot, (name, series) in enumerate(sorted(series_by_name.items())):
        color = _MULTI_PLOT_COLORS[slot % len(_MULTI_PLOT_COLORS)]
        points = " ".join(f"{x_for(index):.1f},{y_for(value):.1f}" for index, (_, value) in enumerate(series))
        polylines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = 40 + slot * 16
        legend.append(
            f'<line x1="{width - 195}" y1="{legend_y}" x2="{width - 170}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
            f'<text x="{width - 163}" y="{legend_y + 4}">{name}</text>'
        )
    labels = next(iter(sorted(series_by_name.items())))[1] if series_by_name else []
    ticks = "".join(
        f'<text x="{x_for(index):.1f}" y="{height - 18}" text-anchor="middle">{label}</text>'
        for index, (label, _) in enumerate(labels)
        if length <= 30 or index % max(1, length // 15) == 0
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<style>text{{font:12px sans-serif}}</style>'
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-size="18">{title}</text>'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>'
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>'
        f'<text x="8" y="{margin}">{hi:.2f}</text><text x="8" y="{height - margin}">{lo:.2f}</text>'
        f'{"".join(polylines)}{"".join(legend)}{ticks}</svg>'
    )


def _svg_plot(path: Path, title: str, series: list[tuple[str, float]]) -> None:
    width, height, margin = 800, 420, 55
    values = [value for _, value in series] or [0]
    lo, hi = min(values), max(values)
    if lo == hi:
        lo -= 1
        hi += 1
    points = []
    for index, (_, value) in enumerate(series):
        x = margin + index * (width - 2 * margin) / max(1, len(series) - 1)
        y = height - margin - (value - lo) * (height - 2 * margin) / (hi - lo)
        points.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4"/><text x="{x:.1f}" y="{height - 18}" text-anchor="middle">{series[index][0]}</text>'
        for index, (x, y) in enumerate(points)
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<style>text{{font:12px sans-serif}} polyline{{fill:none;stroke:#2563eb;stroke-width:3}} circle{{fill:#dc2626}}</style>'
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-size="18">{title}</text>'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>'
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>'
        f'<text x="8" y="{margin}">{hi:.2f}</text><text x="8" y="{height - margin}">{lo:.2f}</text>'
        f'<polyline points="{polyline}"/>{dots}</svg>'
    )
