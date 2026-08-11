from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import AgentConfig, FixedAgent, OpenRouterClient
from .data import discover_datasets, validate_dataset
from .domain import to_dict
from .experiment import ExperimentConfig, run_experiment
from .persistence import analyze_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="obh", description="Self-improving order-book harness")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-data"); validate.add_argument("--data-dir", default="data"); validate.add_argument("--full", action="store_true")
    smoke = sub.add_parser("smoke-agent", help="make one pinned OpenRouter completion and print parser/token telemetry")
    smoke.add_argument("--upstream-provider", required=True)
    analyze = sub.add_parser("analyze", help="write derived analysis under an existing run without modifying its published artifacts or making API calls")
    analyze.add_argument("run_dir", help="path to an existing runs/<run_id> directory containing generations.jsonl")
    run = sub.add_parser("run")
    run.add_argument("--data-dir", default="data")
    run.add_argument("--output-dir", default="runs")
    run.add_argument("--depth", type=int, default=10)
    run.add_argument("--tickers", nargs="+", metavar="TICKER", default=["AAPL", "AMZN", "GOOG", "INTC", "MSFT"], help="one to five instruments; defaults to the five-instrument research universe")
    run.add_argument("--episodes-per-dataset", type=int, default=30)
    run.add_argument("--generations", type=int, default=3)
    run.add_argument("--spacing-ms", type=int, default=900_000, help="decision-point spacing; default 15 minutes spans the session")
    run.add_argument("--queue-model", choices=("conservative", "l3_fifo"), default="conservative", help="deterministic passive-order queue model")
    run.add_argument("--agent", choices=("offline", "openrouter"), default="offline")
    run.add_argument("--upstream-provider", default="", help="required with --agent openrouter; provider is pinned and fallbacks remain disabled")
    run.add_argument("--factory-backend", choices=("heuristic", "llm"), default="heuristic")
    run.add_argument("--factory-upstream-provider", default="", help="required with --factory-backend llm; may equal the separately configured decision provider")
    run.add_argument("--factory-max-output-tokens", type=int, default=16_384)
    run.add_argument("--promotion-rule", choices=("unconditional", "gated"), default="unconditional", help="gated enables a non-causal first-measurement decision-regret rollback heuristic for newly promoted successors (V0.3)")
    run.add_argument("--evidence-window", type=int, default=5, help="trailing completed-episode window aggregated into factory evidence")
    args = parser.parse_args(argv)
    if args.command == "validate-data":
        results = [validate_dataset(p, full=args.full) for p in discover_datasets(args.data_dir)]; print(json.dumps(results, indent=2)); return 0
    if args.command == "analyze":
        summary = analyze_run(Path(args.run_dir))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "smoke-agent":
        config = AgentConfig(upstream_provider=args.upstream_provider)
        agent = FixedAgent(config, OpenRouterClient())
        messages = [
            {"role": "system", "content": "Choose exactly one passive offset from [0,1,2]. Return only JSON: {\"offset_ticks\": integer}."},
            {"role": "user", "content": json.dumps({"parent_order": {"side": "BUY", "quantity": 100}, "book": {"bid_prices": [10000], "bid_sizes": [500], "ask_prices": [10100], "ask_sizes": [400]}, "features": {"spread_ticks": 1.0}, "allowed_offsets": [0, 1, 2]})},
        ]
        action, record = agent.act(messages, "smoke-harness")
        passed = action is not None and record.error is None
        print(json.dumps(to_dict({"passed": passed, "action": action, "inference_record": record}), indent=2, sort_keys=True))
        return 0 if passed else 1
    config = ExperimentConfig(data_dir=args.data_dir, output_dir=args.output_dir, depth=args.depth, tickers=tuple(args.tickers), agent=args.agent, upstream_provider=args.upstream_provider, max_episodes_per_dataset=args.episodes_per_dataset, max_generations=args.generations, spacing_ms=args.spacing_ms, queue_model=args.queue_model, factory_backend=args.factory_backend, factory_upstream_provider=args.factory_upstream_provider, factory_max_output_tokens=args.factory_max_output_tokens, promotion_rule=args.promotion_rule, evidence_window=args.evidence_window)
    root = run_experiment(config); print(root); return 0


if __name__ == "__main__": raise SystemExit(main())
