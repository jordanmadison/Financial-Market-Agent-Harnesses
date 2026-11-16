# Full-session AAPL mechanism demonstration

## Scope and configuration

This is the primary MVP demonstration run. It uses one AAPL trading session, the default `conservative` queue model, 30-second replay and markout horizons, a 100-share order, passive offsets `{0, 1, 2}`, and the existing reward. Each of 26 generations used one chronological 15-minute episode, one logical decision-agent call, and one logical LLM-factory call.

Command:

```sh
uv run --env-file .env obh run \
  --agent openrouter \
  --upstream-provider DeepInfra \
  --factory-backend llm \
  --factory-upstream-provider DeepInfra \
  --queue-model conservative \
  --tickers AAPL \
  --episodes-per-dataset 30 \
  --generations 26
```

The configured episode limit exceeds the 26 eligible AAPL windows; the generation limit therefore consumes the complete sequence. Timestamps span 09:30:00.888 through 15:45:08.601 New York session time.

## Execution summary

| Measure | Result |
|---|---:|
| Generations / measured harness executions | 26 |
| Distinct active harness IDs | 20 |
| Promoted successors | 23 |
| Invalid proposals retained at parent | 3 |
| Constant-outcome episodes | 18 |
| Outcome-differentiated episodes | 8 |
| Mean reward | -11,953.85 |
| Mean reward, generations 0–12 | -16,369.23 |
| Mean reward, generations 13–25 | -7,538.46 |
| Mean regret, differentiated episodes | 113,881.25 |
| Decision calls, logical / physical | 26 / 28 |
| Factory calls, logical / physical | 26 / 27 |
| Decision tokens, input / output | 19,690 / 93,553 |
| Factory tokens, input / output | 49,161 / 56,380 |

The extra physical requests were retries, not additional logical generations. Generation 4's decision call timed out twice at the enforced 180-second wall-clock deadline and then succeeded on its third attempt. Generation 2's factory call succeeded after one retry. No generation was lost.

## Lineage and transitions

The initial harness was `e41aadfa6cfc3df994526f43`; the final promoted, not-yet-measured successor was `d613235a104e09681b35e022`. The full machine-readable lineage is in `lineage.json`, with one transition per generation in `transitions.jsonl`.

Promoted mutations, in generation order:

| Generation | Mutation |
|---:|---|
| 0 | Set `observation.book_levels` to 3 |
| 1 | Set procedure to `compare_all_actions` |
| 2 | Add `level1_imbalance` |
| 3 | Add `recent_trade_imbalance` |
| 5 | Add `microprice` |
| 6 | Remove offset 0 |
| 7 | Add offset 0 |
| 8 | Set `observation.book_levels` to 5 |
| 9 | Remove offset 0 |
| 11 | Add offset 0 |
| 12 | Add `short_term_volatility` |
| 13 | Set procedure to `analyze_then_choose` |
| 14 | Remove offset 2 |
| 15 | Add `multilevel_imbalance` |
| 16 | Set event lookback to 25 |
| 17 | Set procedure to `compare_all_actions` |
| 18 | Remove offset 0 |
| 19 | Add offset 0 |
| 20 | Add `cancel_rate` |
| 21 | Remove offset 0 |
| 23 | Add offset 0 |
| 24 | Add offset 2 |
| 25 | Set procedure to `propose_critic_revise` |

Generations 4, 10, and 22 retained their parent because the factory returned an out-of-range confidence value. Promotions mean operational validation passed; they do not mean reward improvement was demonstrated. Several add/remove changes revisit earlier specifications, which is visible in the lineage rather than hidden.

## Performance interpretation

The later half had a less-negative mean reward than the first half. Among the eight differentiated episodes, mean regret fell from 186,800 over the first four such episodes to 40,962.50 over the last four. This is suggestive, but it is not sufficient evidence of learned decision-quality improvement: opportunity magnitudes differ substantially by episode, 18 of 26 episodes give every legal action the same reward, and nearby observations from one session are not independent regimes.

## Baseline comparison (recomputed from persisted counterfactuals)

Constant-offset baselines derived from this run's own recorded counterfactual rewards (`obh analyze`; no additional replay or inference):

| Policy | Mean reward / generation | Cumulative reward |
|---|---:|---:|
| Always offset 0 | -14,744.23 | -383,350 |
| Always offset 1 | -9,750.00 | -253,500 |
| Always offset 2 | +21,207.69 | +551,400 |
| Evolving harness (agent) | -11,953.85 | -310,800 |
| Per-episode oracle | +23,086.54 | +600,250 |

The evolving system underperformed the static "always place 2 ticks deep" baseline by 33,161.54 mean reward per generation in this session; that baseline captured nearly the entire oracle value. The apparent first-half/second-half improvement is dominated by single-episode outliers (generation 13 at +172,000 and generation 25 at -160,000). This run therefore demonstrates the self-modification mechanism, audit trail, and telemetry — it does not demonstrate performance self-improvement. Any future performance claim must beat these constant-offset baselines, which are now computed and plotted (`plots/baseline_comparison.svg`, `report.json` `performance_summary.baseline_summary`) for every run.

Only generation 13 selected a positive-reward action. A differentiated episode can still have zero regret when the agent selects the best action and all available rewards are non-positive. The final successor has no assigned performance because it was generated after the last measured execution.

## Evaluation definitions

- `chosen reward`: reward produced by the selected offset under deterministic replay.
- `best-action reward`: maximum counterfactual reward among the three legal offsets.
- `decision regret`: `best_action_reward - chosen_action_reward`.
- `constant_outcome`: all three counterfactual rewards are equal.
- `outcome_differentiated`: at least two counterfactual rewards differ.
- Expanding-mean differentiated regret includes only generations where `outcome_differentiated` is true.

## Artifacts

- `generations.jsonl`: per-generation timestamp, active harness/version, selected action and reward, all counterfactual rewards, regret, outcome classification, successor, transition reason, usage, latency, and errors.
- `harnesses.jsonl`, `successors.jsonl`, `transitions.jsonl`, `lineage.json`: auditable harness specifications and lineage.
- `experiences.jsonl`, `factory_calls.jsonl`: decision trajectories and factory-call telemetry.
- `report.json`, `progress.json`: aggregate and plotting inputs.
- `plots/reward.svg`, `plots/decision_regret.svg`, `plots/expanding_mean_reward.svg`, and `plots/expanding_mean_differentiated_regret.svg`: requested performance views generated from persisted telemetry without rerunning harnesses.

## Limitations

This is a one-session mechanism demonstration. Nearby episodes are not independent market regimes. The conservative queue model gives a lower-bound fill estimate. `l3_fifo` remains experimental and its results are excluded from this primary claim. These results do not establish trading profitability or cross-session or cross-regime generalization.
