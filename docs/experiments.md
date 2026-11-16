# Running and analyzing experiments

Start with the [README](../README.md) for installation, data setup, and a three-generation offline example. The [architecture guide](architecture.md) explains how the loop works.

## Configuration used for the blog

The recorded primary experiment used one AAPL session from June 21, 2012:

| Setting                     | Value                                            |
| --------------------------- | ------------------------------------------------ |
| Data                        | Matching 10-level LOBSTER message and book files |
| Episode count               | 26, approximately 09:30–15:45 New York time      |
| Decision spacing            | 15 minutes                                       |
| Order quantity              | 100 shares                                       |
| Passive offsets             | 0, 1, 2 ticks                                    |
| Replay and markout horizons | 30 seconds each                                  |
| Tick size                   | 100 integer price units, or one cent             |
| Queue model                 | `conservative`                                   |
| Promotion rule              | `unconditional`                                  |
| Decision and Factory model  | `deepseek/deepseek-v4-flash-0731`                |
| Inference access / upstream | OpenRouter / DeepInfra                           |

These are the settings behind the [blog](https://jmadison.me/posts/self-improving-harness-implementation/) and [saved report](../runs/20260810T212427Z-18ae798e/delivery_report.md). Provider/model availability and fresh LLM outputs can change; a new invocation is not guaranteed to reproduce the recorded decisions or rewards. The current Factory prompt is `factory-v5`; the recorded run used `factory-v4`. Current windowed evidence and reversal checks also differ from that historical run. Its saved artifacts remain the evidence for the blog's numerical claims.

## Start a live run

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`. The environment file is ignored by Git. Credentials are read at runtime and are not saved in run artifacts.

To check the decision adapter with one remote completion:

```bash
uv run --env-file .env obh smoke-agent --upstream-provider DeepInfra
```

To use the full-session configuration:

```bash
uv run --env-file .env obh run \
  --agent openrouter --upstream-provider DeepInfra \
  --factory-backend llm --factory-upstream-provider DeepInfra \
  --queue-model conservative --promotion-rule unconditional \
  --tickers AAPL --episodes-per-dataset 30 --generations 26
```

The episode cap of 30 allows all 26 eligible AAPL windows; the generation limit consumes that sequence. A normal generation makes one logical decision call and one logical Factory call. Transient retries are additional physical requests and are recorded separately. Invalid model output is rejected without a repair call or silent model/provider substitution.

Use `uv run obh run --help` for all current options. The default run includes five tickers, so use `--tickers AAPL` when only AAPL data is installed. `--depth` selects the input CSV depth; it does not set the active harness's visible depth, which starts at one level and may change through mutations.

## Inspect the saved experiment without inference

The tracked primary run is [`runs/20260810T212427Z-18ae798e/`](../runs/20260810T212427Z-18ae798e/). You can read its report and plots without installing dependencies or downloading market data.

To recalculate derived analysis from its recorded counterfactual rewards:

```bash
uv run obh analyze runs/20260810T212427Z-18ae798e
```

This reads `generations.jsonl` and writes `analysis/report.json` and `analysis/plots/` inside that run directory. It does not modify the original result artifacts, replay market data, or call a model.

| Metric                 | Meaning                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ |
| Chosen reward          | Score for the action selected during the generation                                  |
| Best-action reward     | Maximum recorded counterfactual reward across the three offsets                      |
| Decision regret        | Best-action reward minus chosen reward                                               |
| Constant outcome       | All three offsets have the same reward                                               |
| Outcome differentiated | At least two offsets have different rewards                                          |
| Per-episode oracle     | Best offset chosen after observing outcomes; an upper bound, not a deployable policy |

The primary run has 18 constant-outcome episodes and 8 differentiated episodes. Its selected orders filled in 3 episodes: 1 full fill and 2 partial fills. Only one selected action earned a positive reward. The evolving harness scored −11,953.85 mean reward versus +21,207.69 for always offset 2. These scores do not establish profitable trading or performance improvement from adaptation.

## Output files

Each run creates a new `runs/<UTC timestamp>-<config hash>/` directory. Generated runs and downloaded data are ignored by Git; the primary demonstration run is retained as evidence.

| File                                        | Use                                                                    |
| ------------------------------------------- | ---------------------------------------------------------------------- |
| `run.json`                                  | Run status and fixed configuration, including model/provider settings  |
| `report.json`                               | Evaluation definitions and aggregate results                           |
| `generations.jsonl`                         | Per-episode actions, rewards, counterfactuals, regret, and transitions |
| `experiences.jsonl`                         | Decision inputs, trajectories, and observed outcomes                   |
| `factory_calls.jsonl`                       | Factory prompts, responses, attempts, usage, and errors                |
| `harnesses.jsonl`, `lineage.json`           | Configurations and the sequence of active harnesses                    |
| `successors.jsonl`, `transitions.jsonl`     | Candidate validation and acceptance/rejection                          |
| `episodes.jsonl`, `environment_config.json` | Episode sequence and replay configuration                              |
| `plots/`                                    | Plots derived from generation telemetry                                |

[`example-llm-factory-generation.json`](example-llm-factory-generation.json) is an illustrative record with placeholder identifiers from an earlier implementation, not the first actual mutation from the published run. Use the saved run for exact evidence.

## Optional research settings

The following controls are separate from the primary result:

- **`--promotion-rule gated`:** roll a newly promoted harness back after its first outcome-differentiated measurement if decision regret is positive. This uses within-episode counterfactual outcomes, not a causal comparison with the previous harness.
- **`--queue-model l3_fifo`:** experimental reconstruction using identifiable order IDs and eligible cancellations/deletions. Matching full-session depth-30 or depth-50 exports were unavailable for validation; these results are excluded from the main claim.
- **`--evidence-window N`:** average up to N completed experiences for the Factory, alongside the latest episode. Default: 5. This evidence can span multiple harness versions.
- **`--tickers ...`:** select one to five instruments. The supported sample universe is AAPL, AMZN, GOOG, INTC, and MSFT when matching files are installed. Multi-instrument support does not make the reported AAPL result a multi-market experiment.

The heuristic Factory follows a finite deterministic mutation ladder for offline checks. Exhausting it records an invalid transition with an exhaustion reason; it does not switch to an LLM. Neither the offline heuristic nor an operationally valid successor is evidence of performance improvement.

## Checks

```bash
uv run python -m unittest discover -s tests -v
```

The suite uses synthetic data and fake inference clients. It covers queue replay, action enforcement, features, proposal validation, generation sequencing, persistence, baselines, and experimental rollback. It must not make OpenRouter calls.

For local data validation, `uv run obh validate-data` checks up to 10,000 rows per discovered pair. Add `--full` to check every row.
