# Self-Improving Harness for Limit-Order Placement

The below is AI-Generated. Here is a blog written by me, https://jmadison.me/posts/self-improving-harness-implementation/

A standard-library-only research prototype that evolves a constrained agent harness while placing passive orders against historical LOBSTER order-book data.

Each generation performs exactly this sequence:

```text
active harness
→ one decision-agent call on one chronological episode
→ deterministic market replay and objective outcome
→ deterministic evidence construction
→ one Factory call proposing one atomic HarnessSpec mutation
→ deterministic operational validation
→ promote a runnable successor, otherwise retain the active harness
```

Only one harness is active at a time. The Factory generates a typed mutation—not Python, simulator logic, or reward code. No model weights are trained, and validation makes no LLM calls. Performance is measured when a harness executes during its normal generation; historical harnesses are not rerun for evaluation.

## Scope

The frozen MVP configuration uses:

- aligned 10-level LOBSTER message and order-book files;
- a 100-share passive order;
- offsets `{0, 1, 2}`;
- 30-second replay and markout horizons;
- 15-minute chronological episode spacing;
- the existing reward definition;
- the default `conservative` queue model.

The repository supports AAPL, AMZN, GOOG, INTC, and MSFT sample files when they are present under `data/`. The reported primary experiment uses one AAPL session. It is a mechanism demonstration, not evidence of trading profitability or generalization.

## Setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended:

```bash
uv sync --frozen
uv run obh validate-data
```

LOBSTER files must be matching message/order-book pairs with names such as:

```text
AAPL_2012-06-21_34200000_57600000_message_10.csv
AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
```

For OpenRouter, create a local environment file and add the credential without committing it:

```bash
cp .env.example .env
# Set OPENROUTER_API_KEY in .env.
```

`.env` is ignored. The API key is read from `OPENROUTER_API_KEY` at runtime and is never persisted in run artifacts.

## Run locally without API calls

The default offline agent and deterministic heuristic Factory provide a reproducible end-to-end check:

```bash
uv run obh run \
  --agent offline \
  --factory-backend heuristic \
  --tickers AAPL \
  --episodes-per-dataset 5 \
  --generations 3
```

The heuristic Factory follows a finite deterministic mutation ladder. It emits the first legal non-no-op mutation, never cycles to a previous specification, and records an `INVALID` transition with a distinct exhaustion reason after all rungs are consumed.

## Run with the LLM Factory

Both LLM roles use the fixed dated model `deepseek/deepseek-v4-flash-0731` through explicitly pinned OpenRouter upstream providers. Their prompts, configurations, hashes, usage, latency, retries, and errors remain separate.

Test the decision adapter with one completion:

```bash
uv run --env-file .env obh smoke-agent --upstream-provider DeepInfra
```

Run a small live experiment:

```bash
uv run --env-file .env obh run \
  --agent openrouter \
  --upstream-provider DeepInfra \
  --factory-backend llm \
  --factory-upstream-provider DeepInfra \
  --queue-model conservative \
  --tickers AAPL \
  --episodes-per-dataset 5 \
  --generations 3 \
  --promotion-rule unconditional
```

On the normal path, each generation makes two logical LLM calls: one decision call and one Factory call. Transient retries are additional physical requests and are recorded. Provider substitution, model fallback, malformed proposals, illegal mutations, and model-based output repair are disabled.

## Reproduce the primary experiment

The reported run consumes all 26 eligible AAPL episodes from approximately 09:30 through 15:45:

```bash
uv run --env-file .env obh run \
  --agent openrouter \
  --upstream-provider DeepInfra \
  --factory-backend llm \
  --factory-upstream-provider DeepInfra \
  --queue-model conservative \
  --tickers AAPL \
  --episodes-per-dataset 30 \
  --generations 26 \
  --promotion-rule unconditional
```

The persisted primary run is at [`runs/20260810T212427Z-18ae798e/`](runs/20260810T212427Z-18ae798e/), with a narrative summary in its [`delivery_report.md`](runs/20260810T212427Z-18ae798e/delivery_report.md).

## Primary result

The run demonstrated constrained sequential self-modification, but it did **not** demonstrate performance self-improvement:

- 26 measured generations;
- 23 operationally valid mutation promotions and 3 invalid proposals;
- 20 distinct harness IDs used during the session;
- 18 constant-outcome episodes and only 8 outcome-differentiated episodes;
- 3 selected-action fills: 1 full and 2 partial;
- 1 selected action with positive reward;
- mean differentiated-episode decision regret of `113,881.25`.

Constant-offset baselines are calculated from the same persisted per-generation counterfactual outcomes. They require no additional replay or inference:

| Policy | Mean reward |
| --- | ---: |
| Evolving agent | −11,953.85 |
| Always offset 0 | −14,744.23 |
| Always offset 1 | −9,750.00 |
| Always offset 2 | **+21,207.69** |
| Per-episode oracle | +23,086.54 |

The evolving agent underperformed always-offset-2 by `33,161.54` mean reward per episode. The final generated successor is recorded as operationally valid but unmeasured; it is not labeled the best harness.

## Evaluation definitions

- **Chosen reward:** reward from the action selected during the generation's decision call.
- **Best-action reward:** maximum deterministic counterfactual reward among the legal offsets.
- **Decision regret:** `best_action_reward - chosen_action_reward`.
- **Constant outcome:** every legal offset has the same reward.
- **Outcome differentiated:** at least two legal offsets have different rewards.
- **Per-episode oracle:** the best legal counterfactual action after observing all outcomes; it is a non-causal upper bound, not a deployable policy.

Counterfactual action outcomes are reporting telemetry only. They are not supplied to the decision agent or Factory and do not affect reward, mutation generation, or promotion.

## Promotion rules

`--promotion-rule unconditional` is the default and the rule used for the primary result. Every operationally valid successor becomes the next active harness without a reward comparison. Promotions therefore mean “runnable,” not “demonstrated improvement.”

`--promotion-rule gated` is an optional V0.3 sensitivity rule. It rolls a newly promoted harness back when its first outcome-differentiated measurement has positive within-episode decision regret. This is a deterministic, non-causal rollback heuristic—not evidence that the mutation caused a regression.

Factory evidence uses a trailing completed-episode window configured by `--evidence-window` (default `5`) and includes a separate `latest_episode` slice. Proposal validation prevents an exact inverse of a recently promoted mutation for three attempts.

## Artifacts and analysis

Every run is written to a new, non-overwriting `runs/<UTC timestamp>-<config hash>/` directory. Important files include:

- `run.json`: status and fixed run configuration;
- `generations.jsonl`: active harness, episode outcome, counterfactuals, regret, usage, latency, and errors;
- `experiences.jsonl`: decision trajectories and objective outcomes;
- `factory_calls.jsonl`: Factory prompts, responses, attempts, and telemetry;
- `successors.jsonl` and `transitions.jsonl`: candidate validation and lineage transitions;
- `harnesses.jsonl` and `lineage.json`: canonical harness specifications and active lineage;
- `report.json`: definitions and aggregate results;
- `plots/`: plots derived from persisted generation telemetry.

Generate derived baseline analysis without replay, API calls, or modification of the completed run artifacts:

```bash
uv run obh analyze runs/<run-id>
```

Derived output is written beneath `runs/<run-id>/analysis/`.

## Tests

The deterministic test suite is offline and must not make OpenRouter calls:

```bash
uv run python -m unittest discover -s tests -v
```

## Queue-model limitation

`conservative` is the default and the only queue model used for the primary result. It treats displayed same-price volume at submission as ahead of the hypothetical order and credits visible executions against that queue. Cancellations do not reduce queue-ahead volume, so fills are conservative lower-bound estimates.

`l3_fifo` is an optional experimental sensitivity model using identifiable L3 order IDs and eligible cancellation/deletion credits. Matching full-session depth-30 or depth-50 exports were unavailable for validation, so `l3_fifo` results are excluded from the main claim.

## Known limitations

- The sample covers one trading session; nearby episodes are not independent market regimes.
- Eighteen of the 26 primary episodes contain no action differentiation.
- Historical replay assumes the hypothetical order has no market impact.
- The conservative queue model intentionally understates fills.
- The experiment does not establish trading profitability, cross-session performance, or cross-regime generalization.
