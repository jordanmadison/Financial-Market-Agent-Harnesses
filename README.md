# Self-improving harness for limit-order placement

Can an LLM make better placement decisions by changing the harness around it? This Python research prototype lets a second LLM propose one constrained harness change after each historical market episode.

**The harness adapted, but the reported experiment did not demonstrate performance improvement.** The evolving agent underperformed a static policy that always chose offset 2.

Read the [companion blog](https://jmadison.me/posts/self-improving-harness-implementation/) for the motivation, worked example, results, and future directions. This repository contains the implementation and recorded evidence behind that post.

## How it works

```text
Market snapshot + active harness → decision agent chooses a price offset
→ local replay measures the outcome → Factory proposes one configuration change
→ local validation accepts the successor or keeps the current harness
→ next chronological episode
```

The harness controls visible market data, computed features, decision instructions, allowed offsets, and outcome history. The Factory changes that configuration; it generates no Python code and trains no model weights. In the reported run, acceptance means the successor can run, not that it performs better.

Each episode supplies a buy or sell instruction for 100 shares. The agent chooses a passive offset of 0, 1, or 2 ticks, and the simulator replays 30 seconds of historical events. See [the architecture guide](docs/architecture.md) for the components and a reading path through the code.

## Run an offline example

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/). The runtime uses only the Python standard library.

1. Install the project:

   ```bash
   uv sync --frozen
   ```

2. Place matching 10-level LOBSTER message and order-book CSVs in `data/`. Market data is distributed separately and is not included in a clone. For the example below, use the AAPL sample pair:

   ```text
   data/AAPL_2012-06-21_34200000_57600000_message_10.csv
   data/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
   ```

   Sample data is available from [LOBSTER](https://php.lobsterdata.com/info/DataSamples.php).

3. Validate the files and run three generations:

   ```bash
   uv run obh validate-data
   uv run obh run --tickers AAPL --episodes-per-dataset 5 --generations 3
   ```

The default decision agent and Factory are deterministic offline heuristics. This checks the execution loop without an API key or paid inference; it does not reproduce the blog's LLM decisions. Each invocation prints a new run directory under `runs/`. Start with its `report.json` for results and `generations.jsonl` for individual decisions.

## Run with LLMs

Both LLM roles use `deepseek/deepseek-v4-flash-0731` through OpenRouter. The reported experiment pinned DeepInfra as the upstream provider. Each role has its own prompt, configuration, and telemetry; provider fallbacks are disabled.

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`, then run:

```bash
uv run --env-file .env obh run \
  --tickers AAPL --episodes-per-dataset 5 --generations 3 \
  --agent openrouter --upstream-provider DeepInfra \
  --factory-backend llm --factory-upstream-provider DeepInfra
```

This command makes remote inference requests. The [experiment guide](docs/experiments.md) contains the full-session configuration, analysis commands, artifact reference, and optional research settings. Recorded model/provider identifiers describe the experiment; their present availability is not guaranteed.

## The result behind the blog

The saved [AAPL run](runs/20260810T212427Z-18ae798e/) contains 26 episodes from one trading session. It accepted 23 mutations, rejected 3 proposals, and used 20 distinct harness specifications. All offsets received identical rewards in 18 episodes.

| Policy           | Mean reward per episode |
| ---------------- | ----------------------: |
| Evolving harness |              −11,953.85 |
| Always offset 0  |              −14,744.23 |
| Always offset 1  |               −9,750.00 |
| Always offset 2  |              +21,207.69 |

Reward is an internal score combining execution value, fees, and an unfilled-order penalty; these numbers are not dollar profits. Baselines use the same recorded counterfactual outcomes. See the [experiment report](runs/20260810T212427Z-18ae798e/delivery_report.md) for the evidence and limitations. The run uses conservative fill estimates and assumes no market impact; it does not establish profitability or generalization.

## Develop and explore

```bash
uv run python -m unittest discover -s tests -v
uv run obh --help
uv run obh run --help
```

Tests use small synthetic books and mocked inference, so they require neither downloaded market data nor an API key.

- [Architecture and code map](docs/architecture.md): connect the blog's concepts to the implementation.
- [Running and analyzing experiments](docs/experiments.md): commands, settings, and outputs.
- [Recorded experiment report](runs/20260810T212427Z-18ae798e/delivery_report.md): inspect the published run without executing anything.
- [Original design specification](self-improving-order-book-harness-v0-spec.md): detailed design history, including plans that differ from the current implementation.
