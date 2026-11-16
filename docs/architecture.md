# Architecture and code map

The [blog](https://jmadison.me/posts/self-improving-harness-implementation/) explains the experiment and its findings. This guide connects that explanation to the code. For setup and an offline run, start with the [README](../README.md).

## The system in one paragraph

A local Python process loads LOBSTER market data and selects chronological episodes. A decision agent receives a prompt built by the active harness and returns a price offset. The simulator estimates fills and calculates reward. A separate Factory examines the completed outcome and proposes one allowed harness change. Deterministic local checks either accept that successor for the next episode or retain the current harness. The application saves decisions, proposals, and lineage for analysis.

In a live run, the two LLM roles call OpenRouter through the configured upstream provider. Replay, feature computation, validation, and persistence stay local. The offline adapters replace inference with deterministic heuristics so the same loop can run without network calls.

The blog's Hyperliquid data, continual-learning idea, and multi-action runtime are future directions. This implementation uses LOBSTER data and one placement decision per episode.

## Follow one generation

Read these functions in order; the complete specification is not a prerequisite:

1. [`cli.main`](../src/order_book_harness/cli.py) parses the command into an `ExperimentConfig` and starts the experiment.
2. [`experiment.run_experiment`](../src/order_book_harness/experiment.py) loads the data and adapters, then carries one active specification through the chronological episode sequence.
3. [`generation.execute_generation`](../src/order_book_harness/generation.py) runs the active harness on one episode, builds evidence, requests a Factory proposal, validates a successor, and records the transition.
4. [`evaluation.run_harness`](../src/order_book_harness/evaluation.py) resets the environment, asks the harness to build a prompt, calls the agent, enforces its action, and replays the selected order.
5. [`harness.CompiledHarness`](../src/order_book_harness/harness.py) determines the context visible to the agent and enforces allowed offsets and fallback behavior.
6. [`environment.OrderBookEnvironment`](../src/order_book_harness/environment.py) converts the offset into a price and estimates fills and reward from recorded events.

`RunAccumulator` carries completed experiences, proposals, and lineage between generations. `persistence.py` writes the run artifacts and creates reports and plots.

## What changes, and what stays fixed

`HarnessSpec` is a configuration with five parts:

| Blog concept                | Specification field | Effect                                                             |
| --------------------------- | ------------------- | ------------------------------------------------------------------ |
| What the agent sees         | `observation`       | Select book depth, recent-event lookback, and raw-event visibility |
| Computed context            | `augmentation`      | Select existing features such as spread or imbalance               |
| How the agent is instructed | `decision`          | Select a reasoning instruction and whether to request a rationale  |
| Action constraints          | `constraints`       | Restrict offsets and choose fallback behavior                      |
| Prior outcomes              | `feedback`          | Select history length and displayed metrics                        |

The initial configuration is `H0` in `harness.py`. For example, the first mutation in the blog's run changes visible depth from one level to three:

```json
{ "operation": "SET", "target_path": "observation.book_levels", "value": 3 }
```

[`registry.FIELD_REGISTRY`](../src/order_book_harness/registry.py) defines permitted fields, values, and mutation operations. `apply_harness_mutation` updates the configuration. `compile_harness` validates it and wraps it in the existing implementation. It does not generate or compile Python source.

A decision procedure such as `compare_all_actions` changes instructions within one decision call. It does not create a multi-agent workflow or additional inference calls. Model weights, reward, simulator, and feature implementations remain fixed during a run.

## The Factory and validation

The Factory package separates a few responsibilities:

- [`types.py`](../src/order_book_harness/factory/types.py) defines evidence, proposals, inference records, and the backend interface.
- [`evidence.py`](../src/order_book_harness/factory/evidence.py) builds objective metrics from completed experiences. The current default averages up to five episodes and supplies a separate latest-episode slice.
- [`llm.py`](../src/order_book_harness/factory/llm.py) constructs the Factory prompt and requests one JSON proposal through OpenRouter.
- [`validation.py`](../src/order_book_harness/factory/validation.py) checks structure, evidence references, allowed mutations, and consistency. The current code also rejects an exact inverse of a promoted mutation during a three-attempt cooldown.
- [`heuristic.py`](../src/order_book_harness/factory/heuristic.py) provides a finite mutation ladder for offline checks.

A proposal includes a diagnosis, inspection, expected effect, and one mutation. Diagnoses are model hypotheses; validation does not establish their causal accuracy. Output restrictions are checked after generation rather than enforced through grammar-constrained decoding.

`generation.validate_successor` checks the resulting specification, compiled interface, prompt construction, action enforcement, and fallback. These checks use no LLM calls and do not compare candidate reward with incumbent reward.

With the default `unconditional` promotion rule, every operationally valid successor becomes active on the next episode. `PROMOTED` means operational acceptance. The optional `gated` rollback heuristic is described in the [experiment guide](experiments.md#optional-research-settings).

## Replay and evaluation

The environment supplies side and quantity; the decision agent chooses only offset 0, 1, or 2. A buy price is `best_bid − offset × tick_size`; a sell price is `best_ask + offset × tick_size`.

The default conservative model puts displayed same-price volume ahead of the hypothetical order. Compatible visible executions consume that queue first. Cancellations do not reduce queue-ahead volume. The optional L3 model is a sensitivity experiment, not the queue model behind the blog's result.

Reward is filled quantity multiplied by the signed difference between terminal midpoint and fill price, minus fees and an unfilled-share penalty. Prices retain LOBSTER's integer scale of 10,000 units per dollar. The default fee is zero and the unfilled penalty is 100 internal units per share; an unfilled 100-share order scores −10,000.

For reporting, the environment also replays all three offsets on the same episode. These counterfactuals support constant-offset baselines and decision regret. They are not supplied to either LLM. With the default promotion rule they do not affect transitions; the optional gated rule uses regret for rollback.

A successor is measured when it later executes as the active harness. The last generated successor in a completed run has no subsequent episode and remains unmeasured. Rewards on different chronological episodes do not isolate the causal effect of a mutation.

## Where to look next

- [`agent.py`](../src/order_book_harness/agent.py) and [`completion.py`](../src/order_book_harness/completion.py): decision inference, parsing, timeouts, retries, and telemetry.
- [`data.py`](../src/order_book_harness/data.py) and [`episodes.py`](../src/order_book_harness/episodes.py): aligned CSV input and episode construction.
- [`domain.py`](../src/order_book_harness/domain.py): market observations, actions, trajectories, and outcomes.
- [`tests/test_direct_successor.py`](../tests/test_direct_successor.py): executable examples of generation sequencing and optional rollback.
- [`tests/test_core.py`](../tests/test_core.py): small examples of action prices, queue fills, and reward.
