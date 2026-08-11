# Self-Improving Order-Book Placement Agent — V0 Specification

**Status:** Draft for implementation  
**Version:** 0.2  
**Date:** 2026-08-09

## 1. Purpose

Implement a minimal experimental framework that tests whether a fixed agent can improve its order-book placement performance through iterative modification of its agent harness.

The assignment is:

> Implement a self-improving agent harness for a toy investment program of your choice and show the performance of the self-improving harness over time.

This specification decomposes that assignment into two coupled systems:

1. **Toy investment program:** historical limit-order placement using LOBSTER order-book data.
2. **Self-improving agent harness:** an evidence-driven factory that proposes one constrained `HarnessSpec` successor after each completed episode and promotes it when it is operationally valid.

The primary research object is the **harness**, not the model and not the market simulator.

## 2. Research Question

### Primary question

Under a fixed historical order-book environment, fixed agent, fixed action space, and fixed outcome evaluator, how does performance evolve when an evidence-driven factory iteratively promotes operationally valid harness successors?

### Operational version

Given an externally specified instruction to buy or sell a fixed quantity within a short horizon, can an evolving harness help a fixed agent choose a better passive limit-order price?

### Required attribution

Within an experiment, the following remain fixed:

- historical dataset and preprocessing;
- episode construction;
- market replay and queue assumptions;
- parent-order instructions;
- action semantics;
- OpenRouter inference provider;
- exact base-model slug `deepseek/deepseek-v4-flash-0731`;
- fixed model inference and routing configuration;
- harness interface;
- harness design language;
- deterministic harness compiler;
- factory procedure;
- successor operational-validation rules;
- reward function and evaluation metrics;
- chronological evolution-episode ordering and promotion rule.

Only the following changes between promoted generations:

```text
HarnessSpec_t
```

## 3. Scope

### 3.1 V0 task

V0 is a one-shot, conditional execution task:

```text
Given:
    a historical limit-order-book state,
    recent causal market events,
    and a fixed parent order,

Choose:
    one passive limit-order placement offset.

Then:
    replay a fixed future event horizon,
    simulate the hypothetical order,
    and calculate an objective outcome.
```

Example parent order:

```text
Buy 100 shares within the next 30 seconds.
```

The agent does not choose trade direction or order quantity in V0. It chooses only where to place the order.

### 3.2 V0 non-goals

V0 does not attempt to implement:

- unrestricted market making;
- simultaneous bid and ask placement;
- strategic inventory management across episodes;
- order cancellation or replacement policies;
- aggressive execution or crossing the spread;
- market-impact modeling;
- an exchange-grade matching engine;
- arbitrary generated Python harnesses;
- model fine-tuning, reinforcement learning, LoRA, or weight updates;
- a guarantee that every generated candidate improves;
- online deployment or live trading.

These can be considered after the V0 causal and experimental substrate is validated.

## 4. Data Source and Semantics

V0 uses LOBSTER historical order-book data:

- [LOBSTER output structure](https://php.lobsterdata.com/info/DataStructure.php)
- [LOBSTER data samples](https://lobsterdata.com/info/DataSamples.php)

For each ticker and trading day, LOBSTER provides aligned message and order-book files. Message row `k` describes the event that transforms order-book row `k-1` into row `k`.

### 4.1 Message fields

```text
timestamp
event_type
order_id
size
price
direction
```

Relevant event types:

| Type | Meaning | V0 treatment |
|---:|---|---|
| 1 | New visible limit order | Update historical queue state |
| 2 | Partial cancellation | Apply fixed cancellation model |
| 3 | Total deletion | Apply fixed cancellation model |
| 4 | Visible execution | May consume displayed queue |
| 5 | Hidden execution | Does not directly consume displayed queue |
| 6 | Cross trade | Does not fill the hypothetical passive order |
| 7 | Trading halt indicator | Terminate or invalidate the episode |

Prices are stored as dollar prices multiplied by `10,000`. Direction is `1` for a buy limit order and `-1` for a sell limit order.

### 4.2 Order-book fields

For each requested depth level:

```text
ask_price_i
ask_size_i
bid_price_i
bid_size_i
```

LOBSTER levels are **occupied price levels**, not tick offsets. The gap between adjacent displayed levels may exceed the minimum tick size. Therefore, V0 placement prices are computed from the best quote and the instrument tick size rather than by selecting the next displayed level.

## 5. Fixed Domain Model

```python
from dataclasses import dataclass
from typing import Literal


Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class LOBState:
    timestamp: float
    bid_prices: tuple[int, ...]
    bid_sizes: tuple[int, ...]
    ask_prices: tuple[int, ...]
    ask_sizes: tuple[int, ...]


@dataclass(frozen=True)
class MarketEvent:
    timestamp: float
    event_type: int
    order_id: int
    size: int
    price: int
    direction: int


@dataclass(frozen=True)
class ParentOrder:
    side: Side
    quantity: int
    decision_time: float
    expiration_time: float


@dataclass(frozen=True)
class PlacementAction:
    offset_ticks: Literal[0, 1, 2]


@dataclass(frozen=True)
class OrderOutcome:
    requested_quantity: int
    filled_quantity: int
    fill_ratio: float
    time_to_first_fill: float | None
    time_to_full_fill: float | None
    volume_weighted_fill_price: float | None
    arrival_mid_price: float
    terminal_mid_price: float
    spread_capture: float
    signed_markout: float
    fees: float
    unfilled_penalty: float
    reward: float
```

All outcome fields are calculated by the environment. The harness cannot write, override, reinterpret, or omit environment truth.

## 6. Episode Definition

An `Episode` is a deterministic historical decision point plus a parent-order instruction.

```python
@dataclass(frozen=True)
class Episode:
    episode_id: str
    ticker: str
    trading_date: str
    decision_event_index: int
    parent_order: ParentOrder
    replay_horizon_ms: int
    markout_horizon_ms: int
```

Every episode must satisfy:

- the book is valid and non-crossed at the decision point;
- the selected price offsets are representable;
- enough prior events exist for the maximum permitted lookback;
- enough future events exist for replay and terminal markout;
- no trading halt occurs within the required interval, unless the halt policy explicitly invalidates the episode;
- all agent-visible information has a timestamp less than or equal to the decision timestamp;
- all outcome information after the decision timestamp remains hidden until the action is committed.

Parent-order sides should be generated by a fixed, documented procedure independent of future outcomes. The episode generator must not choose sides using future price movement.

## 7. Action Space

For a buy parent order:

```text
offset 0 → best_bid
offset 1 → best_bid - tick_size
offset 2 → best_bid - 2 × tick_size
```

For a sell parent order:

```text
offset 0 → best_ask
offset 1 → best_ask + tick_size
offset 2 → best_ask + 2 × tick_size
```

The environment performs the mapping:

```python
def action_to_price(
    state: LOBState,
    parent_order: ParentOrder,
    action: PlacementAction,
    tick_size: int,
) -> int:
    if parent_order.side == "BUY":
        return state.bid_prices[0] - action.offset_ticks * tick_size

    return state.ask_prices[0] + action.offset_ticks * tick_size
```

The final agent output must conform to `PlacementAction`. Invalid outputs are rejected or mapped through the fixed fallback policy defined by the active harness specification.

## 8. Historical Replay Assumptions

Historical replay is counterfactual: the hypothetical order was not present in the recorded market. V0 therefore makes the following explicit assumptions.

### 8.1 No-impact assumption

The hypothetical order is sufficiently small that its presence would not have altered subsequent submissions, cancellations, executions, or prices.

The recorded future event stream is therefore replayed unchanged.

### 8.2 Price-time priority

The hypothetical order joins the back of the displayed queue at its price at the placement timestamp.

```python
queue_ahead_at_placement = displayed_quantity_at_action_price
```

Historical orders submitted later at the same price join behind the hypothetical order.

### 8.3 Visible executions

A compatible visible execution at the hypothetical order's price consumes `queue_ahead` first. Once `queue_ahead` reaches zero, further compatible executable quantity fills the hypothetical order until it is complete or the episode expires.

### 8.4 Cancellations and deletions

V0 must select exactly one cancellation model before experiments begin:

1. **Conservative model:** cancellations and deletions do not reduce `queue_ahead`.
2. **Proportional model:** cancellations reduce `queue_ahead` in proportion to estimated ahead-of-order depth.
3. **L3 reconstruction:** historical order IDs and price-time order are reconstructed to identify whether removed quantity was ahead of the hypothetical order.

The chosen model is a fixed environment property and cannot be changed by the harness or factory.

**Implemented default:** `conservative`. It is unchanged, deterministic, and produces a lower-bound fill estimate. The explicit opt-in `l3_fifo` mode reconstructs orders created by observed type-1 messages and classifies those active at hypothetical submission as ahead; later same-price submissions are behind. Only identifiable ahead-order cancellations and deletions reduce queue ahead.

The depth snapshot is authoritative at hypothetical submission. Snapshot depth not represented by identifiable active orders is anonymous inventory that remains ahead and cannot be reduced by an unidentified removal. Because a depth-limited message stream cannot prove that an order ID lacking an observed type-1 event predates the file, such IDs are not treated as identifiable file-boundary orders. When reconstructed identifiable depth exceeds snapshot depth, FIFO allocation is capped at snapshot depth and the excess is recorded as a reconciliation diagnostic; excess orders are not allowed to improve queue position. Hidden executions retain the conservative no-effect behavior.

### 8.5 Hidden executions and cross trades

- Hidden executions do not directly consume the displayed queue ahead of the hypothetical order.
- Cross trades do not fill the hypothetical passive order.
- Trading halts invalidate or terminate the episode according to a fixed policy.

### 8.6 Partial fills and expiration

Partial fills are accumulated until:

- requested quantity is completely filled;
- the parent order reaches `expiration_time`;
- the replay horizon ends; or
- a terminal invalidating event occurs.

Any remaining quantity is unfilled and contributes to the fixed unfilled-quantity penalty.

## 9. Environment Interface and Flow

```python
class OrderBookEnvironment:
    def reset(self, episode: Episode) -> "EnvironmentObservation":
        ...

    def valid_actions(self) -> tuple[PlacementAction, ...]:
        ...

    def step(self, action: PlacementAction) -> OrderOutcome:
        ...
```

Episode flow:

```text
load historical episode
        ↓
construct causal EnvironmentObservation_t
        ↓
Harness_t builds agent context
        ↓
fixed agent selects PlacementAction_t
        ↓
environment validates and inserts hypothetical order
        ↓
environment replays fixed MarketEvents
        ↓
fill / partial fill / expiration
        ↓
environment calculates OrderOutcome_t
        ↓
persist Experience_t
```

The environment owns:

- dataset access and temporal boundaries;
- episode construction;
- parent-order instruction;
- action-to-price conversion;
- market replay;
- queue simulation;
- fill determination;
- fees and transaction costs;
- outcome calculation;
- reward calculation;
- episode validity and termination.

## 10. Objective Function and Metrics

### 10.1 Primary reward

For filled quantity `q_f`, requested quantity `q`, side sign `s`, volume-weighted fill price `p_fill`, and terminal midpoint `m_terminal`:

\[
R = q_f s(m_{terminal} - p_{fill}) - fees - \lambda(q-q_f)
\]

where:

```text
s = +1 for BUY
s = -1 for SELL
λ = fixed unfilled-share penalty
```

The exact fee schedule, terminal horizon, and value of `λ` must be fixed before the first harness experiment.

The primary reward balances:

- execution quality;
- post-fill price movement;
- transaction costs;
- the cost of failing to execute the parent order.

### 10.2 Outcome metrics

The environment records, at minimum:

- reward;
- fill ratio;
- time to first fill;
- time to full fill;
- volume-weighted execution price;
- effective spread or spread captured;
- signed post-trade markout;
- fees;
- unfilled-quantity penalty.

These remain separate objective metrics even when a scalar reward is recorded.

### 10.3 Harness process metrics

Process metrics may include:

- valid-action rate;
- fallback-action rate;
- agent latency;
- token usage;
- output parse success rate;
- action distribution and entropy;
- feature availability or computation failures.

Process metrics must be derived from typed runtime artifacts. No metric may depend on subjective natural-language interpretation.

### 10.4 Metric ownership

```text
Environment metrics:
    Describe what happened in the simulated market.

Harness process metrics:
    Describe how the agent-harness system produced its action.

Factory diagnostics:
    Use the two metric families to hypothesize why performance occurred.
```

## 11. Agent and Harness Boundary

### 11.1 Fixed agent

V0 uses OpenRouter as the inference transport and the following exact dated model as the fixed LLM:

```text
deepseek/deepseek-v4-flash-0731
```

The dated slug must be used directly. A moving alias such as `~deepseek/deepseek-v4-flash-latest` and OpenRouter's automatic model router are prohibited because they may resolve to different models over the lifetime of the experiment.

OpenRouter is not itself the agent. The fixed agent is the combination of:

```text
exact model slug
+ fixed OpenRouter adapter
+ fixed upstream-provider routing
+ fixed inference parameters
+ fixed parsing, retry, and failure behavior
```

The agent and its inference configuration remain fixed across all harness generations, including:

- model identifier;
- model weights;
- upstream inference provider;
- provider fallback behavior;
- temperature and sampling configuration;
- maximum output tokens;
- reasoning-mode configuration, if explicitly enabled;
- structured-output parsing;
- transient retry policy;
- timeout behavior;
- invalid-output behavior outside the harness design language.

Relevant provider references:

- [DeepSeek V4 Flash 0731 on OpenRouter](https://openrouter.ai/deepseek/deepseek-v4-flash-0731)
- [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
- [OpenRouter API parameters](https://openrouter.ai/docs/api_reference/parameters)

### 11.2 Fixed agent configuration

```python
@dataclass(frozen=True)
class AgentConfig:
    provider: Literal["openrouter"]
    model_id: Literal["deepseek/deepseek-v4-flash-0731"]
    upstream_provider: str
    allow_fallbacks: Literal[False]
    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int | None
    timeout_seconds: int
    max_retries: int


AGENT_CONFIG = AgentConfig(
    provider="openrouter",
    model_id="deepseek/deepseek-v4-flash-0731",
    upstream_provider="<selected-and-frozen-before-experiment>",
    allow_fallbacks=False,
    temperature=0.0,
    top_p=1.0,
    max_output_tokens=256,
    seed=None,
    timeout_seconds=60,
    max_retries=2,
)
```

The exact upstream provider must be selected during M6 after a capability and availability smoke test, then frozen before comparative harness experiments begin. If provider pinning is unavailable or unreliable, the actual provider must be recorded for every inference and provider variation must be disclosed as an experimental limitation.

All intended sampling parameters must be sent explicitly rather than relying on upstream defaults. M6 must verify that the selected endpoint supports or consistently handles every submitted parameter.

The OpenRouter API key is supplied through runtime secret configuration, such as `OPENROUTER_API_KEY`. It must never be stored in run artifacts, prompts, trajectories, or logs.

### 11.3 Agent execution interface

V0 uses one model completion per placement episode:

```text
compiled harness builds causal messages
                  ↓
fixed DeepSeek V4 Flash 0731 agent
                  ↓
structured PlacementAction
```

The model does not directly access the replay environment, future events, or reward evaluator. Any derived order-book features are computed by approved harness primitives before inference.

```python
class FixedAgent:
    def __init__(self, config: AgentConfig, client: "OpenRouterClient"):
        self.config = config
        self.client = client

    def act(
        self,
        messages: list[dict],
    ) -> tuple[PlacementAction, "InferenceRecord"]:
        response = self.client.complete(
            model=self.config.model_id,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_output_tokens,
            stream=False,
            provider={
                "only": [self.config.upstream_provider],
                "allow_fallbacks": self.config.allow_fallbacks,
            },
        )

        action = parse_placement_action(response)
        record = build_inference_record(response, action)
        return action, record
```

The required semantic output is:

```json
{
  "offset_ticks": 1
}
```

If strict schema enforcement is unsupported by the selected endpoint, the fixed adapter must parse JSON deterministically and invoke the active harness's fixed invalid-action policy. It must not ask a different model to repair the output.

Transient retries must reuse the exact model, upstream provider, parameters, and messages. When retries are exhausted, the inference failure is recorded explicitly; the system must not silently change model or provider.

### 11.4 Harness definition

The harness determines:

> Given a causal environment observation, what information and permitted derived operations are available to the fixed agent, what decision procedure it performs, what constraints and pre-action checks apply, and how its final action and reusable feedback are constructed.

### 11.5 Harness interface

```python
class Harness:
    def observe(self, state, recent_events):
        """Select and format causal environment data for the agent."""

    def augment(self, observation):
        """Compute permitted features from current and historical data."""

    def constrain(self, context, parent_order):
        """Construct the valid action set and output constraints."""

    def evaluate(self, proposed_action, context):
        """Apply pre-action checks without access to future outcomes."""

    def feedback(self, trajectory, outcome):
        """Construct reusable feedback from completed experience."""
```

`evaluate()` is a pre-action critic. It cannot observe or predict using actual future replay data.

The harness does not own environment truth, reward calculation, or performance-based promotion.

The messages passed to OpenRouter are constructed by the compiled harness. Their information selection, feature presentation, approved instruction template, workflow, and prior-feedback content may therefore change with `HarnessSpec_t`. The OpenRouter transport, exact model, inference configuration, and output adapter do not change.

## 12. Harness Design Language

The harness design space is a typed, declarative domain-specific language rather than arbitrary generated code.

Let:

\[
\mathcal{L}_H = \{s \mid s \text{ satisfies the HarnessSpec schema and semantic invariants}\}
\]

The fixed compiler maps valid specifications to executable harnesses:

\[
Compile: \mathcal{L}_H \rightarrow \{\text{executable Harness objects}\}
\]

A context-free grammar or constrained decoder may enforce syntax, but semantic validators remain necessary for bounds, compatibility, termination, and causal-data constraints.

### 12.1 V0 `HarnessSpec`

```python
from dataclasses import dataclass
from typing import Literal


FeatureName = Literal[
    "spread_ticks",
    "level1_imbalance",
    "multilevel_imbalance",
    "microprice",
    "recent_trade_imbalance",
    "cancel_rate",
    "short_term_volatility",
]


MetricName = Literal[
    "reward",
    "fill_ratio",
    "time_to_fill",
    "spread_capture",
    "signed_markout",
]


@dataclass(frozen=True)
class ObservationSpec:
    book_levels: Literal[1, 3, 5, 10]
    event_lookback: Literal[10, 25, 50, 100]
    include_raw_events: bool


@dataclass(frozen=True)
class AugmentationSpec:
    features: tuple[FeatureName, ...]


@dataclass(frozen=True)
class DecisionSpec:
    procedure: Literal[
        "direct",
        "analyze_then_choose",
        "compare_all_actions",
        "propose_critic_revise",
    ]
    require_action_rationale: bool


@dataclass(frozen=True)
class ConstraintSpec:
    allowed_offsets: tuple[Literal[0, 1, 2], ...]
    reject_invalid_action: bool
    fallback_offset: Literal[0, 1, 2]


@dataclass(frozen=True)
class FeedbackSpec:
    history_window: Literal[0, 10, 25, 50]
    metrics_shown: tuple[MetricName, ...]


@dataclass(frozen=True)
class HarnessSpec:
    observation: ObservationSpec
    augmentation: AugmentationSpec
    decision: DecisionSpec
    constraints: ConstraintSpec
    feedback: FeedbackSpec
```

### 12.2 Semantic invariants

At minimum:

- `allowed_offsets` is non-empty and contains only environment-valid actions;
- `fallback_offset` belongs to `allowed_offsets`;
- feature names exist in the fixed feature registry;
- duplicate features and metrics are rejected or canonicalized deterministically;
- every feature uses only data available at or before the decision timestamp;
- the requested lookback is supported by the episode;
- the decision workflow has a fixed termination bound;
- the compiled harness emits exactly one `PlacementAction`;
- the specification cannot modify environment, reward, splits, model configuration, or compiler behavior.

## 13. Deterministic Harness Compiler

The compiler composes approved primitives. It does not execute generated source code.

```python
def compile_harness(spec: HarnessSpec) -> Harness:
    validate_schema(spec)
    validate_semantics(spec)

    return CompiledHarness(
        observer=ObserverRegistry.build(spec.observation),
        augmenters=tuple(
            FeatureRegistry.get(name)
            for name in spec.augmentation.features
        ),
        decision_workflow=WorkflowRegistry.get(
            spec.decision.procedure
        ),
        constraints=ActionConstraintCompiler.build(
            spec.constraints
        ),
        feedback_builder=FeedbackCompiler.build(
            spec.feedback
        ),
    )
```

Compiler requirements:

- deterministic for a given specification;
- versioned;
- side-effect bounded;
- unable to access future market events;
- unable to mutate environment state except through the valid action interface;
- unable to create capabilities outside the approved registries;
- able to serialize a canonical representation for identity and comparison.

## 14. Experience and Factory Artifacts

Evidence is measured. Diagnosis is inferred. These stages must not be conflated.

```python
@dataclass(frozen=True)
class InferenceRecord:
    request_id: str | None
    requested_model: str
    returned_model: str | None
    upstream_provider: str | None
    agent_config_hash: str
    harness_id: str
    prompt_hash: str
    raw_response: str | None
    parsed_action: PlacementAction | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    retry_count: int
    error: str | None


@dataclass(frozen=True)
class AgentTrajectory:
    agent_observation: object
    augmented_context: object
    proposed_actions: tuple[PlacementAction, ...]
    pre_action_evaluations: tuple[object, ...]
    final_action: PlacementAction
    inference_record: InferenceRecord
    parse_errors: tuple[str, ...]
    latency_ms: int
    token_usage: dict[str, int]


@dataclass(frozen=True)
class Experience:
    experience_id: str
    episode_id: str
    harness_id: str
    trajectory: AgentTrajectory
    outcome: OrderOutcome


@dataclass(frozen=True)
class MetricSlice:
    slice_name: str
    sample_count: int
    metric_name: str
    metric_value: float
    baseline_value: float | None


@dataclass(frozen=True)
class Evidence:
    experience_ids: tuple[str, ...]
    aggregate_metrics: dict[str, float]
    metric_slices: tuple[MetricSlice, ...]


@dataclass(frozen=True)
class Diagnosis:
    affected_metric: str
    evidence_refs: tuple[str, ...]
    suspected_harness_component: str
    suspected_cause: str
    confidence: float


@dataclass(frozen=True)
class Improvement:
    component: str
    rationale: str
    expected_metric_effect: str


@dataclass(frozen=True)
class SuccessorAttempt:
    successor_id: str | None
    parent_harness_id: str
    successor_spec: HarnessSpec | None
    diagnosis_refs: tuple[str, ...]
    mutation: HarnessMutation | None
    validation_results: dict
    decision: Literal["PROMOTED", "INVALID"]
    factory_call: FactoryCallRecord
```

Free-text factory reasoning may appear in `Diagnosis` and `Improvement`, but each claim must reference objective evidence artifacts. `HarnessMutation` is the sole executable change description.

## 15. Self-Improvement Procedure

Harness evolution occurs once after each individual episode. No model weights are trained.

### 15.1 Generation loop

```python
current = compile_harness(H0)

for generation, episode in enumerate(evolution_episodes):
    experience = run_one_episode(fixed_agent, current, episode)
    evidence = derive_episode_evidence(experience, sample_count=1)
    persist_measured_experience(generation, current, experience, evidence)

    proposal = factory.propose(
        current_spec=current.spec,
        trajectory=experience.trajectory,
        outcome=experience.outcome,
        evidence=evidence,
        prior_attempts=prior_attempts,
    )

    successor = validate_and_compile_offline(current.spec, proposal)
    if successor.is_operationally_valid:
        current = successor.harness
        persist_transition(status="PROMOTED")
    else:
        persist_transition(status="INVALID")
```

### 15.2 Factory responsibilities

The factory:

1. consumes objective data from the single completed episode;
2. identifies metric-level failure patterns;
3. inspects the incumbent specification;
4. proposes exactly one evidence-linked atomic change;
5. produces only a declarative successor `HarnessSpec`.

The factory does not:

- modify the environment or evaluator;
- modify the base model;
- write arbitrary executable code;
- decide promotion based on performance;
- inspect future events or any separate evaluation outcomes;
- assume that a successor will perform better.

### 15.3 Promotion rule

A proposed successor is promoted if its response parses, its one atomic mutation is legal and non-no-op, and the resulting specification passes schema, semantic, compilation, interface, causality, and offline smoke validation. These checks establish executability only and use no LLM. There is no reward comparison. Invalid proposals retain the current harness and create no successor harness identity.

## 16. Sequential Evolution Protocol

Episodes form one deterministic chronological `evolution_episodes` sequence. Generation `t` consumes exactly `evolution_episodes[t]`. There are no diagnostic batches, training partitions, validation episodes, held-out episodes, paired candidate evaluations, or post-evolution lineage reruns.

On the normal successful path a generation makes one decision-agent logical call and one factory logical call. Environment replay, evidence construction, proposal validation, compilation, offline smoke validation, persistence, and plotting make zero LLM calls. Retries are exceptional physical requests and are recorded separately.

## 17. Performance Over Time

The project must show both harness lineage and measured performance.

### 17.1 Required generation-level outputs

For every generation:

- active harness ID and episode ID;
- selected action and objective episode outcome;
- decision-agent tokens, latency, parse status, and fallback use;
- proposed successor mutation and operational validation status;
- factory tokens, latency, and retry count;
- promoted successor ID, or the retained parent ID for an invalid proposal;
- evidence and diagnosis references.

### 17.2 Required plots

At minimum:

1. observed reward by generation;
2. observed fill ratio by generation;
3. observed signed markout by generation;
4. observed unfilled penalty by generation.

Plots read persisted generation telemetry directly and never rerun a harness. The same harness may appear in consecutive generations after invalid proposals. After `G` loops, a promoted `H_G` has no performance value until another normal generation uses it; no extra decision call is made to fill that gap.

## 18. Persistence and Lineage

Every run must persist sufficient information for deterministic audit and replay.

Recommended artifact layout:

```text
runs/<run_id>/
    run.json
    environment_config.json
    dataset_manifest.json
    evolution_manifest.json
    episodes.jsonl
    harnesses.jsonl
    successors.jsonl
    factory_calls.jsonl
    transitions.jsonl
    generations.jsonl
    experiences.jsonl
    lineage.json
    plots/
```

`run.json` is published last with `status: complete` only after every other artifact has been staged successfully. Failures publish `status: failed`.

### 18.1 Harness identity

Each harness receives an identity derived from:

- canonical serialized `HarnessSpec`;
- compiler version;
- harness-language version.

### 18.2 Environment identity

Each run records:

- dataset source and checksum;
- ticker and trading dates;
- preprocessing version;
- tick-size source;
- queue and cancellation model;
- fee model;
- replay and markout horizons;
- reward parameters;
- evolution episode manifest.

### 18.3 Lineage invariants

- Every proposed successor references exactly one parent harness.
- Every operationally valid successor is promoted without a performance claim.
- Invalid proposals remain immutable and auditable and create no new harness ID.
- Harness IDs cannot be reused for different specifications.
- An environment configuration change begins a new experiment rather than a new harness generation.

## 19. Validation Requirements

### 19.1 Dataset validation

- message and order-book row counts match;
- timestamps are non-decreasing;
- prices and directions use expected encodings;
- book states are valid at selected decision points;
- event-to-book transitions pass sampled consistency checks;
- invalid sentinel levels are handled explicitly.

### 19.2 Environment validation

- deterministic replay for identical inputs;
- no access to post-decision data before action commitment;
- correct buy/sell symmetry;
- correct tick-offset mapping;
- queue-ahead never becomes negative;
- filled quantity never exceeds requested quantity;
- volume-weighted fill price uses only simulated fills;
- expiration and halt behavior are deterministic;
- reward recomputes exactly from persisted outcome fields.

### 19.3 Harness validation

- specification schema valid;
- semantic invariants valid;
- all referenced primitives registered;
- causal feature computation only;
- bounded termination;
- exactly one valid `PlacementAction` emitted or fixed fallback invoked;
- deterministic compilation;
- canonical serialization stable.

### 19.4 Factory validation

- each diagnosis references evidence;
- each improvement is narrative-only and targets an allowed specification field via `mutation`;
- exactly one successor proposal is emitted per generation;
- no environment or evaluator fields appear in successor diffs;
- factory cannot access held-out future artifacts during synthesis;
- operational validation decides `PROMOTED` vs `INVALID`; there is no reward gate.

## 20. V0 Initial Harness

`H0` should be intentionally minimal but fully functional. Its primitives must not be empty functions.

Example:

```python
H0 = HarnessSpec(
    observation=ObservationSpec(
        book_levels=1,
        event_lookback=10,
        include_raw_events=True,
    ),
    augmentation=AugmentationSpec(
        features=("spread_ticks",),
    ),
    decision=DecisionSpec(
        procedure="direct",
        require_action_rationale=False,
    ),
    constraints=ConstraintSpec(
        allowed_offsets=(0, 1, 2),
        reject_invalid_action=True,
        fallback_offset=1,
    ),
    feedback=FeedbackSpec(
        history_window=0,
        metrics_shown=(),
    ),
)
```

This gives the factory room to introduce additional causal features, broader context, comparative decision procedures, pre-action criticism, or historical feedback without changing the harness interface.

## 21. Implementation Milestones

### M1 — Data ingestion and replay alignment

Implement:

- LOBSTER message and order-book readers;
- fixed-point price conversion;
- aligned event/state iterator;
- dataset validation and manifests.

**Exit condition:** sampled event rows correctly correspond to observed book transitions.

### M2 — Deterministic episode generator

Implement:

- episode eligibility checks;
- causal observation windows;
- fixed parent-order generation;
- chronological split manifests.

**Exit condition:** the same configuration produces the same episode and split IDs.

### M3 — Counterfactual execution simulator

Implement:

- tick-offset price mapping;
- queue-ahead initialization;
- selected cancellation model;
- visible execution handling;
- partial fills, expiration, and terminal markout;
- reward decomposition.

**Exit condition:** hand-constructed replay fixtures pass deterministic fill and reward tests.

### M4 — Fixed baselines and evaluator

Implement:

- constant-offset policies;
- random policy;
- deterministic episode evaluator;
- aggregate metrics and uncertainty reporting.

**Exit condition:** baseline performance is reproducible and persisted.

### M5 — Harness language and compiler

Implement:

- typed `HarnessSpec`;
- registries of approved features and workflows;
- semantic validation;
- deterministic compilation and harness identity.

**Exit condition:** multiple valid specifications compile and invalid specifications fail closed.

### M6 — Fixed agent integration

Implement:

- OpenRouter agent adapter;
- exact model slug `deepseek/deepseek-v4-flash-0731`;
- upstream-provider capability and availability smoke test;
- pinned upstream provider and disabled provider fallbacks;
- fixed inference, timeout, and retry configuration;
- one model completion per episode;
- structured `PlacementAction` output and deterministic parsing;
- trajectory capture;
- parse failure and fallback behavior;
- request, model, provider, token, retry, error, and latency telemetry;
- secret-handling validation that excludes the API key from artifacts.

**Exit condition:** `H0` produces auditable actions over a fixed episode batch, every request uses the exact dated model and frozen routing configuration, and exhausted failures are recorded without silent model or provider substitution.

### M7 — Factory artifact pipeline

Implement:

- one completed `Experience` per generation;
- episode-level objective `Evidence` construction;
- diagnosis and improvement artifacts;
- one atomic successor mutation;
- successor specification diffs.

**Exit condition:** factory successors are evidence-linked and constrained to the harness language.

### M8 — Direct-successor evolution

Implement:

- generation loop;
- offline operational-validation gates;
- one episode and one successor proposal per generation;
- deterministic promote/retain logic;
- complete lineage persistence.

**Exit condition:** valid successors are promoted and invalid proposals retain the current harness without performance comparison.

### M9 — Performance-over-time report

Implement:

- generation telemetry plots;
- sequential lineage;
- reward and outcome metrics;
- limitations and simulator-assumption analysis.

**Exit condition:** the report plots only performance observed during normal generation execution.

## 22. Definition of V0 Completion

V0 is complete when:

1. LOBSTER data can be ingested and validated reproducibly.
2. Episodes enforce a strict causal boundary.
3. The counterfactual simulator has explicit, tested queue and fill semantics.
4. Outcomes and metrics are derivable from the typed data model.
5. `H0` can execute the first evolution episode.
6. A fixed agent can produce valid placement actions through compiled harnesses.
7. The factory receives one completed experience and proposes one atomic `HarnessSpec` successor.
8. Every proposal is validated operationally without an LLM or reward comparison.
9. Promoted and invalid generation transitions are persisted.
10. Performance over generations is shown from existing execution telemetry.
11. Only `HarnessSpec_t` changes within the experiment.

## 23. Known Limitations

- Historical replay assumes the hypothetical order does not affect future market behavior.
- A conservative or proportional cancellation model cannot perfectly recover true queue position.
- LOBSTER data describes recorded exchange activity but cannot reveal counterfactual participant reactions.
- A short evaluation period may make results regime-specific.
- A fixed LLM may exhibit stochastic variance even under identical harnesses.
- One episode per generation produces noisy, non-monotonic performance observations.
- Improvement in simulator reward is not evidence of live profitability.
- The constrained language limits the factory to the capabilities anticipated by the system designer.

These limitations do not invalidate V0. They define the boundary of the claim that the experiment can support.

## 24. Final System Summary

```text
Historical LOBSTER episode + fixed parent order
                       ↓
              fixed Environment
                       ↓
                 HarnessSpec_t
                       ↓
            deterministic Compiler
                       ↓
                 Harness_t
                       ↓
                 fixed Agent
                       ↓
              PlacementAction_t
                       ↓
          deterministic market replay
                       ↓
       Trajectory_t + objective Outcome_t
                       ↓
          one completed Experience
                       ↓
                    Factory
 trajectory + outcome + evidence
                       ↓
        one constrained successor mutation
                       ↓
       deterministic operational validation
                       ↓
       promote successor or retain current
                       ↓
                 HarnessSpec_t+1
```

The self-improving system is therefore not an agent freely rewriting its own runtime. It is a controlled sequential process over a finite, typed, executable harness language. Intermediate performance regressions are allowed and remain visible in generation telemetry.

## 25. Implemented Factory Backends Addendum

V0 provides two selectable factory backends. `heuristic` is the deterministic baseline. `llm` is the primary generative backend and uses the exact dated `deepseek/deepseek-v4-flash-0731` model through a separately hashed and pinned OpenRouter configuration. No model alias, automatic router, provider fallback, alternative model fallback, or silent heuristic fallback is permitted.

The LLM factory receives only the canonical current `HarnessSpec`, the single completed episode's typed trajectory and outcome, programmatically derived episode `Evidence`, objective metric definitions, the finite harness-language allowlist and invariants, and a bounded summary of prior proposal outcomes. It receives no future events, secrets, repository files, or authority over fixed experiment components.

The fixed versioned prompt requires exactly one structured `FactoryProposal` containing an evidence-linked `Diagnosis`, `HarnessInspection`, `Improvement`, and atomic `HarnessMutation`. Factory input publishes an explicit allowlist of aggregate and metric-slice evidence identifiers, and diagnosis references must be copied exactly from it. The mutation is applied by deterministic code. The factory cannot generate source code or feature implementations.

Proposal validation rejects unknown evidence references, forbidden paths, illegal values, no-ops, multi-field representations, compiler-invalid results, and previously invalid identical mutations without new evidence. Malformed or semantically invalid output is not repaired by another model. Transient transport/provider failures reuse the exact request within the fixed retry bound. Exhaustion creates an immutable `INVALID` transition and retains the current harness.

Each generation contains one active harness, one episode, one decision-agent call, deterministic replay and evidence, one factory call returning exactly one proposal, and deterministic offline operational validation. A valid successor is `PROMOTED`; an invalid proposal leaves an `H_t → H_t` transition. No reward gate, ranking, validation split, held-out split, or lineage rerun exists. `factory_calls.jsonl`, `successors.jsonl`, `generations.jsonl`, and `lineage.json` preserve factory input, inference telemetry, validation gates, transition status, and parent/successor identity.

The implemented fixed decision-agent configuration caps total output at 16,384 tokens. A standalone live smoke command remains available, but the evolution runner does not perform a live pre-generation smoke call. Operational successor validation uses a local parsed action fixture and no OpenRouter request.

Implementation limitation: schema/semantic validation and compilation are concrete gates. Interface, causality, and smoke gates currently validate the compiler-produced bounded interface and causal `EnvironmentObservation` path; they are not independent static-analysis systems.

## 26. V0.3 Revision Addendum — Evidence Windows, Baselines, and Promotion Rules

**Status:** Implemented
**Date:** 2026-08-10

V0.3 responds to the primary MVP demonstration, in which the evolving system underperformed a static constant-offset baseline computed from its own recorded counterfactuals. It changes evidence construction, adds mandatory baseline reporting, and introduces an optional promotion rule. Defaults preserve V0.2 behavior except where noted.

### 26.1 Windowed factory evidence (default change)

Factory `Evidence` now aggregates the trailing `evidence_window` completed episodes (default 5, minimum 1; `--evidence-window`), always including the just-completed episode. A `latest_episode` metric slice isolates the newest episode so single-episode diagnosis remains expressible through exact evidence references. Windowed experiences may span several promoted harness versions; per-harness attribution remains in `generations.jsonl`. The factory prompt version advances to `factory-v5`. Setting `--evidence-window 1` reproduces V0.2 single-episode aggregates.

### 26.2 Mandatory baseline reporting (reporting-only change)

Every run's `report.json` includes `performance_summary.baseline_summary`: mean and cumulative reward for each constant-offset policy, the per-episode oracle, and the agent, plus the agent-minus-best-constant delta and outcome-differentiated slices. All values derive from the per-generation counterfactual rewards already persisted during normal execution; no additional replay or inference occurs. `plots/baseline_comparison.svg` plots cumulative reward for all policies. `obh analyze <run_dir>` recomputes these aggregates and plots for any persisted run without API calls. A performance claim for the evolving system is supportable only relative to these baselines.

### 26.3 Oscillation guard (default change)

Proposal validation additionally rejects a mutation that is the exact inverse (ADD/REMOVE, REPLACE old↔new) of a mutation promoted within the last 3 factory attempts. SET mutations carry no derivable inverse and are exempt. This bounds the add/remove thrash observed in the MVP run (the same offset toggled four times in 26 generations) while still allowing a genuine undo after the cooldown. Rejection produces a normal `INVALID` transition.

### 26.4 Optional gated promotion rule (opt-in)

`--promotion-rule` selects between:

- `unconditional` (default; V0.2 behavior): any operationally valid successor is promoted.
- `gated`: adds deterministic revert-on-regression. When the active harness was promoted by the immediately preceding generation and its first measured episode is outcome-differentiated with a strictly lower reward than the parent's measured episode, the lineage reverts to the parent specification and the generation records transition status `REVERTED`. The factory call still executes and its proposal is recorded in `successors.jsonl`, but it is not promoted in a reverted generation. Constant-outcome episodes never trigger reversion because they carry no decision signal.

`transitions.jsonl` and `generations.jsonl` therefore admit three statuses: `PROMOTED`, `INVALID`, `REVERTED`. The reverted-to harness identity appears as the generation's successor. Cross-episode reward comparison is a deterministic, acknowledged approximation; it gates only first measurements of new successors and is restricted to outcome-differentiated episodes.
