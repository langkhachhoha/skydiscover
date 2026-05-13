# LEVI — Detailed Overview

This document is a deep, self-contained note on how the LEVI framework
(`./levi/`, vendored as a git submodule under
[`levi/`](../levi/)) actually works, so future-me does
not need to re-read the source from scratch. It also explains how LEVI links
into SkyDiscover's existing ADRS problems via
[`scripts/run_levi_skydiscover.py`](../scripts/run_levi_skydiscover.py).

> One-line definition (from the README): **LEVI is an LLM-guided evolutionary
> framework for code and prompts that gets AlphaEvolve-level performance from
> tiny / cheap models by treating diversity as an architectural concern
> instead of a model concern.**

---

## 1. Top-level mental model

LEVI is a **CVT-MAP-Elites** behavioral archive plus a small parallel pipeline
that asynchronously feeds it new candidates. Two model tiers are stitched
together by the pipeline:

| Role | Model | What it does |
|---|---|---|
| Mutation (small / cheap) | e.g. `qwen/qwen3-30b-a3b-instruct-2507` | Most edits to elites. Cheap and fast. |
| Paradigm shift (large) | e.g. `openai/gpt-5` | Periodic "throw out the local optimum" jumps. Expensive but rare. |

Cost matches task demand: ~95% of evaluations come from the small model,
the strong model only fires when LEVI detects stagnation or hits a
configurable interval.

The pipeline runs three phases:

1. **Seed & evaluate**, then **init / diversify** — generate a handful of
   structurally different seed programs to populate the archive.
2. **Main evolution loop** — async producer/consumer pair: cheap models
   propose mutations sampled from the archive; the consumer evaluates them in
   a process pool and reinserts surviving variants into the archive.
3. **Punctuated equilibrium** — every N evals, fire the strong model on
   clusters of archive elites to propose new algorithmic paradigms.

A wall-clock / dollar / eval budget can stop the loop at any time.

---

## 2. Public API surface (`levi.__init__`)

The relevant exports are:

```python
levi.evolve_code(
    problem_description: str,
    *,
    function_signature: str,
    seed_program: str | None = None,
    score_fn: Callable[..., dict],          # must return {"score": float, ...}
    inputs: list[Any] | None = None,
    model: ClientSpec | list[ClientSpec] | None = None,
    paradigm_model: ClientSpec | list[ClientSpec] | None = None,
    mutation_model: ClientSpec | list[ClientSpec] | None = None,
    budget_dollars: float | None = None,
    budget_evals: int | None = None,
    budget_seconds: float | None = None,
    target_score: float | None = None,
    resume_snapshot: dict | None = None,
    **kwargs: Any,                          # anything in LeviConfig
) -> LeviResult
levi.evolve_prompts(...)                     # same shape, evolves prompts
levi.LM(model, api_base=..., api_key=..., input_cost_per_token=..., ...)
```

A `ClientSpec` is either a litellm-style model id string
(`"openrouter/openai/gpt-5"`) or a `levi.LM(...)` instance for custom
endpoints / explicit pricing. CLI models (`ClaudeCodeClient`, `CodexClient`)
are also valid client specs and use the user's CLI subscription instead of
API keys.

`LeviResult` carries `best_program`, `best_score`, `total_evaluations`,
`total_cost`, `archive_size`, `runtime_seconds`, and a `score_history`.

---

## 3. Configuration tree (`levi/config/models.py`)

`LeviConfig` is a pydantic model that aggregates a stack of small sub-configs.
The most useful knobs:

```
LeviConfig
├── problem_description, function_signature, seed_program
├── inputs, score_fn
├── budget: BudgetConfig(dollars, evaluations, seconds, target_score)
├── paradigm_models, mutation_models       # list[ClientSpec]
├── sampler_model_pairs                    # auto-generated cross-product of
│                                          #   mutation_models × {0.3,0.7,1.0,1.2}
├── cvt: CVTConfig(n_centroids=50, data_driven_centroids=True)
├── init: InitConfig(enabled=True, n_diverse_seeds=4, n_variants_per_seed=20)
├── meta_advice: MetaAdviceConfig(enabled=True, interval=50)
├── behavior: BehaviorConfig(ast_features=[loop_count, branch_count,
│                                          math_operators, loop_nesting_max])
├── cascade: CascadeConfig(enabled=True, quick_inputs=[], min_score_ratio=0.8)
├── punctuated_equilibrium: PunctuatedEquilibriumConfig(
│         enabled=True, interval=10, n_clusters=3, n_variants=3)
├── prompt_opt: PromptOptConfig(enabled=False, n_trials=12, ...)
├── proxy_benchmark: ProxyBenchmarkConfig(enabled=False, subset_size=15)
└── pipeline: PipelineConfig(n_llm_workers=4, n_eval_processes=4,
                              eval_timeout=60s, max_tokens=16384)
```

A `model_validator` on `LeviConfig` auto-wires defaults: if you only set
`paradigm_models` + `mutation_models`, the validator fills in
`init.diversity_model`, `init.variant_models`, `meta_advice.model`,
`punctuated_equilibrium.heavy_models`, `punctuated_equilibrium.variant_models`,
and `prompt_opt.teacher_model`. It also auto-generates an `output_dir` of
`runs/<timestamp>/` when none is supplied.

---

## 4. The archive — `CVTMAPElitesPool`
(`levi/pool/cvt_map_elites.py`)

The archive is a **MAP-Elites grid in behavior space**, but the cells are
defined by **Centroidal Voronoi Tessellation (CVT)** rather than a regular
grid — i.e. k-means centroids over the behavior space, computed either:

- statically (uniform centroids over the unit cube), or
- *data-driven*: re-clustered from the first batch of init-phase behavior
  vectors so cells actually reflect where solutions live.

For each cell, only the best-scoring **elite** is retained
(`Elite.scores`, `Elite.behavior`, `Elite.program`). New candidates are
mapped to their nearest centroid; if their score beats the resident, they
replace it. Cell stats (`CellStats`) accumulate counts that several samplers
use as exploration signals.

### Samplers

Drawing parents for the next mutation is delegated to `Sampler` strategies
registered with the pool. The built-in ones:

| Sampler | Idea |
|---|---|
| `UCBSampler` | UCB1 over cells using mean elite score + visit count. |
| `SoftmaxSampler` | Softmax over elite scores with a tunable temperature. Default uses `{0.3, 0.7, 1.0, 1.2}`. |
| `CyclicAnnealingSampler` | Cycles temperature over `n_cycles` to exploit-then-explore. |
| `UniformSampler` | Uniform random over occupied cells. |
| `SubscoreSampler` | Sample by a chosen secondary metric key. |

Each `SamplerModelPair` ties a sampler to a particular mutation model and
weight, so an "explore" sampler can be paired with a different model than an
"exploit" sampler. The default in `LeviConfig._auto_wire_models` creates a
softmax pair per `(mutation_model, temperature)` combination.

### Behavior space — `behavior/extractor.py`, `behavior/features.py`

`BehaviorExtractor` builds a `FeatureVector` from each evaluated program. Two
sources:

- **AST features** (built-in, listed in `BehaviorConfig.ast_features`):
  `loop_count`, `loop_nesting_max`, `branch_count`, `math_operators`,
  `comparison_count`, `call_count`, `subscript_count`, …
- **Score-key features**: scrape arbitrary keys out of the score dict.
- **Custom extractors**: any `Callable[[Program], float]`. This is what makes
  `evolve_prompts` work — for non-code artifacts the AST features list is
  emptied and the dimensions are entirely custom.

The behavior vector is what's looked up against the CVT centroids when the
pool decides which cell a new program belongs to.

---

## 5. Init phase — `init/diversifier.py`

Before the main loop starts, `Diversifier.run()`:

1. **If a seed program is provided**, evaluates it.
2. **Generates `init.n_diverse_seeds` structurally different seeds**, using
   the `paradigm_model` (i.e. *temporarily* upgrades — the heavy model is the
   default for diversity prompts because variety matters more than cost at
   `t=0`).
3. **Generates `init.n_variants_per_seed` variants per seed** with the
   `mutation_models`. All variants run through the executor in parallel.
4. Inserts everything into the pool. The init-phase scores feed both:
   - the **data-driven CVT centroids** (cluster the behavior vectors first,
     then keep those centroids for the rest of the run), and
   - the **proxy benchmark** subset selection (see §10).

The total init spend is reported up front before the main loop starts.

---

## 6. Main pipeline — `pipeline/runner.py`,
`pipeline/producer.py`, `pipeline/consumer.py`, `pipeline/state.py`

The main loop is a classic **async producer / consumer with an asyncio
queue**:

```
            ┌──────────────────────────────────┐
parent ─→  │  llm_producer (N llm workers)    │  ─→  candidate strings
sampler    │  - picks samplers (SamplerModel  │
           │    Pair) by weight               │
           │  - builds prompt via PromptBuild │
           │  - calls mutation_model client   │
           └──────────────────────────────────┘
                              │
                              ▼ (asyncio.Queue)
           ┌──────────────────────────────────┐
           │  eval_consumer (M processes)     │  ─→  scored Programs ─→ pool.add(...)
           │  - extracts code from response   │
           │  - submits to ResilientProcessPool│
           │  - validates / scores            │
           │  - meta-advice every interval    │
           └──────────────────────────────────┘
```

Key implementation notes:

- **`PipelineState`** is the single source of truth for budget bookkeeping
  (`total_cost`, `eval_count`, `start_time`), client concurrency, and the
  score history. `state.try_start_evaluation()` is what enforces the budget
  caps cooperatively — a worker that gets `False` shuts down cleanly.
- **`ResilientProcessPool`** is used so a candidate that infinite-loops or
  segfaults takes only one worker down instead of the whole run.
- **Cascade evaluation** (`CascadeConfig`) optionally evaluates a candidate
  on a small "quick" subset first, only running the full input set if the
  quick score is at least `min_score_ratio` × current best. This is a cost
  saver on slow benchmarks (e.g. transaction scheduling has 600s timeouts).
- **Meta-advice** (`MetaAdviceConfig`): every `interval` evaluations, the
  pipeline asks a model to summarize what's working / what's stuck and feeds
  the advice back into subsequent mutation prompts.
- **Snapshotting**: every batch the pipeline writes
  `output_dir/snapshot.json` containing the archive, the cost so far, and
  enough metadata to **resume** later via
  `evolve_code(..., resume_snapshot=load_json(...))`.

---

## 7. Punctuated equilibrium — `equilibrium/equilibrium.py`

Triggered from `pipeline/runner.py` when `eval_count % pe_config.interval == 0`
(plus a stagnation guard). The implementation:

1. **Cluster** current elites into `n_clusters` groups by their behavior
   vectors (k-means on the FeatureVectors).
2. From each cluster, **pick a representative elite** using a
   `ComponentSelector` (default `"stagnation"` — picks the cluster that has
   gone the longest without improving).
3. Build a *paradigm-shift prompt* containing the representatives and ask the
   **heavy model** (`heavy_models`, defaulting to `paradigm_models`) for
   `n_variants` *fundamentally different* candidates. Reasoning effort can be
   tuned via `reasoning_effort`.
4. Those candidates are enqueued through the same consumer path and tagged
   `is_punctuated_equilibrium=True` for logging / ablation.

This is the single most important architectural lever — it gives the
framework a way to *exit* a local optimum that's invisible to local
mutation, without paying for the heavy model on every step.

---

## 8. Selection strategies — `selection/component.py`

`ComponentSelector` decides which "component" (sampler, prompt variant,
cluster, …) to use next. Implementations:

| Selector | Strategy |
|---|---|
| `RoundRobinComponentSelector` | Cycle through components. |
| `UCBComponentSelector` | UCB1 over reward history. Default for prompt-bundle mutation. |
| `StagnationComponentSelector` | Prefer the component that has gone the longest without producing an improvement. Default for punctuated equilibrium. |

`make_component_selector("ucb" | "round_robin" | "stagnation")` is the
factory.

---

## 9. Prompts & adapters — `prompts/`, `artifacts/`

LEVI cleanly separates *what you're evolving* from *how it's evolved*:

- **`CodeAdapter`** (default): seed is a string of Python source; mutations
  are also Python source. The score function is called with the compiled
  function (and optional `inputs`).
- **`PromptAdapter`** (used by `evolve_prompts`): the artifact is a
  `PromptBundle`. The behavior features default to a custom set, AST
  features are disabled, and the score function takes the prompt(s) as
  input. The runner serializes / deserializes the bundle so the rest of the
  pipeline doesn't need to care.

`PromptBuilder` assembles the full LLM message from:

- `PROBLEM_DESCRIPTION` (frozen, user-supplied)
- `FUNCTION_SIGNATURE`
- An `OutputMode` (`"full"` = "emit the full file", `"diff"` = "emit a unified
  diff against the parent") chosen via `PipelineConfig.output_mode`.
- 0..N parents and inspiration elites drawn from the archive (controlled by
  `n_parents` / `n_inspirations`).
- The current meta-advice block, if any.

---

## 10. `prompt_opt` and `proxy_benchmark` (optional)

`prompt_opt.enabled = True` runs a DSPy / MIPROv2-style optimizer over the
mutation- and paradigm-shift prompts **before** the main loop. The optimized
templates are cached at `prompt_opt.cache_dir` so subsequent runs reuse them.

`proxy_benchmark.enabled = True` lets the framework, after the init phase,
select a small representative subset of `discovery_inputs` to evaluate
against — useful when each evaluation is very expensive and the input list
has redundant problems. `selected_indices` is persisted in the snapshot so
resumed runs use the same subset.

---

## 11. Budget, cost & resume

`BudgetConfig` lets any of dollars / evaluations / seconds / target_score
terminate the run; whichever fires first wins. Cost accounting is per-LM
through `clients/lm.py`: litellm reports `cost`, and `levi.LM` can override
with explicit per-token pricing — which is what the local Qwen example uses
(near-zero cost but non-zero so the cost-based termination still ticks).

Resume is symmetric: read `snapshot.json`, pass it as `resume_snapshot=...`,
and the framework reconstructs the archive (centroids, normalization,
elites), restores prior `total_cost`, and skips re-evaluating the seed.

---

## 12. How this hooks into SkyDiscover

SkyDiscover lays each benchmark out as a directory with three files:

```
benchmarks/<domain>/<task>/
    initial_program.py        # seed code (often wrapped in EVOLVE-BLOCK markers)
    evaluator.py              # plain Python: evaluate(program_path) -> {"combined_score": ..., ...}
    config.yaml               # system_message + harness settings
```

LEVI's API instead wants a `score_fn(fn)` (a callable, not a path), a
`function_signature` string, and a `seed_program` string. A small generic
adapter bridges the two — it works for *any* SkyDiscover benchmark that has
a plain-Python `evaluator.py` (the Docker-only `evaluator/` layout is out of
scope for now):

- **[`scripts/skydiscover_levi_adapter.py`](../scripts/skydiscover_levi_adapter.py)**
  — `load_benchmark(<dir>)` reads `initial_program.py` + `config.yaml` and
  auto-detects the entry-point function (first `def` inside the
  `EVOLVE-BLOCK-START/END` markers, falling back to the first top-level
  `def`). `make_score_fn(<evaluator_path>)` returns a *picklable*
  `functools.partial` LEVI can ship to subprocess workers. Inside the
  worker, the score function recovers the full evolved source from
  `fn.__globals__["__source_code__"]` (LEVI's `evaluate_code` stashes it
  there), writes it to a temp file *inside the benchmark dir* so sibling
  imports resolve, calls `evaluator.evaluate(tmp_path)`, and remaps
  `combined_score → score`.
- **[`scripts/run_levi_skydiscover.py`](../scripts/run_levi_skydiscover.py)**
  — thin CLI: `--benchmark <dir> --evals N`. Both LEVI model slots are
  pinned to OpenRouter. Only the evaluation count is wired; dollar and
  seconds budgets are intentionally left empty (`budget_dollars=None,
  budget_seconds=None`) — the run terminates only when 100 evaluations are
  consumed.
- **Workflows:**
  - [`.github/workflows/levi-circle-packing.yml`](../.github/workflows/levi-circle-packing.yml)
    — circle-packing (`benchmarks/math/circle_packing`).
  - [`.github/workflows/levi-adrs.yml`](../.github/workflows/levi-adrs.yml)
    — picks any ADRS task under `benchmarks/ADRS/` from a dropdown.

The two model tiers used in CI:

| Slot | Model | Role |
|---|---|---|
| `mutation_model` (small) | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | ~95% of LLM calls; cheap mutations. |
| `paradigm_model` (large) | `openrouter/openai/gpt-5` | Fires on punctuated-equilibrium steps and init-phase diversity only. |

Local quick run:

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run python scripts/run_levi_skydiscover.py \
    --benchmark benchmarks/math/circle_packing \
    --evals 100
```

Outputs land under `outputs/levi/<benchmark>/<timestamp>/`:
`snapshot.json` (full LEVI state for resume), `best_program.py`, and
`summary.json`.

---

## 13. Glossary of files / where to look

| Question | File |
|---|---|
| Public API surface | `levi/levi/__init__.py` |
| `evolve_code` / `evolve_prompts` glue | `levi/levi/methods/levi.py` |
| Pipeline producer / consumer / runner | `levi/levi/pipeline/{producer,consumer,runner,state}.py` |
| Archive / samplers | `levi/levi/pool/cvt_map_elites.py` |
| Behavior features | `levi/levi/behavior/{extractor,features}.py` |
| Punctuated equilibrium | `levi/levi/equilibrium/equilibrium.py` |
| Init / diversifier | `levi/levi/init/{diversifier,proxy_benchmark}.py` |
| Component selectors | `levi/levi/selection/component.py` |
| Prompts & bundles | `levi/levi/prompts/` |
| LM clients (litellm, claude-code, codex) | `levi/levi/clients/{lm,claude_code,codex}.py` |
| Config schema | `levi/levi/config/models.py` |
| API-key preflight check | `levi/levi/utils/preflight.py` |
| ADRS problem definitions | `levi/examples/ADRS/<problem>/problem.py` |
