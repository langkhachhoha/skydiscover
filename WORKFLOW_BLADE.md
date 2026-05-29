# WORKFLOW_BLADE — BLADE Lite

> BLADE = **B**ehavior-**L**atent **A**daptive **D**iscovery **E**ngine.
>
> A minimal evolutionary code-search engine built around a two-model
> pipeline (frontier + mutation) with one architectural contribution:
> **adaptive MAP-Elites whose cells are built from a hybrid AST +
> description-embedding behavior signature and re-clustered as the search
> progresses**, plus an ensemble of analysis-driven mutation /
> paradigm-shift prompts and a structured lessons-learnt advisor.

---

## 1. Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│  BLADE Lite                                                           │
│                                                                        │
│  (1) ClusterArchive  ◄────  hybrid AST + description-embedding        │
│      • k = n_cells = 50, FIXED across the run                         │
│      • KMeans not fit until ≥ n_cells programs are admitted           │
│      • after the first fit all 50 centroids stay alive                │
│      • admit iff score > cell incumbent                               │
│                                                                        │
│  (2) RankSampler                                                       │
│      • Zipfian over score rank: P(rank=r) ∝ (r+1)^(-β)                │
│      • β interpolates linearly with stagnation                        │
│      • crossover: second parent from a different cell when possible   │
│                                                                        │
│  (3) Monitor                                                           │
│      • plateau_steps (evals since last NEW BEST)                      │
│      • admit_gap (evals since last admit)                             │
│      • stagnation_level() = max(global, local) ∈ [0, 1]               │
│                                                                        │
│  (4) PromptSampler                                                     │
│      • 3 mutate templates: general / focused_fix / mechanism_swap     │
│      • 2 crossover templates: structural / component_swap             │
│      • drawn uniformly per call                                       │
│                                                                        │
│  (5) Parent Analyzer + Targeted Mutate (accumulating cache)            │
│      • each refresh analyses the first ``analyzer_top_k`` programs    │
│        that do NOT yet have a cached analysis (descending rank)       │
│      • cache entries survive across cycles; only evicted when the     │
│        program leaves the archive (cell incumbent kicked it out)      │
│      • when the chosen parent has a cached analysis,                  │
│        TARGETED_MUTATE_PROMPT fires with probability                  │
│        ``p_targeted_mutate``                                          │
│                                                                        │
│  (6) Structured Meta-Advisor (4-bucket, SeaEvo-style)                  │
│      • every ``meta_advice_interval`` evals, the mutation model       │
│        writes a short WORKING / SATURATED / TRY NEXT / AVOID note     │
│      • input: top-K descriptions, recent admits split into IMPROVING  │
│        (|Δ| > 1e-3 above parent) and SATURATED (|Δ| ≈ 0 or regression)│
│        buckets, typed error taxonomy, previous advice                 │
│      • note is injected verbatim into future mutate / crossover       │
│        prompts with probability ``meta_advice_inject_p``              │
│                                                                        │
│  + Orchestrator: 2-phase bootstrap + main loop + three-mode           │
│      paradigm shift (synthesis / surgical / shift) dispatched by      │
│      stagnation level (low → synthesis, mid → surgical, high → shift)│
└──────────────────────────────────────────────────────────────────────┘
```

Source files:

- [levi/levi/simple/archive.py](levi/levi/simple/archive.py) — ClusterArchive
- [levi/levi/simple/ast_features.py](levi/levi/simple/ast_features.py) — 14-d AST counts
- [levi/levi/simple/rank_sampler.py](levi/levi/simple/rank_sampler.py) — RankSampler
- [levi/levi/simple/monitor.py](levi/levi/simple/monitor.py) — Monitor
- [levi/levi/simple/parser.py](levi/levi/simple/parser.py) — OutputParser + `OUTPUT_FORMAT_INSTRUCTION`
- [levi/levi/blade/orchestrator.py](levi/levi/blade/orchestrator.py) — orchestrator
- [levi/levi/blade/prompts.py](levi/levi/blade/prompts.py) — prompt builders + PromptSampler + `classify_error`
- [scripts/run_blade.py](scripts/run_blade.py) — CLI
- [scripts/smoke_blade_prompts.py](scripts/smoke_blade_prompts.py) — live audit of the 3-section prompt contract

### 1.1 ClusterArchive — invariant cell count

**Behavior signature.** Every admitted program is described by

```text
behavior_vec(p) = standardise([
    ast_features(p.code),         # 14 counts, log1p damped
    PCA_d(embedding(p.description))  # d=8 by default
])
```

Both halves are z-score-standardised online (Welford) before
concatenation so neither dominates KMeans's Euclidean distance. The PCA
basis is re-fit from the live population on each re-cluster — the 8
principal components evolve with the search.

**Cells.** `n_cells` is held **fixed** across the run (default 50).

- KMeans is not fit until the population reaches `n_cells`. Below that
  threshold each admit gets its own `cell_id` and **no coalescing
  happens**, so the archive grows freely up to `n_cells` programs.
- Once `n_cells` programs are admitted, KMeans fits with exactly
  `k = n_cells` clusters. After every fit the centroid grid has
  `n_cells` slots; some Voronoi regions may be empty but the slots
  stay alive, so subsequent admits can grow `num_occupied_cells`
  toward `n_cells` instead of plateauing.
- `min_admits_before_cluster` (default 16) is only a sanity floor;
  the binding threshold for the first fit is `n_cells`.

The runtime invariant pinned by
[test_archive_cells.py](levi/tests/blade/test_archive_cells.py) is

```text
num_occupied_cells ≤ n_cells  (always)
num_occupied_cells → n_cells   (as the search proceeds)
```

### 1.2 RankSampler — score-rank Zipfian

Parameter-free relative to score scale:

```text
P(rank=r) ∝ (r+1)^(-β(stagnation))
β(stagnation) = β_min + (1 - stagnation) · (β_max - β_min)
              = 0.3   when stagnation = 1.0  (≈ uniform)
              = 2.0   when stagnation = 0.0  (top-3 dominate)
```

Three call patterns: `select_parent`, `select_two_parents` (cell-aware
second parent), `select_inspirations`. The sampler does not track
per-cell statistics — it reads `cell_id` only inside
`select_two_parents`, so a re-cluster that shuffles cell IDs only
perturbs one tiebreaker.

### 1.3 Monitor — global + local stagnation

`stagnation_level()` = `max(global_stagnation, local_stagnation)`,
where the global timer ticks until the best score improves and the
local timer ticks until any cell incumbent is replaced. The combined
signal drives the rank sampler's β and the paradigm-shift mode
selector.

### 1.4 PromptSampler — five mutate / crossover templates

The mutation worker draws one of three mutate prompts (or two
crossover prompts) uniformly at random per call. Every template
requires the model to emit **three sections in this exact order**:

```text
## Analysis        # structured reasoning (bullets, headings allowed)
## Description     # 2-4 sentence paragraph, ≤ 80 words, no bullets
## Code            # fenced ```python``` block
```

Only `## Description` and `## Code` are consumed by the search system.
`## Analysis` is for the model's own reasoning — the archive does NOT
embed it, so the rich structured bullets do not pollute the behavior
signature used for cell assignment. This contract is pinned by
[test_prompts_keep_analysis_separate_from_description](levi/tests/blade/test_prompts.py)
and validated end-to-end against the live mutation model by
[scripts/smoke_blade_prompts.py](scripts/smoke_blade_prompts.py).

The five templates:

- **`mutate/general`** — 4-section analysis (components, strengths,
  weaknesses, plan) then a free improvement.
- **`mutate/focused_fix`** — single tightest-constraint fix with an
  explicit preservation list.
- **`mutate/mechanism_swap`** — replace one sub-mechanism (e.g. cooling
  schedule, neighbour proposal) while keeping the algorithm class.
- **`crossover/structural`** — full structural hybrid; component table
  required.
- **`crossover/component_swap`** — base skeleton + one transplanted
  component from the donor.

Choice is uniform random — no learned weights — so the mutation model
sees prompt-level diversity even when the parent pool is narrow.

### 1.5 Parent Analyzer + Targeted Mutate (accumulating cache)

A background monitor wakes every `analyzer_interval` evaluations
(default 30) and asks the mutation model for a short review of
**uncached** top-ranked programs. Each review identifies three
bottlenecks plus three suggested changes.

**Cache policy — accumulate, then churn.** On each refresh:

1. Cache entries for programs that no longer live in the archive
   (kicked out by a better cell incumbent, dropped by re-cluster)
   are evicted.
2. The archive is walked in score-descending order; the first
   `analyzer_top_k` programs that do **not yet** have a cache entry
   get analysed. Already-cached programs are skipped — their analysis
   is untouched.

With `analyzer_top_k = 3` and a frozen top-3, refresh #1 covers ranks
1-3, refresh #2 covers ranks 4-6, refresh #3 covers ranks 7-9, …
until the entire archive has been analysed. From then on refreshes
are no-ops until cell churn injects a new program, at which point
exactly the displaced entry is evicted and the newcomer is analysed.

This replaces an earlier "top-K only, evict everything else" policy
that re-analysed the same three frozen parents every cycle and
starved targeted-mutate of analysis-level diversity. The
accumulating policy means cache size is bounded by `len(archive)`
(at most `n_cells`) and no explicit per-entry cap is needed. The
property is pinned by
[test_analyzer_cache.py](levi/tests/blade/test_analyzer_cache.py).

**Targeted mutate.** When a parent with a cached review is selected
for mutation, the orchestrator picks `TARGETED_MUTATE_PROMPT` with
probability `p_targeted_mutate` (default 0.5) — otherwise it falls
back to the standard `PromptSampler`. The targeted prompt receives
the analysis verbatim and asks the model to commit to ONE suggested
change. The default of 0.5 (down from an earlier 0.7) leaves room
for the `PromptSampler`'s template-level diversity to surface even
when the chosen parent has cached analysis, which it does most of
the time under the Zipfian sampler.

### 1.6 Three-mode paradigm shift (stagnation-routed)

The frontier paradigm-shift call picks one of three prompt templates
based on the current stagnation level. Each mode receives a different
anchor / inspiration configuration:

| Mode | Stagnation range | Anchors (full code) | Inspirations | Asks for |
| --- | --- | --- | --- | --- |
| `synthesis` | `s ≤ 0.4` | 3 (close-in-score) | `paradigm_n_inspirations` | combine 2-3 anchors into one program (`MOVE: SYNTHESIS`) |
| `surgical` | `0.4 < s ≤ 0.7` | 1 (champion only) | `paradigm_surgical_n_inspirations` | one local structural fix to the champion (`MOVE: SURGICAL`) |
| `shift` | `s > 0.7` | 2 | `paradigm_n_inspirations` | propose a genuinely new paradigm class (`MOVE: SHIFT`) |

**Rationale.** The earlier mapping routed *high* stagnation to
`surgical` and mid stagnation to `shift`. Empirically that left the
search trapped: when the champion has plateaued and the same family of
solutions has dominated for many evaluations, asking the frontier for
*another* local fix to the same code rarely escapes the optimum. The
mode that actually broke prior plateaus in our runs was `shift` — a
genuinely new paradigm class. We therefore route the deepest stagnation
to `shift` and reserve `surgical` for the mid-stagnation band, where
the champion still has momentum and a careful local polish can
compound into measurable progress.

The mode thresholds are tunable
(`paradigm_synthesis_max_stagnation`, `paradigm_surgical_max_stagnation`).

### 1.7 Structured Meta-Advisor (4-bucket, SeaEvo-style)

The advisor is a background monitor that wakes every
`meta_advice_interval` evaluations and asks the mutation model for a
short prescriptive note. The note is injected verbatim into the next
mutate / crossover prompts with probability `meta_advice_inject_p`
(default 0.35).

**Four-bucket output schema.** The model must emit exactly four short
paragraphs, in this order, with these literal headers:

```text
WORKING:   <1-2 sentences naming what the IMPROVING admits and leaders
            share; cite an operator (e.g. mutate_focused_fix) if one
            dominates improving admits>
SATURATED: <1-2 sentences naming any strategy family / operator that
            is producing admits but no longer producing IMPROVEMENT.
            Future prompts should de-emphasise this direction. If no
            clear saturation, write "none".>
TRY NEXT:  <2-3 short imperative suggestions building on WORKING and
            EXPLICITLY moving away from SATURATED>
AVOID:     <1-2 anti-patterns that actually appear in the window's
            failure taxonomy>
```

The `SATURATED` bucket is the SeaEvo-style addition. SeaEvo's
Strategic Landscape Navigation tracks effective / saturated /
underexplored strategy *families* at the population level rather than
reasoning about individual programs in isolation. We mirror that
signal by splitting recent admits into `IMPROVING` (real progress
vs parent) and `SATURATED` (accepted-but-not-better, or admitted with
a regression) before they reach the advisor's prompt — see §1.7.1.

**Inject probability.** Lowered from an earlier 0.7 → 0.35. At 0.7
nearly every mutation prompt saw the same advice text, which heavily
biased the mutation model toward whichever direction the last advisor
cycle endorsed and blunted the diversity provided by `PromptSampler`.
0.35 means ~1 in 3 prompts is advice-guided, which still gives the
advisor a meaningful nudge across a paradigm-shift window (50 evals ×
0.35 ≈ 17 advice-guided attempts) without saturating it.

**Five signal streams.** The advisor's prompt is built from:

1. **Top-K archived program descriptions** with their scores
   (`top_descriptions`, K=3). These are the current leaders the
   advisor must reason about under WORKING / TRY NEXT.
2. **Recent admits — IMPROVING bucket.** Admits with
   `Δ vs parent > 1e-3`. These are the ones to amplify.
3. **Recent admits — SATURATED bucket.** Admits with
   `|Δ| ≤ 1e-3` or `Δ < -1e-3`. The operator family is converting
   but no longer making progress — the advisor must call this out
   under SATURATED.
4. **Typed error taxonomy.** Errors from this window are bucketed by
   `classify_error` into seven kinds — `timeout`, `syntax`,
   `constraint`, `shape_mismatch`, `numpy_api`, `name_or_attr`,
   `type_error`, with `other` as fallback — and rendered with counts
   plus a per-source breakdown plus one truncated example per bucket.
5. **Previous advice** carried over verbatim so the advisor can refine
   instead of repeating.

**Mode switch.** `meta_advice_mode` is either `rich` (default — feed
all five signal streams) or `errors_only` (drop top descriptions and
both admit buckets; only the typed error taxonomy + previous advice
flow in). The latter exists exclusively as a paper ablation; in
production runs `rich` is the only sensible setting.

Per-window invariant: after each advisor call the per-window error
queue is cleared so the next cycle sees a fresh taxonomy, while the
recent-admits queue intentionally rolls across cycles so the advisor
can spot operators that are *consistently* productive.

#### 1.7.1 Saturation split — absolute Δ tolerance

The IMPROVING / SATURATED split uses an absolute tolerance of
`1e-3` on the score delta vs parent. We use an absolute, score-scale
tolerance rather than a relative one so the threshold is interpretable
across benchmarks. 1e-3 is small enough to rule out actual
breakthroughs on the benchmarks we care about (circle-packing
breakthroughs were ≥ 0.01) while still catching the long tail of
"+0.0001" admits that signal a family is mined out. Admits with
`Δ = None` (no parent score / non-finite parent score) fall into
IMPROVING — we cannot judge them and would rather over-report
progress than over-report saturation.

---

## 2. End-to-end flow

```text
1. Phase 1 (sequential, frontier model)
   for i in 1..n_diverse_seeds:
     prompt = build_diverse_seed_prompt(existing_seeds=…)
     code   = frontier(prompt)
     score  = evaluate(code)
     archive.add(Program(..., source="init"))

2. Phase 2 (parallel, mutation model)
   prompts = [build_init_variant_prompt(inspirations=sample(seeds, 2))
              for _ in n_diverse_seeds × n_variants_per_seed]
   asyncio.gather(_one_variant(p) for p in prompts)
   → archive.add(…) for each that compiles + evaluates

3. Main loop
   while not budget_exhausted:
     for _ in n_workers concurrently:
       op = "crossover" if rng() < p_crossover else "mutate"
       parent(s) ← sampler.select_parent[_two](archive.programs(), stagnation)
       insps     ← sampler.select_inspirations(...)
       if cached_analysis[parent] and rng() < p_targeted_mutate:
         prompt = build_targeted_mutate_prompt(..., meta_advice=...)
       else:
         label, tmpl = prompt_sampler.pick_mutate(rng) | pick_crossover(rng)
         prompt = build_mutate_prompt(template=tmpl, ..., meta_advice=...)
                | build_crossover_prompt(template=tmpl, ..., meta_advice=...)
       code = mutation(prompt)
       archive.add(Program(..., source=op_label))
     # No repair branch: failed candidates count as rejects and feed
     # the advisor's error taxonomy via _advisor_errors.

   parallel:
     • _pe_monitor   fires _paradigm_shift() every pe_cron_interval evals
     • _meta_advice_monitor refreshes the 4-bucket note every
       meta_advice_interval evals
     • _analyzer_monitor refreshes the parent-analysis cache every
       analyzer_interval evals (accumulating, churn-driven)

4. Paradigm shift
   mode = pick_mode(stagnation)         # synthesis | surgical | shift
   prompt, anchors = _build_paradigm_prompt_for_mode(mode)
   code = frontier(prompt)
   archive.add(Program(..., source="paradigm"))
   if no error: fanout (parallel): n_paradigm_variants × paradigm_variant
```

The archive's re-clustering happens inside `archive.add(...)` whenever
enough admits have accumulated. The orchestrator never touches cluster
bookkeeping directly. Admits and errors are simultaneously appended to
two small rolling queues (`_advisor_admits`, `_advisor_errors`) that
feed the meta-advisor.

---

## 3. Configuration

All defaults below are read from the `@dataclass` declarations in
[orchestrator.py](levi/levi/blade/orchestrator.py),
[archive.py](levi/levi/simple/archive.py),
[monitor.py](levi/levi/simple/monitor.py), and
[rank_sampler.py](levi/levi/simple/rank_sampler.py) — this table is
the canonical reference, not the workflow file.

### 3.1 Models

| Knob | Default | Notes |
| --- | --- | --- |
| `mutation_model` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | Small / fast model. Drives mutate, crossover, init variants, paradigm variants, meta-advisor, analyzer. |
| `paradigm_model` | `openrouter/openai/gpt-5` | Frontier model. Drives diverse-seed phase and per-mode paradigm shift. |
| `embedding_model` | `openrouter/openai/text-embedding-3-small` | Description embedder for the second half of the behavior signature. |

### 3.2 Budget & concurrency

| Knob | Default | Notes |
| --- | --- | --- |
| `budget_dollars` | `None` | Hard $ cap. First budget to trigger wins. |
| `budget_evals` | `None` | Hard evaluation cap. |
| `budget_seconds` | `None` | Hard wall-clock cap. |
| `target_score` | `None` | Early-exit threshold. |
| `n_workers` | `4` | Concurrent mutation workers in the main loop. |
| `n_eval_processes` | `4` | Sandboxed processes for `evaluate_code`. |
| `eval_timeout` | `120.0` s | Per-candidate evaluation timeout. |
| `llm_temperature` | `0.8` | Default temperature for mutate / crossover calls. |
| `llm_max_tokens` | `None` | Cap on the mutation model. `None` = no cap. |
| `paradigm_max_tokens` | `None` | Cap on the paradigm (frontier) model. |
| `llm_call_timeout` | `600.0` s | Per-call timeout for any LLM completion. |
| `shutdown_grace_seconds` | `15.0` s | How long the orchestrator waits for in-flight tasks to honor cancel. |

### 3.3 Bootstrap & paradigm-shift cadence

| Knob | Default | Notes |
| --- | --- | --- |
| `n_diverse_seeds` | `5` | Frontier seeds (phase 1, sequential). |
| `n_variants_per_seed` | `20` | Mutation variants per seed (phase 2, parallel). |
| `init_diversity_temperature` | `0.8` | Frontier temperature for diverse seeds. |
| `init_variant_temperature` | `0.9` | Mutation temperature for phase-2 fanout. |
| `pe_cron_interval` | `50` | Fire paradigm shift every N completed evaluations **after init**. The PE cadence counter is reset to `eval_count` when bootstrap finishes, so phase-1 (frontier seeds) + phase-2 (mutation variants) evals do NOT push PE toward its first trigger. Set to 0 to disable. |
| `paradigm_min_archive_size` | `5` | Skip paradigm shift if `num_occupied_cells <` this. |
| `paradigm_temperature` | `0.8` | Frontier temperature inside paradigm shift. |
| `paradigm_variant_temperature` | `0.85` | Mutation temperature for paradigm-fanout variants. |
| `n_paradigm_variants` | `4` | Variants spun off the new paradigm seed. |

### 3.4 Operator mix

| Knob | Default | Notes |
| --- | --- | --- |
| `p_crossover` | `0.35` | Probability of crossover per main-loop step. The remaining 65% is mutate. There is **no repair branch**: failed candidates count as rejects and surface in the advisor's error taxonomy. |

### 3.5 Three-mode paradigm shift

| Knob | Default | Notes |
| --- | --- | --- |
| `paradigm_synthesis_max_stagnation` | `0.4` | At or below → `synthesis` mode. |
| `paradigm_surgical_max_stagnation` | `0.7` | Between synthesis cap and this → `surgical` mode; above → `shift`. |
| `paradigm_synthesis_n_anchors` | `3` | Full-code anchors in synthesis mode. |
| `paradigm_shift_n_anchors` | `2` | Full-code anchors in shift mode. |
| `paradigm_surgical_n_inspirations` | `5` | Description-only inspirations in surgical mode (alongside the single champion anchor). |
| `paradigm_n_inspirations` | `5` | Description-only inspirations in synthesis / shift modes. |

### 3.6 Targeted-mutate analyzer

| Knob | Default | Notes |
| --- | --- | --- |
| `enable_targeted_mutate` | `True` | Master toggle for the analyzer + TARGETED_MUTATE_PROMPT path. |
| `analyzer_interval` | `30` | Refresh the analysis cache every N completed evaluations. |
| `analyzer_top_k` | `3` | How many *new* (uncached) programs to analyse per refresh. Cache survives across refreshes; only evicted when the program leaves the archive. |
| `analyzer_temperature` | `0.3` | Mutation-model temperature for the analysis call. |
| `analyzer_max_tokens` | `500` | Token cap for one analysis. |
| `p_targeted_mutate` | `0.5` | When the chosen parent has a cached analysis, probability of using TARGETED_MUTATE_PROMPT (else fall back to PromptSampler). |

### 3.7 Meta-advisor

| Knob | Default | Notes |
| --- | --- | --- |
| `enable_meta_advice` | `True` | Master toggle. |
| `meta_advice_interval` | `50` | Regenerate the 4-bucket note every N evaluations. |
| `meta_advice_mode` | `"rich"` | Either `rich` (top descriptions + IMPROVING/SATURATED admit split + typed error taxonomy + previous advice) or `errors_only` (taxonomy + previous advice only). The latter is the paper ablation flag. |
| `meta_advice_inject_p` | `0.35` | Probability of injecting the current note into the next mutate / crossover prompt. |
| `meta_advice_temperature` | `0.4` | Mutation-model temperature for the advisor call. |
| `meta_advice_max_tokens` | `400` | Token cap for one advisor note. |

### 3.8 Archive (ClusterArchive)

| Knob | Default | Notes |
| --- | --- | --- |
| `n_cells` | `50` | **Held fixed across the run.** Target centroid count for KMeans. |
| `embedding_dim` | `8` | PCA target dimension for the description-embedding half. |
| `recluster_every` | `30` | Re-fit KMeans every N admits (post the first fit). |
| `min_admits_before_cluster` | `16` | Sanity floor; the *binding* threshold for the first fit is `n_cells`. |
| `use_ast` | `True` | Toggle the 14-d AST features half (ablation A2 off). |
| `use_embedding` | `True` | Toggle the PCA description-embedding half (ablation A1 off). |
| `adaptive_recluster` | `True` | Re-fit KMeans periodically (ablation A3 off → fit once, freeze). |

### 3.9 RankSampler

| Knob | Default | Notes |
| --- | --- | --- |
| `beta_max` | `2.0` | Zipfian exponent when `stagnation = 0` (top-3 dominate). |
| `beta_min` | `0.3` | Zipfian floor when `stagnation = 1` (near-uniform). |
| `n_inspirations` | `3` | Number of description-only inspirations surfaced per mutate / crossover call. |

### 3.10 Monitor

| Knob | Default | Notes |
| --- | --- | --- |
| `plateau_max` | `100` | Denominator for global stagnation (evals since last NEW BEST). |
| `admit_gap_max` | `20` | Denominator for local stagnation (evals since last admit). |
| `accept_window_size` | `50` | Rolling window for the acceptance-rate diagnostic. |
| `score_eps` | `1e-9` | Tie-break tolerance for NEW BEST detection. |

### 3.11 Workflow / CLI override surface

The dispatch workflow ([.github/workflows/blade.yml](.github/workflows/blade.yml))
exposes models, budget, concurrency, `pe_interval`, `n_diverse_seeds`,
`n_variants_per_seed`, and `ablation` directly. Everything else goes
through the JSON `advanced_options` input, e.g.

```json
{
  "n_cells": 64,
  "recluster_every": 30,
  "embedding_dim": 8,
  "n_paradigm_variants": 4,

  "meta_advice_interval": 50,
  "meta_advice_mode": "rich",
  "meta_advice_inject_p": 0.35,
  "meta_advice_disabled": false,
  "targeted_mutate_disabled": false,

  "analyzer_interval": 30,
  "analyzer_top_k": 3,
  "p_targeted_mutate": 0.5,
  "p_crossover": 0.35,

  "paradigm_synthesis_max_stagnation": 0.4,
  "paradigm_surgical_max_stagnation": 0.7,
  "paradigm_synthesis_n_anchors": 3,
  "paradigm_shift_n_anchors": 2,
  "paradigm_surgical_n_inspirations": 5
}
```

The CLI exposes the same knobs as `--n-cells`, `--analyzer-interval`,
`--p-targeted-mutate`, `--p-crossover`, `--meta-advice-mode`,
`--meta-advice-inject-p`, `--paradigm-synthesis-max-stagnation`,
`--paradigm-surgical-max-stagnation`, etc. Run
`uv run python scripts/run_blade.py --help` for the full list.

---

## 4. Ablation protocol (paper-facing)

The workflow exposes eight mutually-exclusive ablation choices via the
`ablation` dropdown. Three target the behavior signature / clustering
side (A1–A3); four target individual operator / loop components
(A4–A7); the remaining slot is the un-ablated default (`full`).

| Ablation | Toggle (workflow) | Toggle (CLI) | What is removed |
| --- | --- | --- | --- |
| **full** | `ablation: full` | (default) | None — every component on. |
| **A1 — emb-only** | `ablation: emb_only` | `--emb-only` | The 14-d AST half of the behavior signature. |
| **A2 — ast-only** | `ablation: ast_only` | `--ast-only` | The PCA description-embedding half. |
| **A3 — static-cells** | `ablation: static_cells` | `--static-cells` | Adaptive re-clustering. KMeans fit once at `n_cells`, then frozen. |
| **A4 — no-meta-advice** | `ablation: no_meta_advice` | `--no-meta-advice` | The lessons-learnt advisor entirely. |
| **A5 — meta-errors-only** | `ablation: meta_errors_only` | `--meta-advice-mode errors_only` | The success-side signals (top descriptions + IMPROVING/SATURATED admits). Advisor still runs but sees only the error taxonomy. Isolates the contribution of the success-side feed. |
| **A6 — no-targeted-mutate** | `ablation: no_targeted_mutate` | `--no-targeted-mutate` | The parent analyzer + `TARGETED_MUTATE_PROMPT`. The PromptSampler still produces the other 5 templates. |
| **A7 — no-crossover** | `ablation: no_crossover` | `--no-crossover` | The crossover branch (sets `p_crossover = 0`). |
| **A8 — no-paradigm** | `ablation: no_paradigm` | `--pe-interval 0` | The frontier paradigm-shift loop. |

The three behavior-signature toggles (A1, A2, A3) are mutually
orthogonal to the operator-side toggles (A4–A8), so reporting results
as two separate ablation tables is reasonable. The previous
`--no-repair` toggle is gone: the repair branch was removed
unconditionally (failed candidates count as rejects and surface in
the advisor's error taxonomy), so there is no longer an ablation slot
for it.

---

## 5. Output artifacts

Per run, `output_dir/`:

- `best.py` — top-scoring program (also written as `best_program.py`).
- `summary.json` — run metadata + budget + ablation block (records the
  exact flag combination — `no_meta_advice`, `meta_advice_mode`,
  `no_targeted_mutate`, `no_crossover`, `p_crossover`, plus the
  behavior-signature ablation booleans).
- `snapshot.json` — full archive dump:
  - `monitor` — `eval_count, best_score, plateau_steps, admit_gap,
    global_stagnation, local_stagnation, stagnation_level, accept_rate`
  - `paradigm_trials` — `idx, accepted, score, delta, description`
  - `cells` — one entry per occupied cell, sorted by score desc
  - `ablation` — block reflecting the three signature toggles + the
    low-level archive parameters
  - `meta_advice` — `{enabled, interval, mode, inject_p,
    trigger_count, current}` (the `current` field carries the
    WORKING/SATURATED/TRY/AVOID text from the most recent advisor cycle)
  - `analyzer` — `{enabled, interval, top_k, trigger_count,
    targeted_mutate_count, p_targeted_mutate, cache_size}`
  - `paradigm_modes` — `{synthesis_max_stagnation,
    surgical_max_stagnation, synthesis_n_anchors, shift_n_anchors,
    surgical_n_inspirations}`

---

## 6. Logging conventions

- `[Eval #N] {model} {status} | source: ... | score: ... | best: ... | $cost`
- Sources include the prompt-variant label, e.g. `mutate_focused_fix`,
  `crossover_component_swap`, `mutate_targeted`, `paradigm_synthesis`,
  `paradigm_surgical`, `paradigm_shift`.
- `[Status] Cost: ... | Evals: ... | Archive: N (cells K) | Best: ... | Elapsed: ...s`
- `[BLADE PE] mode=shift anchors=2 stagnation=0.82 best=2.6133`
- `[BLADE PE] fanout: K variants from paradigm seed (score=..., accepted=...)`
- `[BLADE analyzer] trigger #N at eval=M (analysing up to K uncached programs; cache=C)`
- `[BLADE advisor] new advice (N chars) at eval=M`
- `[Archive] recluster: n=... → K/n_cells occupied cells (admits since last recluster=...)`

---

## 7. Quick start

```bash
# Default run (every component on)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5

# Bigger archive (n_cells controls the cell-count invariant)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --n-cells 64

# Disable targeted mutate (analyzer + TARGETED_MUTATE_PROMPT off)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --no-targeted-mutate

# Force shift earlier (push the surgical→shift boundary down to 0.5)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --paradigm-surgical-max-stagnation 0.5

# Increase advice influence (60% of mutate prompts see the advice block)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --meta-advice-inject-p 0.6

# Paper ablation: errors-only meta-advisor
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --meta-advice-mode errors_only \
  --output-dir runs/ablation-meta-errors-only

# Paper ablation: AST behavior signature only
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --ast-only --output-dir runs/ablation-ast-only
```

On GitHub Actions: dispatch `.github/workflows/blade.yml` and pick the
`ablation` dropdown.

---

## 8. Verification

Two pre-merge gates pin the contracts the system relies on:

```bash
# Offline unit tests
cd levi && uv run python -m pytest tests/blade/ --tb=short
# Coverage: archive invariants + prompt templates + paradigm
# modes (mode dispatch + per-mode anchors) + orchestrator E2E on
# fake LLMs + meta-advisor signal wiring (incl. IMPROVING/SATURATED
# split + 4-bucket schema) + analyzer accumulating-cache policy
# (5 tests) + error taxonomy = 33 offline tests.

# Live audit: one real call per prompt variant against the production
# mutation model, checks that every response contains
# ## Analysis → ## Description → ## Code in that order, with no bullet
# leakage into the embedded description. Costs ~$0.01-0.02 per run.
uv run python scripts/smoke_blade_prompts.py
```

The unit tests are deterministic and cheap; the smoke script is the
guard against silent format drift when a new model release ships.
