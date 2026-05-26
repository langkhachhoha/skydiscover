# WORKFLOW_BLADE — BLADE Lite

> BLADE = **B**ehavior-**L**atent **A**daptive **D**iscovery **E**ngine.
>
> A minimal evolutionary code-search engine built on top of a two-model
> pipeline (frontier + mutation) with one architectural contribution:
> **adaptive MAP-Elites whose cells are built from a hybrid AST +
> description-embedding behavior signature and re-clustered as the search
> progresses**, plus a small ensemble of analysis-driven mutation /
> paradigm-shift prompts.

---

## 1. Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│  BLADE Lite                                                           │
│                                                                        │
│  (1) ClusterArchive  ◄────  hybrid AST + embedding behavior          │
│      • k = n_cells = 50, FIXED across the run                        │
│      • KMeans not fit until ≥ n_cells programs are admitted          │
│      • after the first fit, all 50 centroids stay alive (empty       │
│        Voronoi regions are kept open for future admits)              │
│      • admit iff score > cell incumbent                              │
│                                                                        │
│  (2) RankSampler                                                       │
│      • Zipfian over score rank: P(rank=r) ∝ (r+1)^(-β)               │
│      • β interpolates linearly with stagnation                       │
│      • crossover: second parent from a different cell when possible  │
│                                                                        │
│  (3) Monitor                                                           │
│      • plateau_steps (evals since last new best)                     │
│      • admit_gap (evals since last admit)                            │
│      • stagnation_level() = max(global, local) ∈ [0, 1]              │
│                                                                        │
│  (4) PromptSampler                                                     │
│      • 3 mutate templates: general / focused_fix / mechanism_swap    │
│      • 2 crossover templates: structural / component_swap             │
│      • drawn uniformly per call                                       │
│                                                                        │
│  (5) Parent Analyzer (Đề xuất 1)                                       │
│      • mutation model writes a 250-word review of top-K parents      │
│      • cached by id(parent), refreshed every analyzer_interval evals │
│      • when cached, the orchestrator uses TARGETED_MUTATE_PROMPT     │
│        with probability p_targeted_mutate                             │
│                                                                        │
│  + Orchestrator: standard 2-phase bootstrap + main loop +             │
│      three-mode paradigm shift (synthesis / shift / surgical)         │
│      dispatched by stagnation level.                                  │
└──────────────────────────────────────────────────────────────────────┘
```

Source files:

- [levi/levi/simple/archive.py](levi/levi/simple/archive.py) — ClusterArchive
- [levi/levi/simple/ast_features.py](levi/levi/simple/ast_features.py) — 14-d AST counts
- [levi/levi/simple/rank_sampler.py](levi/levi/simple/rank_sampler.py) — RankSampler
- [levi/levi/simple/monitor.py](levi/levi/simple/monitor.py) — Monitor
- [levi/levi/simple/parser.py](levi/levi/simple/parser.py) — OutputParser + `OUTPUT_FORMAT_INSTRUCTION`
- [levi/levi/blade/orchestrator.py](levi/levi/blade/orchestrator.py) — orchestrator
- [levi/levi/blade/prompts.py](levi/levi/blade/prompts.py) — prompt builders + PromptSampler
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
basis is re-fit from the live population on each re-cluster — i.e. the
8 principal components evolve with the search.

**Cells.** Previously KMeans was fit with `k = min(n_cells, n)`, which
combined with the post-recluster coalesce (keep best per cell) caused
the archive to collapse to ~16 cells for the entire run on
`circle_packing_rect`. The new contract:

- `n_cells` is held **fixed** across the run (default 50).
- KMeans is not fit until the population reaches `n_cells`. Below that
  threshold each admit gets its own `cell_id` and **no coalescing
  happens**, so the archive grows freely up to `n_cells` programs.
- Once `n_cells` programs are admitted, KMeans fits with
  exactly `k = n_cells` clusters. After every fit the centroid grid
  has `n_cells` slots; some Voronoi regions may be empty but the
  slots stay alive, so subsequent admits can grow
  `num_occupied_cells` toward `n_cells` instead of plateauing.
- `min_admits_before_cluster` (default 16) is now only a sanity floor;
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
second parent), `select_inspirations`. The sampler does **not** track
per-cell statistics — it reads `cell_id` only inside
`select_two_parents`, so even a re-cluster that shuffles cell IDs only
perturbs one tiebreaker. No fix is required; this is documented here
so future readers do not chase the symptom.

### 1.3 Monitor — global + local stagnation

`stagnation_level()` = `max(global_stagnation, local_stagnation)`,
where the global timer ticks until the best score improves and the
local timer ticks until any cell incumbent is replaced. The combined
signal drives the rank sampler's β.

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
embed it, so the rich structured bullets (Components, Strengths,
Weaknesses, etc.) do not pollute the behavior signature used for cell
assignment. This contract is pinned by
[test_prompts_keep_analysis_separate_from_description](levi/tests/blade/test_prompts.py)
and validated end-to-end against the live Qwen-30B model by
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

No learned weights — the choice is uniform random — so the mutation
model gets prompt-level diversity even when the parent pool is narrow.

### 1.5 Parent Analyzer + Targeted Mutate (Đề xuất 1)

A background monitor wakes every `analyzer_interval` evaluations
(default 30) and asks the mutation model for a short review of the top
`analyzer_top_k` programs (default 3). Each review is 250 words and
identifies three bottlenecks plus three suggested changes. Reviews are
cached by `id(parent)` and dropped when the parent leaves the top-K.

When a parent with a cached review is selected for mutation, the
orchestrator picks `TARGETED_MUTATE_PROMPT` with probability
`p_targeted_mutate` (default 0.5) — otherwise it falls back to the
standard `PromptSampler`. The targeted prompt receives the analysis
verbatim and asks the model to commit to ONE suggested change.

### 1.6 Three-mode paradigm shift (Đề xuất 8)

The frontier paradigm-shift call now picks one of three prompt
templates based on the current stagnation level. Each mode receives a
different anchor / inspiration configuration:

| Mode | Stagnation range | Anchors (full code) | Inspirations | Asks for |
| --- | --- | --- | --- | --- |
| `synthesis` | `s ≤ 0.4` | 3 (close-in-score) | `paradigm_n_inspirations` | combine 2-3 anchors into one program (`MOVE: SYNTHESIS`) |
| `shift` | `0.4 < s ≤ 0.7` | 2 | `paradigm_n_inspirations` | propose a genuinely new paradigm class (`MOVE: SHIFT`) |
| `surgical` | `s > 0.7` | 1 (champion only) | `paradigm_surgical_n_inspirations` (descriptions only) | one local structural fix to the champion (`MOVE: SURGICAL`) |

Surgical mode is the lever for late-run plateaus: when previous
paradigm trials are not contributing to the best score, the frontier
is told explicitly to focus on the champion and write a small,
local, structural improvement — not a new paradigm and not a
constant-sweep.

The mode thresholds are tunable
(`paradigm_synthesis_max_stagnation`, `paradigm_shift_max_stagnation`).

---

## 2. End-to-end flow

```text
1. Phase 1 (sequential, frontier model)
   for i in 1..n_diverse_seeds:
     prompt = build_diverse_seed_prompt(existing_seeds=…)
     code  = frontier(prompt)
     score = evaluate(code)
     archive.add(Program(..., source="init"))

2. Phase 2 (parallel, mutation model)
   prompts = [build_init_variant_prompt(inspirations=sample(seeds, 2))
              for _ in n_diverse_seeds × n_variants_per_seed]
   asyncio.gather(_one_variant(p) for p in prompts)
   → archive.add(…) for each that compiles + evaluates

3. Main loop
   while not budget_exhausted:
     for _ in n_workers concurrently:
       parent(s) ← sampler.select_parent[_two](archive.programs(), stagnation)
       insps     ← sampler.select_inspirations(...)
       if cached_analysis[parent] and rng() < p_targeted_mutate:
         prompt = build_targeted_mutate_prompt(...)
       else:
         label, tmpl = prompt_sampler.pick_mutate(rng) | pick_crossover(rng)
         prompt = build_mutate_prompt(template=tmpl, ...) | build_crossover_prompt(template=tmpl, ...)
       code = mutation(prompt)
       archive.add(Program(...))
     if error_buffer: _repair_one()    # one-shot

   parallel:
     • _pe_monitor   fires _paradigm_shift() every pe_cron_interval evals
     • _meta_advice_monitor refreshes lessons-learnt every meta_advice_interval evals
     • _analyzer_monitor refreshes top-K parent analyses every analyzer_interval evals

4. Paradigm shift
   mode = pick_mode(stagnation)  # synthesis | shift | surgical
   prompt, anchors = _build_paradigm_prompt_for_mode(mode)
   code = frontier(prompt)
   archive.add(Program(..., source="paradigm"))
   if no error: fanout (parallel): n_paradigm_variants × paradigm_variant
```

The archive's re-clustering happens inside `archive.add(...)` whenever
enough admits have accumulated. The orchestrator never touches cluster
bookkeeping directly.

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
| `mutation_model` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | Small / fast model. Drives mutate, crossover, repair, init variants, paradigm variants, meta-advisor, analyzer. |
| `paradigm_model` | `openrouter/openai/gpt-5` | Frontier model. Drives the diverse-seed phase and the per-mode paradigm shift. |
| `embedding_model` | `openrouter/openai/text-embedding-3-small` | Used by `DescriptionEmbedder` for the embedding half of the behavior signature. |

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
| `pe_cron_interval` | `50` | Fire paradigm shift every N completed evaluations. |
| `paradigm_min_archive_size` | `5` | Skip paradigm shift if `num_occupied_cells <` this. |
| `paradigm_temperature` | `0.8` | Frontier temperature inside paradigm shift. |
| `paradigm_variant_temperature` | `0.85` | Mutation temperature for paradigm-fanout variants. |
| `n_paradigm_variants` | `4` | Variants spun off the new paradigm seed. |

### 3.4 Operator mix & repair

| Knob | Default | Notes |
| --- | --- | --- |
| `p_crossover` | `0.35` | Probability of crossover (else mutate) per main-loop step. |
| `enable_repair` | `True` | One-shot repair branch for error candidates. |

### 3.5 Three-mode paradigm shift

| Knob | Default | Notes |
| --- | --- | --- |
| `paradigm_synthesis_max_stagnation` | `0.4` | At or below → `synthesis` mode. |
| `paradigm_shift_max_stagnation` | `0.7` | Between synthesis cap and this → `shift` mode; above → `surgical`. |
| `paradigm_synthesis_n_anchors` | `3` | Full-code anchors in synthesis mode. |
| `paradigm_shift_n_anchors` | `2` | Full-code anchors in shift mode. |
| `paradigm_surgical_n_inspirations` | `5` | Description-only inspirations in surgical mode (alongside the single champion anchor). |
| `paradigm_n_inspirations` | `5` | Description-only inspirations in synthesis / shift modes. |

Legacy knob: `paradigm_n_anchors` (default `4`) is still on
`BladeConfig` for backwards compatibility, but the active code path
ignores it — each mode now reads its own `paradigm_*_n_anchors`
field. Future cleanup may remove it.

### 3.6 Targeted-mutate analyzer (Đề xuất 1)

| Knob | Default | Notes |
| --- | --- | --- |
| `enable_targeted_mutate` | `True` | Master toggle for the analyzer + TARGETED_MUTATE_PROMPT path. |
| `analyzer_interval` | `30` | Refresh the analysis cache every N completed evaluations. |
| `analyzer_top_k` | `3` | Number of top-ranked programs analysed per refresh. |
| `analyzer_temperature` | `0.3` | Mutation-model temperature for the analysis call. |
| `analyzer_max_tokens` | `500` | Token cap for one analysis. |
| `p_targeted_mutate` | `0.5` | When the chosen parent has a cached analysis, probability of using TARGETED_MUTATE_PROMPT (else fall back to PromptSampler). |

### 3.7 Meta-advisor

| Knob | Default | Notes |
| --- | --- | --- |
| `enable_meta_advice` | `True` | Toggle the lessons-learnt advisor. |
| `meta_advice_interval` | `50` | Regenerate the lessons-learnt note every N evaluations. |
| `meta_advice_inject_p` | `0.7` | Probability of injecting the current note into the next mutate / crossover prompt. |
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
`n_variants_per_seed`, and ablation directly. Everything else goes
through the JSON `advanced_options` input, e.g.

```json
{
  "n_cells": 64,
  "recluster_every": 30,
  "embedding_dim": 8,
  "paradigm_n_inspirations": 5,
  "n_paradigm_variants": 4,
  "meta_advice_interval": 50,
  "repair_disabled": false,
  "meta_advice_disabled": false,

  "targeted_mutate_disabled": false,
  "analyzer_interval": 30,
  "analyzer_top_k": 3,
  "p_targeted_mutate": 0.5,

  "paradigm_synthesis_max_stagnation": 0.4,
  "paradigm_shift_max_stagnation": 0.7,
  "paradigm_synthesis_n_anchors": 3,
  "paradigm_shift_n_anchors": 2,
  "paradigm_surgical_n_inspirations": 5
}
```

The CLI exposes the same knobs as `--n-cells`, `--analyzer-interval`,
`--p-targeted-mutate`, `--paradigm-synthesis-max-stagnation`, etc.
Run `uv run python scripts/run_blade.py --help` for the full list.

---

## 4. Ablation protocol (the *only* paper-facing knobs)

Three toggles, unchanged. Each ablation isolates one component of the
hybrid behavior signature:

| Ablation | Toggle (CLI) | Toggle (workflow) | What is removed |
| --- | --- | --- | --- |
| **A1 — emb-only** | `--emb-only` | `ablation: emb_only` | The 14-d AST half. |
| **A2 — ast-only** | `--ast-only` | `ablation: ast_only` | The PCA description-embedding half. |
| **A3 — static-cells** | `--static-cells` | `ablation: static_cells` | Adaptive re-clustering. KMeans fit once at `n_cells`, then frozen. |

Default config (the "full" variant) = A1 + A2 + A3 all ON.

---

## 5. Output artifacts

Per run, `output_dir/`:

- `best.py` — top-scoring program.
- `summary.json` — run metadata + budget + ablation block.
- `snapshot.json` — full archive dump:
  - `monitor` (eval_count, best_score, plateau_steps, admit_gap,
    global_stagnation, local_stagnation, stagnation_level, accept_rate)
  - `paradigm_trials` (idx, accepted, score, delta, description)
  - `cells` — one entry per occupied cell, sorted by score desc
  - `ablation` — block reflecting the three toggles + low-level
    parameters
  - `meta_advice` — current lessons-learnt block + trigger count
  - `analyzer` — `{enabled, interval, top_k, trigger_count,
    targeted_mutate_count, p_targeted_mutate, cache_size}`
  - `paradigm_modes` — `{synthesis_max_stagnation,
    shift_max_stagnation, synthesis_n_anchors, shift_n_anchors,
    surgical_n_inspirations}`

---

## 6. Logging conventions

- `[Eval #N] {model} {status} | source: ... | score: ... | best: ... | $cost`
- Sources include the prompt-variant label, e.g.
  `mutate_focused_fix`, `crossover_component_swap`, `mutate_targeted`,
  `paradigm_synthesis`, `paradigm_surgical`.
- `[Status] Cost: ... | Evals: ... | Archive: N (cells K) | Best: ... | Elapsed: ...s`
- `[BLADE PE] mode=surgical anchors=1 stagnation=0.82 best=2.2871`
- `[BLADE PE] fanout: K variants from paradigm seed (score=..., accepted=...)`
- `[BLADE analyzer] trigger #N at eval=M (refreshing top-K)`
- `[Archive] recluster: n=... → K/n_cells occupied cells (admits since last recluster=...)`

---

## 7. Quick start

```bash
# Default run (all components on)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5

# Bigger archive (n_cells controls the cell-count invariant)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --n-cells 64

# Disable targeted mutate (Đề xuất 1 off)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --no-targeted-mutate

# Push surgical mode earlier (force exploit-heavy late phase)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --paradigm-shift-max-stagnation 0.5

# Ablation: AST only
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --ast-only --output-dir runs/ablation-ast-only
```

On GitHub Actions: dispatch `.github/workflows/blade.yml` and pick the
`ablation` dropdown.

---

## 8. Verification

Two pre-merge gates pin the contracts that previous runs broke:

```bash
# Offline unit tests (4 archive invariant + 8 prompts + 4 paradigm modes
# + 3 orchestrator e2e on fake LLMs = 19 tests, ~30s).
cd levi && uv run python -m pytest tests/blade/ --tb=short

# Live audit: one real call per prompt variant against the production
# mutation model, checks that every response contains
# ## Analysis → ## Description → ## Code in that order, with no bullet
# leakage into the embedded description. Costs ~$0.01-0.02 per run.
uv run python scripts/smoke_blade_prompts.py
```

The unit tests are deterministic and cheap; the smoke script is the
guard against silent format drift when a new model release ships.
