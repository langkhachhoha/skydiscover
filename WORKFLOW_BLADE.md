# WORKFLOW_BLADE — BLADE Lite

> BLADE = **B**ehavior-**L**atent **A**daptive **D**iscovery **E**ngine.
>
> A minimal evolutionary code-search engine built on top of a two-model
> pipeline (frontier + mutation) with one architectural contribution:
> **adaptive MAP-Elites whose cells are built from a hybrid AST +
> description-embedding behavior signature and re-clustered as the search
> progresses**. Three components total, three paper-facing ablation toggles.

---

## 1. Architecture

Three components, in dependency order.

```text
┌──────────────────────────────────────────────────────────────────────┐
│  BLADE Lite                                                           │
│                                                                        │
│  (1) ClusterArchive  ◄────  hybrid AST + embedding behavior          │
│      • cells = KMeans clusters over behavior vectors                  │
│      • re-cluster every K admits, warm-started from prev centroids   │
│      • admit iff score > cell incumbent                              │
│                                                                        │
│  (2) RankSampler                                                       │
│      • Zipfian over score rank: P(rank=r) ∝ (r+1)^(-β)               │
│      • β interpolates linearly with stagnation                       │
│      • crossover: second parent from a different cell when possible  │
│                                                                        │
│  (3) Monitor                                                           │
│      • plateau_steps (evals since last new best)                     │
│      • accept-rate sliding window                                    │
│      • stagnation_level() ∈ [0, 1] feeds the sampler β               │
│                                                                        │
│  + Orchestrator: standard 2-phase bootstrap + main loop +             │
│      periodic paradigm shift. Frontier + mutation model split.        │
└──────────────────────────────────────────────────────────────────────┘
```

Source files:

- [levi/levi/simple/archive.py](levi/levi/simple/archive.py) — ClusterArchive
- [levi/levi/simple/ast_features.py](levi/levi/simple/ast_features.py) — 14-d AST counts
- [levi/levi/simple/rank_sampler.py](levi/levi/simple/rank_sampler.py) — RankSampler
- [levi/levi/simple/monitor.py](levi/levi/simple/monitor.py) — Monitor
- [levi/levi/blade/orchestrator.py](levi/levi/blade/orchestrator.py) — orchestrator
- [levi/levi/blade/prompts.py](levi/levi/blade/prompts.py) — prompt builders
- [scripts/run_blade.py](scripts/run_blade.py) — CLI

### 1.1 ClusterArchive — the only sophisticated component

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

**Cells.** Initially each admit gets its own cell id. Once the
population crosses `min_admits_before_cluster` (default 16), KMeans is
fit with `n_cells=32` clusters. Subsequent admits assign cell id by
nearest-centroid. Every `recluster_every` admits (default 30), KMeans is
re-fit — warm-started from the previous centroids so cell identity is
continuous.

**Admission rule.** A candidate is admitted iff its score is **strictly
greater** than the current incumbent of its cell. Otherwise it is
dropped (`reason="dropped_worse"`). There is no other admission rule:
no grace period, no Hall-of-Fame, no quota, no near-duplicate dedup at
the embedding level (because the cell structure already provides it —
two truly-similar programs land in the same cell and the lower score is
dropped).

**The re-clustering is the diversity mechanism.** When the population
collapses around one paradigm, the next re-cluster finds many tight
clusters *within* that paradigm and the cells shrink to subspaces of
that paradigm — so workers keep refining it. When the frontier injects
a brand-new paradigm via a paradigm shift, the next re-cluster opens a
fresh cell for it (warm-started centroids reposition).

### 1.2 RankSampler — score-rank Zipfian

Parameter-free relative to score scale:

```text
P(rank=r) ∝ (r+1)^(-β(stagnation))
β(stagnation) = β_min + (1 - stagnation) · (β_max - β_min)
              = 0.3   when stagnation = 1.0  (≈ uniform)
              = 2.0   when stagnation = 0.0  (top-3 dominate)
```

Three call patterns:

- `select_parent(programs, stagnation, rng)` — one parent for mutate.
- `select_two_parents(programs, stagnation, rng)` — two parents for
  crossover; second parent comes from a *different cell* when one is
  available so the mutation model sees two paradigms to hybridise.
- `select_inspirations(programs, exclude, stagnation, rng, k)` — k draws
  without replacement; description-only.

Rank alone encodes everything the sampler needs — there is no UCB term,
no recency bonus, no diversity penalty.

### 1.3 Monitor — global + local stagnation

`stagnation_level()` is the only signal the sampler reads, and it
combines two independent timers:

```text
global_stagnation = plateau_steps / plateau_max          # since last NEW BEST
local_stagnation  = admit_gap     / admit_gap_max        # since last ADMIT
stagnation_level  = max(global_stagnation, local_stagnation)
```

- **Global** ticks until the **best score** improves. NEW BEST is a
  rare event (a handful per multi-hour run), so this timer fires only
  when the search has truly halted.
- **Local** ticks until any cell incumbent gets **replaced**. Admits
  happen much more often than NEW BEST — they capture the fact that a
  cell took one small step forward, which is the unit of progress the
  rank sampler should react to.

Using ``max`` of the two means **either signal saturating is enough**
to drive β toward exploration. Tuning them independently
(``admit_gap_max ≈ plateau_max / 5``) is intentional: admits are ~5×
more frequent than new-best events, so we want the admit-side timer to
fire on a tighter horizon.

`accept_rate` is logged and fed to the meta-advisor prompt but does not
switch any control pathway.

### 1.4 Frontier ↔ mutation model collaboration

The two models keep distinct roles:

- **Frontier (GPT-5)** writes:
  - Phase-1 diverse seeds (sequential, each one shown previously
    accepted seeds, pushed for paradigm diversity).
  - One paradigm-shift seed every `pe_cron_interval` evaluations.
- **Mutation (Qwen-30B)** writes:
  - Phase-2 variants of the diverse seeds (parallel).
  - Main-loop mutate / crossover / repair (parallel).
  - Paradigm-shift fanout (parallel variants of the new frontier seed).
  - Meta-advice (lessons-learnt summary).

The **paradigm-shift prompt** is the only place where the frontier sees
the full archive structure. It receives:

- ``paradigm_n_anchors`` (default 4) **cell representatives** — one
  top-score program per occupied cell, ranked by score, capped at the
  knob. Full code + description + score.
- ``paradigm_n_inspirations`` (default 5) **description-only**
  representatives from other cells.
- A **Strategy Log** of the last 6 paradigm trials (description, score,
  delta-vs-best).

The prompt asks the model to **explicitly choose** between two moves:
**(A) synthesise** a hybrid from the anchors or **(B) propose a
fundamentally different paradigm**. The first line of the response must
start with `MOVE: A` or `MOVE: B`. The model is *not* told which to
pick; we surface `n_cells` and `stagnation` as numeric context and let
the frontier calibrate.

---

## 2. End-to-end flow

```text
1. Phase 1 (sequential, frontier model)
   for i in 1..n_diverse_seeds:
     prompt = build_diverse_seed_prompt(existing_seeds=…)
     code  = frontier(prompt)
     score = evaluate(code)
     archive.add(Program(code, description, score, embedding, source="init"))

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
       prompt    ← build_mutate_prompt | build_crossover_prompt
       code      ← mutation(prompt)
       archive.add(Program(...))
     if error_buffer: _repair_one()    # one-shot

   parallel:
     • _pe_monitor wakes every 2s, fires _paradigm_shift() every pe_cron_interval evals
     • _meta_advice_monitor refreshes lessons-learnt every meta_advice_interval evals

4. Paradigm shift
   anchors = top-score program per occupied cell (capped at paradigm_n_anchors)
   prompt  = build_paradigm_prompt(anchors, inspirations, recent_trials, stagnation)
   code    = frontier(prompt)       # MOVE: A (synthesise) or MOVE: B (shift)
   archive.add(Program(..., source="paradigm"))
   if accepted:
     fanout (parallel): n_paradigm_variants × mutation(build_paradigm_variant_prompt)
```

The archive's re-clustering happens *inside* `archive.add(...)`
whenever enough admits have accumulated. The orchestrator never touches
cluster bookkeeping directly.

---

## 3. Configuration

The dispatch workflow ([.github/workflows/blade.yml](.github/workflows/blade.yml))
exposes:

| Input | Default | Meaning |
| --- | --- | --- |
| `benchmark` | `circle_packing` | LEVI example dir name |
| `evaluations` / `dollars` / `seconds` | — / — / 10800 | Budget caps |
| `mutation_model` | `qwen3-30b-a3b-instruct-2507` | Worker model |
| `paradigm_model` | `gpt-5` | Frontier model |
| `workers` | 4 | Concurrent mutation workers |
| `pe_interval` | 50 | Paradigm-shift cadence (evals) |
| `n_diverse_seeds` | 5 | Phase-1 seeds |
| `n_variants_per_seed` | 20 | Phase-2 variants per seed |
| `ablation` | `full` | One of `full` / `ast_only` / `emb_only` / `static_cells` |
| `advanced_options` | `""` | JSON with low-level overrides |

Low-level knobs reachable via `advanced_options` JSON:

```json
{
  "n_cells": 32,
  "recluster_every": 30,
  "embedding_dim": 8,
  "paradigm_n_anchors": 4,
  "paradigm_n_inspirations": 5,
  "n_paradigm_variants": 4,
  "meta_advice_interval": 50,
  "repair_disabled": false,
  "meta_advice_disabled": false
}
```

---

## 4. Ablation protocol (the *only* paper-facing knobs)

Three toggles. Each ablation isolates one component:

| Ablation | Toggle (CLI) | Toggle (workflow) | What is removed |
| --- | --- | --- | --- |
| **A1 — emb-only** | `--emb-only` | `ablation: emb_only` | The 14-d AST half. Behavior = PCA-reduced description embedding only. Tests "is AST necessary on top of the LLM's semantic signal?" |
| **A2 — ast-only** | `--ast-only` | `ablation: ast_only` | The PCA description-embedding half. Behavior = 14-d AST counts only. Tests "is the embedding necessary on top of AST counts?" |
| **A3 — static-cells** | `--static-cells` | `ablation: static_cells` | Adaptive re-clustering. KMeans fit once at `min_admits_before_cluster`, then frozen. Tests "is the periodic re-cluster necessary, or is one fit enough?" |

Default config (the "full" variant) = A1 + A2 + A3 all ON.

**Recommended experiment plan** for the paper:

| Run | Config | Why |
| --- | --- | --- |
| LEVI baseline | unchanged | External baseline |
| BLADE-full | default | The proposed system |
| BLADE-ast-only | `--ast-only` | Ablate embedding |
| BLADE-emb-only | `--emb-only` | Ablate AST |
| BLADE-static-cells | `--static-cells` | Ablate adaptive re-clustering |

Two-three seeds per config × one or two benchmarks. Report best_score,
total_evaluations, total_cost, num_occupied_cells, plus a plot of
best_score vs evaluations.

The snapshot.json carries an ``ablation`` block reflecting exactly
which toggles were active, so post-hoc analysis cannot mix runs.

---

## 5. Output artifacts

Per run, `output_dir/`:

- `best.py` — top-scoring program (`code` of the highest-score elite).
- `summary.json` — run metadata + budget + ablation block.
- `snapshot.json` — full archive dump:
  - `monitor` (eval_count, best_score, plateau_steps, admit_gap, global_stagnation, local_stagnation, stagnation_level, accept_rate)
  - `paradigm_trials` (idx, accepted, score, delta, description)
  - `cells` — one entry per occupied cell, sorted by score desc
  - `ablation` — block reflecting the three toggles + low-level parameters
  - `meta_advice` — current lessons-learnt block + trigger count

---

## 6. Logging conventions

- `[Eval #N] {model} {status} | source: ... | score: ... | best: ... | $cost`
- `[Status] Cost: ... | Evals: ... | Archive: N (cells K) | Best: ... | Elapsed: ...s`
- `[BLADE PE] trigger #N at eval=... | stagnation=... (global=... local=...) | best=... | cells=...`
- `[BLADE PE] fanout: K variants from paradigm seed (score=..., accepted=...)`
- `[Archive] recluster: n=... → K cells (admits since last recluster=...)`

---

## 7. Quick start

```bash
# Default run (all three components on)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5

# Ablation: AST only
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --ast-only --output-dir runs/ablation-ast-only

# Ablation: embedding only
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --emb-only --output-dir runs/ablation-emb-only

# Ablation: static cells (KMeans fit once)
uv run python scripts/run_blade.py \
  --example-dir levi/examples/circle_packing_rect \
  --seconds 10800 --dollars 5 \
  --static-cells --output-dir runs/ablation-static-cells
```

On GitHub Actions: dispatch `.github/workflows/blade.yml` and pick the
``ablation`` dropdown.
