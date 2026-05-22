# BLADE — Behavior-Latent Adaptive Discovery Engine

> An LLM-driven evolutionary code search. Two models cooperate: a small
> mutation model produces many cheap variants, a large frontier model
> occasionally rewrites the strategy from scratch. Diversity is enforced
> in *idea space* via sentence embeddings of natural-language paradigm
> descriptions, not in code space. This document walks through BLADE
> from first principles — assume you know Python and roughly what an
> evolutionary algorithm is, nothing more.

---

## Table of contents

1. [What problem are we solving?](#1-what-problem-are-we-solving)
2. [Why "evolution with LLMs" needs a structural change](#2-why-evolution-with-llms-needs-a-structural-change)
3. [The three quantities BLADE actually tracks](#3-the-three-quantities-blade-actually-tracks)
4. [Pipeline at a glance](#4-pipeline-at-a-glance)
5. [Component by component](#5-component-by-component)
6. [How a single iteration unfolds](#6-how-a-single-iteration-unfolds)
7. [Running BLADE](#7-running-blade)
8. [Outputs and snapshots](#8-outputs-and-snapshots)
9. [Tuning knobs](#9-tuning-knobs)
10. [Relation to LEVI](#10-relation-to-levi)

---

## 1. What problem are we solving?

We want to find a Python program `P*` that maximises a black-box score:

```text
P* = argmax_{P ∈ valid_programs}  score(P)
```

The "score" is provided by the user as a scoring function (`score_fn`)
that runs `P` on some hidden test set and returns a float. Constraints:

* The score function is **expensive** (compiling and running candidate
  programs takes seconds), so the total number of evaluations is small —
  typically tens to a few hundred per run.
* LLM calls are **expensive** too: a single frontier call to GPT-5 can
  cost as much as a hundred mutation calls to Qwen.
* The search space is enormous (any syntactically valid Python). There
  is no gradient, no obvious neighbourhood structure.

This is the regime where evolutionary methods using LLMs as
"mutation operators" — FunSearch, AlphaEvolve, LEVI — have shown
results, but the existing methods are fragile and over-parametrised.

BLADE's bet: keep what works (frontier + mutation cooperation,
3-phase paradigm prompts, self-repair, async parallelism) and replace
three subsystems that ablation studies in LEVI repeatedly flagged as
too heavy (Voronoi-tessellated MAP-Elites, a multi-term stagnation
formula, a 4-D Thompson bandit). The replacements are simple and rely
on one ingredient LEVI did not use: **embeddings of natural-language
paradigm descriptions**.

---

## 2. Why "evolution with LLMs" needs a structural change

The standard recipe goes: keep an archive of past programs, repeatedly
pick a parent, ask the LLM to mutate it, evaluate the child, decide
whether it joins the archive. Two pathologies dominate:

**Pathology 1 — paradigm collapse.** Without an explicit diversity
mechanism, the archive fills up with minor edits of one early winner.
The LLM, asked to mutate that winner, mostly produces near-duplicates.
The search becomes hill-climbing on a single local optimum.

**Pathology 2 — syntactic diversity is a lie.** Naïve fixes (e.g.
keep the top-K programs sorted by AST diversity) overweight surface
features: two textually different DP solutions look "diverse" to the
archive but are the same idea. Meanwhile a genuinely novel approach
written in similar syntax gets rejected as too close.

BLADE addresses both with one move: **every LLM call emits a short
description of its paradigm alongside the code, and the archive
clusters programs by description embedding**. Two DP variants with
slightly different code now correctly cluster together (their
descriptions are near-identical in embedding space); a fresh BFS
written in similar Python lands in its own cluster.

The frontier model, when called, is given only the descriptions of
representatives — never their source — so it cannot fall into the
"copy and tweak" attractor that plagues LEVI's frontier calls.

---

## 3. The three quantities BLADE actually tracks

Everything in BLADE flows from three quantities. Knowing them is enough
to read the rest of this document.

### 3.1 Semantic distance between programs

For two programs `i` and `j` with descriptions `d_i` and `d_j`, embed
each via `text-embedding-3-small`:

```text
e_i = embed(d_i)                                (1536-dim, L2-normalised)
e_j = embed(d_j)
sim(i, j) = e_i · e_j                           (cosine, in [-1, 1])
```

Two descriptions of the same paradigm sit around `sim ≈ 0.85`; cross-
paradigm pairs sit around `sim ≈ 0.55`. Thresholds on this single
scalar drive the whole archive.

### 3.2 Stagnation level

A scalar in [0, 1] that says "how stuck is the search right now?":

```text
stagnation(t) = min(1, plateau_steps(t) / plateau_max)
```

`plateau_steps(t)` is the number of evaluations since the global best
score last improved. `plateau_max` defaults to 100. The number is
*not* used to trigger anything by itself — only to pick which of three
frontier prompts to use (early / mid / late) when the cron fires.

### 3.3 Selection priority

When the search picks a parent or an inspiration program, every
candidate `p` in the pool gets a priority score against the set `S` of
already-picked programs:

```text
priority(p; S) = score_norm(p)                                # exploit
              + α · sqrt( log(1+N) / (1+uses_count(p)) )      # novelty (UCB)
              + β · exp( −(N − created_at(p)) / τ )           # recency
              − γ · max_{q ∈ S} sim(p, q)                     # diversity penalty
```

* `score_norm(p)` is the score linearly rescaled to [0, 1] over the
  current pool.
* `N` is the running evaluation count; `uses_count(p)` is how many
  times `p` has already been chosen.
* `created_at(p)` is the evaluation at which `p` entered the pool, so
  the recency term `exp(−age/τ)` decays toward zero as `p` ages.
* `α, β, γ` swap from healthy weights `(0.5, 0.3, 0.4)` to stuck
  weights `(0.8, 0.5, 0.7)` when the monitor flags a stall.

This single formula plays the role of an explore/exploit policy. The
novelty term keeps the search from re-picking the same hot parent
forever; the recency term protects freshly added programs (especially
frontier paradigm shifts) from being buried by long-tenured top scorers;
the diversity penalty ensures a batch of picks (a parent + k
inspirations, or two crossover parents) does not all come from the same
family.

---

## 4. Pipeline at a glance

```text
                ┌─────────────────────────────────────────────────────┐
                │  Frontier model  (heavy, ~5-15% of LLM budget)      │
                │                                                     │
                │  One call every pe_cron_interval evals (default 50) │
                │  Stage ∈ {early, mid, late} picked by stagnation.   │
                │  Sees 3 representatives — description + score only. │
                │  Plus a strategy log of the last 5 paradigm trials. │
                └───────────────────┬─────────────────────────────────┘
                                    │ paradigm_code (description + code)
                                    ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  Mutation workers  (light, ~85-95% of LLM budget) — N parallel   │
   │                                                                  │
   │  Operator chosen at random each step:                            │
   │    mutate     (1 parent + k=3 inspiration descriptions)          │
   │    crossover  (2 parents from far-apart families)                │
   │  Temperature & operator mix flip when monitor flags is_stuck.    │
   │                                                                  │
   │  On runtime crash → push to error buffer → one-shot self-repair  │
   │  On code-only output → fallback summariser writes a description  │
   └───────────────────┬──────────────────────────────────────────────┘
                       │ candidates (code + description, ready to score)
                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  Sandboxed evaluator pool (ResilientProcessPool)                 │
   └───────────────────┬──────────────────────────────────────────────┘
                       │ (score, error?)
                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  Pool — top-K = 100, niching on description embeddings           │
   │   Three nested admission rules:                                  │
   │     1. semantic dedup       (cosine ≥ 0.92 → replace best)       │
   │     2. family cap            (≤ 8 per family @ cosine ≥ 0.72)    │
   │     3. global top-K          (drop lowest-score when over)       │
   └───────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  Monitor — three sliding-window signals                          │
   │   plateau_steps     — evals since last new best                  │
   │   accept_window     — fraction admitted in last 50 evals         │
   │   diversity_window  — mean pairwise cosine over last 20 admits   │
   │                                                                  │
   │  Routes 3-phase prompt + flips temperature / operator mix.       │
   │  Does NOT trigger frontier calls — the cron does.                │
   └──────────────────────────────────────────────────────────────────┘
```

---

## 5. Component by component

### 5.1 Pool

**File:** [`levi/levi/simple/pool.py`](../levi/levi/simple/pool.py).

The pool is a Python list with three admission rules layered on top.

**Semantic dedup** (cosine threshold 0.92). When a new candidate
arrives, we find its nearest neighbour by cosine. If the cosine is
above 0.92, this is a near-duplicate description — we replace the
incumbent only if the new program's score is higher, otherwise drop.

**Family cap.** We maintain a union-find over the pool with
single-linkage at cosine 0.72. Each maximal cluster is a "family"
(typically one algorithmic paradigm). The pool refuses to hold more
than `max_per_family = 8` programs per family — when an 9th lands, the
weakest member of that family is evicted, not the global weakest.
This prevents a single hot paradigm from squeezing out fresh ideas.

**Top-K cap.** If the pool grows beyond K = 100, drop the lowest-score
program.

The pool also exposes `representatives(stage, n)` for the frontier:

* **early** — greedy MMR with low score weight (λ = 0.2). Picks 3
  mutually distant programs so the frontier sees the broadest possible
  spread of paradigms already in the archive.
* **mid** — one top-score anchor + 2 MMR-diverse complements
  (λ = 0.5). The frontier sees the best so far plus two angles to
  synthesise from.
* **late** — top 3 by score, unfiltered. The frontier is being asked
  to surgically refine the strongest incumbent, so we want the
  unfiltered top of the pool, not a diverse spread.

### 5.2 Monitor

**File:** [`levi/levi/simple/monitor.py`](../levi/levi/simple/monitor.py).

A tiny dataclass with three sliding-window quantities updated on every
eval:

* **plateau_steps** — counter, reset whenever an accepted program
  improves the global best.
* **accept_window** — deque of length 50 of booleans (was the eval
  admitted to the pool?).
* **diversity_window** — deque of length 20 of floats (mean pairwise
  cosine between the most recently admitted program and the previous
  members of this window).

From these three the monitor produces:

```text
stagnation_level() = min(1, plateau_steps / plateau_max)         # ∈ [0, 1]
is_stuck()         = plateau_steps > 80 OR accept_rate < 0.08
is_collapsing()    = mean(diversity_window) > 0.78               # too similar
```

Crucially, the monitor **does not** trigger frontier calls. Those fire
on a fixed cron (`pe_cron_interval`, default 50 evals). The monitor
only routes the frontier's prompt to early / mid / late and toggles a
"stuck" flag that the selector and operator dispatcher react to.

This separation matters: LEVI's event-driven PE made cost forecasting
hard because pathological runs could trigger many expensive frontier
calls. BLADE's frontier budget is bounded by `total_evals /
pe_cron_interval` — predictable.

### 5.3 Selector

**File:** [`levi/levi/simple/selector.py`](../levi/levi/simple/selector.py).

Implements the priority formula from §3.3. Three public methods:

* `select_parent(programs)` — returns the single highest-priority
  candidate.
* `select_two_parents(programs)` — picks `p1` normally, then picks `p2`
  with a bias toward `sim(p1, p2) < 0.65` (cross-family crossover).
* `select_inspirations(programs, exclude, k=3)` — greedy batched: pick
  one, add it to the set, repeat. The diversity penalty automatically
  keeps the inspirations from clumping.

### 5.4 Embedder

**File:** [`levi/levi/simple/embedder.py`](../levi/levi/simple/embedder.py).

Thin wrapper around litellm's embedding endpoint. Defaults to
`openrouter/openai/text-embedding-3-small` (1536 dimensions, ~$0.02 /
1M tokens). Two niceties: results are L2-normalised at intake so
cosine is just a dot product; results are cached by SHA1 of the input
text so identical descriptions cost nothing on repeat.

### 5.5 Parser

**File:** [`levi/levi/simple/parser.py`](../levi/levi/simple/parser.py).

Every LLM call asks for output in this shape: a `## Description`
heading followed by 2-4 sentences describing the paradigm, key data
structures, and the distinguishing trick; then a `## Code` heading
followed by a fenced Python block.

Parser tries five recognised shapes in order, taking the first that
yields code:

1. Both `## Description` and `## Code` headers present.
2. `## Description` present, no `## Code` header, code in a fence right
   after.
3. **Header-less prose-before-fence**: the LLM forgot the
   `## Description` header but wrote a prose paragraph before the
   fenced code block. Treat the paragraph as the description. This
   branch alone takes the format-compliance rate from ~50 % (header
   strict) to ~100 % (header tolerant) on real LLM output.
4. Just a code fence — description empty. Caller invokes the fallback
   summariser (a cheap mutation-model call) to write one.
5. Raw Python (text starts with `def` / `class` / `import` / `from`) —
   code only, no description; fallback summariser kicks in.

### 5.6 Operator prompts and orchestrator

**Files:** [`levi/levi/blade/prompts.py`](../levi/levi/blade/prompts.py)
and [`levi/levi/blade/orchestrator.py`](../levi/levi/blade/orchestrator.py).

Four prompt builders cover every LLM call BLADE makes:

* `build_mutate_prompt` — one parent (full code) + k inspirations
  (description + score only).
* `build_crossover_prompt` — two parents (full code) + a few
  inspirations.
* `build_repair_prompt` — broken code + stack trace; asks the mutation
  model to patch.
* `build_paradigm_prompt` — wraps LEVI's three-phase frontier
  templates from `levi.equilibrium.prompts`, but plugs in
  description-only representatives and a short log of the last 5
  paradigm attempts.

The orchestrator is one async event loop that runs `n_workers`
mutation tasks in parallel, fires one paradigm-shift task in the
background every `pe_cron_interval` evals, and pulls from an error
buffer for at-most-one-in-flight self-repair. Frontier and repair are
*background* tasks deliberately — they cannot stall the mutation
workers even when GPT-5 takes 60 seconds to think.

### 5.7 The frontier — reasoning-model gotcha

Reasoning-heavy frontier models (GPT-5, o1) spend most of any fixed
token budget on internal thinking. A `max_tokens = 1200` cap causes
them to return an empty content block and `finish_reason = "length"`.
BLADE handles this by **not setting `max_tokens` at all** for the
frontier call — the LM client drops the field, the provider applies
its own large default, and visible output comes through. This is a
single setting (`paradigm_max_tokens = None` in `BladeConfig`) but
without it the whole frontier branch returns nothing.

---

## 6. How a single iteration unfolds

To make the moving parts concrete, here's a play-by-play of one
mutation step:

1. **Mutation worker wakes up.** It checks the budget — if the cap is
   exhausted, signal stop and return.
2. **Operator dice roll.** Crossover with probability 0.3 (healthy)
   or 0.7 (stuck), else mutate.
3. **Selector picks parents.** For mutate: `select_parent` returns the
   highest-priority program by §3.3. For crossover: `select_two_parents`
   biases the second pick toward a different family.
4. **Selector picks inspirations.** `select_inspirations` returns 3
   programs the LLM has not seen in this batch; only their descriptions
   and scores will be exposed.
5. **Prompt is built** by `build_mutate_prompt` (or
   `build_crossover_prompt`) and sent to the mutation model at
   temperature 0.8 (or 1.1 if stuck).
6. **Output is parsed** — description + code extracted (or fallback
   summariser fires).
7. **Code is evaluated** in the subprocess pool against `score_fn`. If
   it crashes, the broken code plus error tail is pushed to the error
   buffer (a repair worker will pick it up later); the monitor records
   a rejected eval and the worker is done.
8. **Description is embedded** asynchronously (the embedder runs on a
   thread so the event loop is not blocked).
9. **Pool admission.** Three layered checks: dedup, family cap,
   top-K. The pool returns `(accepted, reason)` for diagnostics.
10. **Monitor updates** plateau_steps / accept_window / diversity_window
    based on whether the program was admitted.

Meanwhile, every `pe_cron_interval` evals, the orchestrator spawns a
**paradigm task** in parallel that picks 3 representatives from the
pool by current stage, calls the frontier model, parses the output,
evaluates it, admits or records the trial. The frontier task does not
block mutation workers — they keep producing through the whole frontier
call.

---

## 7. Running BLADE

Three entry points exist for three audiences.

### 7.1 From Python

The most direct call mirrors `levi.evolve_code` so existing problem
modules drop straight in:

```python
import levi

result = levi.evolve_code_blade(
    problem_description="Maximise solve(0)+solve(1)+solve(2).",
    function_signature="def solve(x):",
    score_fn=my_scoring_fn,
    seed_program=optional_starter_code,         # may be None
    mutation_model="openrouter/qwen/qwen3-30b-a3b-instruct-2507",
    paradigm_model="openrouter/openai/gpt-5",
    embedding_model="openrouter/openai/text-embedding-3-small",
    budget_evals=100,
    n_workers=4,
    pe_cron_interval=50,
    output_dir="runs/my-experiment",
)

print(result.best_score, result.total_cost, result.pool_size)
for trial in result.paradigm_trials:
    print(trial.stage, trial.accepted, trial.score, trial.description[:80])
```

Default models match `_levi.yml` so BLADE and LEVI are
budget-comparable out of the box.

### 7.2 From the CLI

[`scripts/run_blade.py`](../scripts/run_blade.py) accepts any example
directory that follows the LEVI problem-module shape: a `problem.py`
exporting `PROBLEM_DESCRIPTION` (string), `FUNCTION_SIGNATURE`
(string), `score_fn` (callable), and optionally `SEED_PROGRAM` and
`INPUTS`.

```bash
# Demo on the coin-change benchmark in this repo
python scripts/run_blade.py \
    --example-dir levi/examples/blade_demo \
    --evals 50 \
    --workers 4 \
    --pe-interval 50

# Run on any other LEVI example unchanged
python scripts/run_blade.py \
    --example-dir levi/examples/circle_packing \
    --evals 200 \
    --dollars 5.0 \
    --pe-interval 25
```

The driver writes `summary.json`, `snapshot.json`, and
`best_program.py` to the chosen `--output-dir`.

### 7.3 From GitHub Actions

[`.github/workflows/blade.yml`](../.github/workflows/blade.yml) is a
reusable workflow that mirrors `_levi.yml`. Trigger it manually
through the GitHub UI (`workflow_dispatch`) or call it from a
problem-specific workflow with `uses:`:

```yaml
jobs:
  bench:
    uses: ./.github/workflows/blade.yml
    with:
      example_dir: levi/examples/blade_demo
      evaluations: "100"
      mutation_model: openrouter/qwen/qwen3-30b-a3b-instruct-2507
      paradigm_model: openrouter/openai/gpt-5
      # Everything else is a JSON blob, matching `_levi.yml`'s pattern:
      advanced_options: '{"pe_interval":25,"family_threshold":0.70}'
    secrets: inherit
```

The workflow exposes 8 top-level inputs (well under GitHub's 21-input
limit). The remaining knobs — problem module name, target score,
embedding model, eval-process count, eval timeout, pe interval, pool K,
niche/family thresholds, max-per-family, repair toggle — live inside
the `advanced_options` JSON. Each run uploads its
`outputs/github-actions/blade_<run_id>` directory as the artifact
`blade-<run_id>` for 14 days.

---

## 8. Outputs and snapshots

Every run writes three files into the output directory:

* **`snapshot.json`** — final state. Top-level keys:
  `method` (= `"blade"`), `best_score`, `total_evaluations`,
  `total_cost`, `pool_size`, `runtime_seconds`, `monitor` (live
  signals at exit), `paradigm_trials` (one record per frontier call:
  stage, accepted, score, delta vs prev best, description), and
  `elites` (the pool sorted by score, each with code, description,
  source tag, family id, uses_count).
* **`best.py`** (orchestrator path) or **`best_program.py`** (driver
  path) — the best-scoring program as a standalone module.
* **`summary.json`** (driver path only) — flat summary keys for
  benchmark aggregators.

The snapshot schema is intentionally close to LEVI's so benchmark
tooling that already reads LEVI's `snapshot.json` requires minimal
changes.

---

## 9. Tuning knobs

Sensible defaults are baked in; you should rarely need to override
them. Listed roughly in order of impact:

| Knob                             | Default | What it does                                                                                                  |
| -------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------- |
| `pe_cron_interval`               | 50      | Frontier paradigm shift fires every N evals. Smaller = more frontier cost; larger = less diversity injection. |
| `n_workers`                      | 4       | Mutation workers in parallel. Caps simultaneous LLM calls.                                                    |
| `pool.K`                         | 100     | Maximum programs retained.                                                                                    |
| `pool.family_cosine_threshold`   | 0.72    | Single-linkage threshold for "same family". Lower = more permissive merging; higher = more families.          |
| `pool.max_per_family`            | 8       | How many programs from one family may co-exist.                                                               |
| `selector.recency_tau`           | 30      | Half-life-ish of the recency boost.                                                                           |
| `monitor.plateau_max`            | 100     | Denominator of `stagnation_level()` — flips routing to the "late" frontier prompt when reached.               |

All are exposed via `BladeConfig` (Python entry) or the
`advanced_options` JSON (workflow entry). Most production papers will
just touch `evaluations`, `pe_cron_interval`, and the model IDs.

---

## 10. Relation to LEVI

BLADE is a deliberate redesign of three subsystems within an otherwise
shared architecture. Everything else — frontier 3-phase prompts,
async producer/consumer, error-archive self-repair, meta-advisor
prompt slot, evaluator sandbox — is kept verbatim.

| Subsystem          | LEVI                                                          | BLADE                                                               |
| ------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------- |
| Behavioral archive | CVT-MAP-Elites (1000 centroids, hand-crafted AST descriptors) | Top-K (= 100) pool keyed on description embeddings + family caps    |
| Stagnation signal  | PPS formula driven by sparse new-best events                  | Three dense sliding-window statistics                               |
| Sampler            | 4-D Thompson bandit (sampler × model × prompt × temperature)  | UCB-style priority (score + novelty + recency − diversity penalty)  |

In rough numbers: BLADE has about 80 % fewer hyperparameters than LEVI,
and the orchestrator + supporting modules are ~600 + ~700 lines instead
of LEVI's ~3500 across the equivalent set of files. Same execution
shape, smaller surface, embedding-aware diversity.

For the design rationale and ablation thinking that led here see
[SIMPLE_EVO.md](SIMPLE_EVO.md) and
[SIMPLE_EVO_IMPLEMENTATION.md](SIMPLE_EVO_IMPLEMENTATION.md).
