# SpecEvo: Speculative Evolution with Large Language Models for Cost-Efficient Scientific Discovery

> Paper-style design report. This document is written as a method draft: an abstract, a motivated
> introduction, and a detailed description of the framework and its components. All parameter values
> are the system's defaults and can be lifted directly into an *Implementation Details* section.

---

## Abstract

Scientific discovery is the work of a **lab**, not a lone genius — yet LLM-driven evolutionary search is
built as though it were a lone genius. The dominant recipe asks a single frontier model to carry *every*
step of the search: the computational equivalent of hiring one brilliant researcher and making them
personally run every experiment, judge every result, and choose every next move, flat-out like an ox until
the budget runs out. It is expensive — in dollars and in wall-clock time — and it is a poor imitation of
how discovery actually happens, especially under tight resources. A real lab divides the labor: a few
cheap, eager juniors run many experiments at the bench in parallel; a senior labmate periodically reviews
the whole body of work — failures included — and tells the juniors what to stop doing and what to build on;
and a principal investigator, whose time is the lab's scarcest resource, steps in only at the hard
junctures to read the accumulated evidence and set the next direction. **SpecEvo is this low-resource lab,
made mechanical.** A small, fast **Speculator** plays the juniors, exploring many directions at once into a
shared behavioral archive that serves as the lab's notebook; a frontier **Navigator** plays the PI, woken
only for the rare, hard step, reading the archive as a *map* and proposing a directed move — one of three
classes, escalating with how stalled the search is — rather than another blind edit; and an **Advisor**
plays the senior labmate, continually turning the whole trajectory — including the failures and near-misses
a scalar score throws away — into concise natural-language feedback that flows back to the juniors: *avoid
this error, stop crowding this exhausted corner, build on this working idea.* The roles reinforce one
another the way a functioning lab does: cheap parallel hands make breadth affordable, treating their output
as shared evidence makes the PI's rare calls decisive, and the senior's mentoring keeps the juniors from
collapsing into repetition or noise. The resulting cost-efficiency is not a trick bolted on — it falls out
of faithfully modeling how a resource-constrained team divides labor. Across mathematical-discovery and
system-engineering benchmarks, on both GPT-5 and Kimi-K2 backbones, SpecEvo matches or exceeds the
strongest baselines on most tasks while spending **2.0–3.4× less** than the average baseline.

---

## 1. Introduction

LLM-guided evolutionary methods now produce competitive — sometimes novel — results in mathematics,
heuristic design, and systems optimization. The recipe is familiar: a user supplies a problem and a
scoring function; an evolutionary loop repeatedly asks an LLM to mutate candidate programs, evaluates
them, and keeps the promising ones. The difficulty is cost. The strongest published systems route
*every* mutation through an expensive frontier model, and a single run can cost tens of dollars and
hours of wall-clock time. We argue that this is not just a billing problem but a *picture* problem: the
field has implicitly modeled discovery as a single expert mind grinding away alone. SpecEvo starts from a
different and, we think, more faithful picture — and the architecture is what follows once that picture
is taken seriously.

### 1.1 Discovery is a lab, not a lone genius

The frontier-only recipe is the computational form of a fantasy: hire one brilliant researcher, sit them
at a desk, and have them personally run every experiment, judge every result, and decide every next move,
working flat-out like an ox until the money runs out. No real discovery happens this way — and certainly
not in the labs that produce most of science under ordinary, limited budgets. A working lab is a *team
with a structure*, and that structure exists precisely because no single mind, however capable, is the
cheapest way to do every job.

Picture a small lab with three kinds of people:

- **The juniors** — students at the bench. There are several of them, they are inexpensive, and they
  work in parallel. They run a large number of experiments across many directions, most of which are
  ordinary, some broken, a few surprising. They are not expected to be brilliant; they are expected to be
  *prolific and diverse*, and to leave a clear record of everything they tried.
- **The principal investigator (PI)** — the professor. Their time is the scarcest and most expensive
  resource in the lab, so they do not sit at the bench. They appear at the hard junctures: they read what
  the lab has accumulated, see the shape of the whole effort, and propose the next direction — a
  hypothesis, not a tweak. One good call from the PI redirects a dozen bench-experiments.
- **The senior labmate** — the postdoc or "đàn anh" who has been here longer. They write little code
  themselves; their job is to look across the juniors' whole body of work — the failures and dead-ends
  included — and hand back concrete guidance: *that mistake keeps recurring, stop making it; that corner
  is exhausted, stop digging there; that idea is working, build on it.*

This is the essence of how a **low-resource team** actually does research: breadth comes from many cheap
hands, depth from a rare and expensive judgment, and what keeps the cheap hands from wandering is
continual mentoring grounded in the lab's own record. SpecEvo is a deliberate, mechanical scale-model of
this lab. The **Speculator** is the juniors; the **Navigator** is the PI; the **Advisor** is the senior
labmate; and a shared **behavioral archive** is the lab notebook they all read and write. The rest of
this section traces how each role is forced by a real constraint of doing science cheaply — and the
cost savings reported above are not a clever trick layered on top, but the natural consequence of letting
each job be done by the cheapest agent capable of it.

### 1.2 Why one expert cannot also be every student: the cost–time dilemma

The first constraint is the one that breaks the lone-genius picture outright. When the only capable tool
is an expensive model, every framework is forced into one of two regimes, and each fails in its own way —
exactly as a lab would fail if it had only the professor and no one else.

**The sequential regime.** A single line of refinement, where each step waits on the previous one and
calls the frontier model to "improve the current best." This is the professor working alone at the bench:
cheap in *coordination* but catastrophic in *time*. Frontier reasoning is slow per step, and — worse —
early in the run the model has almost no context about which directions have already been tried, so it
re-derives the search from scratch. The quality curve takes off *late*: only after enough steps
accumulate does the run become good. Under a short budget, sequential frontier search simply has not had
time to find its footing. This is a *timing* failure, not merely a monetary one.

**The parallel regime.** Population- or island-based search that explores many directions at once,
covering the design space early. This is the right division of labor — but if you staff every bench with
a professor, i.e. route each parallel lineage through the frontier model, cost grows with the number of
lineages and *explodes*. This is a *cost* failure.

SpecEvo's first principle is the lab's first principle: **breadth and depth are different jobs with
different costs, and should be staffed by different people.** Breadth — running many directions in
parallel — is exactly what small, fast, cheap models (the juniors) are good at; depth — the rare, hard
step that genuinely needs frontier-scale reasoning — is reserved for the frontier model (the PI) and run
sequentially so it never throttles throughput. Asking *one* model to be both the professor and every
student is what makes existing systems either slow or expensive; giving each job to the agent suited for
it dissolves the dilemma.

### 1.3 The lab notebook: discovery runs on collective evidence

Hiring cheap juniors only helps if their work is genuinely *used* — and that depends less on how *much*
they generate than on what the lab does with it. Here the standard loop leaves its largest value on the
table: it treats search as one model iterating `improve → improve → improve`, keeping only the running
best and discarding the rest. No real lab works this way. Discovery is a **generate-then-reflect**
cycle: the juniors run many cheap experiments across many directions; then someone steps back and looks
*across the whole body of attempts*, successes and failures alike, to judge what to try next. It is the
panoramic view over *collective evidence* — the lab notebook, not any one person's memory — that makes
the next round intelligent.

SpecEvo institutionalizes this. The Speculator is the bench: its value lies as much in the evidence it
leaves behind as in the wins it scores, so even its near-misses are written into the notebook rather than
thrown away. The Navigator is the PI: when it wakes, it does not look at a single parent program but
reads the *map* of everything the lab has tried — which directions are occupied, which are paying off,
how stalled the search has become — and proposes a grounded hypothesis about where to go. A system that
keeps only the best score so far has, in effect, thrown out the lab notebook and asked the professor to
plan the next experiment from memory.

### 1.4 Juniors must be mentored, not merely told to work harder

The reflective layer is only as good as the stream of work it reads — and cheap juniors, left
unsupervised, produce a poor stream. The usual way to drive a mutation model is a single instruction,
"improve this version," trusting model strength for diversity. That works passably for a frontier model
but breaks for a cheap one, in two ways a good senior labmate knows to correct.

First, **inexperienced hands collapse to repetition.** Given the same generic "improve" prompt, a cheap
model falls back on a few familiar patterns, producing near-duplicate proposals that cover the design
space narrowly — the junior who keeps running slight variants of the one protocol they know. SpecEvo
counters this the way a mentor assigns varied tasks: it drives the Speculator with a *set of distinct
exploration operators*, each reframing the work from a different angle — fix one weakness, swap one
mechanism, fuse two structures, or follow a pre-computed analysis. The operators force breadth back into
the stream, turning the cheap model's tendency to repeat into a structured sweep of the space.

Second, **every result is information — even the failed ones.** Cheap models frequently emit code that
will not run, violates constraints, or rehashes a stale idea. A system that reads only scalar scores
throws this context away — like a lab that files only its successful experiments. SpecEvo harvests it
through the Advisor, the senior labmate's role made explicit: it aggregates recurring errors, identifies
exhausted regions, and names what is paying off, then expresses all of it as concise, code-shaped
**natural-language feedback** injected back into the Speculator's prompt — *avoid this error class, stop
crowding this exhausted region, build on this working mechanism.* Language preserves the context that a
`Δscore` destroys. This is mentoring, not scoring: it is what lets a few cheap, error-prone hands be
*directed* rather than left to guess.

### 1.5 The Speculate–Navigate–Advise loop and contributions

The three roles assemble into a single loop with three distinct tempos — the daily rhythm of the lab.
The **Speculator** (juniors) runs almost continuously — cheap, parallel, the source of most steps and all
evidence. Periodically, the **Navigator** (PI) wakes for a single expensive call to read the population
map and propose a direction matched to the current degree of stagnation. Also periodically, the
**Advisor** (senior labmate) distills the trajectory into language and feeds it back. All three are
anchored to a **self-organizing behavioral archive** — the lab notebook — that serves simultaneously as a
diverse solution memory and as the raw material for reflection. We call this the
**Speculate–Navigate–Advise (SNA)** loop.

Our contributions are:

1. **A lab-structured, two-tempo architecture** that separates cheap parallel breadth (the juniors) from
   expensive sequential depth (the PI), resolving the cost–time dilemma of frontier-only search and
   reaching near-SOTA quality at 2.0–3.4× lower cost.
2. **A stagnation-routed Navigator** that, like a PI reading the lab's results, treats the population as a
   map and proposes *hypotheses* — choosing among three move-classes, **Synthesis**, **Surgical**, and
   **Reframe** — rather than issuing undifferentiated edits, with the counter-intuitive but data-driven
   rule that the *deepest* stagnation triggers the most aggressive move.
3. **An Advisor that recovers the signal scalar fitness discards.** A fitness number is a lossy summary
   of what an evaluation revealed: a crash says what to avoid, a saturated region says where to stop
   digging, a winner's description says what mechanism is paying off. Playing the senior reviewer, the
   Advisor re-reads the whole trajectory — including the failures and near-misses other systems throw
   away — and re-expresses this discarded context as concise natural-language feedback that steers the
   Speculator. This *verbal* mentoring is what lets a team of cheap, error-prone models be directed rather
   than left to guess.

---

## 2. The SpecEvo Framework

### 2.1 Overview

SpecEvo takes a `SpecEvoConfig` — a `problem_description`, a `function_signature`, an executable
`score_fn`, the target `fn_name`, the evaluation `inputs`, and an optional user `seed_program` — and
returns the highest-scoring program found within a budget. Budgets may be expressed in dollars,
evaluations, wall-clock seconds, or a target score; the run stops when any is reached.

The framework instantiates the SNA loop over a shared behavioral archive (Figure: *Overall*). After an
initialization phase establishes a diverse, behaviorally embedded population, the loop proceeds with
the Speculator generating candidates continuously, the Navigator escalating rarely on stagnation, and
the Advisor reflecting at fixed intervals. We describe each component in turn.

### 2.2 Initialization and the Embedded Population

Before cheap speculation is useful, the search needs a diverse, verified set of starting strategies to
build on. Initialization is therefore not a trivial warm-up but the construction of the initial
hypothesis space, in two phases.

**Generating seed programs (frontier, sequential).** The frontier model (`navigator_model`, default
`gpt-5`) writes `n_diverse_seeds = 5` seed programs, each carrying a short natural-language description
alongside its executable code. Seeds are generated *sequentially*: each new seed is shown all
previously accepted seeds and explicitly instructed to differ from them in strategy, with up to three
retries on parse or runtime failure. Generating genuinely different starting strategies is the
prototypical *hard* step, so it is exactly where frontier-scale reasoning is worth its price.

**Generating variants (Speculator, parallel).** For each seed, the cheap model (`speculator_model`,
default `qwen3-30b-a3b-instruct`) generates `n_variants_per_seed = 20` variants in parallel, each
prompt seeded with one or two sibling programs as inspiration. After initialization the population
already holds up to `5 + 5×20 = 105` candidates — the first large, cheap, broad sweep, and the data
from which the archive is calibrated.

**The embedded population.** Each program is encoded as a 22-dimensional **hybrid behavior descriptor**
that captures *what kind of program this is* along two complementary axes:

```text
behavior_vec = [ AST features (14-d) | PCA(description embedding) (8-d) ]   → 22-d
               └── z-score (Welford) ┘ └──────── z-score (Welford) ────────┘
```

- *Structural axis (14-d).* Fourteen AST features — AST depth, cyclomatic complexity, loop count and
  nesting, branch/function/comprehension/call/comparison/subscript counts, numeric-literal and
  math-op counts, import count, and code length — each passed through `log1p` so large counts do not
  swamp small ones.
- *Semantic axis (8-d).* The program's description is embedded with `text-embedding-3-small`
  (1536-d) and reduced to 8 dimensions by PCA, recomputed at each re-clustering.
- The two halves are independently standardized online (Welford) before concatenation, so neither
  dominates the distance geometry, and each can be ablated independently.

Two programs of equal fitness but different underlying ideas thus land in different regions of the
descriptor space — the property the rest of the framework relies on to maintain diversity.

### 2.3 The Adaptive Behavioral Archive

SpecEvo maintains an **adaptive, MAP-Elites-style behavioral archive** whose niches are *not fixed in
advance*. Niches are discovered online by KMeans (`n_cells = 50`) over the learned descriptor: the
first fit occurs once `min_admits_before_cluster = 16` programs exist, and the archive is
**re-clustered every `recluster_every = 30` admissions** (warm-started from the previous centroids).
The cell boundaries therefore *track the search* rather than being imposed at the outset — this online,
periodically re-fit clustering over a *learned* descriptor is the archive's defining feature.

Admission is elitist: a program enters the archive *iff* its score exceeds that of its cell's current
incumbent, and each cell retains only its single best program. There is no hall-of-fame, quota, or
grace period. A single escape valve, `force_add`, is reserved for the Reframe move-class (§2.5): if a
genuinely new strategy is rejected by an occupied cell, the globally weakest program is evicted to make
room — when the search is deeply stalled, losing a weak slot is cheaper than discarding a fresh
direction.

The archive plays a dual role throughout the run: it is both the Speculator's diverse solution memory
and the substrate the Navigator and Advisor read to reason about the search.

### 2.4 The Speculator: cheap, massive, directed exploration

The Speculator is the engine of breadth. Rather than a single "improve" instruction, it samples at
each step (`_generate_one`) from a set of distinct operators, the mechanism that forces a cheap model
to cover the space:

- A coin flip (`p_crossover = 0.35`) chooses **crossover** or **mutation**.
- **Mutation** draws uniformly from three templates: a general improvement (`MUTATE_PROMPT_GENERAL`),
  a focused single-weakness fix (`MUTATE_PROMPT_FOCUSED_FIX`), and a single-mechanism swap
  (`MUTATE_PROMPT_MECHANISM_SWAP`, e.g. greedy → Metropolis acceptance, fixed step → line search).
- **Crossover** draws from two templates: a structural blend (`CROSSOVER_PROMPT_STRUCTURAL`) and a
  single-component graft from a donor onto a skeleton (`CROSSOVER_PROMPT_COMPONENT_SWAP`).
- **Targeted mutation.** When a parent already has a cached analysis (below) and a coin flip
  (`p_targeted_mutate = 0.5`) succeeds, `TARGETED_MUTATE_PROMPT` forces the model to act on exactly
  one item from that analysis rather than inventing a fresh direction.

Every operator requires the model to write an `## Analysis` block (components, strengths, weaknesses,
plan) *before* producing code, committing it to a hypothesis instead of rewriting blindly; the analysis
is reused on later passes.

The main loop runs `n_workers = 4` concurrent workers continuously until the budget is spent. Each
offspring selects a parent, generates code, is scored in a sandboxed process pool, and is offered to
the archive. The archive *is* the acceptance test: an offspring is admitted only if it beats its cell's
incumbent; otherwise it is recorded as a near-miss (`dropped_worse`) — still counted as an evaluation
and, importantly, **still retained as evidence** for the Advisor. This is where the principle that
"even waste is information" is realized concretely.

**Parent selection** uses a Zipfian rank sampler, `P(rank) ∝ rank^(−β)`, that depends only on rank and
is thus scale-invariant. The exponent adapts to stagnation, `β = β_min + (1−s)(β_max − β_min)` with
`β_max = 2.0` (fresh: exploit the top few) and `β_min = 0.3` (stalled: spread across the tail). A second
parent is preferentially drawn from a *different* cell to expose two strategies to crossover, and three
inspirations are passed as *description and score only* (never code) to avoid copying and token bloat.

**Analyzer.** A background task every `analyzer_interval = 30` evaluations selects the
`analyzer_top_k = 3` archived programs lacking an analysis and asks the cheap model for a short review
(algorithm summary, top-three ranked bottlenecks, three suggested changes; under 250 words), cached per
program. The policy is accumulating — once the top-k are covered it analyzes ranks 4–6, then 7–9 —
until the whole archive carries an analysis, giving targeted mutation a sharper aim.

### 2.5 The Navigator: rare, expensive, hypothesis-driven escalation

The Navigator supplies depth. A background monitor wakes it every `navigator_interval = 50`
evaluations (after initialization); if fewer than `navigator_min_cells = 5` cells are occupied it
abstains, since the population map is not yet meaningful. Each waking is one expensive frontier call
followed by a cheap fan-out.

When it wakes, the Navigator reads the population map: the **anchors** are the top-scoring cells'
representatives, passed as *full code with description and score*, while the remaining cells are passed
as *description and score only*. It then proposes a hypothesis as one of three **move-classes**, routed
by a stagnation signal `s = max(global, local)`, where `global = min(1, plateau_steps / 100)` measures
evaluations since the last global best and `local = min(1, admit_gap / 20)` measures evaluations since
the last admission:

| Stagnation `s` | Move-class | Anchors | Hypothesis the model is asked to form |
|---|---|---|---|
| `s ≤ 0.4` | **Synthesis** | 3 close-in-score + 5 inspirations | Combine 2–3 concrete mechanisms from different anchors into one coherent program that beats each individually. No constant retuning. |
| `0.4 < s ≤ 0.7` | **Surgical** | 1 (the champion) + 5 inspirations | Make exactly one precise, local, *structural* change to the champion (score-aware acceptance, better repair, exploit existing slack). "Be the careful surgeon, not the wild inventor." |
| `s > 0.7` | **Reframe** | 2 | Switch to a genuinely different strategy family absent from all anchors and inspirations. Forbidden: new constants, sub-routine grafting (that is Synthesis), variable renaming. |

The routing is deliberately counter-intuitive — *deeper* stagnation triggers the most aggressive move.
This is data-driven: once a champion has plateaued, asking the frontier model for yet another local
patch rarely escapes the basin; only a change of strategy family breaks the plateau. Moderate
stagnation, where the champion still has momentum, rewards careful local polishing; a fresh search
rewards synthesizing nearby contenders. (The third move-class is named **Reframe** rather than the
figure's preliminary "Shift" label; the figure asset should be updated to match.)

**Closing the loop.** Each accepted Navigator hypothesis (an expensive seed) is fanned into
`n_navigator_variants = 4` cheap variants spread across temperatures (`[center−0.25, center+0.15]`,
exploit → explore) so the four do not merely retune the same constants. One expensive depth step is
thereby amplified into many cheap breadth steps — the "Navigate → Speculate" beat that closes the loop.

**Navigator memory: the Strategy Log.** The Navigator keeps a rolling `deque(maxlen=6)` of its six most
recent attempts, each rendered as `[#idx] ✓/✗ score=… Δ=… :: description`. This block is injected into
every Navigator prompt under `## Strategy Log (recent attempts)`, with the standing instruction to
*"Avoid any strategy whose Strategy-Log entry has delta ≤ 0 — that approach has already failed."* The
Navigator thus knows which directions it has tried and whether they paid off, and never repeats a
dead-end — short-term, rolling context rather than blind guessing.

### 2.6 The Advisor: population-level reflection in language

The Advisor is the reflective layer. It writes no code; it reads the entire trajectory and distills
*lessons* that are injected back into the Speculator's prompt. Three design choices make it more than a
score tracker.

**Credit by behavioral niche and content, not by operator.** The mutation/crossover templates are drawn
uniformly at random; "the improvement came from template X" is a process artifact, not a causal signal.
The Advisor therefore ignores which operator produced an admission and instead attributes progress to
*behavioral niches* (archive cells) and the *content* of their incumbents' descriptions — what the
algorithm does, not which template wrote it. The measurement is performed by cheap, deterministic,
reproducible counters; the LLM only verbalizes the resulting niche map.

**Region signals without absolute thresholds.** Every `advisor_interval = 50` evaluations, the Advisor
reads the live archive and labels niches using two cheap, threshold-free quantities: when a niche's
frontier last advanced (the incumbent's `created_at_eval`, which survives re-clustering because it is
attached to the program object), and how *busy* a niche is (attempts falling into the cell within the
window, counting near-misses, since cell-ids are assigned before the incumbent comparison — removing
the survivorship bias of counting only admissions). With `window_start = eval_count − advisor_interval`,
this yields four buckets:

| Niche | Condition | Bucket |
|---|---|---|
| top-3 by score (each a different cell) | — | **LEADING** |
| incumbent advanced this window | ≤ 4 | **IMPROVING** |
| frontier stuck this window but ≥ 1 attempt (busy-but-stale) | top by #attempts, ≤ 3 | **SATURATED** |
| occupied, neither leading nor improving, fewest attempts | ≤ 3 | **UNDER-EXPLORED** |

This answers the question "is using the score principled?" without ever bucketing individual admissions
by Δ-magnitude: it asks only whether a niche's frontier is *moving* and whether it is being *mined* —
a region-based, attempt-level, non-stationary estimate independent of the benchmark's score scale.

**Errors as accumulated, portable knowledge.** Every error is collected — both structural/code errors
(syntax, name/attribute, type, shape) and evaluator-contract errors (timeout, infinite loop, constraint
violation, malformed return). Errors are keyed by an `error_signature` (lowercased, digits and
punctuation stripped, first ≤ 8 words), so "Overlap between circles 0 and 2" and "… 3 and 5" merge into
one counted entry — the recurring text *is* the taxonomy, with no hand-built keyword table and full
portability to new problems. The knowledge base is merged by signature and persists across cycles but is
bounded (`error_knowledge_max = 24`): when full, the rarest mode is evicted, so one-off errors cycle out
while recurring ones survive and accumulate.

**Output and injection.** The Advisor emits exactly four short sections (under 140 words total):
**WORKING** (the mechanism that LEADING and IMPROVING niches share, described as an algorithm),
**SATURATED** (busy-but-stale niches to de-emphasize, or "none"), **TRY NEXT** (2–3 code-shaped
suggestions that build on WORKING and push into under-explored niches), and **AVOID** (anti-patterns
from the accumulated error knowledge, recurrent ones first). This block is injected verbatim into
mutation and crossover prompts under `## Lessons learnt so far` with probability `advice_inject_p = 0.35`
(≈17 injections per 50 evaluations — enough to steer without homogenizing the operator stream). At the
end of each cycle the attempt counters reset but the error knowledge persists.

A known limitation, worth stating honestly: the busy-ness counter is keyed by *current* cell-id, so a
mid-cycle re-clustering can relabel cells and slightly blur the signal. With re-clustering every 30
admissions and the Advisor running every 50 evaluations (≈1–2 re-clusters per cycle), the blur is
bounded; the frontier-staleness signal is immune because it tracks the incumbent object's
`created_at_eval`.

---

## 3. Why each component exists

Each component answers a specific asymmetry rather than being a generic add-on:

| Component | The asymmetry that forces it |
|---|---|
| **Two tempos (cheap parallel Speculator + expensive sequential Navigator)** | Breadth and depth have very different marginal costs; serving both with one model is either expensive (parallel frontier) or slow (sequential frontier). |
| **Adaptive behavioral archive (online clustering, hybrid descriptor)** | Cheap guessing pays off only atop a verified diverse base; since the strategy families are unknown a priori, niches must be discovered online rather than fixed up front. |
| **Diverse exploration operators** | Cheap models collapse to a few patterns under a generic prompt; varied operators force breadth back. |
| **Stagnation-routed three move-class Navigator** | Different degrees of stalling call for different moves; and discovery advances by surveying the whole map and forming a hypothesis, not by repeating one edit. |
| **Strategy Log** | Each depth step is expensive; repeating a dead-end is doubly wasteful, so the Navigator remembers its last six trials and avoids non-improving ones. |
| **Advisor (niche-based, four-bucket, error knowledge)** | Every candidate — including failures and near-misses — is evidence; a scalar Δ destroys the context, so verbal feedback is needed to let cheap speculation aim. |

---

## Appendix — Default configuration (Implementation Details)

| Group | Parameter | Default |
|---|---|---|
| Models | speculator / navigator / embedding | `qwen3-30b-a3b-instruct` / `gpt-5` / `text-embedding-3-small` |
| Init | `n_diverse_seeds` / `n_variants_per_seed` | 5 / 20 |
| Archive | `n_cells` / AST dims / `embedding_dim` (PCA) / vector | 50 / 14 / 8 / 22-d |
| Archive | `min_admits_before_cluster` / `recluster_every` / adaptive | 16 / 30 admits / True |
| Main loop | `n_workers` / `p_crossover` | 4 / 0.35 |
| Operators | mutate templates / crossover templates / `p_targeted_mutate` | 3 / 2 / 0.5 |
| Analyzer | `analyzer_interval` / `analyzer_top_k` (accumulating) | 30 / 3 |
| Sampler | Zipfian `β_max → β_min` (stagnation-adaptive) / `k` inspirations | 2.0 → 0.3 / 3 |
| Stagnation | `global = min(1, plateau/100)` · `local = min(1, admit_gap/20)` · `max(·,·)` | 100 / 20 |
| Navigator | `navigator_interval` / `navigator_min_cells` / `n_navigator_variants` | 50 / 5 / 4 |
| Navigator | synthesis ≤ `0.4` (3 anchors) · surgical ≤ `0.7` (1 anchor) · reframe > `0.7` (2 anchors) | 0.4 / 0.7 |
| Advisor | `advisor_interval` / `advice_inject_p` / `error_knowledge_max` | 50 / 0.35 / 24 |
