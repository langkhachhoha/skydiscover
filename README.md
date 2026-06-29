# SpecEvo: Speculative Evolution with Large Language Models for Cost-Efficient Scientific Discovery

**Operating loop: Speculate → Navigate → Advise.**

---

## Abstract

LLM-driven evolutionary search is powerful but expensive, and the strongest systems pay for their
results twice over — in **dollars and in time** — because they lean on a single frontier model for every
step. SpecEvo dissolves this with three coupled ideas. First, in algorithm discovery *breadth is cheap
and parallel, while depth is expensive and sequential*: a small, fast **Speculator** explores many
directions at once into a self-organizing behavioral archive, while a frontier **Navigator** is woken
only occasionally for the rare, hard step — so the search gets broad early coverage cheaply *and* deep
reasoning when it matters, without the latency of a sequential loop or the cost of a parallel one.
Second, discovery is a *generate-then-reflect* process: the Speculator generates a large body of
attempts whose value lies as much in the evidence they leave behind as in the wins they score, and the
Navigator reads that evidence as a *map* to propose a directed hypothesis (one of three move-classes,
escalating with stagnation). Third, cheap models must be *steered, not merely repeated*: a set of
distinct exploration operators forces the Speculator to cover the space, and an **Advisor** turns the
whole trajectory — including the failures and near-misses a scalar score discards — into concise
natural-language guidance injected back into the Speculator.

Across mathematical-discovery and system-engineering benchmarks, on both GPT-5 and Kimi-K2 backbones,
SpecEvo matches or exceeds the strongest baselines on most tasks while spending **2.0–3.4× less** than
the average baseline.

---

## Motivation

SpecEvo's design responds to three observations about how existing LLM-driven discovery systems spend
their resources.

1. **The cost–time dilemma of frontier-only search.** With only an expensive model in hand, a system
   must pick one of two regimes, and each fails in its own way. A *sequential* refinement loop is cheap
   to coordinate but slow: frontier reasoning is slow per step, and early on the model has little
   context about what has been tried, so the quality curve takes off late — a *timing* failure under
   short budgets. A *parallel* population covers the space early but, if every lineage calls the
   frontier model, cost *explodes* — a *cost* failure. SpecEvo assigns breadth (parallel) to cheap fast
   models and depth (rare, sequential, hard) to the frontier model, getting early broad coverage cheaply
   *and* occasional deep reasoning without throttling throughput.

2. **Discovery is collective evidence, not a single mind iterating.** Real research is
   *generate-then-reflect*: many cheap experiments are run broadly, then someone steps back and looks
   across the whole body of attempts — successes *and* failures — to decide what to try next. A loop of
   `improve → improve → improve` discards exactly that panoramic evidence. SpecEvo institutionalizes the
   division of labor: the Speculator is the bench, the Navigator is the principal investigator forming
   hypotheses from the map, and the Advisor is the reviewer distilling lessons.

3. **Cheap models must be steered, not merely repeated.** Under a generic "improve this" prompt, small
   models collapse to a few familiar patterns and cover the space narrowly. SpecEvo drives the
   Speculator with a *set of distinct exploration operators* that force breadth back. And it treats
   every output as information — including the failures and near-misses a scalar score throws away. A
   fitness number is a lossy summary of what an evaluation revealed: a crash says what to avoid, a
   saturated region says where to stop digging, a winner's description says what mechanism is paying off.
   The **Advisor** recovers this discarded context as code-shaped **natural-language feedback** that
   steers the next round — what a `Δscore` alone cannot say.

---

## Framework

![SpecEvo overall framework](image/Overall.png)

*An **Init phase** seeds a diverse, behaviorally embedded population. The **Speculator** (cheap model)
explores many directions in parallel into an adaptive MAP-Elites archive; the **Navigator** (frontier
model) wakes on stagnation to propose a directed move; and the **Advisor** distills population-level
lessons that are injected back into the Speculator's prompt. The archive's elites converge to the
returned **best code**.*

### Components

- **Init phase — seeding the strategy map.**
  The frontier model writes a handful of *diverse seed programs* sequentially (each conditioned on the
  previous to push for novelty); the cheap model then fans each seed into many parallel variants. Every
  program is encoded as a **hybrid behavior descriptor** — structural AST features concatenated with a
  PCA-reduced embedding of its natural-language description — so programs of equal fitness but different
  ideas occupy different niches.

- **Population — adaptive, online-clustered MAP-Elites.**
  Niches are *discovered online* by KMeans over the learned descriptor and **periodically re-clustered**
  as the search moves, rather than fixed up front. Each niche keeps a single elite incumbent; a program
  is admitted only if it beats its niche's incumbent.

- **Speculator — cheap, massive, parallel exploration.**
  The low-cost model issues many proposals at once across the archive, driven by a set of distinct
  operators (focused fix, mechanism swap, structural/component crossover, analysis-guided mutation). A
  proposal is valuable not only when it wins, but because its outcome — admits, near-misses, recurring
  errors — becomes *evidence* for the Navigator and Advisor.

- **Navigator — rare, expensive, hypothesis-driven escalation.**
  The frontier model wakes only at fixed intervals, reads the cells' representatives, and is routed by a
  stagnation signal to one of three move-classes of increasing intensity — **Synthesis** (hybridize
  nearby strong ideas), **Surgical** (precisely tune the champion), or **Reframe** (switch to a
  genuinely different strategy family) — with the counter-intuitive but data-driven rule that the
  *deepest* stagnation triggers a **Reframe**. A short *Strategy Log* of its last six attempts keeps it
  from repeating dead-ends.

- **Advisor — population-level reflection in language.**
  At intervals, cheap deterministic counters label niches as **leading**, **improving**, **saturated**,
  or **under-explored** (by whether a niche's frontier moved and how heavily it is being mined — no
  absolute thresholds), and aggregate **recurring errors** into bounded, portable knowledge. An LLM
  verbalizes this map into concise advice injected back into the Speculator's prompt.

See [SPECEVO_FRAMEWORK.md](SPECEVO_FRAMEWORK.md) for the full method write-up and default configuration.

---

## Results

SpecEvo is evaluated against OpenEvolve, GEPA, AdaEvolve, and EvoX over three seeds, reporting
mean ± std and best.

**Mathematical-discovery benchmarks** (circle packing, Heilbronn, MinMax distance, signal processing):

![Mathematical-discovery results](image/Math_result.png)

**System-engineering benchmarks** (EPLB, LLM-SQL, Transaction, PRISM):

![System-engineering results](image/System_result.png)

Across both suites and both backbones, SpecEvo attains the best or near-best score on the majority of
tasks, confirming that aggressive cheap speculation under rare frontier navigation does not sacrifice
solution quality.

---

## Cost

The advantage is in *cost*. For each method we sum per-task cost into a total per seed, then report the
mean ± std over three seeds. SpecEvo is consistently the cheapest method by a wide margin, while the
baselines cluster together near the top.

| | GPT-5 | Kimi-K2 |
|---|---|---|
| **Math** | ![Math cost, GPT-5](image/math_gpt_cost.png) | ![Math cost, Kimi-K2](image/math_kimi_cost.png) |
| **System** | ![System cost, GPT-5](image/system_gpt_cost.png) | ![System cost, Kimi-K2](image/system_kimi_cost.png) |

Relative to the **average baseline**, SpecEvo is **3.4×** cheaper on math (GPT-5), **2.0×** on math
(Kimi-K2), **2.8×** on system (GPT-5), and **2.6×** on system (Kimi-K2) — near-SOTA quality at a
fraction of the cost.
