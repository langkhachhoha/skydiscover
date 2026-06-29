# SpecEvo: Speculative Evolution with Large Language Models for Cost-Efficient Scientific Discovery

*A low-resource research lab, emulated with large language models: many cheap students (**Speculator**), a
rarely-available professor (**Navigator**), and a senior labmate who mentors them (**Advisor**).*

**Operating loop: Speculate → Navigate → Advise.**

---

## Abstract

Scientific discovery is the work of a **lab**, not a lone genius — yet LLM-driven evolutionary search is
built as though it were a lone genius. The dominant recipe asks a single frontier model to carry *every*
step of the search: the computational equivalent of hiring one brilliant researcher and making them
personally run every experiment, judge every result, and choose every next move, flat-out like an ox until
the budget runs out. It is expensive — in dollars and in wall-clock time — and a poor imitation of how
discovery actually happens, especially under tight resources. A real lab divides the labor: a few cheap,
eager juniors run many experiments at the bench in parallel; a senior labmate periodically reviews the
whole body of work — failures included — and tells them what to stop doing and what to build on; and a
principal investigator, whose time is the lab's scarcest resource, steps in only at the hard junctures to
read the accumulated evidence and set direction. **SpecEvo is this low-resource lab, made mechanical.** A
small, fast **Speculator** plays the juniors, exploring many directions at once into a shared behavioral
archive that serves as the lab's notebook; a frontier **Navigator** plays the PI, woken only for the rare,
hard step, reading the archive as a *map* and proposing a directed move — one of three classes, escalating
with how stalled the search is — instead of another blind edit; and an **Advisor** plays the senior
labmate, continually turning the whole trajectory — including the failures and near-misses a scalar score
throws away — into concise natural-language feedback for the juniors: *avoid this error, stop crowding this
exhausted corner, build on this working idea.* The cost-efficiency is not a trick bolted on; it falls out
of faithfully modeling how a resource-constrained team divides labor.

Across mathematical-discovery and system-engineering benchmarks, on both GPT-5 and Kimi-K2 backbones,
SpecEvo matches or exceeds the strongest baselines on most tasks while spending **2.0–3.4× less** than
the average baseline.

---

## Motivation

The frontier-only recipe is the computational form of a fantasy: hire one brilliant researcher, sit them at
a desk, and have them personally run every experiment, judge every result, and decide every next move,
working flat-out until the money runs out. No real discovery happens this way — least of all in the
ordinary, budget-limited labs that produce most of science. A working lab is a *team with a structure*, and
that structure exists precisely because no single mind, however capable, is the cheapest way to do every
job. SpecEvo asks what that structure looks like when the lab is poor — few experts, little compute — and
builds the framework as a mechanical scale-model of it, with three roles tied together by a shared lab
notebook.

The first lesson of a low-resource lab is that *one expert cannot also be every student*. Staffing the
problem with only a frontier model forces the two failure modes a one-person lab would suffer: work
sequentially, and the search is slow to find its footing under a budget — each step re-derives context the
loop never recorded; parallelize by putting a professor at every bench, and the cost explodes. So SpecEvo
separates the jobs by who can afford to do them. Breadth — running many directions at once — is what cheap,
fast models (the juniors, our **Speculator**) are good at; depth — the rare step that genuinely needs
frontier-scale reasoning — is reserved for the expensive model (the PI, our **Navigator**), woken
sequentially so it never throttles the bench.

But cheap juniors only help if their work is *used*, and discovery runs on collective evidence, not on one
mind's memory. A loop of `improve → improve → improve` keeps the running best and discards the rest;
a real lab keeps a notebook and reads across the whole body of work — successes and failures alike — before
deciding what to try next. SpecEvo's archive is that notebook: the Speculator's attempts are valued as much
for the evidence they leave behind as for the wins they score, and when the Navigator wakes it does not
stare at a single program but reads the *map* of everything tried — which directions are occupied, which
are paying off, how stalled the search has become — and proposes a grounded hypothesis about where to go.

And juniors must be *mentored*, not just told to work harder. Left under a generic "improve this" prompt, a
cheap model collapses to a few familiar patterns — the student who keeps re-running the one protocol they
know — so SpecEvo assigns varied work through a set of distinct exploration operators that force breadth
back in. More importantly, every result is information, even the failures: a crash says what to avoid, an
exhausted region says where to stop digging, a winning program says what idea is paying off. A scoreboard
throws all of this away; the senior labmate does not. The **Advisor** reads the whole trajectory and hands
the juniors concrete, code-shaped guidance in plain language — what a `Δscore` alone can never say.

---

## Framework

![SpecEvo overall framework](image/Overall.png)

*The lab in one pass. An **Init phase** lays out a few diverse starting projects and a behaviorally
embedded population. The **Speculator** (the juniors — a cheap model) explores many directions in parallel
into an adaptive MAP-Elites archive that serves as the **lab notebook**; the **Navigator** (the PI — a
frontier model) wakes on stagnation to read the notebook and propose a directed move; and the **Advisor**
(the senior labmate) distills lessons from the whole trajectory and injects them back into the juniors'
prompts. The archive's best entry is returned as the **best code**.*

### Components

- **Init phase — laying out the starting projects.**
  The professor (frontier model) sketches a handful of *diverse seed programs* sequentially — each
  conditioned on the previous to push for novelty — and the juniors (cheap model) fan each seed into many
  parallel variants. Every program is encoded as a **hybrid behavior descriptor** — structural AST features
  concatenated with a PCA-reduced embedding of its natural-language description — so programs of equal
  fitness but different ideas occupy different niches.

- **The lab notebook — an adaptive, online-clustered MAP-Elites archive.**
  Niches are *discovered online* by KMeans over the learned descriptor and **periodically re-clustered** as
  the search moves, rather than fixed up front. Each niche keeps a single elite incumbent; a program is
  admitted only if it beats its niche's incumbent. This is the shared record every role reads and writes.

- **Speculator (the juniors) — cheap, massive, parallel exploration.**
  The low-cost model issues many proposals at once across the archive, driven by a set of distinct
  operators (focused fix, mechanism swap, structural/component crossover, analysis-guided mutation). A
  proposal is valuable not only when it wins, but because its outcome — admits, near-misses, recurring
  errors — becomes *evidence* for the Navigator and Advisor.

- **Navigator (the PI) — rare, expensive, hypothesis-driven direction.**
  The frontier model wakes only at fixed intervals, reads the cells' representatives like a PI reviewing the
  lab's results, and is routed by a stagnation signal to one of three move-classes of increasing intensity —
  **Synthesis** (hybridize nearby strong ideas), **Surgical** (precisely tune the champion), or **Reframe**
  (switch to a genuinely different strategy family) — with the counter-intuitive but data-driven rule that
  the *deepest* stagnation triggers a **Reframe**. A short *Strategy Log* of its last six visits keeps it
  from re-proposing dead-ends.

- **Advisor (the senior labmate) — mentorship in language.**
  Between the PI's rare visits, cheap deterministic counters label niches as **leading**, **improving**,
  **saturated**, or **under-explored** (by whether a niche's frontier moved and how heavily it is being
  mined — no absolute thresholds), and aggregate **recurring errors** into bounded, portable knowledge. An
  LLM verbalizes this map into concise advice injected back into the juniors' prompts.

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
