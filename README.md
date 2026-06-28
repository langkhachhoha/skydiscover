# SpecEvo: Speculative Scientific Discovery

*A two-tier evolutionary framework for cost-efficient automated algorithm design with LLMs.*
**Core mechanism: Speculate-then-Escalate.**

---

## Abstract

Automated algorithm design with large language models is powerful but expensive,
largely because frontier models are invoked *uniformly* at every search step. We
observe that algorithmic discovery is **non-uniform in difficulty**: the vast
majority of improvement steps are local refinements — *normal science* — that a
cheap model can readily guess, while only a small minority are genuine
*paradigm shifts* that demand frontier-scale reasoning. **SpecEvo** exploits this
asymmetry. A cheap **Speculator** continuously proposes large batches of diverse
refinements, and an expensive **Navigator** is escalated *only* when cheap
speculation stalls — a principle we call **Speculate-then-Escalate**. The design
rests on two anchors: the *spirit* of speculative decoding (guess cheaply, pay
the expensive cost only when needed) and Kuhn's account of normal science
punctuated by rare paradigm shifts. Across mathematical-discovery and
system-engineering benchmarks, on both GPT-5 and Kimi-K2 backbones, SpecEvo
matches or exceeds the strongest baselines on most tasks while spending
**2.0–3.4× less** than the average baseline.

---

## Framework

![SpecEvo overall framework](image/Overall.png)

*SpecEvo pipeline. An **Init phase** seeds a diverse, behaviorally embedded
population; the **Speculator** (cheap model) explores many directions in
parallel into an adaptive MAP-Elites archive; the **Navigator** (frontier model)
is woken on stagnation to propose paradigm-scale moves; and the **Advisor**
distills population-level lessons that are injected back into the Speculator's
prompt. The archive's elites converge to the returned **best code**.*

### Components

- **Init Phase — seeding the paradigm map.**
  A frontier model writes a handful of *diverse paradigm seeds* sequentially
  (each conditioned on the previous to push for novelty); the cheap model then
  fans each seed into many parallel variants. Every program is encoded as a
  **hybrid behavior descriptor** — structural AST features concatenated with a
  PCA-reduced embedding of its natural-language description — so that programs of
  equal fitness but different ideas occupy different niches.

- **Population — adaptive re-clustered MAP-Elites.**
  Niches are *discovered online* by KMeans over the learned descriptor and
  **periodically re-clustered** as the search moves, rather than fixed up front
  (i.e., **not** CVT-MAP-Elites). Each niche keeps a single elite incumbent; a
  program is admitted only if it beats its niche's incumbent.

- **Speculator — cheap, massive, parallel *normal science*.**
  The low-cost model issues many speculative refinements at once across the
  archive. A proposal is valuable not only when it wins but also because its
  outcome (admits, near-misses, recurring errors) becomes *evidence* that steers
  the Navigator and Advisor.

- **Navigator — rare, expensive, directed escalation.**
  The frontier model wakes only at fixed intervals and reads cluster
  representatives. A stagnation signal routes it to one of three move-classes of
  increasing intensity — **Synthesis** (hybridize nearby strong ideas),
  **Surgical** (precisely tune the champion), or **Shift** (jump to a genuinely
  new algorithm class) — with the counter-intuitive but data-driven rule that the
  *deepest* stagnation triggers a paradigm **Shift**.

- **Advisor — population-level reflection.**
  At intervals, cheap deterministic counters label niches as **leading**,
  **improving**, **saturated**, or **under-explored**, and aggregate **recurring
  errors** into bounded, portable knowledge. An LLM verbalizes this niche map into
  concise advice that is injected back into the Speculator's prompt.

---

## Results

SpecEvo is evaluated against OpenEvolve, GEPA, AdaEvolve, and EvoX over three
seeds, reporting mean ± std and best.

**Mathematical-discovery benchmarks** (circle packing, Heilbronn, MinMax
distance, signal processing):

![Mathematical-discovery results](image/Math_result.png)

**System-engineering benchmarks** (EPLB, LLM-SQL, Transaction, PRISM):

![System-engineering results](image/System_result.png)

Across both suites and both backbones, SpecEvo attains the best or
near-best score on the majority of tasks, confirming that aggressive cheap
speculation with rare frontier escalation does not sacrifice solution quality.

---

## Cost

The advantage is in *cost*. For each method we sum per-task cost into a total per
seed, then report the mean ± std over three seeds. SpecEvo is consistently the
cheapest method by a wide margin, while the baselines cluster together near the
top.

| | GPT-5 | Kimi-K2 |
|---|---|---|
| **Math** | ![](image/math_gpt_cost.png) | ![](image/math_kimi_cost.png) |
| **System** | ![](image/system_gpt_cost.png) | ![](image/system_kimi_cost.png) |

Relative to the **average baseline**, SpecEvo is **3.4×** cheaper on math (GPT-5),
**2.0×** on math (Kimi-K2), **2.8×** on system (GPT-5), and **2.6×** on system
(Kimi-K2) — near-SOTA quality at a fraction of the cost.
