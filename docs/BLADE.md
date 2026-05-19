# BLADE

**Budget-Limited Adaptive Discovery for Resource-Efficient LLM Evolution**

> An LLM-driven evolutionary code search that stays cheap and recovers
> from partial failures. Every adaptive behaviour is driven by a single
> mathematically calibrated signal — the *posterior stagnation depth*.

---

## Table of Contents

1. [Problem Setting](#1-problem-setting)
2. [The Story Behind the Design](#2-the-story-behind-the-design)
3. [PPS — Posterior-Plateau Stagnation](#3-pps--posterior-plateau-stagnation)
4. [The Zipfian Rank Sampler — choosing the parent cell](#4-the-zipfian-rank-sampler--choosing-the-parent-cell)
5. [The Sampler-Model Bandit — choosing how to mutate](#5-the-sampler-model-bandit--choosing-how-to-mutate)
6. [Punctuated Equilibrium with Phase-Staged Prompts](#6-punctuated-equilibrium-with-phase-staged-prompts)
7. [Strategy History — Memory-Augmented Heavy Prompts](#7-strategy-history--memory-augmented-heavy-prompts)
8. [Code Error Repair](#8-code-error-repair)
9. [Adaptive Island Expansion](#9-adaptive-island-expansion)
10. [How the Pieces Fit Together](#10-how-the-pieces-fit-together)

---

## 1. Problem Setting

We optimize a black-box scalar objective over Python programs:

$$P^* = \arg\max_{P \in \mathcal{P}} f(P)$$

subject to a hard resource cap. Let $C(t)$ be dollars spent, $N(t)$ be
evaluations consumed, and $T(t)$ be wall-clock seconds elapsed at step
$t$. Define the *dominant budget ratio*

$$b(t) = \max\!\left(\frac{C(t)}{C_{\max}},\; \frac{N(t)}{N_{\max}},\; \frac{T(t)}{T_{\max}}\right) \in [0, 1]$$

where any of $C_{\max}, N_{\max}, T_{\max}$ may be $\infty$. The search
terminates at the first $t$ with $b(t) = 1$.

The framework maintains an *archive* $\mathcal{A}(t)$ realised as a
CVT-MAP-Elites tessellation of a behaviour space $\mathcal{B} \subset \mathbb{R}^d$
into $K(t)$ centroids $\{\mu_1, \ldots, \mu_K\}$. Each occupied cell
holds one **elite** — the highest-scoring program whose behaviour
vector $\phi(P) \in \mathcal{B}$ falls in that cell. Write

$$f^*(t) = \max_{P \in \mathcal{A}(t)} f(P)$$

for the running best score, and let $\tau_{\text{new}}(t)$ be the
evaluation index at which $f^*(t)$ was first reached.

Each step the framework spends one LLM call to produce a candidate
$P'$, one evaluation to compute $f(P')$, and one archive update.

---

## 2. The Story Behind the Design

Pure MAP-Elites + LLM mutation has three pathologies:

**Pathology A — uncalibrated stagnation signals.**
The naive plateau ratio $p(t) = (N(t) - \tau_{\text{new}}(t))/\tau$ and
the naive budget ratio $b(t)$ are *statistically wrong* as stagnation
proxies. At $t = 0$ with a fixed evaluation budget, $b(t)$ is already
positive even though no information has been collected. Late in the
budget, $b(t) \to 1$ even when the run is genuinely improving.

**Pathology B — fragmented sampling, twice.**
LLM-driven evolution actually has *two* sampling problems, not one.
First: **which parent cell** do we mutate? Softmax-over-scores fails
when score distributions are heavy-tailed and forces a brittle
temperature fan-out. Second: **how do we mutate it** — which model,
which prompt, which generation temperature? Without a bandit over this
choice, every call uses an arbitrary fixed configuration and the
system never learns which generative regime actually produces accepts
on *this* problem.

**Pathology C — wasted work.**
A failed LLM call costs the same as a successful one. Every syntax-error
candidate, every paradigm shift the heavy model has *already tried*, and
every novel-but-slightly-worse paradigm gets discarded. The expected
cost-per-improvement grows monotonically as the run progresses.

BLADE addresses these with **six core mechanisms** plus one shared
calibrated signal $s(t)$:

| Pathology | Mechanism | Section |
| --- | --- | --- |
| A — uncalibrated stagnation | PPS — Posterior-Plateau Stagnation $s(t)$ | §3 |
| B — fragmented parent sampling | Zipfian rank sampler with $\beta(t) = f(s(t))$ | §4 |
| B — fragmented mutation policy | Sampler-Model bandit (Thompson + NEW BEST bonus) | §5 |
| Heavy-model myopia | Phase-staged PE prompts + Strategy History | §6, §7 |
| C — wasted broken candidates | Code Error Repair (one-shot light fix) | §8 |
| C — archive saturation | Adaptive Island Expansion | §9 |

The story is: **one calibrated signal, six mechanisms it drives**.

---

## 3. PPS — Posterior-Plateau Stagnation

### 3.1 The naive baseline

The legacy stagnation signal is

$$s_{\text{legacy}}(t) = \max\bigl(p(t),\; b(t)\bigr)$$

with $p(t) = \min(1,\; (N(t) - \tau_{\text{new}}(t))/\tau)$ and $b(t)$
as in §1. We want a signal with two calibration properties:

- **Late + improving** ⇒ $s(t)$ stays low. A long run that keeps
  producing NEW BEST events should not trigger panic mechanisms just
  because $b(t) \to 1$.
- **Late + truly stuck** ⇒ $s(t) \to 1$. A long run that has stopped
  improving should reliably trigger paradigm shifts and island
  expansions.

The legacy max-formula has neither property.

### 3.2 Setup — modelling NEW BEST as a Poisson process

Let $B(t)$ be the cumulative budget consumed (in whichever unit
dominates $b(t)$ — dollars, evals, or seconds). Treat each strict NEW
BEST event as an arrival of a non-homogeneous Poisson process on the
$B$-axis with unknown rate $\lambda(B)$.

Maintain a bounded sliding history

$$\mathcal{H}(t) = \{(u_i, B_i)\}_{i=1}^{k_W}$$

where $u_i$ is the eval index at which the $i$-th tracked NEW BEST
occurred and $B_i$ is the cumulative budget consumed at that moment.
$k_W = |\mathcal{H}(t)|$ is bounded ($\leq 32$).

Let

- $B_W(t) = B(t) - B_1$ — budget consumed since the window's oldest
  tracked NEW BEST;
- $B_{\text{total}}$ — the total budget cap on the dominant axis;
- $B_{\text{rem}}(t) = B_{\text{total}} - B(t)$ — budget remaining.

### 3.3 Derivation — Laplace-smoothed hazard estimate

Under a Poisson model with constant rate $\lambda$ over the window, the
maximum-likelihood estimator is

$$\hat{\lambda}_{\text{MLE}} = \frac{k_W}{B_W}$$

which collapses to $0$ when no NEW BEST has been observed yet
($k_W = 0$), making the survival probability degenerate at $1$. We
regularize via Laplace smoothing with a $\text{Gamma}(1, B_W)$
conjugate prior on $\lambda$; the posterior mean is

$$\hat{\lambda}(t) = \frac{k_W + 1}{B_W + \varepsilon}, \qquad \varepsilon = \max(10^{-9},\; 10^{-3}\,B_{\text{total}})$$

The $\varepsilon$ floor is numerical safety against $B_W = 0$ at run
start. The "+1" ensures $\hat{\lambda}(t) > 0$ even with $k_W = 0$.

The probability that *no further NEW BEST appears* in the remaining
budget, conditional on the constant-rate posterior, is the survival
function of an exponential inter-arrival distribution:

$$S(t) = \mathbb{P}(\text{no NEW BEST in } B_{\text{rem}}) = \exp\!\bigl(-\hat{\lambda}(t)\,B_{\text{rem}}(t)\bigr) \in (0, 1]$$

### 3.4 Derivation — the posterior-stuck score

The plateau term $p(t)$ acts as a *gate*: if we are not even on a
plateau (i.e., we just observed a NEW BEST), then "stuck" has no
weight. The gated posterior is

$$\Pi(t) := p(t) \cdot S(t)$$

This already has the right qualitative behaviour:

- Improving fast ⇒ $\hat\lambda$ large ⇒ $S \to 0$ ⇒ $\Pi \to 0$.
- Stuck for long ⇒ $p \to 1$, $S$ also large (rate is small) ⇒ $\Pi \to 1$.

But $\hat{\lambda}(t)$ is a *noisy* estimator early in the run. We
should not trust it when the window contains few events and little
budget. Define a **confidence weight** that grows quadratically in
budget consumed:

$$\alpha(t) = b(t)^2 \in [0, 1]$$

The quadratic rather than linear shape means confidence stays low for
the first third of the budget and rises sharply only when both event
count and budget are non-trivial.

The final stagnation signal is the convex blend of the two terms:

$$s(t) = \bigl(1 - \alpha(t)\bigr)\,p(t) \;+\; \alpha(t)\,\Pi(t)$$

### 3.5 Safety floor

At the very end of the run, $\hat{\lambda}$ can still produce small
$S(t)$ if the window happens to contain a recent improvement, even
when $p(t)$ is saturated. We do not want $s(t)$ to *under-report*
a strict end-of-run plateau, so we clamp:

$$s(t) \leftarrow \max\bigl(s(t),\; p(t)\bigr) \quad \text{whenever} \quad b(t) \geq 0.95 \;\wedge\; p(t) \geq 0.95$$

### 3.6 Putting it together

$$\boxed{\;
\begin{aligned}
p(t) &= \min\!\left(1,\; \frac{N(t) - \tau_{\text{new}}(t)}{\tau}\right) \\[4pt]
\hat{\lambda}(t) &= \frac{k_W + 1}{B_W + \varepsilon} \\[4pt]
S(t) &= \exp\!\bigl(-\hat{\lambda}(t)\,B_{\text{rem}}(t)\bigr) \\[4pt]
\Pi(t) &= p(t) \cdot S(t) \\[4pt]
\alpha(t) &= b(t)^2 \\[4pt]
s(t) &= (1 - \alpha)\,p + \alpha\,\Pi \quad\text{(then apply floor)}
\end{aligned}
\;}$$

### 3.7 Properties

| Regime | Behaviour |
| --- | --- |
| Early run, $\alpha \approx 0$ | $s(t) \approx p(t)$ — pure plateau term. |
| Late + improving, $\hat\lambda B_{\text{rem}}$ large | $S \to 0$, $\Pi \to 0$, $s \to (1-\alpha)p$. Plateau alone, no panic. |
| Late + stuck, $\hat\lambda B_{\text{rem}}$ small | $S \to 1$, $\Pi \to p$, $s \to p$. |
| End of run, all caps saturate | Safety floor → $s \geq p$. |
| $\beta \to 0$ in §4 below | $s$ has no effect on sampler. |

### 3.8 Take-away

$s(t)$ is a **calibrated posterior** on the event "no further NEW BEST
in the remaining budget", gated by the plateau term and weighted by
confidence in the rate estimate. It is one number; every downstream
adaptive mechanism reads from this one number.

---

## 4. The Zipfian Rank Sampler — choosing the parent cell

### 4.1 Motivation

Parent selection should be **score-scale invariant** — independent of
the absolute magnitudes of scores, depending only on the order of
elites. Softmax sampling

$$P_{\text{softmax}}(c) \propto \exp(f_c / T)$$

fails this: when scores have a heavy tail, the softmax collapses to a
near-degenerate point mass on the global best regardless of $T$. Worse,
to cover the explore/exploit spectrum, one fans out across multiple
temperatures, which divides any downstream bandit posterior across
arms.

### 4.2 Setup

Sort the occupied cells by primary score in descending order. Let
$r(c) \in \{0, 1, 2, \ldots\}$ be the 0-indexed rank of cell $c$.
Define a **stagnation-driven Zipfian exponent**:

$$\beta(t) = \max\bigl(\beta_{\min},\; \beta_{\max}(1 - s(t))\bigr), \qquad (\beta_{\min}, \beta_{\max}) = (0.2,\; 2.0)$$

### 4.3 The sampler

Cell $c$ is sampled with probability

$$P(c \mid t) = \frac{(r(c) + 1)^{-\beta(t)}}{\displaystyle\sum_{c'} (r(c') + 1)^{-\beta(t)}}$$

The denominator is a finite Zipf-type partition function over the
currently occupied cells.

### 4.4 Limit behaviour

| Regime | $s(t)$ | $\beta(t)$ | Distribution |
| --- | --- | --- | --- |
| Not stuck | $\to 0$ | $\to \beta_{\max} = 2.0$ | Heavy peak on top ranks (exploit). |
| Stuck | $\to 1$ | $\to \beta_{\min} = 0.2$ | Nearly uniform (explore). |
| Hypothetical | — | $\to 0$ | Exactly uniform. |
| Hypothetical | — | $\to \infty$ | Argmax (deterministic best). |

**Score-scale invariance.** Substituting $f_c \mapsto a\,f_c + b$ for
any $a > 0$ preserves the ranks $r(c)$, hence preserves $P(c \mid t)$.

### 4.5 Take-away

One knob — the stagnation depth $s(t)$ — collapses the entire
explore/exploit spectrum into a single Zipfian exponent. No
temperature fan-out, no score-scale fragility.

---

## 5. The Sampler-Model Bandit — choosing how to mutate

### 5.1 The second sampling problem

§4 chose **which parent cell** to mutate. There is a second, orthogonal
question: given the chosen parent, **which LLM should produce the
offspring, with which prompt template, and at which temperature?**

A single mutation call is parameterised by four discrete choices:

$$
\text{call config} = (\text{sampler},\;\text{model},\;\text{prompt\_id},\;\text{llm\_temperature})
$$

where the four entries are, respectively: the **parent rule**, **which
LLM is called**, **which prompt template is rendered**, and the
**generation temperature**.

The four dimensions trade off independently:

- **sampler** — the parent-selection rule from §4 (or a subscore-aware
  variant). Cheap to swap, no LLM cost difference.
- **model** — `qwen3-30b` (light, ~$0.001/call) vs `gpt-5` (heavy,
  ~$0.10/call). Two orders of magnitude in cost.
- **prompt_id** — a template from the prompt bank, e.g.
  `surgical_local_refine`, `borrow_from_inspiration`,
  `aggressive_rewrite`, `tune_hyperparameters`, `diversify_initialization`.
  Each conditions the model on a different *mutation operator*.
- **llm_temperature** — sampling temperature of the generation call,
  e.g. $\{0.5, 0.8, 1.1\}$. Low → conservative; high → diverse.

The full **arm space** is the Cartesian product:

$$
\mathcal{A}_{\text{bandit}} = \mathcal{S} \times \mathcal{M} \times \mathcal{T} \times \mathcal{L}
$$

where $\mathcal{S}, \mathcal{M}, \mathcal{T}, \mathcal{L}$ are the sets
of available samplers, models, prompt templates, and temperatures. A
typical configuration has $|\mathcal{S}| = 1$, $|\mathcal{M}| = 1$,
$|\mathcal{T}| = 5$, $|\mathcal{L}| = 3$, giving $|\mathcal{A}_{\text{bandit}}| = 15$
arms.

### 5.2 Why a bandit

We don't know *a priori* which arm produces the most accepts on this
problem at this stagnation depth. The right algorithmic family — and
the right temperature regime — depends on the score landscape, which
we only learn through observation.

This is a **multi-armed bandit** problem under a stationary reward:
which arm has the highest accept rate? Three properties dictate the
choice of algorithm:

1. **Cheap to play every arm** — every call is bounded by the budget.
2. **Reward is binary** — the candidate is either accepted into the
   archive or not.
3. **No prior on which arm is best** — the bank composition can
   change run-to-run.

This combination points directly to **Thompson sampling on a
Beta-Bernoulli posterior**.

### 5.3 The Beta-Bernoulli posterior

For each arm $i \in \mathcal{A}_{\text{bandit}}$, maintain two
sufficient statistics:

$$
\alpha_i = (\text{accepts on arm } i) + 1, \qquad \beta_i = (\text{rejects on arm } i) + 1
$$

The "+1" is a $\text{Beta}(1,1)$ uniform prior on the accept rate
$\theta_i$. After observing accepts and rejects, the posterior is

$$
\theta_i \mid \text{data} \;\sim\; \text{Beta}(\alpha_i,\;\beta_i)
$$

with posterior mean $\mathbb{E}[\theta_i] = \alpha_i / (\alpha_i + \beta_i)$
and variance shrinking as $1/(\alpha_i + \beta_i)$.

### 5.4 The NEW BEST bonus

Accept rate alone is an undersold signal — most accepts are *marginal*
improvements that fill in cells. A NEW BEST is qualitatively different:
it advances the run-best. We want arms that produce NEW BEST events to
be selected disproportionately.

For each arm $i$, track a counter $n_i \in \mathbb{Z}_{\geq 0}$ of NEW
BEST events produced by that arm. Multiply the posterior sample by

$$
\text{bonus}_i = (1 + \gamma\,n_i)^{1 + s(t)}, \qquad \gamma = 0.5
$$

Two properties of this shape:

- **Bonus grows linearly inside, polynomially outside.** A single NEW
  BEST gives factor $1.5^{1+s}$; ten NEW BEST give $6^{1+s}$.
- **Bonus exponent depends on stagnation.** When the run is fresh
  ($s \to 0$), $\text{bonus} = 1 + \gamma n$ — linear. When stuck
  ($s \to 1$), $\text{bonus} = (1 + \gamma n)^2$ — squared, sharpening
  the preference for past winners.

The stagnation-dependent exponent is the bridge from PPS into the
bandit. When the run is healthy, exploration of un-tried arms is
worthwhile; when stuck, doubling down on what has historically broken
plateaus is more valuable.

### 5.5 Thompson sampling with a probability floor

At each producer step, sample one $\theta_i$ from each arm's posterior
and form the raw scores:

$$
\text{raw}_i = \theta_i \cdot \text{bonus}_i = \theta_i \cdot (1 + \gamma\,n_i)^{1 + s(t)}, \qquad \theta_i \sim \text{Beta}(\alpha_i, \beta_i)
$$

Normalize to a probability distribution:

$$
\tilde{p}_i = \frac{\text{raw}_i}{\sum_j \text{raw}_j}
$$

A pure Thompson scheme would now sample $i$ with probability $\tilde{p}_i$
and stop. But Beta posteriors can be *very* sharp once $\alpha_i + \beta_i$
is large, and the bonus can compound that further — risking complete
collapse of $\tilde{p}_i$ onto one arm even when other arms might still
become useful at a future stagnation depth.

We mix the Thompson distribution with a uniform floor:

$$
\boxed{\;
w_i = w_{\min} + (1 - N \cdot w_{\min})\,\tilde{p}_i
\;}
$$

where $N = |\mathcal{A}_{\text{bandit}}|$ and $w_{\min} = 0.05$. Two
properties:

- **Floor preserved.** $w_i \geq w_{\min}$ for every $i$ — every arm is
  always playable.
- **Mass conserved.** $\sum_i w_i = N w_{\min} + (1 - N w_{\min}) \sum_i \tilde{p}_i = N w_{\min} + (1 - N w_{\min}) = 1$.

The arm is finally drawn as $i \sim \text{Categorical}(w)$.

### 5.6 Why the floor matters

Without the floor, Thompson + multiplicative bonus collapses too
aggressively. Consider an arm $i^*$ that produces a single NEW BEST in
the first 20 evaluations. Its bonus jumps to $1.5^{1 + 0.1} \approx 1.55$;
its posterior mean is around $0.5$ (one accept, no rejects yet).
Competing arms with mean $0.3$ and no NEW BEST get $\text{raw} = 0.3$;
arm $i^*$ gets $\approx 0.775$. After normalization, $i^*$ already has
$\sim 60\%$ probability — and *every NEW BEST event compounds*.

With $w_{\min} = 0.05$ and $N = 15$ arms, $i^*$ is capped at
$0.05 + 0.95 \cdot 0.60 = 0.62$; the other 14 arms collectively retain
$\geq 0.05 \cdot 14 = 0.70 - 0.62 = 0.38$ of the mass. They keep
exploring and updating their posteriors.

### 5.7 Posterior updates

After the evaluation of the offspring produced by arm $i$:

$$
\begin{aligned}
\text{if accepted: }   \quad &\alpha_i \mathrel{+}= 1 \\
\text{if rejected: }   \quad &\beta_i  \mathrel{+}= 1 \\
\text{if NEW BEST: }   \quad &n_i \mathrel{+}= 1
\end{aligned}
$$

Note that NEW BEST is *additionally* an accept, so $\alpha_i$ also
increments. The two signals carry independent information: $\alpha_i$
counts accept events, $n_i$ counts the strict subset of those that
advanced the run-best.

### 5.8 Why four dimensions, not one or two

A natural question: why parameterise mutation as a 4-tuple instead of
folding `prompt_id` and `llm_temperature` into the `model` choice or
the `sampler`?

The answer is **posterior fragmentation vs. expressive capacity**:

- One arm per model: too coarse. The same model can be a great
  surgical refiner with `surgical_local_refine` at $T = 0.5$ but a
  terrible aggressive rewriter at $T = 1.1$. Conflating them into one
  arm averages out the signal.
- One arm per (model, prompt): better, but ignores temperature. Two
  candidates from the same `(model, prompt)` at different $T$ have
  fundamentally different acceptance distributions.
- One arm per 4-tuple: maximally expressive. Each arm targets a
  *specific generative regime*. Posterior shrinkage is the only price
  paid — and the floor in §5.5 plus the NEW BEST bonus in §5.4 control
  it.

The legacy framework collapsed the "sampler" dimension to a softmax
temperature on raw scores ($\mathcal{S}$ contained four arms per
sampler). After §4 introduced the stagnation-driven Zipfian sampler,
$|\mathcal{S}|$ dropped to one, and the freed posterior mass moved to
the more informative `(prompt_id, llm_temperature)` axes — where it
actually changes the offspring's generative behaviour.

### 5.9 Take-away

Thompson sampling on a Beta-Bernoulli posterior with a NEW BEST
multiplicative bonus and a probability floor gives an adaptive,
stagnation-aware policy over the
$(\text{sampler}, \text{model}, \text{prompt\_id}, \text{llm\_temperature})$
arm space. Each LLM call is now a deliberate choice of generative
regime, not a uniform draw, and the policy gets sharper as the run
collects more evidence.

---

## 6. Punctuated Equilibrium with Phase-Staged Prompts

### 5.1 The PE event

Every $\nu = 10$ evaluations the framework fires a **Punctuated
Equilibrium** event: a heavy-model call designed to inject a
fundamentally different solution into the archive.

The mechanics:

1. **Cluster** the occupied cells into $k = 3$ groups via $k$-means
   over their centroid vectors $\{\mu_c\}$.
2. **Select representatives** — pick the highest-score elite from each
   cluster, giving three anchors.
3. **Generate paradigm** — call the heavy model with the three anchors
   and a stage-dependent prompt.
4. **Generate variants** — call the light model with the accepted
   paradigm as parent.
5. **Update archive** — add paradigm and variants, applying
   §8 (island expansion) when needed.

### 5.2 The phase router

The stagnation depth $s(t)$ routes the heavy call to one of three
distinct prompt templates:

$$\text{stage}(t) = \begin{cases}
\text{early} & s(t) < \sigma_{\text{mid}} \\
\text{mid}   & \sigma_{\text{mid}} \leq s(t) < \sigma_{\text{late}} \\
\text{late}  & s(t) \geq \sigma_{\text{late}}
\end{cases} \qquad (\sigma_{\text{mid}}, \sigma_{\text{late}}) = (0.35,\; 0.70)$$

with templates:

- **early** — "pick a paradigm class **not represented** in the
  archive". The model is told to diversify.
- **mid** — "**synthesize** strengths from each cluster; fix one
  weakness". The model is told to combine.
- **late** — "**targeted refinement** on the best solution; do not
  rewrite". The model is told to polish.

These are written as different tasks, not the same task with
cosmetic differences. The information density of each prompt is
calibrated to what the search actually needs at that stagnation
depth.

### 5.3 Farthest-first selection under deep stagnation

When $s(t) \geq 0.85$ and two consecutive PE events have produced no
NEW BEST, we trigger **Hard-PE**. Representative selection then
switches from "highest score" to *farthest-first* in behaviour space:

$$e_{\mathcal{C}_j} = \arg\max_{e \in \mathcal{C}_j} \bigl\lVert \phi(e) - \bar{\phi}_{j-1} \bigr\rVert$$

where $\bar{\phi}_{j-1}$ is the mean behaviour vector of previously
picked representatives. This diversifies the heavy model's context
away from score-similar anchors that have already failed to produce a
fresh paradigm.

### 5.4 Take-away

PE is not "one paradigm-shift prompt". It is a stage-routed
intervention whose information content adapts to where the search
currently is on the stagnation curve.

---

## 7. Strategy History — Memory-Augmented Heavy Prompts

### 6.1 The cost of forgetting

Without memory, the heavy model's $i$-th paradigm proposal is
conditionally independent of its $(i-1)$-th given the current archive.
The expected progress from PE event $i$ is

$$\mathbb{E}[\Delta f^*_i] = \mathbb{P}(\text{novel}_i) \cdot \mathbb{E}[\Delta f^* \mid \text{novel}] + \mathbb{P}(\text{rehash}_i) \cdot \mathbb{E}[\Delta f^* \mid \text{rehash}]$$

The rehash term is approximately zero — the paradigm has already been
rejected at this score level. Reducing $\mathbb{P}(\text{rehash}_i)$
directly increases expected progress.

If the search has $M$ "distinct usable paradigms" for the problem and
the heavy model samples i.i.d. uniformly without memory, then

$$\mathbb{P}(\text{rehash}_i \mid \text{no memory}) = 1 - \left(\frac{M-1}{M}\right)^{i-1}$$

For $M = 6$ and $i = 5$, that is already $58\%$. By the late phase of
a typical run, more than half of PE calls without memory rehash a
previous approach.

### 6.2 The two-line summary

After each PE event, a **light** model produces a strictly two-line
summary of the paradigm-shift code:

```
IDEA:    <algorithmic family + concrete tactic used here>
QUALITY: <where it should win + the most likely failure mode>
```

Total length $\leq 60$ words. The format is enforced by the prompt;
the light call costs $\leq 150$ output tokens.

### 6.3 The strategy record

For each PE event $i$ we store a record

$$r_i = \bigl(\text{event\_id}_i,\; \text{stage}_i,\; \text{summary}_i,\; f^*_{i-1},\; f_{\text{paradigm},i},\; \Delta_i,\; \text{accepted}_i\bigr)$$

where $\Delta_i = f^*_i - f^*_{i-1}$ is the run-best delta induced by
the entire PE event (paradigm + variants). Records are kept in a
bounded FIFO of capacity $12$.

### 6.4 Injection into the next heavy prompt

The next heavy prompt is pre-pended with the last $8$ records,
rendered as:

```
## Strategy Log (already tried in this run)

### PE #5 [mid] — Δ=+0.012, score=2.604, accepted
  IDEA:    Simulated annealing on overlap-penalty energy …
  QUALITY: Strong when radii are diverse; weak under symmetric configs.

### PE #6 [late] — Δ=-0.000, score=2.601, rejected
  IDEA:    Gradient ascent with KKT projection …
  …

Do NOT propose an approach whose summary appears above with Δ ≤ 0.
Aim for something structurally different from every rejected entry.
```

The heavy model can perform mechanical string-similarity checks
against prior `IDEA:` lines *before* generating any output.

### 6.5 Take-away

A 60-word summary per PE buys long-term memory for the heavy model at
$\leq 0.04\%$ of the typical $\$3$ run budget. The expected reduction
in rehash probability directly increases expected progress per PE
event.

---

## 8. Code Error Repair

### 7.1 The expected-value argument

Every candidate that fails evaluation (syntax error, runtime
exception, score parse error) has already cost one LLM call
$c_{\text{parent}}$. Discarding the broken code forfeits that sunk
cost. A *small* light-model repair call costs $c_{\text{light}} \ll c_{\text{parent}}$,
so its expected value is

$$\mathbb{E}[V_{\text{repair}}] = \mathbb{P}(\text{fix succeeds}) \cdot f_{\text{parent}} \;-\; c_{\text{light}}$$

Even at $\mathbb{P}(\text{fix succeeds}) = 0.05$, the repair pays for
itself when $f_{\text{parent}}$ is in the upper half of the archive
(because $c_{\text{light}}$ is one to two orders of magnitude smaller
than $f_{\text{parent}}$ in cost-per-unit-score terms).

### 7.2 The error buffer

Failed candidates are pushed into a bounded deque $\mathcal{E}(t)$ of
size $64$, deduplicated by the first 200 characters of the code (so an
avalanche of identical errors does not crowd out informative variety).
Each record carries

$$\text{ErrorRecord} = \bigl(P_{\text{code}},\; f_{\text{parent}},\; \epsilon,\; c_{\text{cell}}\bigr)$$

where $\epsilon$ is the (tail-truncated) error message and
$c_{\text{cell}}$ is the parent's cell index.

### 7.3 Rank-by-parent-score selection

When a repair fires (see §8.4), we select one record from $\mathcal{E}(t)$
by **Zipfian rank over parent scores**. Sort the buffer by
$f_{\text{parent}}$ in descending order, with rank $r$ for the $r$-th
entry. The chosen record has

$$P_{\text{repair}}(r) = \frac{(r + 1)^{-\beta_{\text{rep}}}}{\displaystyle\sum_{r'} (r' + 1)^{-\beta_{\text{rep}}}}, \qquad \beta_{\text{rep}} = 1.5$$

Higher $\beta_{\text{rep}}$ favours rescuing near-elite parents — the
intuition being that a near-elite with a fixable bug has higher
expected post-fix score than a random low-scoring sibling. The chosen
record is **removed from the buffer** (one-shot — failed repairs are
not retried).

### 7.4 The fire gate

A repair attempt fires at producer step $t$ iff **all** of the
following hold:

$$\text{enabled} \;\wedge\; |\mathcal{E}(t)| > 0 \;\wedge\; n_{\text{repair}}(t) < n_{\max} \;\wedge\; \bigl(N(t) - N_{\text{last\_repair}}\bigr) \geq \nu_{\text{repair}}$$

with defaults $\nu_{\text{repair}} = 8$ and $n_{\max} = 100$. The
schedule guarantees at most one repair per $\nu_{\text{repair}}$
main-loop offspring and at most $n_{\max}$ total.

### 7.5 Take-away

Each broken candidate is treated as a sunk-cost asset that a cheap
light-model call can rehabilitate. The Zipfian rank-by-parent-score
biases the rehabilitation budget toward near-elite errors where the
expected post-fix value is highest.

---

## 9. Adaptive Island Expansion

### 8.1 The saturation problem

CVT-MAP-Elites tessellates $\mathcal{B}$ into $K$ centroids at
initialization. Each candidate $P'$ is admitted only if it beats the
incumbent in its nearest cell:

$$\text{admit}(P') \iff f(P') > f\!\left(\mathcal{A}(t)\!\left[\,\arg\min_k \lVert \phi(P') - \mu_k\rVert\right]\right)$$

Once every cell is occupied with a strong elite, the strict-better
rule biases the search toward marginal improvements on existing
behaviour modes. A genuinely novel paradigm that scores **slightly
worse** than its nearest incumbent is rejected — and the archive
never gets to host the new behaviour for future variants to mutate.

### 8.2 The expansion rule

When a PE candidate $P'$ fails standard admission **and** stagnation
is high, BLADE opens a new cell at the candidate's own behaviour
vector instead of evicting any incumbent. The rule:

$$\boxed{\;
\text{expand}(P') \iff
\begin{cases}
\text{admit}(P') = \text{False} \\
s(t) \geq \sigma_{\text{island}} = 0.7 \\
n_{\text{island}}(t) < n_{\text{island,max}} = 16 \\
K(t) < K_{\max} = 200
\end{cases}
\;}$$

When all four hold, append $\phi(P')$ to the centroid set:

$$\mu_{K(t)+1} \leftarrow \phi(P'), \qquad K(t+1) = K(t) + 1$$

and seed the new cell with $P'$. The old cell and its incumbent are
unchanged.

### 8.3 Why "non-destructive"

Two desirable invariants follow immediately:

- **No archive corruption.** Existing elites are never overwritten by
  weaker candidates.
- **Bounded growth.** $K(t) \leq K_{\max}$ and $n_{\text{island}}(t) \leq n_{\text{island,max}}$,
  so a bad PE batch cannot inflate the archive without bound.

### 8.4 Why "adaptive"

The expansion gate uses the *same* signal $s(t)$ as the PE phase
router and the Zipfian sampler. Expansion fires only when

- the candidate genuinely failed admission (it is novel by
  behaviour), and
- the run is stuck enough that loosening admission is justified.

These conditions occur together precisely when *opening a new island
in behaviour space* is the right move, and not before.

### 8.5 Comparison with prior work

| Prior approach | Cost | Effect |
| --- | --- | --- |
| **AdaEvolve** — periodic re-clustering on buffered behaviours | $O(K\,d)$ per call to $k$-means + buffer maintenance | $K$ may shrink or grow; requires cooldown gate. |
| **CMA-ME** relaxed admission | Constant overhead | Incumbents can be evicted by *worse* candidates; destructive. |
| **BLADE Island Expansion** | $O(d)$ per expansion (one append) | $K$ monotone non-decreasing; incumbents preserved. |

BLADE is the cheapest principled archive growth that is both
non-destructive and stagnation-gated.

### 8.6 Take-away

A novel-but-marginally-worse PE candidate opens a brand-new cell at
its own behaviour vector. The archive grows organically along the
exact axes where the search proved diverse, never along axes the heavy
model never explored.

---

## 10. How the Pieces Fit Together

The six mechanisms above are **not independent fixes stitched
together** — they are a coordinated response to a single signal.

### 10.1 The signal flow

```
                  ┌────────────────────────────┐
                  │  NEW BEST events history   │
                  │       (last 32 events)     │
                  └────────────┬───────────────┘
                               │
                  Laplace-smoothed hazard λ̂(t)
                               │
                               ▼
                  ┌────────────────────────────┐
                  │   PPS signal s(t) ∈ [0,1]  │  §3
                  └──┬──────┬──────┬──────┬────┘
                     │      │      │      │
       ┌─────────────┘      │      │      └──────────────┐
       ▼                    ▼      ▼                     ▼
┌─────────────┐   ┌─────────────────┐   ┌──────────────┐   ┌──────────────────┐
│  β(t) for   │   │  Bandit bonus   │   │  PE phase    │   │  Island          │
│  Zipfian    │§4 │  exponent       │§5 │  router      │§6 │  expansion gate  │§9
│  sampler    │   │  (1+γn)^(1+s)   │   │  early/mid/  │   │  s(t) ≥ 0.7      │
│  picks cell │   │  Thompson pick  │   │  late + Hard │   │                  │
└─────────────┘   └─────────────────┘   └──────────────┘   └──────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────┐
                  │  Strategy Log injection    │  §7
                  │  into heavy prompt         │
                  └────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────┐
                  │  Code Repair fires every   │  §8
                  │  ν_repair offspring        │
                  └────────────────────────────┘
```

§4 picks **which cell** the parent comes from; §5 picks **how to
mutate** it (model + prompt + temperature). Together they replace what
was previously a single brittle softmax knob with two principled,
stagnation-aware policies.

### 10.2 What each mechanism contributes per LLM call

| Mechanism | Cost per fire | Frequency | Cumulative cost share |
| --- | --- | --- | --- |
| PPS evaluation | $O(1)$ in-process | every read | $\approx 0$ |
| Zipfian parent sampler | $O(K \log K)$ in-process | every producer step | $\approx 0$ |
| Sampler-Model bandit pick | $O(N)$ in-process | every producer step | $\approx 0$ |
| Bandit posterior update | $O(1)$ in-process | every evaluation | $\approx 0$ |
| PE (heavy + variants) | 1 heavy + 3 light calls | every $\nu = 10$ evals | dominates the bill |
| Strategy summarisation | 1 light call, $\leq 150$ tokens | every PE | $\leq 0.04\%$ of budget |
| Code repair | 1 light call | every $\nu_{\text{repair}} = 8$ broken-eligible steps | bounded by $n_{\max} = 100$ |
| Island expansion | $O(d)$ in-process | bounded by $n_{\text{island,max}} = 16$ | $\approx 0$ |

### 10.3 The minimal-set claim

The seven mechanisms address seven distinct failure modes:

| Failure mode | Mechanism |
| --- | --- |
| Uncalibrated stagnation → false alarms | PPS (§3) |
| Score-scale fragility in parent picks | Zipfian rank sampler (§4) |
| Uninformed choice of (model, prompt, temperature) | Sampler-Model bandit (§5) |
| Single PE prompt → wrong intervention at wrong time | Phase-staged PE (§6) |
| Heavy model proposes already-failed paradigms | Strategy History (§7) |
| Wasted LLM calls on syntax-trivial errors | Code Repair (§8) |
| Archive saturation → novel paradigms rejected | Island Expansion (§9) |

Each mechanism solves a problem the others do not — and removing any
one re-introduces its corresponding failure mode without affecting the
others. This is what makes the set minimal and non-redundant.

### 10.4 The headline

> **One calibrated signal. Six adaptive mechanisms — two for sampling,
> four for evolution. One non-destructive archive that grows where the
> search proves it must.**

That is BLADE.

---

## Notation Summary

| Symbol | Meaning |
| --- | --- |
| $\mathcal{P}$ | space of admissible Python programs |
| $f(P)$ | scalar objective function |
| $\mathcal{A}(t)$ | archive at step $t$ |
| $K(t)$ | number of centroids at step $t$ |
| $\mathcal{B} \subset \mathbb{R}^d$ | behaviour space |
| $\phi(P) \in \mathcal{B}$ | normalised behaviour vector of program $P$ |
| $\mu_k$ | $k$-th centroid in $\mathcal{B}$ |
| $f^*(t)$ | running best score |
| $\tau_{\text{new}}(t)$ | eval index of the current NEW BEST |
| $C(t), N(t), T(t)$ | dollars / evals / seconds consumed |
| $C_{\max}, N_{\max}, T_{\max}$ | hard caps on the three budget axes |
| $b(t)$ | dominant budget ratio, $\in [0, 1]$ |
| $\mathcal{H}(t)$ | bounded sliding window of NEW BEST events |
| $k_W$ | $\lvert\mathcal{H}(t)\rvert$, NEW BEST count in window |
| $B_W(t)$ | budget consumed since window's first event |
| $B_{\text{rem}}(t)$ | budget remaining on the dominant axis |
| $\hat{\lambda}(t)$ | Laplace-smoothed empirical NEW BEST rate |
| $S(t)$ | survival probability "no NEW BEST in $B_{\text{rem}}$" |
| $p(t)$ | plateau term $\min(1,(N - \tau_{\text{new}})/\tau)$ |
| $\Pi(t)$ | posterior-stuck $= p \cdot S$ |
| $\alpha(t)$ | confidence weight $= b(t)^2$ |
| $s(t)$ | **posterior stagnation depth**, $\in [0, 1]$ |
| $\tau$ | plateau saturation length (default 80) |
| $\beta(t)$ | Zipfian exponent for parent sampling |
| $\beta_{\min}, \beta_{\max}$ | Zipfian bounds $(0.2, 2.0)$ |
| $r(c)$ | 0-indexed rank of cell $c$ by score |
| $\nu$ | PE interval (default 10) |
| $\sigma_{\text{mid}}, \sigma_{\text{late}}$ | PE phase thresholds $(0.35, 0.70)$ |
| $\sigma_{\text{island}}$ | island-expansion stagnation threshold $(0.70)$ |
| $\mathcal{E}(t)$ | bounded error buffer (size 64) |
| $\beta_{\text{rep}}$ | repair-buffer Zipfian exponent $(1.5)$ |
| $\nu_{\text{repair}}, n_{\max}$ | repair fire interval and per-run cap $(8, 100)$ |
| $n_{\text{island,max}}, K_{\max}$ | island caps $(16, 200)$ |
| $\Delta_i$ | run-best delta from PE event $i$ |

---

*BLADE — Budget-Limited Adaptive Discovery for Resource-Efficient LLM Evolution.*
