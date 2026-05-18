# BLADE

**Budget-Limited Adaptive Discovery for Resource-Efficient LLM Evolution**

> A principled, mathematically grounded framework for evolutionary code
> search under hard resource caps. Every adaptive component is driven by
> one calibrated signal — the **posterior stagnation depth** `s(t)` —
> and every provider call is wrapped in a self-throttling token-budget
> retry loop. The result is a search that stays cheap, recovers from
> partial failures, and refuses to waste a single LLM call.

---

## Table of Contents

1. [Problem setting and notation](#1-problem-setting-and-notation)
2. [The single signal — Posterior-Plateau Stagnation](#2-the-single-signal--posterior-plateau-stagnation)
3. [Parent selection — Stagnation-driven Zipfian sampler](#3-parent-selection--stagnation-driven-zipfian-sampler)
4. [Punctuated Equilibrium with phase-staged prompts](#4-punctuated-equilibrium-with-phase-staged-prompts)
5. [Strategy History — memory-augmented heavy prompts](#5-strategy-history--memory-augmented-heavy-prompts)
6. [Code Error Repair — one-shot rescue](#6-code-error-repair--one-shot-rescue)
7. [Adaptive Island Expansion — non-destructive archive growth](#7-adaptive-island-expansion--non-destructive-archive-growth)
8. [Budget-aware resilience — the BLADE retry calculus](#8-budget-aware-resilience--the-blade-retry-calculus)
9. [Elite-as-paradigm fallback](#9-elite-as-paradigm-fallback)
10. [Cost accounting and budget enforcement](#10-cost-accounting-and-budget-enforcement)
11. [Why this is the right minimal set](#11-why-this-is-the-right-minimal-set)

---

## 1. Problem setting and notation

We optimize a black-box scalar objective over Python programs:

$$
P^\* \;=\; \underset{P\,\in\,\mathcal{P}}{\arg\max}\;\; f(P)
$$

subject to a **resource cap**:

$$
g(t)\;:=\;\max\Bigl(\tfrac{C(t)}{C_{\max}},\,\tfrac{N(t)}{N_{\max}},\,\tfrac{T(t)}{T_{\max}}\Bigr)\;\le\;1,
$$

where $C, N, T$ are dollars spent, evaluations consumed, and wall-clock
seconds, and any of $C_{\max}, N_{\max}, T_{\max}$ may be $\infty$. The
search terminates at the first $t$ with $g(t) = 1$.

Let

- $\mathcal{A}(t) \subset \mathcal{P}$ be the **archive** at step $t$
  (a CVT-MAP-Elites tessellation of a behaviour space $\mathcal{B}\subset\mathbb{R}^d$
  into $K(t)$ centroids; $K(t)$ may grow — see §7);
- $f^\*(t) = \max_{P \in \mathcal{A}(t)} f(P)$ be the running best score;
- $\tau_{\text{new}}(t) = \min\{u \le t : f^\*(u) = f^\*(t)\}$ be the
  evaluation index at which the current best was first reached;
- $\mathcal{H}(t) = \{(u_i, C_i)\}_{i=1}^{k}$ be the bounded history of
  *strict* NEW BEST events as `(eval_count, cumulative_cost)` pairs,
  $k \le 32$.

A **call** to an LLM produces an offspring program $P'$ from a parent
selection rule. A call **succeeds** when (a) the provider returns text,
(b) text extraction yields valid code, (c) evaluation returns a finite
score. The cost of a failed call is still debited.

The fundamental quantity that drives all five adaptive mechanisms below
is the *posterior stagnation depth* $s(t) \in [0, 1]$.

---

## 2. The single signal — Posterior-Plateau Stagnation

### 2.1 Motivation

Classical stagnation triggers compute either a plateau ratio
$p(t) = (\text{evals since best})/\tau$ or a budget ratio $b(t)$ and
take their maximum:

$$
s_{\text{legacy}}(t) \;=\; \max\bigl(p(t),\,b(t)\bigr).
$$

This is statistically uncalibrated:

- At $t = 0$ with a fixed evaluation budget $N_{\max}$, $s_{\text{legacy}} > 0$
  even though no information about the search trajectory has been
  collected.
- Near $t = N_{\max}$, $s_{\text{legacy}} \to 1$ even when NEW BEST events
  are arriving rapidly and the run is genuinely improving.

We want a signal that is **near zero when the run is improving** (even
late in the budget) and **near one when the run is genuinely stuck**
(even if budget is fresh and the plateau is long enough to matter).

### 2.2 A Poisson survival formulation

Treat each strict NEW BEST event as a point of a non-homogeneous Poisson
process over the budget axis $B$ (cost or evals, whichever dominates
$g(t)$). The empirical, Laplace-smoothed rate over the bounded window is

$$
\hat{\lambda}(t) \;=\; \frac{k_W + 1}{B_W + \varepsilon},
\qquad
\varepsilon \;=\; \max(10^{-9},\;10^{-3}\,B_{\text{total}}),
$$

where $k_W = |\mathcal{H}(t)|$ is the number of NEW BEST events in the
window and $B_W$ is the budget consumed since the window's oldest event.
The "+1" in the numerator is a Laplace smoothing prior over the
single-event rate; it keeps $\hat{\lambda}(t) > 0$ even when no NEW BEST
has appeared yet.

The **survival probability** that no further NEW BEST appears in the
remaining budget $B_{\text{rem}} = B_{\text{total}} - B_{\text{used}}$
under a homogeneous Poisson with rate $\hat{\lambda}(t)$ is

$$
S(t) \;=\; \exp\!\bigl(-\hat{\lambda}(t)\,B_{\text{rem}}\bigr) \;\in\; (0, 1].
$$

The plateau term

$$
p(t) \;=\; \min\!\Bigl(1,\;\tfrac{N(t) - \tau_{\text{new}}(t)}{\tau}\Bigr)
$$

gates the survival belief — if we are not even on a plateau,
"stuck" carries no weight. The **posterior-stuck** probability is

$$
\Pi(t) \;:=\; p(t)\cdot S(t).
$$

### 2.3 Confidence-weighted blend

Early in the run, $\hat{\lambda}(t)$ is a noisy estimator (few events,
small $B_W$). We weight the posterior by a **confidence** term that
grows quadratically in budget consumed:

$$
\alpha(t) \;=\; b(t)^2,
\qquad
b(t) \;=\; \max\Bigl(\tfrac{C}{C_{\max}},\,\tfrac{N}{N_{\max}},\,\tfrac{T}{T_{\max}}\Bigr).
$$

The final signal is the convex combination

$$
\boxed{\;
s(t) \;=\; (1 - \alpha(t))\,p(t)\;+\;\alpha(t)\,\Pi(t)
\;}
$$

with a **safety floor** for end-of-run plateaus:

$$
s(t) \;\leftarrow\; \max\bigl(s(t),\,p(t)\bigr) \quad\text{whenever}\quad b(t) \ge 0.95\;\wedge\;p(t) \ge 0.95.
$$

### 2.4 Properties (verified by unit tests)

| Regime | Behaviour |
| --- | --- |
| Early run ($\alpha \approx 0$) | $s(t) \approx p(t)$. Plain plateau term, no noisy hazard. |
| Late + improving ($\hat{\lambda}\,B_{\text{rem}}$ large) | $S(t) \to 0 \Rightarrow s(t) \to 0$ — no panic. |
| Late + stuck ($k_W$ small, $p \to 1$) | $\Pi \to p$, safety floor activates. |
| $b(t) = 1$ exactly | Either $b(t)^2 = 1$ and $s = \Pi$, or the floor forces $s \ge p$. |

The estimator $\hat{\lambda}(t)$ is a **maximum a posteriori** estimate
under a $\text{Gamma}(1, B_W)$ conjugate prior on the rate of an
exponential inter-arrival distribution; the Laplace smoothing is its
posterior mean. The survival term is exact under the same model.

---

## 3. Parent selection — Stagnation-driven Zipfian sampler

### 3.1 The problem with score-based sampling

Softmax-over-scores parent sampling

$$
P_{\text{softmax}}(\text{cell } c) \;\propto\; \exp\!\bigl(f_c / T\bigr)
$$

is brittle: when scores have a heavy-tailed distribution, $\exp(f_c/T)$
collapses to a near-degenerate point mass on the global best. The
standard workaround is to fan out across multiple temperatures
$T \in \{0.3, 0.7, 1.0, 1.2\}$, which divides any downstream Thompson
posterior across four arms per sampler.

### 3.2 Rank sampling

Let $r(c)$ denote the 0-indexed rank of cell $c$ by primary score
(descending). Define the **stagnation-driven Zipfian exponent**

$$
\beta(t) \;=\; \max\bigl(\beta_{\min},\;\beta_{\max}\,(1 - s(t))\bigr),
\qquad
(\beta_{\max},\beta_{\min}) \;=\; (2.0,\,0.2),
$$

and sample cells with probability

$$
\boxed{\;
P(c \mid t) \;=\; \frac{(r(c) + 1)^{-\beta(t)}}{\sum_{c'} (r(c') + 1)^{-\beta(t)}}
\;}
$$

### 3.3 Limit behaviour

| Regime | $\beta(t)$ | Distribution |
| --- | --- | --- |
| $s(t) \to 0$ | $\beta \to \beta_{\max} = 2.0$ | Heavy peak on top ranks (exploit). |
| $s(t) \to 1$ | $\beta \to \beta_{\min} = 0.2$ | Nearly uniform (explore). |
| $\beta \to 0$ | — | Exactly uniform. |
| $\beta \to \infty$ | — | Argmax (deterministic best). |

A **single** stagnation-driven exponent collapses the
explore/exploit spectrum, removes the temperature-fan-out
cross-product, and is **score-scale invariant** — it depends only on
the order of cells, not the magnitudes of their scores.

---

## 4. Punctuated Equilibrium with phase-staged prompts

Every $\nu$ evaluations (default $\nu = 10$) the framework fires a
**Punctuated Equilibrium** (PE) event: a heavy-model call designed to
inject a fundamentally different solution.

### 4.1 Cluster representatives

Let $E(t)$ be the set of currently occupied cells. Run $k$-means with
$k = 3$ on the corresponding centroid vectors, yielding clusters
$\mathcal{C}_1,\mathcal{C}_2,\mathcal{C}_3$. From each cluster pick the
elite with maximum score; these three representatives anchor the heavy
prompt.

Under **Hard-PE** (triggered when $s(t) \ge 0.85$ for two consecutive
failed PEs), the representative selection switches to *farthest-first*:

$$
e_{\mathcal{C}_j} \;=\; \underset{e \in \mathcal{C}_j}{\arg\max}\;\bigl\lVert \phi(e) \;-\; \bar{\phi}_{j-1} \bigr\rVert,
$$

where $\phi(e)$ is the elite's normalised behaviour vector and
$\bar{\phi}_{j-1}$ is the centroid of previously picked
representatives. This diversifies the heavy model's context away from
score-similar anchors.

### 4.2 Phase routing

Let $\sigma_{\text{mid}}, \sigma_{\text{late}} \in (0,1)$ be staging
thresholds (default $0.35, 0.7$). The prompt template is chosen by

$$
\text{stage}(t) \;=\;
\begin{cases}
\text{early} & s(t) < \sigma_{\text{mid}} \\
\text{mid}   & \sigma_{\text{mid}} \le s(t) < \sigma_{\text{late}} \\
\text{late}  & s(t) \ge \sigma_{\text{late}}
\end{cases}
$$

- **early** — "pick a paradigm class not represented in the archive."
- **mid** — "synthesize strengths from each cluster; fix one weakness."
- **late** — "targeted refinement; do not rewrite."

The three prompts are written *as different tasks*, not the same task
with different cosmetic phrasing. The information density of each
prompt is calibrated to what the search actually needs at that
stagnation depth.

---

## 5. Strategy History — memory-augmented heavy prompts

### 5.1 The cost of forgetting

Without memory, the heavy model proposes paradigms with
i.i.d. probability conditional only on the current archive. When the
search is stuck, this means *re-proposing approaches that have already
been tried and rejected*. The expected payoff of the next PE event is

$$
\mathbb{E}[\Delta f] \;=\; \mathbb{E}\bigl[\Delta f \mid \text{novel paradigm}\bigr]\cdot \mathbb{P}(\text{novel}) + \mathbb{E}\bigl[\Delta f \mid \text{rehash}\bigr]\cdot \mathbb{P}(\text{rehash}),
$$

with the rehash term contributing approximately zero (the paradigm has
already been rejected once at this score level). Reducing
$\mathbb{P}(\text{rehash})$ directly increases expected progress per PE.

### 5.2 The two-line summary

After each PE event, a **light** model (cheap, fast) produces an
exactly-two-line summary of the paradigm-shift code:

```
IDEA:    <algorithmic family + concrete tactic used here>
QUALITY: <where it should win + the most likely failure mode>
```

Total length $\le 60$ words; sized to comfortably fit in 150 output
tokens including punctuation safety margin.

### 5.3 The deque and the heavy prompt

Strategy records are stored in a bounded FIFO queue of capacity 12. The
next heavy prompt sees the last 8 records as

```
## Strategy Log (already tried in this run)

### PE #5 [mid] — Δ=+0.012, score=2.604, accepted
  IDEA:    Simulated annealing on overlap-penalty energy …
  QUALITY: Strong when radii are diverse; weak under symmetric configs.

### PE #6 [late] — Δ=−0.000, score=2.601, rejected
  IDEA:    Gradient ascent with KKT projection …
  …

Do NOT propose an approach whose summary appears above with Δ ≤ 0.
Aim for something structurally different from every rejected entry.
```

The heavy model can mechanically check rehash via string match against
prior `IDEA:` lines, *before* generating output.

### 5.4 Cost analysis

Light summariser cost per PE:

$$
C_{\text{sum}} \;\le\; \text{output tokens} \cdot \text{output price}
\;\le\; 150 \cdot 0.40 \times 10^{-6}
\;\approx\;\$6 \times 10^{-5}.
$$

For a 200-evaluation run at $\nu = 10$, total summarisation cost is
bounded by $20 \times \$6 \times 10^{-5} = \$1.2 \times 10^{-3}$ —
roughly $0.04\%$ of a typical $\$3$ budget. The cost is negligible
relative to a single heavy-model call.

---

## 6. Code Error Repair — one-shot rescue

### 6.1 The expected-value argument

Every candidate that fails evaluation (syntax error, runtime
exception, score parse error) has already cost one LLM call. Discarding
it forfeits that sunk cost. A *small* light-model call has expected
value

$$
\mathbb{E}[V_{\text{repair}}] \;=\; \mathbb{P}(\text{fix succeeds})\cdot f_{\text{parent}} \;-\; c_{\text{light}}
$$

with $c_{\text{light}} \ll c_{\text{parent\_call}}$. Even at
$\mathbb{P}(\text{fix succeeds}) = 0.05$, the repair pays for itself
when the parent's score sits in the upper half of the archive.

### 6.2 The error buffer

Failed candidates are pushed into a bounded deque
$\mathcal{E}(t)$ of size 64, deduplicated by the first 200 characters
of the code. Each record carries
$(P_{\text{code}}, f_{\text{parent}}, \epsilon, c_{\text{cell}})$ where
$\epsilon$ is the error message tail (see §6.4).

### 6.3 Fire schedule and rank-by-parent-score selection

A repair fires when **all** of the following hold:

$$
\text{enabled}\;\wedge\;|\mathcal{E}(t)| > 0\;\wedge\;n_{\text{repair}}(t) < n_{\max}\;\wedge\;(N(t) - N_{\text{last repair}}) \ge \nu_{\text{repair}}.
$$

Default $\nu_{\text{repair}} = 8$, $n_{\max} = 100$. Given the gate
passes, a record is sampled from $\mathcal{E}(t)$ with **Zipfian
rank-by-parent-score** (analogous to §3 but using parent score as the
ranking signal):

$$
P_{\text{repair}}(\text{rank } r) \;\propto\; (r + 1)^{-\beta_{\text{rep}}},
\qquad \beta_{\text{rep}} = 1.5.
$$

Higher $\beta_{\text{rep}}$ → favour rescuing near-elite parents. The
chosen record is removed (one-shot — failed repairs are not retried).

### 6.4 Error-message tail-truncation

Python tracebacks place the actionable line — exception type and
message — at the **bottom**. Forwarding the head (call-stack frames)
to the repair model wastes tokens. We keep the last 4000 characters:

$$
\epsilon_{\text{forwarded}} \;=\;
\begin{cases}
\epsilon & |\epsilon| \le 4000 \\
\text{"[…truncated head…]\textbackslash n"} \;+\; \epsilon_{\text{last 4000}} & |\epsilon| > 4000
\end{cases}
$$

The marker tells the repair model that the upper part of the stack was
dropped so it does not hallucinate context that isn't there.

---

## 7. Adaptive Island Expansion — non-destructive archive growth

### 7.1 The archive-saturation problem

CVT-MAP-Elites tessellates the behaviour space into $K$ fixed
centroids at initialization. Once every cell is occupied with an
elite, the strict-better admission rule

$$
\text{admit}(P') \;\iff\; f(P') > f\bigl(\mathcal{A}(t)[\text{argmin}_k\,\lVert\phi(P') - \mu_k\rVert]\bigr)
$$

biases the search toward marginal improvements on existing behaviour
modes. A truly novel paradigm with a *slightly* worse score than its
nearest incumbent is rejected — and the archive never gets to host the
new behaviour for future variants to mutate.

### 7.2 The expansion rule

When a PE candidate fails standard admission **and** stagnation is
high, we open a new cell at the candidate's own behaviour vector
instead of evicting any incumbent:

$$
\boxed{\;
\text{expand}\bigl(P', \phi(P')\bigr) \;\iff\;
\begin{cases}
\text{strict\_admission}(P') = \text{False} \\
s(t) \ge \sigma_{\text{island}} = 0.7 \\
n_{\text{island}}(t) < n_{\text{island,max}} = 16 \\
K(t) < K_{\max} = 200
\end{cases}
\;}
$$

When all conditions hold, we append $\phi(P')$ to the centroid set:

$$
\mu_{K(t)+1} \;\leftarrow\; \phi(P'),
\qquad
K(t+1) \;=\; K(t) + 1,
$$

and seed the new cell with $P'$. The old cell — and its incumbent —
are unchanged.

### 7.3 Comparison with prior work

- **AdaEvolve** (Tang et al.) re-clusters periodically with a buffered
  set of recent behaviour vectors. Computationally expensive; requires
  a cooldown gate.
- **MAP-Elites with relaxed thresholds** (e.g. CMA-ME variants) evicts
  incumbents under a softened comparison. Destructive when the new
  candidate is in fact worse.
- **Adaptive Island Expansion** is the cheapest principled archive
  growth: one extra centroid per rescued PE candidate, no re-clustering,
  no buffer, incumbent preserved. Hard caps prevent runaway growth.

---

## 8. Budget-aware resilience — the BLADE retry calculus

### 8.1 The provider pre-authorization problem

Modern LLM gateways (e.g. OpenRouter) charge by

$$
\text{cost}(\text{call}) \;=\; n_{\text{in}}\,p_{\text{in}} \;+\; n_{\text{out}}\,p_{\text{out}}
$$

but **pre-authorize** $n_{\text{out}}^{\max}\,p_{\text{out}}$ before the
call. With gpt-5 ($p_{\text{out}} \approx \$10 / 10^6$) and a default
$n_{\text{out}}^{\max} = 65536$, a single call requires

$$
\text{credit\_required} \;=\; 65536 \times 10^{-5} \;\approx\; \$0.66
$$

of pre-authorized credit. Mid-run, when account credit has decayed
below this threshold, the call is rejected with code 402:

```
This request requires more credits, or fewer max_tokens.
You requested up to 65536 tokens, but can only afford 61223.
```

Hard-coding a small $n_{\text{out}}^{\max}$ avoids the failure but
leaves capacity unused when credit is fresh. We want **adaptive**
behaviour.

### 8.2 The halving loop

Let $m_0$ be the initial `max_tokens` (default: unset, i.e. the
provider's own ceiling). On the $k$-th attempt we use $m_k$, computed
by

$$
m_k \;=\;
\begin{cases}
m_0 & k = 0 \\
m_{\text{hint}} & k = 1\;\text{and a provider hint } m_{\text{hint}} \text{ is present} \\
2^{15} & k = 1\;\text{and no hint} \\
\lfloor m_{k-1} / 2 \rfloor & k \ge 2
\end{cases}
$$

where the **provider hint** $m_{\text{hint}}$ is extracted by regex
from messages of the form `"can only afford N"`, rounded down to a
multiple of $1024$:

$$
m_{\text{hint}} \;=\; \bigl\lfloor N / 1024 \bigr\rfloor \cdot 1024.
$$

The loop terminates when either

- the call succeeds at attempt $k$, returning the response; or
- $m_{k+1} < m_{\min} = 1024$, at which point we give up and re-raise
  the last token-cap error.

### 8.3 Convergence bound

Starting from $m_1$, the loop terminates in at most

$$
k_{\max} \;=\; 1 + \bigl\lceil \log_2(m_1 / m_{\min}) \bigr\rceil
$$

attempts. For $m_1 = 32768, m_{\min} = 1024$: $k_{\max} = 6$. The
worst-case sequence is $\{m_0,\,32768,\,16384,\,8192,\,4096,\,2048\}$,
then terminate.

### 8.4 Detection

A call's exception is classified as a token-cap error iff its
stringified message contains (case-insensitive) any of

$$
\mathcal{S}_{\text{cap}} \;=\;
\bigl\{\text{"fewer max\_tokens"},\;
\text{"requires more credits"},\;
\text{"max\_tokens"},
$$
$$
\text{"afford"},\;
\text{"context length"},\;
\text{"context\_length"},
$$
$$
\text{"maximum context"},\;
\text{"too many tokens"}\bigr\}.
$$

Errors **not** matching any pattern are re-raised immediately — there
is no value in retrying a network blip with a smaller `max_tokens`.

### 8.5 Coverage

The retry wrapper sits in front of every LLM call in the pipeline:
main mutation, code repair, paradigm shift, PE variants, strategy
summariser, meta-advice. This is the property that gives BLADE its
name — **every** request enters and exits via the budget-aware
adaptive layer.

---

## 9. Elite-as-paradigm fallback

When §8 exhausts its retries and a heavy paradigm-shift call still
fails (e.g. account credit is truly depleted), discarding the entire
PE event would forfeit the cost already spent on clustering and
representative selection. BLADE instead substitutes the best cluster
representative as the synthetic paradigm:

$$
P_{\text{paradigm}} \;\leftarrow\; \underset{j}{\arg\max}\;f\bigl(e_{\mathcal{C}_j}\bigr)
\quad\text{(elite already in archive)}.
$$

Because $P_{\text{paradigm}}$ is already in $\mathcal{A}(t)$ with a
known score, no extra evaluation is performed for the paradigm itself.
The variant step proceeds normally:

$$
P_{\text{variant}, i} \;=\; \text{LLM}_{\text{light}}\bigl(\text{prompt}(P_{\text{paradigm}})\bigr),\quad i = 1,\dots,V.
$$

A run-time invariant holds: **a PE event always produces at most $V$
variants worth of work, even when the heavy model is unavailable**.
The strategy log marks rescued events with
`paradigm_source = "elite_fallback"` so post-hoc analysis can
distinguish them from real paradigm shifts.

---

## 10. Cost accounting and budget enforcement

### 10.1 Atomic reservation

Let $C(t), N(t), T(t)$ denote dollars, evaluations, and seconds spent.
Before every LLM call and every evaluation, the framework performs an
**atomic reserve-if-budget-permits** operation:

```
async with budget_lock:
    if g(t) >= 1: raise BudgetLimitReached
    reserve_slot(...)
```

The `BudgetLimitReached` exception propagates through the retry
wrapper unchanged — it is **not** treated as a token-cap error.

### 10.2 Concurrency control

A semaphore caps the number of in-flight client requests at
`max_in_flight` (default 4). When the remaining budget falls below a
threshold

$$
C_{\max} - C(t) \;\le\; \max\bigl(3\,\bar{c}_{\text{call}},\,0.03\,C_{\max},\,\$0.05\bigr)
$$

(where $\bar{c}_{\text{call}}$ is an exponential moving average of
per-call cost) the framework switches to **serial mode**: in-flight
requests are serialized through an additional lock. This prevents the
last few dollars of budget from being overspent by simultaneous calls.

### 10.3 Cost EMA

Per-call cost is tracked with an EMA

$$
\bar{c}_{\text{call}} \;\leftarrow\; 0.8\,\bar{c}_{\text{call}} \;+\; 0.2\,c_{\text{this call}}.
$$

The EMA gives the serial-mode threshold a smoothed reference point
that adapts as model choice and prompt length drift.

---

## 11. Why this is the right minimal set

We claim the **six** mechanisms above (PPS signal, Zipfian sampler,
phase-staged PE, strategy log, code repair, island expansion) plus the
**budget-aware retry wrapper** and the **elite-as-paradigm fallback**
form a minimal, non-redundant set under the resource-cap objective. The
argument has four parts.

### 11.1 No redundancy

Each mechanism solves a problem the others don't:

| Failure mode | Mechanism that addresses it |
| --- | --- |
| Uncalibrated stagnation signal → false PE triggers | §2 (PPS) |
| Softmax temperature fan-out → bandit posterior dilution | §3 (Zipfian sampler) |
| Heavy model proposes already-failed paradigms | §5 (Strategy log) |
| Wasted LLM calls on syntax-trivial errors | §6 (Code repair) |
| Archive saturation rejects novel-but-slightly-worse paradigms | §7 (Island expansion) |
| Provider rejects calls due to credit/token cap | §8 (Retry calculus) |
| Paradigm call fails entirely → PE event aborts | §9 (Elite-as-paradigm) |

### 11.2 One signal drives many decisions

The PPS signal $s(t)$ enters the Zipfian exponent $\beta(t)$ (§3), the
PE phase routing (§4), the island-expansion gate (§7.2), the
contrastive-context augmentation, and the offensive/defensive
meta-advice switch. **A single calibrated estimator informs five
adaptive behaviours** — no separate threshold ad-hoc-ery.

### 11.3 Hard budget bounds at every level

| Resource | Cap |
| --- | --- |
| Repair attempts | $n_{\max} = 100$ (§6.3) |
| Strategy summary tokens | $\le 150$ per PE (§5.4) |
| Island expansions per run | $\le 16$ (§7.2) |
| Total archive size | $\le 200$ centroids (§7.2) |
| LLM retries per call | $\le 1 + \lceil \log_2(m_1/m_{\min}) \rceil$ (§8.3) |
| Concurrent in-flight | $\le 4$ (§10.2) |

The dollar/eval/seconds cap is the only *external* budget; every
internal sub-process has its own bounded contract.

### 11.4 Provable termination and progress

- **Termination.** $g(t)$ is monotone non-decreasing in $t$ and the
  framework stops at the first $t$ with $g(t) \ge 1$. Every internal
  loop (retry, repair, expansion) is bounded by a finite cap.
- **Progress.** A PE event produces at most $V$ variants worth of
  evaluations regardless of heavy-model availability (§9), so the
  expected per-PE progress lower bound is

  $$
  \mathbb{E}[\Delta f^\* \mid \text{PE}] \;\ge\; V \cdot \mathbb{E}[\Delta f^\* \mid \text{variant accepted}]\cdot \mathbb{P}(\text{variant accepted}),
  $$

  strictly positive whenever the light model can produce *any*
  improving variant from an elite parent. The budget-aware retry +
  fallback guarantees the right-hand side does not collapse to zero
  due to provider-side failure modes.

---

## Notation summary

| Symbol | Meaning |
| --- | --- |
| $\mathcal{P}$ | space of admissible Python programs |
| $f(P)$ | scalar objective |
| $\mathcal{A}(t)$ | archive at time $t$ |
| $K(t)$ | number of centroids at time $t$ |
| $\mathcal{B}$ | behaviour space, $\mathcal{B} \subset \mathbb{R}^d$ |
| $\phi(P)$ | normalised behaviour vector of program $P$ |
| $f^\*(t)$ | running best score |
| $g(t)$ | dominant budget ratio |
| $b(t)$ | shorthand for $g(t)$ in the PPS formula |
| $p(t)$ | plateau term, $(N(t)-\tau_{\text{new}}(t))/\tau \wedge 1$ |
| $\hat{\lambda}(t)$ | Laplace-smoothed empirical NEW BEST rate |
| $S(t)$ | survival probability (no further NEW BEST) |
| $\Pi(t)$ | posterior-stuck $= p\cdot S$ |
| $\alpha(t)$ | confidence weight, $b(t)^2$ |
| $s(t)$ | posterior stagnation depth, $\in [0,1]$ |
| $\beta(t)$ | Zipfian exponent for parent sampling |
| $\tau$ | plateau saturation length (default 80) |
| $\nu$ | PE interval (default 10) |
| $\sigma_{\text{mid}}, \sigma_{\text{late}}$ | PE phase thresholds |
| $\sigma_{\text{island}}$ | island-expansion stagnation threshold |
| $\beta_{\text{rep}}$ | repair-buffer Zipfian exponent |
| $\mathcal{E}(t)$ | bounded error buffer |
| $n_{\text{repair}}, n_{\max}$ | running repair count and cap |
| $\nu_{\text{repair}}$ | repair fire interval |
| $m_k$ | `max_tokens` on the $k$-th retry attempt |
| $m_{\min}$ | giveup threshold for retries |
| $\bar{c}_{\text{call}}$ | EMA of per-call cost |
| $V$ | number of variants per PE event |
| $C_{\max}, N_{\max}, T_{\max}$ | hard caps on dollars, evals, seconds |

---

*BLADE — Budget-Limited Adaptive Discovery for Resource-Efficient LLM Evolution.*
