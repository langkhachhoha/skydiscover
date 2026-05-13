# LEVI — Research Directions for a Publishable Improvement

Goal: ship a *small but defensible* paper on top of LEVI within one
internship-scale time budget. The constraint is that LEVI itself is already
strong on the ADRS leaderboard, so an incremental result has to be both
**cheap to demonstrate** and **conceptually clean**, not "yet another
sampler". Below are three concrete directions, ordered by how realistic the
paper feels given the time we have.

Throughout: training-free additions only, OpenRouter for both model tiers
(`qwen/qwen3-30b-a3b-instruct-2507` and `openai/gpt-5`), 100-evaluation
budget per run, run via [the GitHub Actions
workflow](../.github/workflows/levi-adrs.yml).

---

## Direction 1 (most feasible) — *Behavior-aware Punctuated Equilibrium*: deciding **when** the heavy model fires, not just **how often**

### What LEVI does today

`PunctuatedEquilibriumConfig.interval = 10` fires the strong model every
ten evaluations, *regardless* of whether the archive is actually stuck. The
stagnation signal exists only at the cluster level (selecting *which* elite
to mutate). The trigger itself is a fixed clock.

Empirically this means LEVI either:

- **over-fires** the heavy model when the archive is still finding cheap
  wins (wasted gpt-5 dollars), or
- **under-fires** when the archive has plateaued for, say, 25 evals but the
  next paradigm shift is still 5 evals away.

### Proposed change

Replace the fixed interval with a **CUSUM-style drift detector on the
score history**:

```
trigger_pe ←  (evals_since_last_pe ≥ min_gap)
        AND  (best_score plateau detected over window W)
        AND  (CVT cell-occupancy entropy plateau in window W)
```

Concretely:

- Keep a sliding window of the last `W = 20` accepted scores.
- If both `max(window) − max(window[:W/2]) < ε₁` *and* the CVT occupancy
  entropy hasn't grown by more than `ε₂`, fire.
- Maintain `min_gap` (e.g. 5) to prevent oscillation.

This is a ~50-line modification in `pipeline/runner.py::_monitor_pe()`
plus a small helper in `pipeline/state.py` (we already record the score
history and archive occupancy there).

### Why this is publishable

It produces a clean ablation: same archive, same models, same budget — only
the PE trigger changes. We can report:

1. **Heavy-model call savings** (proportion of evaluations that go to gpt-5).
2. **Best-score-vs-budget curve** versus the fixed-interval baseline.
3. **Sensitivity** to `W`, `ε`.

If even one of (a) "fewer heavy-model dollars at same score" or (b)
"higher score at same budget" holds across most ADRS tasks, the paper
writes itself as *"the harness's idle time matters more than its tempo"* —
a natural follow-on to LEVI's own thesis that the *harness* is what
matters, not the model.

### Feasibility / risk

- **Effort:** 1–2 weeks. Code change is small; most of the work is running
  the matrix.
- **Compute:** 100 evals × 7 ADRS tasks × {old, new} × ≥3 seeds = 4,200
  evals. With LEVI's published $4.50/run cost we're inside a few hundred
  dollars even after seed variance.
- **Risk that result is negative:** moderate but recoverable. A negative
  result on "smarter PE triggers don't help" is itself publishable as a
  cautionary note.

### Stretch (only if the basic version works)

Train a tiny linear model on `(score_velocity, archive_entropy,
cells_filled, evals_since_pe) → pe_will_improve?` from the matrix data
above and replace the hand-tuned thresholds with a learned classifier. This
is a 2-paragraph appendix, not a new paper.

---

## Direction 2 — *Cross-problem prior*: warm-start the CVT centroids and meta-advice from a related task

### Observation

Right now every LEVI run starts from scratch: data-driven CVT centroids are
computed from the init phase of *this run only*. But the seven ADRS
problems share structural priors (greedy + small local-search + a
cost-shaped objective). If the centroids and meta-advice from a previous
run on a *neighbour* task were available, the init phase could spend its
budget on exploitation instead of recomputing the lay of the land.

### Proposed change

1. Persist `snapshot.json`'s `metadata.centroids` and the *last* meta-advice
   block alongside `summary.json`.
2. Add a `--warm-start-from <run_dir>` flag to `evolve_code` that:
   - Pre-seeds CVT centroids with the prior run's centroids (re-normalize
     against the new behavior bounds).
   - Pre-loads the prior meta-advice as the initial guidance string.
3. Run an *n × n* transfer matrix across the seven ADRS tasks: "warm-start
   from task i, evolve task j", measure best score and evals-to-target.

This is essentially **MAP-Elites with a non-trivial prior**, which the
quality-diversity literature does discuss but, to my knowledge, no public
LLM-evo framework actually ships.

### Why this is publishable

- It quantifies *transferability* between systems-optimization problems —
  an empirical question with no good current answer.
- It opens up an actually-useful CI workflow: schedule LEVI nightly on a
  hub task, and every related downstream task warm-starts from it.
- The negative cells of the n × n matrix (where warm-start *hurts*) tell
  you which problems are structurally dissimilar — a clean
  characterization story.

### Feasibility / risk

- **Effort:** 2–3 weeks (the snapshot already exists; mainly bookkeeping
  + experiment matrix).
- **Compute:** 7 × 7 = 49 cells × 100 evals × ≥2 seeds. Still tractable.
- **Risk:** larger than D1. Warm-starting could trivially help on some
  cells and trivially hurt on others, making the headline number
  mixed. The paper needs a per-task analysis, not just a mean.

---

## Direction 3 (more ambitious) — *Multi-objective behavior axes from problem statement*

### Observation

LEVI's behavior axes are hand-picked AST features
(`loop_count`, `loop_nesting_max`, …). Two problems:

- They're **AST-syntactic**, so two algorithmically identical solutions
  written in different styles look "diverse"; two algorithmically distinct
  solutions written in the same style look identical.
- They're **fixed at config time** — you can't change them mid-run as you
  learn which axes actually matter.

### Proposed change

Use the heavy model **once at init time** to propose a small set of
*semantic* behavior axes from `PROBLEM_DESCRIPTION` (e.g. for
txn_scheduling: "uses lookahead?", "respects key locality?", "schedules
greedily by length?"). Each axis is realized as a small Python predicate
or LLM-as-judge call applied to a candidate program. Plug those into
`BehaviorConfig.custom_extractors`.

This is essentially **LLM-proposed quality-diversity descriptors** —
related to recent work on automatic novelty descriptors but specialized to
LEVI's archive.

### Why this could be a paper

- It tackles a real LEVI weakness called out in its own README ("diversity
  is an architectural concern") and shows you can do *better* architectural
  diversity than hand-picked AST counters.
- It produces interpretable archives: the cells correspond to *strategies*,
  not *token patterns*.
- The comparison story is clean: same models, same budget, same harness,
  only behavior axes change.

### Feasibility / risk

- **Effort:** 4–6 weeks. Behavior extractors need careful design; the
  LLM-as-judge axis is itself a cost-and-noise source.
- **Compute:** larger because you'd want to run with both LEVI-default and
  LLM-derived axes across all 7 tasks.
- **Risk:** semantic axes can be noisy; the LLM-as-judge calls need to be
  *cheap* (small model) or the cost story collapses. Mitigation: predicates
  first, judge only as a fallback.

---

## Recommendation

**Start with Direction 1.** It is genuinely small, has a 1-week prototype
path, the ablation is the cleanest possible (one knob), and the result —
either way — is interesting. If the result is strong, fold in the learned
trigger from §1's stretch goal and submit as a short workshop paper. If
it's marginal, add Direction 2 as a second contribution in a longer
write-up: *"two cheap, training-free additions to LEVI."*

Avoid starting with Direction 3 unsupervised — it has more moving parts
and the failure modes are harder to debug in time-bounded CI runs.

---

## Concrete first sprint (≈1 week)

1. **Day 1–2.** Implement the CUSUM trigger in
   [`levi/pipeline/runner.py`](../levi/levi/pipeline/runner.py) behind a
   feature flag (`punctuated_equilibrium.trigger="cusum"`). Persist
   per-trigger reasoning into the snapshot so we can plot it later.
2. **Day 3.** Add a `--trigger {fixed,cusum}` flag to
   [`scripts/run_levi_skydiscover.py`](../scripts/run_levi_skydiscover.py)
   and a matching workflow input in
   [`.github/workflows/levi-adrs.yml`](../.github/workflows/levi-adrs.yml).
3. **Day 4–5.** Kick off the 7-task × 2-trigger × 3-seed matrix in GitHub
   Actions. Each run is ≤ 100 evaluations. Collect the `summary.json`s.
4. **Day 6–7.** Plot two curves per task (score-vs-evals, heavy-model-evals
   share). If the headline looks right, draft the workshop paper outline.
