# Running LLM-SRBench (LSR-Synth) on a server

How to run SpecEvo and the baselines on the LSR-Synth half of
[LLM-SRBench](https://github.com/deep-symbolic-mathematics/llm-srbench), a domain
at a time, with checkpointing and resume. For what the benchmark measures see
[`benchmarks/llm_srbench/README.md`](../benchmarks/llm_srbench/README.md).

The reduced configuration described here is **the first 10 problems of each of the
four domains** — 40 problems per method — at 500 search iterations per problem.
Pass `--full` to run every problem instead (§2.1).

---

## 1. One-time setup

```bash
conda activate minhhieu

# Two extra packages, only needed to download and unpack the dataset:
pip install "huggingface_hub>=0.24" "pyarrow>=15.0"
#   equivalently: pip install -r scripts/server/requirements-server.txt

# Fetch the dataset (~9 MB on disk, idempotent — safe to re-run):
python benchmarks/llm_srbench/prepare_data.py
```

The API key lives in `.env` at the repo root:

```text
OPENAI_API_KEY=sk-or-v1-...
```

Check that the key can reach both the chat model and the embedding model SpecEvo
needs for its behavioural archive:

```bash
python scripts/test_openrouter_key.py
```

The per-problem directories for the first 10 problems of each domain are already
in the repo. Regenerate them (for example to change the default model or to widen
past 10 problems) with:

```bash
python benchmarks/llm_srbench/generate_dirs.py --limit 10
```

---

## 2. Run it

One command per (method, domain). `--tmux` detaches so the run survives an SSH
disconnect.

```bash
# SpecEvo across all four domains
for d in chem_react bio_pop_growth phys_osc matsci; do
    ./scripts/server/run_lsr_synth.sh --method specevo --domain "$d" --tmux
done

# A baseline across all four domains
for d in chem_react bio_pop_growth phys_osc matsci; do
    ./scripts/server/run_lsr_synth.sh --method evox --domain "$d" --tmux
done
```

Defaults, all overridable:

| setting | default |
| --- | --- |
| problems per domain | 10 (`--problems N`, or `--full` for all of them) |
| iterations per problem | 500 (`--iterations N`) |
| search signal | `log_nmse` (`--score-mode log_nmse\|inv_nmse`, §5.1) |
| training points used for the fit | all of them (`--max-fit-points N` to subsample) |
| Speculator + Navigator model | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` |
| baseline model | the same qwen3-30b, EvoX's meta level included (see below) |
| time per equation hypothesis | 30 s (`--eval-timeout N`) |
| wall-clock cap per problem | 7200 s (`--problem-timeout N`, `0` disables) |
| conda env | `minhhieu` (`--conda-env NAME`) |

Methods: `specevo` (alias `blade`), `openevolve_native`, `gepa_native`,
`adaevolve`, `evox`.

EvoX needs one extra precaution to stay a like-for-like baseline. Besides the
solution loop it co-evolves its own *search strategy*, and that meta level reads
`skydiscover/search/evox/config/search.yaml`, which pins `openai/gpt-5` (plus
`gpt-5-mini` for its guide). `--model` only rewrites the problem's `config.yaml`,
so by default EvoX would be the one baseline steered by a frontier model — and at
roughly 50x qwen's price per output token, a couple of strategy switches cost more
than an entire 500-evaluation run of any other method. The generated `config.yaml`
therefore carries `search.share_llm: true`, which makes the meta level, its guide
and its evaluator all inherit whatever `--model` was passed. If you hand-write a
config for EvoX, carry that key over.

Everything lands in a directory named after the run, with no timestamp:

```text
outputs/lsr_synth/<method>/<domain>/seed<N>/
├── run.log                      appended across every attempt
├── results.jsonl                one record per finished problem
├── summary.json                 aggregate, refreshed after each problem
└── problems/<pid>/              the search's own output dir
    ├── .done                    written only when the problem is finished
    └── run/checkpoints/...      baselines: resume points, every 10 iterations
```

### 2.1 The full benchmark instead of the first 10

`--full` (equivalently `--problems all`) runs **every** problem in the domain:

```bash
./scripts/server/run_lsr_synth.sh --method specevo --domain matsci --full --tmux
```

| domain | problems | reduced | full |
| --- | --- | --- | --- |
| `chem_react` | 36 | 10 | 36 |
| `phys_osc` | 44 | 10 | 44 |
| `matsci` | 25 | 10 | 25 |
| `bio_pop_growth` | 24 | 10 | 24 |
| **total per method** | **129** | 40 | 129 |

Only the reduced set's problem directories are committed; the rest are generated
automatically at the start of the run (`>>> generating N problem directories`),
so there is no extra setup step. Multiply your reduced-set spend by 3.2 before
launching — at 500 iterations a full sweep is the dominant cost in the project.

`--problem-list` takes arbitrary ids and also generates what it needs, which is
the cheapest way to extend a finished reduced run rather than restart it:

```bash
./scripts/server/run_lsr_synth.sh --method specevo --domain matsci \
    --problem-list matsci10,matsci11,matsci12
```

Note that "full" here is about the number of **problems**. Each problem's own
data has always been used in full: the BFGS fit sees all 4000 training points
unless you deliberately shrink it with `--max-fit-points`, which is a smoke-test
knob and changes the benchmark.

### SpecEvo ignores `--iterations` below ~105

BLADE's bootstrap is a fixed shape — `--n-diverse-seeds` frontier seeds, each
fanned out into `--n-variants-per-seed` parallel variants, ~105 evaluations at the
defaults — and its variants are all dispatched at once, so the eval budget cannot
preempt them. `--iterations 10` therefore still runs ~105 evaluations. (This is
documented BLADE behaviour: `budget_evals` counts the init phase too.)

At the intended 500 it is a non-issue — 105 of the 500 go to bootstrap, exactly as
in the repo's other experiments. It only matters for small smoke tests, where you
should shrink the bootstrap as well:

```bash
./scripts/server/run_lsr_synth.sh --method specevo --domain matsci \
    --problems 10 --iterations 10 \
    --n-diverse-seeds 2 --n-variants-per-seed 2      # ~10 evals for real
```

Baselines have no bootstrap, so their `--iterations` is exact at any size.

### Budget planning — read this before launching the full sweep

The methods run at very different speeds, and the default `--problem-timeout`
(2 h per problem) will cut some of them off well before 500 iterations. Measured
on qwen3-30b with these problems:

| method | seconds per iteration | 500 iterations |
| --- | --- | --- |
| `specevo` | ~3–6 (4 workers, parallel) | ~40–60 min ✓ fits |
| `adaevolve` | ~15 | ~2 h — borderline |
| `openevolve_native` | ~20 | ~2.8 h ✗ truncated at the 2 h cap |
| `evox` | ~110 (it also evolves its search strategy) | ~15 h ✗ truncated |
| `gepa_native` | untimed here | verify with a short run |

The `evox` row was measured *before* `search.share_llm: true` was added, i.e. while
its meta level was still on gpt-5 — a reasoning model, and much of that 110 s was
its thinking time. Re-time it on qwen before trusting the figure.

Almost all of that is LLM latency, not the benchmark: the BFGS fit plus scoring
of one hypothesis takes **0.5–2 s**, while one qwen3-30b completion takes 15–25 s.
So the per-iteration figure is really "how many completions does this method wait
on, one at a time". The baselines run **one iteration at a time by design** —
`max_parallel_iterations = 1` — whereas SpecEvo dispatches `--workers` (4)
completions concurrently, which is the whole of the 4x gap above. Measured on
these problems at 10 iterations per problem: `openevolve_native` ~3.5 min/problem
against `specevo` ~1 min/problem, for the same 10 evaluations and the same model.

Wall clock is therefore not a like-for-like axis between the two families; the
eval budget is. Independent problems and independent domains can always be run as
separate concurrent processes (separate `--output-dir`, or just separate domains),
which costs nothing in comparability because each search remains sequential.

So either raise the cap for the slower baselines:

```bash
./scripts/server/run_lsr_synth.sh --method evox --domain chem_react \
    --problem-timeout 0 --tmux                    # no wall-clock cap
```

or — better for a comparison you intend to publish — equalise on **spend** rather
than iterations, which is what the rest of this repo's experiments do:

```bash
./scripts/server/run_lsr_synth.sh --method evox --domain chem_react \
    --dollars 2 --problem-timeout 21600 --tmux
```

Time one problem first to calibrate:

```bash
./scripts/server/run_lsr_synth.sh --method evox --domain matsci --problems 1 --iterations 20
```

### Adding a spend cap

`--dollars N` applies **per problem**, and the search stops itself gracefully at
the next iteration boundary once tracked spend reaches it. With 10 problems,
`--dollars 2` bounds the domain at roughly \$20.

```bash
./scripts/server/run_lsr_synth.sh --method specevo --domain chem_react --dollars 2 --tmux
```

A problem that hits `--problem-timeout` is not lost: whatever the search found is
kept, scored, and recorded — the run just moves on to the next problem. But a
method that is being truncated at a different point from the others is not a fair
comparison, so check the `evaluations` column in `results.jsonl` before drawing
conclusions.

---

## 3. Watching a run

```bash
# Progress table — no API calls, safe at any time
./scripts/server/run_lsr_synth.sh --method specevo --domain chem_react --status

# Live log
tail -f outputs/lsr_synth/specevo/chem_react/seed1/run.log

# tmux
tmux ls
tmux attach -t lsr_specevo_chem_react_seed1     # detach: Ctrl-b then d
```

`--status` prints which problems are finished, their ID and OOD NMSE, and where
an unfinished problem's newest checkpoint sits.

---

## 4. Interrupting and resuming

This is designed around running out of API credit mid-sweep.

**To stop:** `Ctrl-C` in the pane, or `tmux kill-session -t <session>`. Both are
handled (`SIGINT`, `SIGTERM` and `SIGHUP`): the running search is terminated and
the current problem is abandoned rather than recorded, so nothing half-finished
ends up in the results.

Avoid `kill -9` on the runner. `SIGKILL` cannot be trapped, so the search process
survives and keeps spending credit. If it happens anyway, the next run of the
same command detects the survivor and terminates it before resuming — but you
will have paid for the gap.

**To resume: re-run the identical command.** The output directory is derived from
(method, domain, seed) with no timestamp, so:

```bash
./scripts/server/run_lsr_synth.sh --method specevo --domain chem_react --tmux
```

picks up where it stopped. Specifically:

* problems with a `.done` marker are skipped;
* an interrupted **baseline** problem resumes from its newest checkpoint and is
  given only the iterations it still owes against `--iterations` (skydiscover's
  own `--iterations` counts *additional* iterations, so the runner does that
  subtraction for you). How much a problem has already had is measured from the
  program database inside the checkpoint, not from the checkpoint's name: on an
  early shutdown skydiscover still writes a final checkpoint numbered by the
  *requested* budget, and trusting that name would silently truncate the problem
  to whatever it managed before the interrupt (see `scripts/lsr_resume_plan.py`);
* an interrupted **SpecEvo** problem restarts, because BLADE has no mid-search
  resume. The partial attempt is moved to `problems/<pid>/prev_attempt_<ts>/`
  rather than deleted, and its spend is still counted in the reported cost.

The exit status is `3` when problems remain unfinished, `0` when the domain is
complete — useful in a wrapper loop.

To throw a run away and start over: `--fresh`.

---

## 5. Results

```bash
# One run
python scripts/lsr_summarize.py outputs/lsr_synth/specevo/chem_react/seed1

# Every method and domain that has finished anything
python scripts/lsr_summarize.py outputs/lsr_synth

# For the paper
python scripts/lsr_summarize.py outputs/lsr_synth --csv lsr_synth.csv
```

Illustrative shape of the output (numbers are made up):

```text
method             domain          n fail    ID NMSE   OOD NMSE  ID Acc OOD Acc ID w/in OOD w/in    ID R2   OOD R2   cost $
---------------------------------------------------------------------------------------------------------------------------
specevo            Chemistry      10    0  4.110e-06  3.020e-04   30.0%   20.0%   81.3%    64.9%   0.9999   0.9981    3.412
evox               Chemistry      10    1  9.110e-05  2.770e-04   10.0%   10.0%   62.0%    48.2%   0.9998   0.9902    5.887
```

* **ID** is the held-out in-domain test split, **OOD** the out-of-domain split
  (later times, higher temperatures/strains). Neither is ever used to steer the
  search.
* `n` is problems attempted; `fail` is how many produced no usable equation.
  Failures stay in `n` and count as 0 in `Acc`/`w/in`, so a method cannot look
  good by failing often.
* NMSE **and R²** columns are **medians** over the problems that produced an
  equation, as in the paper's Table 1. R² is unbounded below, so a single equation
  that diverges off the training range would drag a mean to -1e9 and hide the
  other nine; `ood_r2_worst` in the JSON/CSV shows that tail deliberately.
* `Acc` is the paper's `Acc(0.1)`: the percentage of problems whose *maximum*
  point-wise relative error stays within 10%. It is all-or-nothing per problem,
  so one near-zero target point can zero it out for an otherwise excellent
  equation. `w/in` is the graded companion — the mean fraction of points inside
  the tolerance. Read the two together.

Each `results.jsonl` record also keeps the full discovered program and the
ground-truth expression, so symbolic accuracy can be judged offline later without
re-running any search.

### 5.1 What the search optimises — `--score-mode`

Every method searches on **training NMSE only**, computed by
`benchmarks/llm_srbench/lsr_eval.py`:

1. the candidate's `params` are fitted on the train split by BFGS on MSE;
2. with those parameters frozen, `train_nmse = SSE / Σ(y - ȳ)²` on the same split;
3. the ID and OOD splits are measured too, but *only* recorded — never selected on.

The scalar handed to the searchers (`combined_score`) is a monotone transform of
that NMSE, so the *ranking* of hypotheses is the paper's ranking (`-MSE`, since
NMSE only divides by the constant `var(y_train)`) under either mode:

| mode | formula | NMSE 1e-3 | 1e-5 | 1e-7 |
| --- | --- | --- | --- | --- |
| `log_nmse` **(default)** | `log10(1 + 1/NMSE)` | 3.0004 | 5.0000 | 7.0000 |
| `inv_nmse` | `1 / (1 + NMSE)` | 0.9990 | 1.0000 | 1.0000 |

`inv_nmse` was the original scale and it is where `score: 0.999 / best: 1.000000`
in the logs came from. It is monotone, so selection was never *wrong*, but it
saturates — and both runners print the score to 3–4 decimals in the prompts they
build for the LLM (`levi/levi/blade/prompts.py`,
`skydiscover/context_builder/default/builder.py`), so past NMSE ≈ 1e-4 the model
was shown `1.0000` for every parent and could not tell a good equation from an
excellent one, nor see that its last edit helped.

`log_nmse` reads as **decades of NMSE below 1** — 3.0 is NMSE 1e-3, 7.0 is 1e-7 —
which stays informative at four decimals over the whole useful range. It is
capped at 16 (`LSR_SCORE_LOG_CAP`), the point where NMSE is float64 round-off,
and stays strictly positive for any finite NMSE so that **0.0 still means "the
hypothesis did not run"**: a candidate that raises, returns the wrong shape,
produces NaN/inf, or exceeds the 30 s limit scores exactly 0.0.

Nothing downstream assumes a `[0, 1]` score — the databases min-max scale or rank
whatever they are given — and `train_nmse` / `neg_log10_train_nmse` are recorded
under both modes, so the NMSE behind any score is always in the record. Runs made
under different modes are still comparable **on the reported ID/OOD metrics**,
which are pure NMSE/R²/Acc and independent of the search scale. Use
`--score-mode inv_nmse` only to reproduce a run made before this flag existed.

### 5.2 How each reported metric is computed

All of them come from the *final* program, re-evaluated from scratch by
`scripts/lsr_finalize.py` after the search ends — the number is pinned to the
program on disk, not to whatever the search believed at the time. Per split
(`train`, `id_test`, `ood_test`), with `ŷ` the prediction and `y` the target:

| metric | definition | notes |
| --- | --- | --- |
| `nmse` | `Σ(ŷ-y)² / Σ(y-ȳ)²` | 0 is perfect; 1 is as good as predicting the mean. This is the paper's headline number. |
| `r2` | `1 - nmse` | Identical information to NMSE, opposite direction; unbounded below. |
| `mse` | `Σ(ŷ-y)² / n` | Unnormalised, so not comparable across problems. |
| `acc` → `Acc(0.1)` | `1` iff `max\|ŷ-y\|/\|y\| ≤ 0.1` over the whole split, else `0` | The paper's `Acc_τ`. All-or-nothing per problem. |
| `frac_within` → `w/in` | mean of `\|ŷ-y\|/\|y\| ≤ 0.1` per point | The graded companion to `Acc(0.1)`. |
| `mape` | mean of `\|ŷ-y\|/\|y\|` | Blows up wherever `y ≈ 0`. |
| `max_rel_error` | `max \|ŷ-y\|/\|y\|` | The quantity `Acc(0.1)` thresholds; useful for seeing *how close* a 0 was. |

Points whose prediction is non-finite are dropped from NMSE/R² but recorded in
`num_valid_points` vs `num_points`, and any dropped point forces `acc = 0` — a
method cannot earn accuracy by predicting NaN on the hard points.

Then, across problems, `scripts/lsr_summarize.py` reports per (method, domain):

* `ID NMSE`, `OOD NMSE`, `ID R2`, `OOD R2` — **medians** over the problems that
  produced a usable equation, as in the paper's Table 1. Means are in the
  JSON/CSV (`*_mean`) along with `ood_r2_worst`; R² is unbounded below, so one
  equation that diverges off the training range would drag a mean to -1e9 and
  hide every other problem.
* `ID Acc`, `OOD Acc` — mean of the per-problem 0/1 `Acc(0.1)`, as a percentage,
  over **all** attempted problems.
* `ID w/in`, `OOD w/in` — mean of `frac_within`, again over all attempted
  problems.
* `n` / `fail` — problems attempted, and how many left no usable equation.
  Failures stay in `n` and count as 0 in `Acc`/`w/in`, so failing often cannot
  flatter a method. They are excluded from the NMSE/R² medians (there is no
  number to include), which is why `fail` must be read next to them.
* `cost $` — the sum of the per-call `cost_log.jsonl`, which accumulates across
  resumed attempts, plus any archived SpecEvo attempt.

`total_evaluations` in the JSON/CSV is how many hypotheses were actually scored.
Check it before comparing methods: a method truncated by `--problem-timeout` at a
different point from the others is not a fair comparison.

### 5.3 Symbolic accuracy — the paper's other axis

`lsr_summarize.py` reports data fidelity only. Symbolic accuracy — GPT-4o judging
whether the discovered equation is mathematically equivalent to the ground truth
(paper §2.3 / App. B.2, Fig. 11) — is scored separately, from what the records
already hold. No search is re-run:

```bash
# Everything that has finished, all four domains, per-method table:
./scripts/server/symbolic_accuracy.sh --workers 16

# Only the methods that have run so far, plus a CSV for the paper table:
./scripts/server/symbolic_accuracy.sh --methods specevo,openevolve_native \
    --csv outputs/symbolic_accuracy/table.csv

# See the task counts and one example prompt without spending anything:
./scripts/server/symbolic_accuracy.sh --dry-run
```

It prints per-domain SA (% of attempted problems judged equivalent) plus a pooled
and macro-averaged total per method, and writes
`outputs/symbolic_accuracy/symbolic_accuracy.json` alongside
`judgments.jsonl` — one line per problem with the verdict, the judge's reasoning
and the exact strings it compared. Domains that do not yet cover every problem in
the dataset are marked `*`, so a partial sweep cannot be mistaken for the
full-dataset number.

Judgements are cached by (ground truth, hypothesis, model), so re-running after
another baseline finishes only pays for the new baseline, and an interrupted
sweep resumes for free. `--fresh` re-judges everything. The judge defaults to
`gpt-4o` — via OpenRouter as `openai/gpt-4o` when `.env` holds an `sk-or-` key,
matching how the runs themselves are configured.

---

## 6. Running a single problem

The per-problem directories are ordinary benchmark directories, so the generic
runners work on them — handy for debugging one problem:

```bash
./scripts/server/run_bench.sh baseline \
    --benchmark-dir benchmarks/llm_srbench/matsci/matsci0 --iterations 20

./scripts/server/run_bench.sh blade \
    --benchmark levi/examples/llm_srbench/matsci/matsci0 --evaluations 20
```

These give no aggregation and no resume; that is what `run_lsr_synth.sh` adds.

---

## 7. Troubleshooting

**`could not list problems — has the dataset been prepared?`**
Run `python benchmarks/llm_srbench/prepare_data.py`, then
`python benchmarks/llm_srbench/prepare_data.py --check`.

**`Cannot access gated repo` from HuggingFace**
The default source is the ungated mirror, so this only appears if you passed
`--repo nnheui/llm-srbench`. Either request access on the dataset page, or drop
the flag. On a machine with no internet, download the parquet elsewhere and use
`--from-local <dir>`.

**Every hypothesis scores 0 with `Timeout (30s)`**
The parameter fit is not completing. Confirm with a single problem:

```bash
python - <<'PY'
import sys; sys.path.insert(0, "benchmarks/llm_srbench")
import lsr_eval as L
print(L.evaluate_source("matsci", "matsci0", L.seed_program("matsci", "matsci0"))["feedback"])
PY
```

The linear seed should score in well under a second. If it does not, raise
`--eval-timeout`; that departs from the paper's 30 s, so say so in the writeup.

**A problem hangs**
`--problem-timeout` (default 2 h) kills it and moves on, keeping whatever was
found. Set `0` to disable.

**Costs look too low after a resume**
They should not: the reported spend sums the per-call `cost_log.jsonl`, which
accumulates across attempts, plus any archived SpecEvo attempt. A run that was
never interrupted reports the runner's own total.
