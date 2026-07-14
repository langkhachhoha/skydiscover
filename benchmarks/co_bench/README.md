# CO-Bench in SkyDiscover

This directory integrates [**CO-Bench**](https://github.com/sunnweiwei/CO-Bench)
(Sun et al., 2025 — *Benchmarking Language Model Agents in Algorithm Search for
Combinatorial Optimization*) into SkyDiscover so that CO-Bench problems can be
run by **both** discovery paths in this repo:

- the **baselines** (`openevolve_native`, `gepa_native`, `adaevolve`, `evox`)
  via `.github/workflows/baseline.yml` / `skydiscover-run`; and
- **SpecEvo / BLADE** via `.github/workflows/blade.yml` / `scripts/run_blade.py`.

The evaluation protocol follows the paper: each problem instance is solved under
a **hard 10-second per-instance time limit**, the raw objective is **normalized
against the best-known solution** (score `1.0` = best-known / optimal, higher is
better), and any program error, constraint violation, or timeout scores **0.0**
for that instance.

## The 8 representative problems

CO-Bench spans 8 problem categories (36 problems total). To keep runs cheap we
integrate **one representative problem per category**:

| Slug | CO-Bench task | Category |
|------|---------------|----------|
| `bin_packing_1d`           | Bin packing - one-dimensional      | Packing |
| `non_guillotine_cutting`   | Constrained non-guillotine cutting | Cutting |
| `warehouse_location_uncap` | Uncapacitated warehouse location   | Facility location |
| `flow_shop`                | Flow shop scheduling               | Scheduling |
| `tsp`                      | Travelling salesman problem        | Routing |
| `gap`                      | Generalised assignment problem     | Assignment |
| `steiner`                  | Euclidean Steiner problem          | Tree |
| `graph_coloring`           | Graph colouring                    | Graph & set |

## Layout

```
benchmarks/co_bench/
├── cobench_eval.py          # shared evaluation engine (used by BOTH paths)
├── data/<CO-Bench task>/    # vendored config.py (load_data/eval_func/norm_score/get_dev) + instance files
└── <slug>/                  # baseline benchmark dir, one per problem
    ├── initial_program.py   #   seed `solve` inside an EVOLVE-BLOCK + problem description
    ├── config.yaml          #   skydiscover-run config
    ├── evaluator.py         #   evaluate(program_path) -> {combined_score, ...}
    └── requirements.txt

levi/examples/co_bench/<slug>/
└── problem.py               # BLADE/LEVI example: PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, score_fn
```

Both paths import the single engine `cobench_eval.py`, so a candidate `solve`
gets scored **identically** whether it is discovered by a baseline or by BLADE.

### How the engine works (`cobench_eval.py`)

- Loads the vendored task `config.py` and iterates its test-case files. For each
  file, `load_data()` yields instances; each instance runs `solve(**instance)`
  then `eval_func(**instance, **solution)`.
- **Per-instance isolation + timeout.** Non-daemon callers (the baseline
  evaluator) run each instance in a **forked subprocess** and hard-kill it at the
  limit. Daemon callers (BLADE workers, which may not spawn child processes) run
  each instance in-process under **`SIGALRM`**. Both enforce the same 10s limit;
  a runaway/looping `solve` scores 0 and never hangs the search.
- Applies the task's `norm_score` (normalize vs. best-known). The search signal
  `combined_score` = `overall_score` = mean over **every instance actually
  evaluated** (robust under any cap). `dev_score` / `test_score` reproduce
  CO-Bench's dev/test split and are informational — only meaningful at the full
  set (under a small cap the split can be tiny/empty). Also returns `valid_rate`.

## Runtime knobs (env vars)

| Variable | Meaning | Default |
|----------|---------|---------|
| `COBENCH_TIMEOUT` | per-instance time limit, seconds (paper: 10) | `10` |
| `COBENCH_MAX_CASES` | max test-case files per evaluation (`0` = all) | all |
| `COBENCH_MAX_INSTANCES` | max instances per file (`0` = all) | `3` |

**Default = all files × 3 instances/file** — ~6–48 instances per problem (TSP 6,
gap 48, graph_coloring 30…). In practice each iteration is **dominated by the
LLM call** (~10–30s); evaluation is usually well under a second because good
solves finish fast and bad ones fail fast — the instance count only bites when a
candidate is *valid but slow* (near 10s × instances). To run the **full
CO-Bench test set** (faithful to the paper, slower), set `COBENCH_MAX_INSTANCES=0`;
lower `COBENCH_MAX_CASES` to use fewer files for a quicker smoke test.

## Running locally

Set your OpenRouter key (already in `.env`) and route OpenAI-style calls to
OpenRouter:

```bash
set -a; . ./.env; set +a
export OPENAI_API_BASE=https://openrouter.ai/api/v1
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export COBENCH_MAX_CASES=1 COBENCH_MAX_INSTANCES=2   # quick smoke; unset for defaults
```

### Baselines

```bash
skydiscover-run \
  benchmarks/co_bench/tsp/initial_program.py \
  benchmarks/co_bench/tsp/evaluator.py \
  --config benchmarks/co_bench/tsp/config.yaml \
  --search openevolve_native \
  --model openrouter/openai/gpt-5 \
  --iterations 100 \
  --output outputs/cobench/tsp
```

Swap `tsp` for any slug and `--search` for any of
`openevolve_native | gepa_native | adaevolve | evox`.

### BLADE / SpecEvo

```bash
python scripts/run_blade.py \
  --example-dir levi/examples/co_bench/tsp \
  --mutation-model openrouter/qwen/qwen3-30b-a3b-instruct-2507 \
  --paradigm-model openrouter/openai/gpt-5 \
  --evals 64 \
  --output-dir outputs/blade/cobench_tsp
```

## Running via GitHub Actions

Both workflows accept CO-Bench directly — no code changes needed:

- **Baselines** (`Baselines Smoke`): set `benchmark_dir` to
  `benchmarks/co_bench/<slug>` and pick a `baseline`.
- **BLADE** (`BLADE`): set `benchmark` to `levi/examples/co_bench/<slug>`.

Both expose `cobench_timeout`, `cobench_max_cases`, `cobench_max_instances`
inputs (they map to the env vars above; blank = the defaults). They are ignored
by non-CO-Bench benchmarks.

## Seed sanity check (no LLM)

Each problem ships a simple but **feasible** seed `solve` (nearest-neighbour for
TSP, first-fit for bin packing, greedy+repair for GAP, greedy colouring, …). The
seed scores below were produced by the shared engine on a bounded sample
(`MAX_CASES=3, MAX_INSTANCES=5`) and are the starting point the search improves on:

| Problem | Seed dev score | Notes |
|---------|:--:|-------|
| bin_packing_1d           | ~0.99 | first-fit decreasing |
| warehouse_location_uncap | ~0.90 | assign each customer to cheapest warehouse |
| gap                      | ~0.86 | least-consumption + overflow repair |
| flow_shop                | ~0.82 | identity permutation |
| tsp                      | ~0.80 | nearest-neighbour |
| graph_coloring           | ~0.75 | greedy largest-first |
| steiner                  | ~0.11 | no Steiner points (MST baseline) |
| non_guillotine_cutting   | 0.00  | places nothing — genuinely hard feasibility, for the search to crack |

Reproduce:

```bash
python - <<'PY'
import sys; sys.path.insert(0, "benchmarks/co_bench")
import cobench_eval as ce
for slug, task in ce.TASKS.items():
    # baseline seed lives in the initial_program.py of each slug
    src = open(f"benchmarks/co_bench/{slug}/initial_program.py").read()
    r = ce.evaluate_source(task, src, max_cases=3, max_instances=5)
    print(f"{slug:26s} dev={r['score']:.3f} test={r['test_score']:.3f} valid={r['valid_rate']:.2f}")
PY
```

## Adding more CO-Bench problems

1. Download the task from the [CO-Bench HF dataset](https://huggingface.co/datasets/CO-Bench/CO-Bench)
   into `benchmarks/co_bench/data/<task>/` (it must contain `config.py` + instance files).
2. Add a `slug -> task` entry to `TASKS` (and `CATEGORY`) in `cobench_eval.py`.
3. Create `benchmarks/co_bench/<slug>/` (baseline) and
   `levi/examples/co_bench/<slug>/problem.py` (BLADE), mirroring an existing
   problem — only the `TASK` name and the seed `solve` change.
