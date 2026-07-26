#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_bench.sh — run BLADE or a baseline on a server, exactly like the GitHub
# Actions workflows .github/workflows/blade.yml and baseline.yml do, but
# locally, under conda, and (optionally) detached inside tmux so the run keeps
# going after you close the SSH connection or shut down your laptop.
#
#   ./scripts/server/run_bench.sh blade    [options]      # mirrors blade.yml
#   ./scripts/server/run_bench.sh baseline [options]      # mirrors baseline.yml
#
# Every workflow_dispatch input of the two YAML files is exposed as a flag with
# the same name and the same default, so anything you used to type into the
# GitHub "Run workflow" form you can type here.
#
# Quick start:
#   ./scripts/server/run_bench.sh blade --tmux \
#       --benchmark levi/examples/circle_packing --evaluations 100
#   tmux attach -t <session shown on screen>
#
# See docs/SERVER_GUIDE.md for a step-by-step walkthrough.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ===========================================================================
# Defaults — identical to the `default:` values in the two workflow files.
# ===========================================================================
MODE=""

# --- shared -----------------------------------------------------------------
SEED="1"
COBENCH_TIMEOUT_IN="10"
COBENCH_MAX_CASES_IN=""
COBENCH_MAX_INSTANCES_IN=""

# --- blade.yml inputs -------------------------------------------------------
BENCHMARK="levi/examples/circle_packing"
EVALUATIONS=""
DOLLARS=""
SECONDS_CAP=""
MUTATION_MODEL="openrouter/qwen/qwen3-30b-a3b-instruct-2507"
PARADIGM_MODEL="openrouter/openai/gpt-5"
WORKERS="4"
PE_INTERVAL="10"
EVAL_TIMEOUT="600"
N_DIVERSE_SEEDS="5"
N_VARIANTS_PER_SEED="20"
ABLATION="full"
ADVANCED_OPTIONS=""

# --- baseline.yml inputs ----------------------------------------------------
BASELINE="evox"
BENCHMARK_DIR="benchmarks/math/circle_packing"
ITERATIONS="100"
MODEL="openrouter/openai/gpt-5"
# baseline.yml caps the job with `timeout-minutes: 180`; mirror that as the
# default here. blade.yml has no such cap, so blade defaults to no timeout
# (it has its own graceful --seconds budget instead).
BASELINE_TIMEOUT_DEFAULT="10800"   # 3h, == timeout-minutes: 180
TIMEOUT_IN="__unset__"

# --- server-only knobs ------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-minhhieu}"
USE_TMUX=0
SESSION=""
OUTPUT_DIR=""
RUN_ID=""
INSTALL_DEPS=1
DRY_RUN=0
USE_CONDA=1

VALID_ABLATIONS="full ast_only emb_only static_cells no_meta_advice meta_errors_only no_targeted_mutate no_crossover no_paradigm"
VALID_BASELINES="openevolve_native gepa_native adaevolve evox"

# ===========================================================================
usage() {
    cat <<'EOF'
Usage:
  run_bench.sh blade    [options]      # mirrors .github/workflows/blade.yml
  run_bench.sh baseline [options]      # mirrors .github/workflows/baseline.yml

Common options
  --seed N                    Experiment seed label; tags run name + output dir. (default: 1)
  --cobench-timeout N         CO-Bench per-instance time limit, seconds. (default: 10)
  --cobench-max-cases N       CO-Bench max test-case files. (default: all)
  --cobench-max-instances N   CO-Bench max instances per file. (default: 3; 0 = all)
  --timeout SPEC              Hard wall-clock cap; the run is SIGTERM'd (then SIGKILL'd
                              30s later) when it expires and the exit status becomes 124.
                              Accepts seconds, or a 3h / 180m / 600s suffix.
                              Use 0 / none to disable.
                              (default: 3h for baseline — same as baseline.yml's
                               timeout-minutes: 180 — and none for blade, matching blade.yml)

  --tmux                      Run detached inside a tmux session (survives SSH disconnect).
  --session NAME              tmux session name. (default: auto, = run id)
  --output-dir DIR            Where to write results. (default: outputs/server/<run-id>)
  --run-id NAME               Override the auto-generated run id.
  --conda-env NAME            Conda env to activate. (default: $CONDA_ENV or "minhhieu")
  --no-conda                  Do not touch conda; use whatever python is active.
  --no-install-deps           Skip the automatic per-benchmark dependency install.
  --dry-run                   Print the command that would run, then exit. No API calls.
  -h, --help                  This help.

blade options (defaults match blade.yml)
  --benchmark PATH            LEVI example dir. (default: levi/examples/circle_packing)
  --evaluations N             Max evaluations. (default: unset = no cap)
  --dollars N                 Max USD spend. (default: unset)
  --seconds N                 Wall-clock cap, seconds. (default: unset)
  --mutation-model ID         Small/mutation model. (default: openrouter/qwen/qwen3-30b-a3b-instruct-2507)
  --paradigm-model ID         Frontier model. (default: openrouter/openai/gpt-5)
  --workers N                 Concurrent LLM workers. (default: 4)
  --pe-interval N             Paradigm-shift cadence. (default: 10)
  --eval-timeout N            Per-candidate eval timeout, seconds. (default: 600)
  --n-diverse-seeds N         Phase-1 diverse seeds. (default: 5)
  --n-variants-per-seed N     Phase-2 variants per seed. (default: 20)
  --ablation NAME             full | ast_only | emb_only | static_cells | no_meta_advice
                              | meta_errors_only | no_targeted_mutate | no_crossover
                              | no_paradigm   (default: full)
  --advanced-options JSON     Low-level overrides, e.g. '{"n_cells":50,"p_crossover":0.2}'

baseline options (defaults match baseline.yml)
  --baseline NAME             openevolve_native | gepa_native | adaevolve | evox (default: evox)
  --benchmark-dir DIR         Benchmark directory. (default: benchmarks/math/circle_packing)
  --iterations N              Iterations to run. (default: 100)
  --model ID                  OpenRouter model id. (default: openrouter/openai/gpt-5)
  --dollars N                 Max USD spend; the run stops itself at the next iteration
                              boundary once tracked LLM cost reaches it, keeping every
                              result written so far. Soft cap — in-flight calls still
                              finish, so the total can overshoot by ~one iteration.
                              (default: unset = no cap)

Examples
  ./scripts/server/run_bench.sh blade --tmux \
      --benchmark levi/examples/co_bench/tsp --evaluations 200 --seed 2
  ./scripts/server/run_bench.sh baseline --tmux \
      --baseline openevolve_native --benchmark-dir benchmarks/math/heilbronn_triangle \
      --iterations 100 --model openrouter/openai/gpt-5
  ./scripts/server/run_bench.sh baseline --tmux \
      --baseline evox --benchmark-dir benchmarks/co_bench/tsp \
      --iterations 500 --dollars 5      # stops itself at $5, whichever comes first
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ===========================================================================
# Argument parsing
# ===========================================================================
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
case "$1" in
    blade|baseline) MODE="$1"; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) die "first argument must be 'blade' or 'baseline' (got '$1')" ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        # shared
        --seed)                  SEED="$2"; shift 2 ;;
        --cobench-timeout)       COBENCH_TIMEOUT_IN="$2"; shift 2 ;;
        --cobench-max-cases)     COBENCH_MAX_CASES_IN="$2"; shift 2 ;;
        --cobench-max-instances) COBENCH_MAX_INSTANCES_IN="$2"; shift 2 ;;
        --timeout)               TIMEOUT_IN="$2"; shift 2 ;;
        --dollars)               DOLLARS="$2"; shift 2 ;;   # both modes
        # blade
        --benchmark)             BENCHMARK="$2"; shift 2 ;;
        --evaluations|--evals)   EVALUATIONS="$2"; shift 2 ;;
        --seconds)               SECONDS_CAP="$2"; shift 2 ;;
        --mutation-model)        MUTATION_MODEL="$2"; shift 2 ;;
        --paradigm-model)        PARADIGM_MODEL="$2"; shift 2 ;;
        --workers)               WORKERS="$2"; shift 2 ;;
        --pe-interval)           PE_INTERVAL="$2"; shift 2 ;;
        --eval-timeout)          EVAL_TIMEOUT="$2"; shift 2 ;;
        --n-diverse-seeds)       N_DIVERSE_SEEDS="$2"; shift 2 ;;
        --n-variants-per-seed)   N_VARIANTS_PER_SEED="$2"; shift 2 ;;
        --ablation)              ABLATION="$2"; shift 2 ;;
        --advanced-options)      ADVANCED_OPTIONS="$2"; shift 2 ;;
        # baseline
        --baseline)              BASELINE="$2"; shift 2 ;;
        --benchmark-dir)         BENCHMARK_DIR="$2"; shift 2 ;;
        --iterations)            ITERATIONS="$2"; shift 2 ;;
        --model)                 MODEL="$2"; shift 2 ;;
        # server-only
        --tmux)                  USE_TMUX=1; shift ;;
        --session)               SESSION="$2"; shift 2 ;;
        --output-dir)            OUTPUT_DIR="$2"; shift 2 ;;
        --run-id)                RUN_ID="$2"; shift 2 ;;
        --conda-env)             CONDA_ENV="$2"; shift 2 ;;
        --no-conda)              USE_CONDA=0; shift ;;
        --no-install-deps)       INSTALL_DEPS=0; shift ;;
        --dry-run)               DRY_RUN=1; shift ;;
        -h|--help)               usage; exit 0 ;;
        *) die "unknown option '$1' (try --help)" ;;
    esac
done

# ===========================================================================
# Validation + run identity
# ===========================================================================
cd "$REPO_ROOT"

# --timeout accepts a bare number of seconds or a 600s / 180m / 3h suffix;
# 0 / none / off disables the cap. Echoes seconds, or "" for disabled.
parse_duration() {
    local v="$1" n suf
    case "$v" in
        ""|0|none|off|no|false) echo ""; return 0 ;;
    esac
    n="${v%[smhSMH]}"
    suf="${v#"$n"}"
    [[ "$n" =~ ^[0-9]+$ ]] || die "invalid --timeout '$v' (use e.g. 10800, 600s, 180m, 3h, or 0 to disable)"
    case "$suf" in
        ""|s|S) echo "$n" ;;
        m|M)    echo $(( n * 60 )) ;;
        h|H)    echo $(( n * 3600 )) ;;
        *)      die "invalid --timeout unit in '$v' (use s, m or h)" ;;
    esac
}

if [[ "$TIMEOUT_IN" == "__unset__" ]]; then
    # Mirror the workflows: baseline.yml caps at 180 minutes, blade.yml does not cap.
    [[ "$MODE" == "baseline" ]] && TIMEOUT_IN="$BASELINE_TIMEOUT_DEFAULT" || TIMEOUT_IN=""
fi
TIMEOUT_SECS="$(parse_duration "$TIMEOUT_IN")"

# --dollars is a plain USD amount; blank/0 means "no cap". Reject anything
# non-numeric here rather than letting it surface hours into a run.
if [[ -n "$DOLLARS" ]]; then
    [[ "$DOLLARS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || die "invalid --dollars '$DOLLARS' (use a number of USD, e.g. 5 or 2.50, or leave it unset)"
fi

if [[ "$MODE" == "blade" ]]; then
    grep -qw -- "$ABLATION" <<<"$VALID_ABLATIONS" || die "invalid --ablation '$ABLATION' (one of: $VALID_ABLATIONS)"
    [[ -d "$BENCHMARK" ]] || die "--benchmark directory not found: $BENCHMARK"
    TARGET="$BENCHMARK"
    TAG="$ABLATION"
else
    grep -qw -- "$BASELINE" <<<"$VALID_BASELINES" || die "invalid --baseline '$BASELINE' (one of: $VALID_BASELINES)"
    [[ -d "$BENCHMARK_DIR" ]] || die "--benchmark-dir not found: $BENCHMARK_DIR"
    TARGET="$BENCHMARK_DIR"
    TAG="$BASELINE"
fi

# Artifact-safe form of the benchmark path ("/" -> "_"), same idea as the
# "Sanitise benchmark path for artifact name" step in blade.yml.
BENCHMARK_SAFE="${TARGET//\//_}"

if [[ -z "$RUN_ID" ]]; then
    # The timestamp only has second resolution, so two runs launched within the
    # same second (a for-loop over ablations, say) would otherwise share a run
    # id — and therefore an output dir and a run.log, silently interleaving two
    # experiments into one file. Add a numeric suffix until the name is free.
    BASE_RUN_ID="${MODE}_${BENCHMARK_SAFE}_${TAG}_seed${SEED}_$(date +%Y%m%d-%H%M%S)"
    RUN_ID="$BASE_RUN_ID"
    n=2
    while [[ -e "outputs/server/$RUN_ID" ]]; do
        RUN_ID="${BASE_RUN_ID}_$n"
        n=$((n + 1))
    done
fi
[[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="outputs/server/$RUN_ID"
if [[ -e "$OUTPUT_DIR/run.log" ]]; then
    echo "WARNING: $OUTPUT_DIR/run.log already exists — this run will append to it." >&2
fi
[[ -n "$SESSION" ]]    || SESSION="$RUN_ID"
# tmux session names may not contain '.' or ':'
SESSION="${SESSION//./_}"; SESSION="${SESSION//:/_}"

LOG_FILE="$OUTPUT_DIR/run.log"

# ===========================================================================
# tmux re-exec: relaunch this exact script inside a detached tmux session.
# ===========================================================================
if [[ "$USE_TMUX" == "1" && "${RUN_BENCH_INSIDE_TMUX:-0}" != "1" ]]; then
    command -v tmux >/dev/null 2>&1 || die "tmux is not installed (sudo apt-get install -y tmux, or: conda install -y -c conda-forge tmux)"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        die "a tmux session named '$SESSION' already exists (use --session NAME, or: tmux kill-session -t $SESSION)"
    fi

    mkdir -p "$OUTPUT_DIR"

    # Rebuild the original command line, minus --tmux, pinning the resolved
    # run id / output dir so the inner run lands where we just announced.
    INNER=("$REPO_ROOT/scripts/server/run_bench.sh" "$MODE")
    INNER+=(--seed "$SEED" --cobench-timeout "$COBENCH_TIMEOUT_IN")
    INNER+=(--cobench-max-cases "$COBENCH_MAX_CASES_IN" --cobench-max-instances "$COBENCH_MAX_INSTANCES_IN")
    INNER+=(--timeout "${TIMEOUT_SECS:-0}" --dollars "$DOLLARS")
    if [[ "$MODE" == "blade" ]]; then
        INNER+=(--benchmark "$BENCHMARK" --evaluations "$EVALUATIONS"
                --seconds "$SECONDS_CAP" --mutation-model "$MUTATION_MODEL"
                --paradigm-model "$PARADIGM_MODEL" --workers "$WORKERS"
                --pe-interval "$PE_INTERVAL" --eval-timeout "$EVAL_TIMEOUT"
                --n-diverse-seeds "$N_DIVERSE_SEEDS" --n-variants-per-seed "$N_VARIANTS_PER_SEED"
                --ablation "$ABLATION" --advanced-options "$ADVANCED_OPTIONS")
    else
        INNER+=(--baseline "$BASELINE" --benchmark-dir "$BENCHMARK_DIR"
                --iterations "$ITERATIONS" --model "$MODEL")
    fi
    # --session is passed through purely so the pane can record it in the log
    # header: the run dir is named after the run id, but what a human remembers
    # months later is the session name. scripts/server/collect_logs.sh maps one
    # back to the other, and needs the name to survive the session being killed.
    INNER+=(--session "$SESSION")
    INNER+=(--run-id "$RUN_ID" --output-dir "$OUTPUT_DIR" --conda-env "$CONDA_ENV")
    [[ "$USE_CONDA"    == "0" ]] && INNER+=(--no-conda)
    [[ "$INSTALL_DEPS" == "0" ]] && INNER+=(--no-install-deps)
    [[ "$DRY_RUN"      == "1" ]] && INNER+=(--dry-run)

    QUOTED="$(printf '%q ' "${INNER[@]}")"
    # `rc=$?` must be the very first thing after the run — any command in
    # between (even `echo`) would overwrite it and the pane would always
    # report success. The trailing `read` keeps the pane alive afterwards so
    # the result stays readable when you attach later.
    # A pane inherits the environment of the *tmux server*, which was fixed when
    # that server first started — possibly from a shell that had no conda on
    # PATH. Pin the PATH (and HOME) we can see right now so `conda` is findable
    # inside the pane regardless of how old the tmux server is.
    TMUX_CMD="RUN_BENCH_INSIDE_TMUX=1 PATH=$(printf '%q' "$PATH") HOME=$(printf '%q' "$HOME") ${QUOTED}; rc=\$?; echo; echo \"=== run finished (exit \$rc) — press Enter to close this pane ===\"; read -r _"
    tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$TMUX_CMD"

    cat <<EOF

Started in tmux.
  session     : $SESSION
  output dir  : $OUTPUT_DIR
  log file    : $LOG_FILE

Watch it live   :  tmux attach -t $SESSION      (detach again with Ctrl-b then d)
Follow the log  :  tail -f $LOG_FILE
Final result    :  ./scripts/server/result.sh $LOG_FILE
List sessions   :  tmux ls
Stop the run    :  tmux kill-session -t $SESSION

EOF
    exit 0
fi

# ===========================================================================
# Conda activation (works both interactively and inside tmux)
# ===========================================================================
if [[ "$USE_CONDA" == "1" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV" \
            || die "could not activate conda env '$CONDA_ENV' — run: bash scripts/server/setup_env.sh"
    else
        die "conda not found on PATH. Run scripts/server/setup_env.sh first, or pass --no-conda."
    fi
fi

mkdir -p "$OUTPUT_DIR"

# ===========================================================================
# Environment variables — same set the workflows export.
# ===========================================================================
# .env is gitignored, so it must be created by hand on the server; see
# docs/SERVER_GUIDE.md. scripts/run_blade.py and skydiscover.config both read
# it, but we also load it here so the validation below sees the key.
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

# The repo accepts either name; OpenRouter keys start with "sk-or-".
if [[ -z "${OPENROUTER_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
    export OPENROUTER_API_KEY="$OPENAI_API_KEY"
fi
if [[ -z "${OPENAI_API_KEY:-}" && -n "${OPENROUTER_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="$OPENROUTER_API_KEY"
fi
export OPENROUTER_API_KEY OPENAI_API_KEY
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export PYTHONUNBUFFERED=1

# CO-Bench knobs, read by levi/examples/co_bench/*/problem.py and
# benchmarks/co_bench/*/evaluator.py; harmless everywhere else.
export COBENCH_TIMEOUT="$COBENCH_TIMEOUT_IN"
export COBENCH_MAX_CASES="$COBENCH_MAX_CASES_IN"
export COBENCH_MAX_INSTANCES="$COBENCH_MAX_INSTANCES_IN"

if [[ "$DRY_RUN" != "1" && -z "${OPENROUTER_API_KEY:-}" ]]; then
    die "no API key found. Create $REPO_ROOT/.env with a line: OPENAI_API_KEY=sk-or-v1-... (see docs/SERVER_GUIDE.md)"
fi

# ===========================================================================
# Header
# ===========================================================================
{
    echo "============================================================"
    echo " run id      : $RUN_ID"
    echo " mode        : $MODE"
    echo " target      : $TARGET"
    if [[ "$MODE" == "blade" ]]; then
        echo " ablation    : $ABLATION"
        echo " models      : mutation=$MUTATION_MODEL"
        echo "               paradigm=$PARADIGM_MODEL"
        echo " budget      : evals='${EVALUATIONS}' dollars='${DOLLARS}' seconds='${SECONDS_CAP}'"
    else
        echo " baseline    : $BASELINE"
        echo " model       : $MODEL"
        echo " iterations  : $ITERATIONS"
        echo " budget      : dollars='${DOLLARS}'"
    fi
    echo " seed label  : $SEED"
    if [[ "${RUN_BENCH_INSIDE_TMUX:-0}" == "1" ]]; then
        echo " tmux session: $SESSION"
    fi
    echo " timeout     : ${TIMEOUT_SECS:-none}${TIMEOUT_SECS:+s}"
    echo " python      : $(command -v python)  ($(python -V 2>&1))"
    echo " conda env   : ${CONDA_DEFAULT_ENV:-<none>}"
    echo " output dir  : $OUTPUT_DIR"
    echo " started at  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"
} | tee -a "$LOG_FILE"

# ===========================================================================
# Dependency install (mirrors the workflows' install steps)
# ===========================================================================
install_blade_extra() {
    # blade.yml maps the benchmark to a `uv sync --extra ...` group; under conda
    # the equivalent is a plain pip install of that group's packages.
    local pkgs=()
    case "$BENCHMARK" in
        *eplb*)      pkgs=("torch>=2.9.1") ;;
        *cloudcast*) pkgs=("networkx>=3.0" "pandas>=2.3.3") ;;
        *llm_sql*)   pkgs=("pandas>=2.3.3") ;;
        *ifbench*)   pkgs=("nltk>=3.9" "datasets>=3.0" "litellm>=1.50" "spacy>=3.7" "emoji>=2.0"
                           "syllapy>=0.7" "langdetect>=1.0.9" "immutabledict>=4.0" "packaging>=24.0") ;;
        *hover*)     pkgs=("bm25s>=0.3.6" "datasets<4" "pystemmer>=3.0.0") ;;
        *)           pkgs=() ;;
    esac
    if [[ ${#pkgs[@]} -eq 0 ]]; then
        echo ">>> No extra dependencies needed for $BENCHMARK"
        return 0
    fi
    echo ">>> Installing extra dependencies for $BENCHMARK: ${pkgs[*]}"
    python -m pip install --quiet "${pkgs[@]}"
}

download_datasets() {
    # Some benchmarks keep their data out of git and ship download_dataset.sh.
    local root="$1"
    local found=0 script
    while IFS= read -r script; do
        found=1
        echo ">>> Running dataset download: $script"
        bash "$script"
    done < <(find "$root" -name download_dataset.sh -type f 2>/dev/null | sort)
    [[ "$found" -eq 0 ]] && echo ">>> No download_dataset.sh under $root — data is self-contained."
    return 0
}

if [[ "$INSTALL_DEPS" == "1" && "$DRY_RUN" != "1" ]]; then
    if [[ "$MODE" == "blade" ]]; then
        install_blade_extra 2>&1 | tee -a "$LOG_FILE"
        # blade.yml downloads only eplb / llm_sql datasets (from benchmarks/ADRS/...).
        case "$BENCHMARK" in
            *eplb*)    download_datasets "benchmarks/ADRS/eplb" 2>&1 | tee -a "$LOG_FILE" ;;
            *llm_sql*) download_datasets "benchmarks/ADRS/llm_sql" 2>&1 | tee -a "$LOG_FILE" ;;
        esac
    else
        echo ">>> Installing benchmark requirements for $BENCHMARK_DIR" | tee -a "$LOG_FILE"
        python scripts/install_benchmark_requirements.py "$BENCHMARK_DIR" 2>&1 | tee -a "$LOG_FILE"
        download_datasets "$BENCHMARK_DIR" 2>&1 | tee -a "$LOG_FILE"
    fi
fi

# ===========================================================================
# Build and run the command
# ===========================================================================
if [[ "$MODE" == "blade" ]]; then
    # ---- advanced_options JSON -> ADV_* variables (same as blade.yml) -------
    if [[ -n "$ADVANCED_OPTIONS" ]]; then
        ADV_EXPORTS="$(python3 -c '
import json, shlex, sys
raw = sys.argv[1]
try:
    opts = json.loads(raw)
except Exception as exc:
    sys.stderr.write(f"Error parsing --advanced-options: {exc}\n")
    sys.exit(1)
if not isinstance(opts, dict):
    sys.stderr.write("--advanced-options must be a JSON object\n")
    sys.exit(1)
for k, v in opts.items():
    print(f"export ADV_{k.upper()}={shlex.quote(str(v))}")
' "$ADVANCED_OPTIONS")" || die "could not parse --advanced-options"
        eval "$ADV_EXPORTS"
    fi

    PROBLEM_MODULE="${ADV_PROBLEM_MODULE:-problem}"
    EMBEDDING_MODEL="${ADV_EMBEDDING_MODEL:-openrouter/openai/text-embedding-3-small}"
    EVAL_PROCESSES="${ADV_EVAL_PROCESSES:-4}"

    EXTRA_OPTS=()
    [[ -n "${ADV_TARGET_SCORE:-}" ]]                     && EXTRA_OPTS+=(--target-score "$ADV_TARGET_SCORE")
    [[ -n "${ADV_N_PARADIGM_VARIANTS:-}" ]]              && EXTRA_OPTS+=(--n-paradigm-variants "$ADV_N_PARADIGM_VARIANTS")
    [[ -n "${ADV_PARADIGM_N_ANCHORS:-}" ]]               && EXTRA_OPTS+=(--paradigm-n-anchors "$ADV_PARADIGM_N_ANCHORS")
    [[ -n "${ADV_PARADIGM_N_INSPIRATIONS:-}" ]]          && EXTRA_OPTS+=(--paradigm-n-inspirations "$ADV_PARADIGM_N_INSPIRATIONS")
    [[ -n "${ADV_N_CELLS:-}" ]]                          && EXTRA_OPTS+=(--n-cells "$ADV_N_CELLS")
    [[ -n "${ADV_RECLUSTER_EVERY:-}" ]]                  && EXTRA_OPTS+=(--recluster-every "$ADV_RECLUSTER_EVERY")
    [[ -n "${ADV_EMBEDDING_DIM:-}" ]]                    && EXTRA_OPTS+=(--embedding-dim "$ADV_EMBEDDING_DIM")
    [[ -n "${ADV_META_ADVICE_INTERVAL:-}" ]]             && EXTRA_OPTS+=(--meta-advice-interval "$ADV_META_ADVICE_INTERVAL")
    [[ -n "${ADV_META_ADVICE_INJECT_P:-}" ]]             && EXTRA_OPTS+=(--meta-advice-inject-p "$ADV_META_ADVICE_INJECT_P")
    [[ "${ADV_META_ADVICE_DISABLED:-}" == "true" ]]      && EXTRA_OPTS+=(--no-meta-advice)
    [[ "${ADV_TARGETED_MUTATE_DISABLED:-}" == "true" ]]  && EXTRA_OPTS+=(--no-targeted-mutate)
    [[ -n "${ADV_ANALYZER_INTERVAL:-}" ]]                && EXTRA_OPTS+=(--analyzer-interval "$ADV_ANALYZER_INTERVAL")
    [[ -n "${ADV_ANALYZER_TOP_K:-}" ]]                   && EXTRA_OPTS+=(--analyzer-top-k "$ADV_ANALYZER_TOP_K")
    [[ -n "${ADV_P_TARGETED_MUTATE:-}" ]]                && EXTRA_OPTS+=(--p-targeted-mutate "$ADV_P_TARGETED_MUTATE")
    [[ -n "${ADV_P_CROSSOVER:-}" ]]                      && EXTRA_OPTS+=(--p-crossover "$ADV_P_CROSSOVER")
    [[ -n "${ADV_META_ADVICE_MODE:-}" ]]                 && EXTRA_OPTS+=(--meta-advice-mode "$ADV_META_ADVICE_MODE")
    [[ -n "${ADV_PARADIGM_SYNTHESIS_MAX_STAGNATION:-}" ]] && EXTRA_OPTS+=(--paradigm-synthesis-max-stagnation "$ADV_PARADIGM_SYNTHESIS_MAX_STAGNATION")
    [[ -n "${ADV_PARADIGM_SURGICAL_MAX_STAGNATION:-}" ]]  && EXTRA_OPTS+=(--paradigm-surgical-max-stagnation "$ADV_PARADIGM_SURGICAL_MAX_STAGNATION")
    [[ -n "${ADV_PARADIGM_SYNTHESIS_N_ANCHORS:-}" ]]      && EXTRA_OPTS+=(--paradigm-synthesis-n-anchors "$ADV_PARADIGM_SYNTHESIS_N_ANCHORS")
    [[ -n "${ADV_PARADIGM_SHIFT_N_ANCHORS:-}" ]]          && EXTRA_OPTS+=(--paradigm-shift-n-anchors "$ADV_PARADIGM_SHIFT_N_ANCHORS")
    [[ -n "${ADV_PARADIGM_SURGICAL_N_INSPIRATIONS:-}" ]]  && EXTRA_OPTS+=(--paradigm-surgical-n-inspirations "$ADV_PARADIGM_SURGICAL_N_INSPIRATIONS")

    # Paper-facing ablation toggles (identical mapping to blade.yml).
    PE_INTERVAL_EFFECTIVE="$PE_INTERVAL"
    case "$ABLATION" in
        ast_only)           EXTRA_OPTS+=(--ast-only) ;;
        emb_only)           EXTRA_OPTS+=(--emb-only) ;;
        static_cells)       EXTRA_OPTS+=(--static-cells) ;;
        no_meta_advice)     EXTRA_OPTS+=(--no-meta-advice) ;;
        meta_errors_only)   EXTRA_OPTS+=(--meta-advice-mode errors_only) ;;
        no_targeted_mutate) EXTRA_OPTS+=(--no-targeted-mutate) ;;
        no_crossover)       EXTRA_OPTS+=(--no-crossover) ;;
        no_paradigm)        PE_INTERVAL_EFFECTIVE="0" ;;
        full|*)             ;;
    esac

    # blade.yml runs with working-directory: levi and --output-dir relative to
    # it; make the output dir absolute so it lands where we announced.
    ABS_OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
    [[ "$OUTPUT_DIR" = /* ]] && ABS_OUTPUT_DIR="$OUTPUT_DIR"

    CMD=(python "$REPO_ROOT/scripts/run_blade.py"
         --example-dir "$BENCHMARK"
         --problem-module "$PROBLEM_MODULE"
         --mutation-model "$MUTATION_MODEL"
         --paradigm-model "$PARADIGM_MODEL"
         --embedding-model "$EMBEDDING_MODEL"
         --evals "$EVALUATIONS"
         --dollars "$DOLLARS"
         --seconds "$SECONDS_CAP"
         --workers "$WORKERS"
         --eval-processes "$EVAL_PROCESSES"
         --eval-timeout "$EVAL_TIMEOUT"
         --pe-interval "$PE_INTERVAL_EFFECTIVE"
         --n-diverse-seeds "$N_DIVERSE_SEEDS"
         --n-variants-per-seed "$N_VARIANTS_PER_SEED"
         --output-dir "$ABS_OUTPUT_DIR"
         ${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"})  # bash 3.2-safe empty-array expansion
    WORKDIR="$REPO_ROOT/levi"
else
    # ---- resolve the evaluator, same as baseline.yml -----------------------
    for f in initial_program.py config.yaml; do
        [[ -f "$BENCHMARK_DIR/$f" ]] || die "expected $BENCHMARK_DIR/$f to exist"
    done
    if [[ -f "$BENCHMARK_DIR/evaluator.py" ]]; then
        EVAL_FILE="$BENCHMARK_DIR/evaluator.py"
    elif [[ -f "$BENCHMARK_DIR/evaluator/evaluator.py" ]]; then
        EVAL_FILE="$BENCHMARK_DIR/evaluator/evaluator.py"
    else
        die "no evaluator at $BENCHMARK_DIR/evaluator.py or $BENCHMARK_DIR/evaluator/evaluator.py"
    fi
    echo ">>> Using evaluator: $EVAL_FILE" | tee -a "$LOG_FILE"

    # Per-LLM-call OpenRouter usage.cost, appended by skydiscover.llm.cost_tracker.
    export SKYDISCOVER_COST_LOG="$OUTPUT_DIR/cost_log.jsonl"

    CMD=(skydiscover-run "$BENCHMARK_DIR/initial_program.py" "$EVAL_FILE"
         --config "$BENCHMARK_DIR/config.yaml"
         --search "$BASELINE"
         --model "$MODEL"
         --iterations "$ITERATIONS"
         --dollars "$DOLLARS"
         --output "$OUTPUT_DIR/run")
    WORKDIR="$REPO_ROOT"
fi

{
    echo ""
    echo ">>> working dir: $WORKDIR"
    echo ">>> command:"
    printf '     '; printf ' %q' "${CMD[@]}"; echo
    echo ""
} | tee -a "$LOG_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
    echo ">>> --dry-run: not executing." | tee -a "$LOG_FILE"
    exit 0
fi

# ===========================================================================
# Wall-clock watchdog (--timeout). Implemented in plain bash rather than via
# coreutils `timeout` so the behaviour is identical on every machine — macOS
# has no `timeout` binary, which would otherwise leave this path untested.
# ===========================================================================
TIMEOUT_MARK="$OUTPUT_DIR/.timed_out"
rm -f "$TIMEOUT_MARK"
WATCHDOG_PID=""

# All PIDs descending from $1, deepest first, so children die before parents.
_descendants() {
    local parent="$1" child
    for child in $(pgrep -P "$parent" 2>/dev/null); do
        _descendants "$child"
        echo "$child"
    done
}

# Signal every process belonging to this run. Processes are matched by RUN_ID,
# which appears in the --output-dir / --output argument of the actual worker.
# The wrapper script itself (and its subshells, which share its command line)
# is skipped so the footer below still gets written.
_signal_run() {
    local sig="$1" p d cmdline
    for p in $(pgrep -f -- "$RUN_ID" 2>/dev/null); do
        cmdline="$(ps -o command= -p "$p" 2>/dev/null || true)"
        case "$cmdline" in
            *run_bench.sh*) continue ;;
        esac
        for d in $(_descendants "$p"); do kill "-$sig" "$d" 2>/dev/null || true; done
        kill "-$sig" "$p" 2>/dev/null || true
    done
}

if [[ -n "$TIMEOUT_SECS" ]]; then
    echo ">>> wall-clock timeout: ${TIMEOUT_SECS}s" | tee -a "$LOG_FILE"
    (
        slept=0
        while [[ "$slept" -lt "$TIMEOUT_SECS" ]]; do
            sleep 5
            slept=$(( slept + 5 ))
        done
        {
            echo ""
            echo ">>> TIMEOUT — wall-clock limit of ${TIMEOUT_SECS}s reached."
            echo ">>> Sending SIGTERM to the run; SIGKILL follows in 30s."
        } | tee -a "$LOG_FILE"
        : > "$TIMEOUT_MARK"
        _signal_run TERM
        sleep 30
        _signal_run KILL
    ) &
    WATCHDOG_PID=$!
fi

set +e
( cd "$WORKDIR" && "${CMD[@]}" ) 2>&1 | tee -a "$LOG_FILE"
STATUS="${PIPESTATUS[0]}"
set -e

if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
fi

TIMED_OUT=0
if [[ -e "$TIMEOUT_MARK" ]]; then
    TIMED_OUT=1
    STATUS=124          # same convention as coreutils `timeout`
    rm -f "$TIMEOUT_MARK"
fi

# ===========================================================================
# Cost summary (the "Cost summary" step of baseline.yml) + footer
# ===========================================================================
{
    echo ""
    echo "============================================================"
    TOTALS="$OUTPUT_DIR/cost_log.totals.json"
    if [[ -f "$TOTALS" ]]; then
        echo " cost totals:"
        cat "$TOTALS"
        echo ""
    fi
    if [[ -f "$OUTPUT_DIR/cost_log.jsonl" ]]; then
        echo " LLM calls recorded: $(wc -l < "$OUTPUT_DIR/cost_log.jsonl" | tr -d ' ')"
    fi
    if [[ -n "$DOLLARS" ]]; then
        echo " spend budget: \$$DOLLARS"
    fi
    echo " finished at : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    if [[ "$TIMED_OUT" == "1" ]]; then
        echo " exit status : $STATUS  (TIMED OUT after ${TIMEOUT_SECS}s — partial results kept)"
    else
        echo " exit status : $STATUS"
    fi
    echo " results in  : $OUTPUT_DIR"
    echo "============================================================"
} | tee -a "$LOG_FILE"

exit "$STATUS"
