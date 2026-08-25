#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_relay.sh — run RelayEvolve or one of its cheap/strong routing baselines
# on a server, optionally detached inside tmux so the job survives an SSH
# disconnect.
#
#   ./scripts/server/run_relay.sh --method relayevolve --tmux \
#       --benchmark-dir benchmarks/math/circle_packing \
#       --iterations 300 --dollars 2 --seed 1
#
# Every method shares the same OpenEvolve backend, evaluator, generation cap
# and dollar cap; only the model schedule differs. Defaults follow the paper's
# setup as ported to this repo:
#
#   strong model : openrouter/moonshotai/kimi-k2
#   cheap model  : openrouter/qwen/qwen3-30b-a3b-instruct-2507
#   iterations   : 300
#   dollars      : 2       (hard stop — the run ends gracefully once reached)
#   eval timeout : 60s
#   workers      : 8       (generations in flight at once)
#
# See docs/SERVER_GUIDE.md §6c for a step-by-step walkthrough.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ===========================================================================
# Defaults
# ===========================================================================
METHOD="relayevolve"
BENCHMARK_DIR="benchmarks/math/circle_packing"
CHEAP_MODEL="openrouter/qwen/qwen3-30b-a3b-instruct-2507"
STRONG_MODEL="openrouter/moonshotai/kimi-k2"
ITERATIONS="300"
DOLLARS="2"
WORKERS="8"
EVAL_TIMEOUT="60"
RETRIES="1"
SEED="1"
TIMEOUT_IN=""
EXTRA_ARGS=()

CONDA_ENV="${CONDA_ENV:-minhhieu}"
USE_TMUX=0
SESSION=""
OUTPUT_DIR=""
RUN_ID=""
INSTALL_DEPS=1
DRY_RUN=0
USE_CONDA=1

VALID_METHODS="relayevolve all_cheap all_strong fixed_switch random bandit"

usage() {
    cat <<'EOF'
Usage:
  run_relay.sh [options]

Method and target
  --method NAME             relayevolve | all_cheap | all_strong | fixed_switch
                            | random | bandit                (default: relayevolve)
  --benchmark-dir DIR       Benchmark with initial_program.py + config.yaml + evaluator
                            (default: benchmarks/math/circle_packing)

Budget (all three caps apply; whichever binds first wins)
  --iterations N            Generation cap.                    (default: 300)
  --dollars N               USD spend cap; graceful stop.      (default: 2)
  --timeout SPEC            Hard wall-clock cap (3h / 180m / 600s / 0 = off).
                            (default: off)

Models
  --cheap-model ID          (default: openrouter/qwen/qwen3-30b-a3b-instruct-2507)
  --strong-model ID         (default: openrouter/moonshotai/kimi-k2)

Execution
  --workers N               Generations in flight at once.     (default: 8)
  --eval-timeout N          Per-candidate evaluation timeout, seconds. (default: 60)
  --retries N               LLM attempts per generation. 1 (default) = no retry: an
                            invalid program spends its generation and the search moves
                            on, so one generation is always one model call.
  --seed N                  Seed; tags the run name and output dir. (default: 1)

RelayEvolve knobs (passed straight through to scripts/run_relay.py)
  --strong-reserve F        Share of the dollar budget kept for the strong model (0.85)
  --block-size N            Generations per Grow/Deepen block h (5)
  --max-trajectories N      Cheap trajectory cap (5)
  --trajectory-horizon N    Blocks per trajectory (6)
  --bank-size N             Relay bank / handoff seed count k (8)
  --relay-lambda F          Quality vs coverage weight in F_C(S) (0.5)
  --epsilon-rel F           Relay-Gain saturation threshold (0.02)
  --patience N              Consecutive low-gain blocks before handoff (3)
  --curation MODE           full | quality | diversity | random
  --relay-control MODE      full | random | no_stop | random_no_stop
  --switch-fraction F       Fixed-switch handover point (0.5)
  --p-strong F              Random baseline P(strong) (0.5)
  --advanced-options JSON   Any other search.database override, e.g. '{"ucb_c":0.8}'

Server
  --tmux                    Run detached inside a tmux session.
  --session NAME            tmux session name.               (default: the run id)
  --output-dir DIR          Results directory.               (default: outputs/server/<run-id>)
  --run-id NAME             Override the generated run id.
  --conda-env NAME          Conda env to activate.           (default: $CONDA_ENV or minhhieu)
  --no-conda                Use whatever python is already active.
  --no-install-deps         Skip the per-benchmark dependency install.
  --dry-run                 Print the command and exit. No API calls, no spend.
  -h, --help                This help.
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ===========================================================================
# Argument parsing
# ===========================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)               METHOD="$2"; shift 2 ;;
        --benchmark-dir)        BENCHMARK_DIR="$2"; shift 2 ;;
        --cheap-model)          CHEAP_MODEL="$2"; shift 2 ;;
        --strong-model)         STRONG_MODEL="$2"; shift 2 ;;
        --iterations)           ITERATIONS="$2"; shift 2 ;;
        --dollars)              DOLLARS="$2"; shift 2 ;;
        --workers)              WORKERS="$2"; shift 2 ;;
        --eval-timeout)         EVAL_TIMEOUT="$2"; shift 2 ;;
        --retries)              RETRIES="$2"; shift 2 ;;
        --seed)                 SEED="$2"; shift 2 ;;
        --timeout)              TIMEOUT_IN="$2"; shift 2 ;;
        --strong-reserve|--block-size|--max-trajectories|--trajectory-horizon|\
        --bank-size|--relay-lambda|--epsilon-rel|--patience|--curation|\
        --relay-control|--switch-fraction|--p-strong|--embedding-backend|\
        --embedding-model|--advanced-options)
                                EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --tmux)                 USE_TMUX=1; shift ;;
        --session)              SESSION="$2"; shift 2 ;;
        --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
        --run-id)               RUN_ID="$2"; shift 2 ;;
        --conda-env)            CONDA_ENV="$2"; shift 2 ;;
        --no-conda)             USE_CONDA=0; shift ;;
        --no-install-deps)      INSTALL_DEPS=0; shift ;;
        --dry-run)              DRY_RUN=1; shift ;;
        -h|--help)              usage; exit 0 ;;
        *)                      die "unknown option: $1 (try --help)" ;;
    esac
done

cd "$REPO_ROOT"

grep -qw -- "$METHOD" <<<"$VALID_METHODS" || die "invalid --method '$METHOD' (one of: $VALID_METHODS)"
[[ -d "$BENCHMARK_DIR" ]] || die "--benchmark-dir not found: $BENCHMARK_DIR"
[[ -f "$BENCHMARK_DIR/initial_program.py" ]] || die "missing $BENCHMARK_DIR/initial_program.py"
[[ -f "$BENCHMARK_DIR/config.yaml" ]] || die "missing $BENCHMARK_DIR/config.yaml"
if [[ -n "$DOLLARS" ]]; then
    [[ "$DOLLARS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || die "invalid --dollars '$DOLLARS' (a number of USD, e.g. 2 or 1.50)"
fi

parse_duration() {
    local v="$1" n suf
    case "$v" in ""|0|none|off|no|false) echo ""; return 0 ;; esac
    n="${v%[smhSMH]}"; suf="${v#"$n"}"
    [[ "$n" =~ ^[0-9]+$ ]] || die "invalid --timeout '$v' (e.g. 10800, 600s, 180m, 3h, or 0)"
    case "$suf" in
        ""|s|S) echo "$n" ;;
        m|M)    echo $(( n * 60 )) ;;
        h|H)    echo $(( n * 3600 )) ;;
        *)      die "invalid --timeout unit in '$v' (use s, m or h)" ;;
    esac
}
TIMEOUT_SECS="$(parse_duration "$TIMEOUT_IN")"

BENCHMARK_SAFE="${BENCHMARK_DIR//\//_}"
if [[ -z "$RUN_ID" ]]; then
    BASE_RUN_ID="relay_${METHOD}_${BENCHMARK_SAFE}_seed${SEED}_$(date +%Y%m%d-%H%M%S)"
    RUN_ID="$BASE_RUN_ID"
    n=2
    while [[ -e "outputs/server/$RUN_ID" ]]; do
        RUN_ID="${BASE_RUN_ID}_$n"; n=$((n + 1))
    done
fi
[[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="outputs/server/$RUN_ID"
[[ -n "$SESSION" ]]    || SESSION="$RUN_ID"
SESSION="${SESSION//./_}"; SESSION="${SESSION//:/_}"
LOG_FILE="$OUTPUT_DIR/run.log"

# ===========================================================================
# tmux re-exec
# ===========================================================================
if [[ "$USE_TMUX" == "1" && "${RUN_RELAY_INSIDE_TMUX:-0}" != "1" ]]; then
    command -v tmux >/dev/null 2>&1 \
        || die "tmux is not installed (sudo apt-get install -y tmux, or conda install -y -c conda-forge tmux)"
    tmux has-session -t "$SESSION" 2>/dev/null \
        && die "a tmux session named '$SESSION' already exists (use --session NAME, or tmux kill-session -t $SESSION)"

    mkdir -p "$OUTPUT_DIR"

    INNER=("$REPO_ROOT/scripts/server/run_relay.sh"
           --method "$METHOD" --benchmark-dir "$BENCHMARK_DIR"
           --cheap-model "$CHEAP_MODEL" --strong-model "$STRONG_MODEL"
           --iterations "$ITERATIONS" --dollars "$DOLLARS"
           --workers "$WORKERS" --eval-timeout "$EVAL_TIMEOUT" --retries "$RETRIES"
           --seed "$SEED"
           --timeout "${TIMEOUT_SECS:-0}"
           --session "$SESSION" --run-id "$RUN_ID" --output-dir "$OUTPUT_DIR"
           --conda-env "$CONDA_ENV")
    INNER+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
    [[ "$USE_CONDA"    == "0" ]] && INNER+=(--no-conda)
    [[ "$INSTALL_DEPS" == "0" ]] && INNER+=(--no-install-deps)
    [[ "$DRY_RUN"      == "1" ]] && INNER+=(--dry-run)

    QUOTED="$(printf '%q ' "${INNER[@]}")"
    # A tmux pane inherits the environment of the tmux *server*, which may
    # predate conda being on PATH; pin the PATH we can see right now.
    TMUX_CMD="RUN_RELAY_INSIDE_TMUX=1 PATH=$(printf '%q' "$PATH") HOME=$(printf '%q' "$HOME") ${QUOTED}; rc=\$?; echo; echo \"=== run finished (exit \$rc) — press Enter to close this pane ===\"; read -r _"
    tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$TMUX_CMD"

    cat <<EOF

Started in tmux.
  method      : $METHOD
  session     : $SESSION
  output dir  : $OUTPUT_DIR
  log file    : $LOG_FILE

Watch it live   :  tmux attach -t $SESSION      (detach with Ctrl-b then d)
Follow the log  :  tail -f $LOG_FILE
Spend so far    :  cat $OUTPUT_DIR/cost_log.totals.json
Relay report    :  cat $OUTPUT_DIR/relay_summary.json
List sessions   :  tmux ls
Stop the run    :  tmux kill-session -t $SESSION

EOF
    exit 0
fi

# ===========================================================================
# Conda
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
# Environment
# ===========================================================================
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi
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
export SKYDISCOVER_COST_LOG="$OUTPUT_DIR/cost_log.jsonl"

[[ "$DRY_RUN" == "1" || -n "${OPENROUTER_API_KEY:-}" ]] \
    || die "no API key found. Create $REPO_ROOT/.env with: OPENAI_API_KEY=sk-or-v1-... (see docs/SERVER_GUIDE.md)"

# ===========================================================================
# Header
# ===========================================================================
{
    echo "============================================================"
    echo " run id      : $RUN_ID"
    echo " method      : $METHOD"
    echo " benchmark   : $BENCHMARK_DIR"
    echo " models      : cheap=$CHEAP_MODEL"
    echo "               strong=$STRONG_MODEL"
    echo " budget      : iterations=$ITERATIONS dollars=\$${DOLLARS} eval-timeout=${EVAL_TIMEOUT}s"
    echo " retries     : $RETRIES (1 = no retry)"
    echo " workers     : $WORKERS"
    echo " seed        : $SEED"
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo " extra       : ${EXTRA_ARGS[*]}"
    [[ "${RUN_RELAY_INSIDE_TMUX:-0}" == "1" ]] && echo " tmux session: $SESSION"
    echo " timeout     : ${TIMEOUT_SECS:-none}${TIMEOUT_SECS:+s}"
    echo " python      : $(command -v python)  ($(python -V 2>&1))"
    echo " conda env   : ${CONDA_DEFAULT_ENV:-<none>}"
    echo " output dir  : $OUTPUT_DIR"
    echo " started at  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"
} | tee -a "$LOG_FILE"

# ===========================================================================
# Benchmark dependencies
# ===========================================================================
if [[ "$INSTALL_DEPS" == "1" && "$DRY_RUN" != "1" ]]; then
    echo ">>> Installing benchmark requirements for $BENCHMARK_DIR" | tee -a "$LOG_FILE"
    python scripts/install_benchmark_requirements.py "$BENCHMARK_DIR" 2>&1 | tee -a "$LOG_FILE"
    while IFS= read -r script; do
        echo ">>> Running dataset download: $script" | tee -a "$LOG_FILE"
        bash "$script" 2>&1 | tee -a "$LOG_FILE"
    done < <(find "$BENCHMARK_DIR" -name download_dataset.sh -type f 2>/dev/null | sort)
fi

# ===========================================================================
# Command
# ===========================================================================
CMD=(python "$REPO_ROOT/scripts/run_relay.py"
     --method "$METHOD"
     --benchmark-dir "$BENCHMARK_DIR"
     --cheap-model "$CHEAP_MODEL"
     --strong-model "$STRONG_MODEL"
     --iterations "$ITERATIONS"
     --dollars "$DOLLARS"
     --workers "$WORKERS"
     --eval-timeout "$EVAL_TIMEOUT"
     --retries "$RETRIES"
     --seed "$SEED"
     --output "$OUTPUT_DIR"
     ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

{
    echo ""
    echo ">>> command:"
    printf '     '; printf ' %q' "${CMD[@]}"; echo
    echo ""
} | tee -a "$LOG_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
    echo ">>> --dry-run: not executing." | tee -a "$LOG_FILE"
    exit 0
fi

# ===========================================================================
# Wall-clock watchdog (--timeout), same convention as run_bench.sh
# ===========================================================================
TIMEOUT_MARK="$OUTPUT_DIR/.timed_out"
rm -f "$TIMEOUT_MARK"
WATCHDOG_PID=""

_descendants() {
    local parent="$1" child
    for child in $(pgrep -P "$parent" 2>/dev/null); do
        _descendants "$child"
        echo "$child"
    done
}

_signal_run() {
    local sig="$1" p d cmdline
    for p in $(pgrep -f -- "$RUN_ID" 2>/dev/null); do
        cmdline="$(ps -o command= -p "$p" 2>/dev/null || true)"
        case "$cmdline" in *run_relay.sh*) continue ;; esac
        for d in $(_descendants "$p"); do kill "-$sig" "$d" 2>/dev/null || true; done
        kill "-$sig" "$p" 2>/dev/null || true
    done
}

if [[ -n "$TIMEOUT_SECS" ]]; then
    echo ">>> wall-clock timeout: ${TIMEOUT_SECS}s" | tee -a "$LOG_FILE"
    (
        slept=0
        while [[ "$slept" -lt "$TIMEOUT_SECS" ]]; do sleep 5; slept=$(( slept + 5 )); done
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
( cd "$REPO_ROOT" && "${CMD[@]}" ) 2>&1 | tee -a "$LOG_FILE"
STATUS="${PIPESTATUS[0]}"
set -e

if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
fi

TIMED_OUT=0
if [[ -e "$TIMEOUT_MARK" ]]; then
    TIMED_OUT=1
    STATUS=124
    rm -f "$TIMEOUT_MARK"
fi

# ===========================================================================
# Footer
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
    SUMMARY="$OUTPUT_DIR/relay_summary.json"
    if [[ -f "$SUMMARY" ]]; then
        python - "$SUMMARY" <<'PY' || true
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
tiers = s.get("llm_calls_by_tier") or {}
print(" relay summary:")
print(f"   method          : {s.get('method')}")
print(f"   best score      : {s.get('best_score')}")
print(f"   generations     : {s.get('iterations_used')}")
print(f"   llm calls       : cheap={tiers.get('cheap', 0)}  strong={tiers.get('strong', 0)}")
if s.get("handoff_iteration") is not None:
    print(f"   handoff at gen  : {s['handoff_iteration']}  ({s.get('handoff_reason')})")
if s.get("cheap_iterations") is not None:
    print(f"   cheap/strong gen: {s.get('cheap_iterations')} / {s.get('strong_iterations')}")
if s.get("seeds"):
    print(f"   handoff seeds   : {len(s['seeds'])}")
PY
    fi
    echo " spend budget: \$$DOLLARS"
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
