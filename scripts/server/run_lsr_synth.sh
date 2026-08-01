#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_lsr_synth.sh — run SpecEvo or a baseline over one LSR-Synth domain of
# LLM-SRBench, one problem at a time, with checkpointing and resume.
#
#   ./scripts/server/run_lsr_synth.sh --method specevo --domain chem_react
#   ./scripts/server/run_lsr_synth.sh --method evox    --domain matsci
#
# LSR-Synth is a *set* of independent equation-discovery problems, so a "run"
# here means: for each of the domain's first N problems (10 by default), search
# for an equation with the chosen method, then re-evaluate the best program on
# the held-out in-domain (ID) and out-of-domain (OOD) test splits. Per-problem
# results land in results.jsonl; the aggregate lands in summary.json.
#
# RESUME
#   The output directory is derived from (method, domain, seed) with no
#   timestamp, so re-running the *same command* after an interruption — an
#   exhausted API key, a dropped SSH session, Ctrl-C — picks up where it left
#   off. Problems that already finished are skipped; a baseline problem that was
#   interrupted mid-search resumes from its newest checkpoint and only runs the
#   iterations it still owes.
#
#   Check progress at any time without launching anything:
#       ./scripts/server/run_lsr_synth.sh --method specevo --domain chem_react --status
#
# See benchmarks/llm_srbench/README.md for the benchmark itself.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ===========================================================================
# Defaults
# ===========================================================================
METHOD="specevo"
DOMAIN=""
N_PROBLEMS="10"
PROBLEM_LIST=""
ITERATIONS="500"
SEED="1"

QWEN="openrouter/qwen/qwen3-30b-a3b-instruct-2507"
MODEL="$QWEN"                # baselines
MUTATION_MODEL="$QWEN"       # SpecEvo Speculator (small model)
PARADIGM_MODEL="$QWEN"       # SpecEvo Navigator (frontier model)
EMBEDDING_MODEL="openrouter/openai/text-embedding-3-small"

SCORE_MODE="log_nmse"        # search signal: log_nmse | inv_nmse (see lsr_eval.py)
MAX_FIT_POINTS="0"           # 0 = fit on the full training split (the benchmark)
EVAL_TIMEOUT="30"            # seconds per equation hypothesis (paper: T = 30s)
PROBLEM_TIMEOUT="7200"       # wall-clock cap per problem; 0 disables
DOLLARS=""                   # USD cap per problem
SECONDS_CAP=""               # SpecEvo-only wall cap per problem, seconds
WORKERS="4"
EVAL_PROCESSES="4"
PE_INTERVAL="10"
N_DIVERSE_SEEDS="5"
N_VARIANTS_PER_SEED="20"
ABLATION="full"

CONDA_ENV="${CONDA_ENV:-minhhieu}"
USE_CONDA=1
USE_TMUX=0
SESSION=""
OUTPUT_DIR=""
INSTALL_DEPS=1
DRY_RUN=0
FRESH=0
STATUS_ONLY=0

VALID_METHODS="specevo openevolve_native gepa_native adaevolve evox"
VALID_DOMAINS="chem_react bio_pop_growth phys_osc matsci"

usage() {
    cat <<'EOF'
Usage:
  run_lsr_synth.sh --method METHOD --domain DOMAIN [options]

Required
  --method NAME        specevo | openevolve_native | gepa_native | adaevolve | evox
                       ("blade" is accepted as an alias for specevo)
  --domain NAME        chem_react | bio_pop_growth | phys_osc | matsci

Problem selection
  --problems N         First N problems of the domain. (default: 10)
                       Use "all" for every problem in the domain — the full
                       LSR-Synth benchmark (25-44 problems per domain). Missing
                       problem directories are generated on demand.
  --full               Shorthand for --problems all.
  --problem-list LIST  Comma-separated problem ids instead, e.g. crk0,crk3,crk7.
                       Overrides --problems.

Data and scoring
  --score-mode MODE    log_nmse (default) | inv_nmse. Both rank hypotheses by
                       training NMSE; log_nmse reports it as decades below 1
                       (3.0 = NMSE 1e-3), which stays legible in the 4-decimal
                       score both runners put in their prompts. inv_nmse is the
                       older 1/(1+NMSE) scale, kept to reproduce earlier runs.
  --max-fit-points N   Subsample the training split to N points before the BFGS
                       fit. (default: 0 = the full split, i.e. the benchmark as
                       published; anything else is a smoke test, not a result)

Budget (applied per problem)
  --iterations N       Search iterations / evaluations per problem. (default: 500)
                       Exact for the baselines. For specevo, BLADE's bootstrap
                       (--n-diverse-seeds x --n-variants-per-seed, ~105 evals at
                       the defaults) always runs to completion first, so a value
                       below ~105 does not shrink the run — lower the two
                       bootstrap knobs as well for a small smoke test.
  --dollars N          USD cap per problem; the search stops gracefully at it.
  --seconds N          Wall-clock cap per problem in seconds (SpecEvo only).
  --problem-timeout N  Hard wall-clock cap per problem; the search is killed and
                       the run moves on, keeping whatever it found.
                       (default: 7200 = 2h; 0 disables)
  --eval-timeout N     Seconds allowed per equation hypothesis, i.e. the BFGS
                       parameter fit plus test-split scoring. (default: 30,
                       the LLM-SRBench paper's T = 30s per program hypothesis)

Models
  --model ID           Baseline model. (default: qwen3-30b-a3b-instruct-2507)
  --mutation-model ID  SpecEvo Speculator / small model. (default: same qwen3-30b)
  --paradigm-model ID  SpecEvo Navigator / frontier model. (default: same qwen3-30b)
  --embedding-model ID SpecEvo description embedder. (default: openai/text-embedding-3-small)

SpecEvo shape (ignored by baselines)
  --workers N          Concurrent LLM workers. (default: 4)
  --eval-processes N   Concurrent evaluator processes. (default: 4)
  --pe-interval N      Navigator (paradigm-shift) cadence. (default: 10)
  --n-diverse-seeds N  Phase-1 diverse seeds. (default: 5)
  --n-variants-per-seed N   Phase-2 variants per seed. (default: 20)
  --ablation NAME      full | ast_only | emb_only | static_cells | no_meta_advice
                       | meta_errors_only | no_targeted_mutate | no_crossover
                       | no_paradigm   (default: full)

Run control
  --seed N             Seed label; part of the output path. (default: 1)
  --output-dir DIR     Override the output directory.
                       (default: outputs/lsr_synth/<method>/<domain>/seed<N>)
  --fresh              Delete any existing results for this (method, domain, seed)
                       and start over. Without it, an existing run is RESUMED.
  --status             Print progress for this run and exit. No API calls.
  --tmux               Run detached in tmux (survives SSH disconnect).
  --session NAME       tmux session name. (default: derived from the run)
  --conda-env NAME     Conda env to activate. (default: $CONDA_ENV or minhhieu)
  --no-conda           Use whatever python is already active.
  --no-install-deps    Skip the dataset/dependency preparation step.
  --dry-run            Print the commands that would run, then exit.
  -h, --help           This help.

Examples
  # SpecEvo on all four domains, 500 evaluations per problem, detached:
  for d in chem_react bio_pop_growth phys_osc matsci; do
      ./scripts/server/run_lsr_synth.sh --method specevo --domain "$d" --tmux
  done

  # One baseline, cheaper smoke run:
  ./scripts/server/run_lsr_synth.sh --method evox --domain matsci \
      --problems 2 --iterations 20

  # The FULL benchmark: every problem in the domain (25-44), not just the first 10:
  ./scripts/server/run_lsr_synth.sh --method specevo --domain matsci --full --tmux

  # Interrupted by an exhausted API key? Re-run the identical command:
  ./scripts/server/run_lsr_synth.sh --method specevo --domain chem_react --tmux
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ===========================================================================
# Arguments
# ===========================================================================
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)              METHOD="$2"; shift 2 ;;
        --domain)              DOMAIN="$2"; shift 2 ;;
        --problems)            N_PROBLEMS="$2"; shift 2 ;;
        --full)                N_PROBLEMS="all"; shift ;;
        --problem-list)        PROBLEM_LIST="$2"; shift 2 ;;
        --score-mode)          SCORE_MODE="$2"; shift 2 ;;
        --max-fit-points)      MAX_FIT_POINTS="$2"; shift 2 ;;
        --iterations)          ITERATIONS="$2"; shift 2 ;;
        --dollars)             DOLLARS="$2"; shift 2 ;;
        --seconds)             SECONDS_CAP="$2"; shift 2 ;;
        --problem-timeout)     PROBLEM_TIMEOUT="$2"; shift 2 ;;
        --eval-timeout)        EVAL_TIMEOUT="$2"; shift 2 ;;
        --model)               MODEL="$2"; shift 2 ;;
        --mutation-model)      MUTATION_MODEL="$2"; shift 2 ;;
        --paradigm-model)      PARADIGM_MODEL="$2"; shift 2 ;;
        --embedding-model)     EMBEDDING_MODEL="$2"; shift 2 ;;
        --workers)             WORKERS="$2"; shift 2 ;;
        --eval-processes)      EVAL_PROCESSES="$2"; shift 2 ;;
        --pe-interval)         PE_INTERVAL="$2"; shift 2 ;;
        --n-diverse-seeds)     N_DIVERSE_SEEDS="$2"; shift 2 ;;
        --n-variants-per-seed) N_VARIANTS_PER_SEED="$2"; shift 2 ;;
        --ablation)            ABLATION="$2"; shift 2 ;;
        --seed)                SEED="$2"; shift 2 ;;
        --output-dir)          OUTPUT_DIR="$2"; shift 2 ;;
        --fresh)               FRESH=1; shift ;;
        --status)              STATUS_ONLY=1; shift ;;
        --tmux)                USE_TMUX=1; shift ;;
        --session)             SESSION="$2"; shift 2 ;;
        --conda-env)           CONDA_ENV="$2"; shift 2 ;;
        --no-conda)            USE_CONDA=0; shift ;;
        --no-install-deps)     INSTALL_DEPS=0; shift ;;
        --dry-run)             DRY_RUN=1; shift ;;
        -h|--help)             usage; exit 0 ;;
        *) die "unknown option '$1' (try --help)" ;;
    esac
done

[[ "$METHOD" == "blade" ]] && METHOD="specevo"
grep -qw -- "$METHOD" <<<"$VALID_METHODS" || die "invalid --method '$METHOD' (one of: $VALID_METHODS)"
[[ -n "$DOMAIN" ]] || die "--domain is required (one of: $VALID_DOMAINS)"
grep -qw -- "$DOMAIN" <<<"$VALID_DOMAINS" || die "invalid --domain '$DOMAIN' (one of: $VALID_DOMAINS)"
[[ "$ITERATIONS" =~ ^[0-9]+$ ]] || die "--iterations must be a whole number (got '$ITERATIONS')"
case "$N_PROBLEMS" in
    all|full) N_PROBLEMS="all" ;;
    *) [[ "$N_PROBLEMS" =~ ^[0-9]+$ ]] || die "--problems must be a whole number or 'all' (got '$N_PROBLEMS')" ;;
esac
case "$SCORE_MODE" in
    log_nmse|inv_nmse) ;;
    *) die "invalid --score-mode '$SCORE_MODE' (one of: log_nmse inv_nmse)" ;;
esac
[[ "$MAX_FIT_POINTS" =~ ^[0-9]+$ ]] || die "--max-fit-points must be a whole number (got '$MAX_FIT_POINTS')"
[[ "$PROBLEM_TIMEOUT" =~ ^[0-9]+$ ]] || die "--problem-timeout must be seconds (got '$PROBLEM_TIMEOUT')"
if [[ -n "$DOLLARS" ]]; then
    [[ "$DOLLARS" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--dollars must be a number of USD (got '$DOLLARS')"
fi
VALID_ABLATIONS="full ast_only emb_only static_cells no_meta_advice meta_errors_only no_targeted_mutate no_crossover no_paradigm"
grep -qw -- "$ABLATION" <<<"$VALID_ABLATIONS" || die "invalid --ablation '$ABLATION' (one of: $VALID_ABLATIONS)"

cd "$REPO_ROOT"

RUN_TAG="${METHOD}_${DOMAIN}_seed${SEED}"
[[ "$METHOD" == "specevo" && "$ABLATION" != "full" ]] && RUN_TAG="${METHOD}-${ABLATION}_${DOMAIN}_seed${SEED}"
if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ "$METHOD" == "specevo" && "$ABLATION" != "full" ]]; then
        OUTPUT_DIR="outputs/lsr_synth/${METHOD}-${ABLATION}/${DOMAIN}/seed${SEED}"
    else
        OUTPUT_DIR="outputs/lsr_synth/${METHOD}/${DOMAIN}/seed${SEED}"
    fi
fi
RESULTS="$OUTPUT_DIR/results.jsonl"
LOG_FILE="$OUTPUT_DIR/run.log"
PGID_FILE="$OUTPUT_DIR/.running_pgid"
[[ -n "$SESSION" ]] || SESSION="lsr_$RUN_TAG"
SESSION="${SESSION//./_}"; SESSION="${SESSION//:/_}"

PY="${PYTHON:-python}"

# ===========================================================================
# --status: report and exit (no conda, no API, no side effects)
# ===========================================================================
if [[ "$STATUS_ONLY" == "1" ]]; then
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        echo "No run found at $OUTPUT_DIR"
        exit 0
    fi
    echo "Run directory : $OUTPUT_DIR"
    "$PY" "$REPO_ROOT/scripts/lsr_summarize.py" "$OUTPUT_DIR" --progress || true
    exit 0
fi

# ===========================================================================
# tmux re-exec
# ===========================================================================
if [[ "$USE_TMUX" == "1" && "${LSR_INSIDE_TMUX:-0}" != "1" ]]; then
    command -v tmux >/dev/null 2>&1 || die "tmux is not installed (sudo apt-get install -y tmux)"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        die "a tmux session named '$SESSION' is already running" \
            "(attach with: tmux attach -t $SESSION, or pick another name with --session NAME)"
    fi
    mkdir -p "$OUTPUT_DIR"

    INNER=("$REPO_ROOT/scripts/server/run_lsr_synth.sh"
           --method "$METHOD" --domain "$DOMAIN"
           --problems "$N_PROBLEMS" --problem-list "$PROBLEM_LIST"
           --iterations "$ITERATIONS" --dollars "$DOLLARS" --seconds "$SECONDS_CAP"
           --problem-timeout "$PROBLEM_TIMEOUT" --eval-timeout "$EVAL_TIMEOUT"
           --score-mode "$SCORE_MODE" --max-fit-points "$MAX_FIT_POINTS"
           --model "$MODEL" --mutation-model "$MUTATION_MODEL"
           --paradigm-model "$PARADIGM_MODEL" --embedding-model "$EMBEDDING_MODEL"
           --workers "$WORKERS" --eval-processes "$EVAL_PROCESSES"
           --pe-interval "$PE_INTERVAL" --n-diverse-seeds "$N_DIVERSE_SEEDS"
           --n-variants-per-seed "$N_VARIANTS_PER_SEED" --ablation "$ABLATION"
           --seed "$SEED" --output-dir "$OUTPUT_DIR" --conda-env "$CONDA_ENV")
    [[ "$USE_CONDA"    == "0" ]] && INNER+=(--no-conda)
    [[ "$INSTALL_DEPS" == "0" ]] && INNER+=(--no-install-deps)
    [[ "$FRESH"        == "1" ]] && INNER+=(--fresh)
    [[ "$DRY_RUN"      == "1" ]] && INNER+=(--dry-run)

    QUOTED="$(printf '%q ' "${INNER[@]}")"
    # A tmux pane inherits the environment of the tmux *server*, which may predate
    # conda being on PATH; pin the PATH/HOME we can see now.
    TMUX_CMD="LSR_INSIDE_TMUX=1 PATH=$(printf '%q' "$PATH") HOME=$(printf '%q' "$HOME") ${QUOTED}; rc=\$?; echo; echo \"=== run finished (exit \$rc) — press Enter to close ===\"; read -r _"
    tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$TMUX_CMD"

    cat <<EOF

Started in tmux.
  session     : $SESSION
  output dir  : $OUTPUT_DIR
  log file    : $LOG_FILE

Watch it live  :  tmux attach -t $SESSION      (detach: Ctrl-b then d)
Follow the log :  tail -f $LOG_FILE
Progress       :  ./scripts/server/run_lsr_synth.sh --method $METHOD --domain $DOMAIN --seed $SEED --status
Stop the run   :  tmux kill-session -t $SESSION
Resume later   :  re-run the exact same command (finished problems are skipped)

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

if [[ "$FRESH" == "1" && -d "$OUTPUT_DIR" ]]; then
    echo ">>> --fresh: removing $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR/problems"

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
# Read by benchmarks/llm_srbench/lsr_eval.py on both integration paths.
export LSR_EVAL_TIMEOUT="$EVAL_TIMEOUT"
export LSR_SCORE_MODE="$SCORE_MODE"
export LSR_MAX_FIT_POINTS="$MAX_FIT_POINTS"

if [[ "$DRY_RUN" != "1" && -z "${OPENROUTER_API_KEY:-}" ]]; then
    die "no API key found. Create $REPO_ROOT/.env containing: OPENAI_API_KEY=sk-or-v1-..."
fi

# ===========================================================================
# Problem list
# ===========================================================================
if [[ -n "$PROBLEM_LIST" ]]; then
    PROBLEMS="${PROBLEM_LIST//,/ }"
else
    PROBLEMS="$(LSR_DOMAIN="$DOMAIN" LSR_LIMIT="$N_PROBLEMS" "$PY" - <<'PYEOF'
import os, sys
sys.path.insert(0, "benchmarks/llm_srbench")
import lsr_eval as L
raw = os.environ["LSR_LIMIT"]
ids = L.problem_ids(os.environ["LSR_DOMAIN"])
print(" ".join(ids if raw == "all" else ids[: int(raw)]))
PYEOF
)" || die "could not list problems — has the dataset been prepared? (python benchmarks/llm_srbench/prepare_data.py)"
fi
[[ -n "${PROBLEMS// /}" ]] || die "no problems selected"
N_TOTAL="$(wc -w <<<"$PROBLEMS" | tr -d ' ')"

# ===========================================================================
# Header
# ===========================================================================
{
    echo "============================================================"
    echo " LLM-SRBench / LSR-Synth"
    echo " method        : $METHOD${ABLATION:+ (ablation: $ABLATION)}"
    echo " domain        : $DOMAIN"
    echo " problems      : $N_TOTAL  ($PROBLEMS)"
    echo " iterations    : $ITERATIONS per problem"
    echo " score mode    : $SCORE_MODE"
    echo " train points  : $([[ "$MAX_FIT_POINTS" == "0" ]] && echo 'full split' || echo "subsampled to $MAX_FIT_POINTS (SMOKE TEST, not a result)")"
    echo " eval timeout  : ${EVAL_TIMEOUT}s per hypothesis"
    echo " problem cap   : ${PROBLEM_TIMEOUT}s wall-clock$([[ "$PROBLEM_TIMEOUT" == "0" ]] && echo ' (disabled)')"
    if [[ "$METHOD" == "specevo" ]]; then
        echo " models        : speculator=$MUTATION_MODEL"
        echo "                 navigator  =$PARADIGM_MODEL"
        echo "                 embedding  =$EMBEDDING_MODEL"
    else
        echo " model         : $MODEL"
    fi
    echo " budget/problem: dollars='${DOLLARS}' seconds='${SECONDS_CAP}'"
    echo " seed label    : $SEED"
    echo " python        : $(command -v "$PY")  ($("$PY" -V 2>&1))"
    echo " conda env     : ${CONDA_DEFAULT_ENV:-<none>}"
    echo " output dir    : $OUTPUT_DIR"
    echo " started at    : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "============================================================"
} | tee -a "$LOG_FILE"

# ===========================================================================
# Dataset + dependency preparation
# ===========================================================================
if [[ "$INSTALL_DEPS" == "1" && "$DRY_RUN" != "1" ]]; then
    # Prepare every domain, not just this one: the manifest is a single file, and
    # a per-domain preparation would have to merge into it while three sibling
    # runs may be doing the same. Preparing all four is ~9 MB, idempotent, and
    # writes nothing once complete.
    echo ">>> Preparing LSR-Synth data" | tee -a "$LOG_FILE"
    "$PY" benchmarks/llm_srbench/prepare_data.py 2>&1 | tee -a "$LOG_FILE"
fi

# ===========================================================================
# Problem directories
#
# Only the reduced set (the first 10 problems of each domain) is committed, so a
# --full or --problem-list run needs the rest generating. They are also
# score-mode-specific: the baseline config.yaml embeds the task prompt, which
# quotes the active scoring rule, so a directory generated under a different
# --score-mode must be rewritten or the model would be told the wrong rule. (The
# SpecEvo path builds its prompt at import time and is never stale.)
# ===========================================================================
ensure_problem_dirs() {
    # The revision generate_dirs.py currently writes. A directory stamped with
    # anything older predates a change to the emitted config (prompt trimming,
    # parallelism, ...) and is regenerated rather than run as-is.
    local want_rev
    want_rev="$("$PY" -c 'import sys; sys.path.insert(0, "benchmarks/llm_srbench"); import generate_dirs as g; print(g.CONFIG_REV)')"

    local missing=() pid cfg
    for pid in $PROBLEMS; do
        cfg="$BDIR_ROOT/$pid/config.yaml"
        if [[ ! -f "$cfg" ]] \
           || [[ ! -f "$BDIR_ROOT/$pid/evaluator.py" ]] \
           || [[ ! -f "$BDIR_ROOT/$pid/initial_program.py" ]] \
           || [[ ! -f "$LEVI_ROOT/$pid/problem.py" ]] \
           || ! grep -q "^# score-mode: $SCORE_MODE\$" "$cfg" \
           || ! grep -q "^# config-rev: $want_rev\$" "$cfg"; then
            missing+=("$pid")
        fi
    done
    [[ ${#missing[@]} -eq 0 ]] && return 0
    echo ">>> generating ${#missing[@]} problem director$([[ ${#missing[@]} -eq 1 ]] && echo y || echo ies)" \
         "for $DOMAIN (score-mode=$SCORE_MODE)" | tee -a "$LOG_FILE"
    local args=(--domain "$DOMAIN" --iterations "$ITERATIONS" --model "$MODEL")
    for pid in "${missing[@]}"; do args+=(--problem "$pid"); done
    "$PY" benchmarks/llm_srbench/generate_dirs.py "${args[@]}" 2>&1 | tee -a "$LOG_FILE"
}

# ===========================================================================
# Graceful interrupt: stop the current search, record nothing for it, and start
# nothing new — so the next run of the same command resumes cleanly.
#
# HUP matters as much as INT/TERM here: `tmux kill-session` and a dropped SSH
# connection both send SIGHUP, and without a handler bash exits immediately and
# leaves the search running as an orphan — still spending API credit with nobody
# left to collect its results.
# ===========================================================================
STOP_REQUESTED=0
CURRENT_CHILD=""
on_signal() {
    STOP_REQUESTED=1
    {
        echo ""
        echo ">>> interrupt received — stopping the current problem and exiting"
    } | tee -a "$LOG_FILE"
    [[ -n "$CURRENT_CHILD" ]] && kill -TERM "-$CURRENT_CHILD" 2>/dev/null || true
}
on_exit() {
    # Belt and braces for every other way this script can die (an unhandled
    # signal, `set -e`, an OOM kill of the driver): never outlive the search.
    [[ -n "$CURRENT_CHILD" ]] && kill -TERM "-$CURRENT_CHILD" 2>/dev/null || true
    return 0
}
trap on_signal INT TERM HUP
trap on_exit EXIT

# Run "$@" with its output going to both the console and the run log, under an
# optional wall-clock cap. Returns 124 on timeout, mirroring coreutils `timeout`.
#
# Two details matter for interrupt handling:
#   * the child is launched with `set -m` so it gets its OWN process group, which
#     lets us signal it (and everything it spawned) without signalling ourselves;
#   * the tee is a process substitution rather than a pipeline, because
#     `run_guarded ... | tee` would run run_guarded in a subshell — CURRENT_CHILD
#     would be set in that subshell and the signal handler in this shell would
#     have nothing to kill.
run_guarded() {
    local limit="$1"; shift
    set -m
    "$@" > >(tee -a "$LOG_FILE") 2>&1 &
    local pid=$!
    set +m
    CURRENT_CHILD="$pid"
    # Leave a breadcrumb so the next run can clean up after a SIGKILL'd driver
    # (see reap_orphan): two searches writing one output directory would corrupt
    # both. `set -m` makes the pid equal to the process-group id.
    printf '%s' "$pid" > "$PGID_FILE"
    local waited=0 rc=0
    while kill -0 "$pid" 2>/dev/null; do
        if [[ "$limit" != "0" && "$waited" -ge "$limit" ]]; then
            echo ">>> problem timeout after ${limit}s — terminating" | tee -a "$LOG_FILE"
            kill -TERM "-$pid" 2>/dev/null || true
            sleep 15
            kill -KILL "-$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            CURRENT_CHILD=""
            return 124
        fi
        sleep 5
        waited=$((waited + 5))
    done
    wait "$pid" || rc=$?
    CURRENT_CHILD=""
    rm -f "$PGID_FILE"
    # The tee in the process substitution is not our child, so we cannot wait for
    # it; give it a moment to drain the pipe before the caller writes its own
    # lines, or the log ends up slightly out of order.
    sleep 1
    return "$rc"
}

# Kill a search left behind by a driver that was SIGKILL'd (or a machine that
# went down mid-run). Only ever touches a process whose command line still looks
# like one of our searches, so a recycled pid cannot be mistaken for one.
reap_orphan() {
    [[ -f "$PGID_FILE" ]] || return 0
    local pgid cmd
    pgid="$(cat "$PGID_FILE" 2>/dev/null || true)"
    rm -f "$PGID_FILE"
    [[ "$pgid" =~ ^[0-9]+$ ]] || return 0
    kill -0 "$pgid" 2>/dev/null || return 0
    cmd="$(ps -o command= -p "$pgid" 2>/dev/null || true)"
    case "$cmd" in
        *skydiscover-run*|*run_blade.py*) ;;
        *) return 0 ;;
    esac
    echo ">>> a previous search (pgid $pgid) is still running — terminating it before resuming" \
        | tee -a "$LOG_FILE"
    kill -TERM "-$pgid" 2>/dev/null || true
    sleep 10
    kill -KILL "-$pgid" 2>/dev/null || true
}

# Keep only the newest two baseline checkpoints. Each one serialises the entire
# program database, so 500 iterations at one checkpoint per 10 would otherwise
# leave ~50 snapshots per problem on disk; resume only ever reads the newest, and
# the runner-up is kept in case the newest was written during a kill.
prune_checkpoints() {
    local run_dir="$1" keep=2 dirs=() n
    for d in "$run_dir"/run/checkpoints/checkpoint_*; do
        [[ -d "$d" ]] || continue
        n="${d##*checkpoint_}"
        [[ "$n" =~ ^[0-9]+$ ]] && dirs+=("$n")
    done
    [[ ${#dirs[@]} -le $keep ]] && return 0
    # Numeric sort, drop the tail we want to keep, remove the rest.
    local sorted
    sorted="$(printf '%s\n' "${dirs[@]}" | sort -n | head -n $(( ${#dirs[@]} - keep )))"
    for n in $sorted; do
        rm -rf "$run_dir/run/checkpoints/checkpoint_$n"
    done
}

# ===========================================================================
# Main loop
# ===========================================================================
reap_orphan     # defined above; must run before we start writing to this run dir

IDX=0
N_DONE=0
N_SKIPPED=0
N_FAILED=0
BDIR_ROOT="benchmarks/llm_srbench/$DOMAIN"
LEVI_ROOT="levi/examples/llm_srbench/$DOMAIN"

if [[ "$DRY_RUN" != "1" ]]; then
    ensure_problem_dirs
fi

for PID_NAME in $PROBLEMS; do
    IDX=$((IDX + 1))
    if [[ "$STOP_REQUESTED" == "1" ]]; then
        echo ">>> stopping before $DOMAIN/$PID_NAME (interrupted)" | tee -a "$LOG_FILE"
        break
    fi

    PDIR="$OUTPUT_DIR/problems/$PID_NAME"
    if [[ -f "$PDIR/.done" ]]; then
        echo ">>> [$IDX/$N_TOTAL] $DOMAIN/$PID_NAME already finished — skipping" | tee -a "$LOG_FILE"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    mkdir -p "$PDIR"

    {
        echo ""
        echo "------------------------------------------------------------"
        echo ">>> [$IDX/$N_TOTAL] $DOMAIN/$PID_NAME   ($METHOD)   $(date '+%H:%M:%S')"
    } | tee -a "$LOG_FILE"

    ITER_THIS="$ITERATIONS"
    RESUME_ARGS=()

    if [[ "$METHOD" == "specevo" ]]; then
        EDIR="$LEVI_ROOT/$PID_NAME"
        [[ -d "$EDIR" ]] || die "missing $EDIR (ensure_problem_dirs should have created it)"
        # BLADE cannot resume a partial search, so an interrupted attempt is moved
        # aside (never deleted) and the problem restarts.
        if [[ -e "$PDIR/summary.json" || -e "$PDIR/snapshot.json" || -e "$PDIR/best_program.py" ]]; then
            ARCHIVE="$PDIR/prev_attempt_$(date +%Y%m%d-%H%M%S)"
            mkdir -p "$ARCHIVE"
            find "$PDIR" -maxdepth 1 -mindepth 1 ! -name 'prev_attempt_*' -exec mv {} "$ARCHIVE"/ \;
            echo ">>> previous partial attempt archived to $ARCHIVE (SpecEvo restarts the problem)" \
                | tee -a "$LOG_FILE"
        fi
        # Our own 30s-per-hypothesis alarm must fire before BLADE's outer process
        # timeout, so the hypothesis is graded 0 rather than the worker being killed.
        BLADE_EVAL_TIMEOUT=$((EVAL_TIMEOUT + 60))
        # The no_paradigm ablation is expressed by silencing the Navigator cadence.
        PE_INTERVAL_EFFECTIVE="$PE_INTERVAL"
        [[ "$ABLATION" == "no_paradigm" ]] && PE_INTERVAL_EFFECTIVE="0"
        CMD=("$PY" "$REPO_ROOT/scripts/run_blade.py"
             --example-dir "$EDIR"
             --problem-module problem
             --mutation-model "$MUTATION_MODEL"
             --paradigm-model "$PARADIGM_MODEL"
             --embedding-model "$EMBEDDING_MODEL"
             --evals "$ITER_THIS"
             --dollars "$DOLLARS"
             --seconds "$SECONDS_CAP"
             --workers "$WORKERS"
             --eval-processes "$EVAL_PROCESSES"
             --eval-timeout "$BLADE_EVAL_TIMEOUT"
             --pe-interval "$PE_INTERVAL_EFFECTIVE"
             --n-diverse-seeds "$N_DIVERSE_SEEDS"
             --n-variants-per-seed "$N_VARIANTS_PER_SEED"
             --output-dir "$REPO_ROOT/$PDIR")
        case "$ABLATION" in
            ast_only)           CMD+=(--ast-only) ;;
            emb_only)           CMD+=(--emb-only) ;;
            static_cells)       CMD+=(--static-cells) ;;
            no_meta_advice)     CMD+=(--no-meta-advice) ;;
            meta_errors_only)   CMD+=(--meta-advice-mode errors_only) ;;
            no_targeted_mutate) CMD+=(--no-targeted-mutate) ;;
            no_crossover)       CMD+=(--no-crossover) ;;
            no_paradigm|full)   ;;   # no_paradigm is handled via PE_INTERVAL_EFFECTIVE
        esac
    else
        BDIR="$BDIR_ROOT/$PID_NAME"
        [[ -d "$BDIR" ]] || die "missing $BDIR (ensure_problem_dirs should have created it)"
        for f in initial_program.py config.yaml evaluator.py; do
            [[ -f "$BDIR/$f" ]] || die "missing $BDIR/$f"
        done
        # Also prune here, not just after the search: a run killed hard (SIGKILL,
        # reboot) never reaches the post-search prune, and its checkpoints would
        # otherwise accumulate across attempts.
        prune_checkpoints "$PDIR"
        # skydiscover-run's --iterations counts ADDITIONAL iterations on top of a
        # checkpoint, so a resumed problem is given only what it still owes. How
        # much it has already had is measured from the checkpoint's program
        # database rather than its name — see scripts/lsr_resume_plan.py for why
        # the name cannot be trusted after an early shutdown.
        LSR_CKPT=""; LSR_COMPLETED=0; LSR_REMAINING="$ITERATIONS"
        PLAN="$("$PY" "$REPO_ROOT/scripts/lsr_resume_plan.py" \
                    --run-dir "$PDIR" --iterations "$ITERATIONS" 2>/dev/null || true)"
        if [[ -n "$PLAN" ]]; then
            eval "$PLAN"
        fi
        ITER_THIS="$LSR_REMAINING"
        if [[ -n "$LSR_CKPT" ]]; then
            RESUME_ARGS=(--checkpoint "$LSR_CKPT")
            if [[ "$ITER_THIS" -le 0 ]]; then
                echo ">>> already ran $LSR_COMPLETED/$ITERATIONS iterations — finalising without more search" \
                    | tee -a "$LOG_FILE"
                ITER_THIS=0
            else
                echo ">>> resuming from $(basename "$LSR_CKPT")" \
                     "($LSR_COMPLETED/$ITERATIONS done, $ITER_THIS to go)" | tee -a "$LOG_FILE"
            fi
        fi
        export SKYDISCOVER_COST_LOG="$PDIR/cost_log.jsonl"
        CMD=(skydiscover-run "$BDIR/initial_program.py" "$BDIR/evaluator.py"
             --config "$BDIR/config.yaml"
             --search "$METHOD"
             --model "$MODEL"
             --iterations "$ITER_THIS"
             --dollars "$DOLLARS"
             --output "$PDIR/run"
             ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"})
    fi

    {
        echo ">>> command:"
        printf '     '; printf ' %q' "${CMD[@]}"; echo
    } | tee -a "$LOG_FILE"

    if [[ "$DRY_RUN" == "1" ]]; then
        continue
    fi

    RC=0
    if [[ "$ITER_THIS" -gt 0 ]]; then
        set +e
        run_guarded "$PROBLEM_TIMEOUT" "${CMD[@]}"
        RC=$?
        set -e
        if [[ "$RC" == "124" ]]; then
            echo ">>> $DOMAIN/$PID_NAME hit the wall-clock cap; keeping partial results" | tee -a "$LOG_FILE"
        elif [[ "$RC" != "0" ]]; then
            echo ">>> $DOMAIN/$PID_NAME search exited $RC" | tee -a "$LOG_FILE"
        fi
    fi

    [[ "$METHOD" == "specevo" ]] || prune_checkpoints "$PDIR"

    # An interrupt during the search must NOT be recorded as a finished problem:
    # leave it unmarked so the next run resumes (baselines) or retries (SpecEvo).
    if [[ "$STOP_REQUESTED" == "1" ]]; then
        echo ">>> $DOMAIN/$PID_NAME left unfinished (interrupted) — re-run to continue" | tee -a "$LOG_FILE"
        break
    fi

    set +e
    "$PY" "$REPO_ROOT/scripts/lsr_finalize.py" \
        --domain "$DOMAIN" --problem "$PID_NAME" --method "$METHOD" --seed "$SEED" \
        --run-dir "$PDIR" --results "$RESULTS" --iterations "$ITERATIONS" 2>&1 | tee -a "$LOG_FILE"
    FRC="${PIPESTATUS[0]}"
    set -e

    if [[ "$FRC" == "0" ]]; then
        N_DONE=$((N_DONE + 1))
    else
        N_FAILED=$((N_FAILED + 1))
        echo ">>> $DOMAIN/$PID_NAME produced no usable equation (recorded as failed)" | tee -a "$LOG_FILE"
    fi
    # Marked done either way: the problem was attempted to its full budget and its
    # outcome is in results.jsonl. Only an interrupt leaves it unmarked.
    date '+%Y-%m-%dT%H:%M:%S%z' > "$PDIR/.done"

    # Refresh the aggregate after every problem so a run that is killed later
    # still leaves a valid summary on disk.
    "$PY" "$REPO_ROOT/scripts/lsr_summarize.py" "$OUTPUT_DIR" \
        --json "$OUTPUT_DIR/summary.json" --quiet || true
done

# ===========================================================================
# Footer
# ===========================================================================
{
    echo ""
    echo "============================================================"
    echo " finished at : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo " problems    : $N_DONE scored, $N_SKIPPED already done, $N_FAILED without an equation"
    echo " results     : $RESULTS"
    echo " summary     : $OUTPUT_DIR/summary.json"
    echo "============================================================"
} | tee -a "$LOG_FILE"

if [[ "$DRY_RUN" != "1" ]]; then
    "$PY" "$REPO_ROOT/scripts/lsr_summarize.py" "$OUTPUT_DIR" \
        --json "$OUTPUT_DIR/summary.json" 2>&1 | tee -a "$LOG_FILE" || true
fi

if [[ "$DRY_RUN" == "1" ]]; then
    # Nothing was attempted, so "unfinished" is not a meaningful verdict.
    exit 0
fi

REMAINING=0
for PID_NAME in $PROBLEMS; do
    [[ -f "$OUTPUT_DIR/problems/$PID_NAME/.done" ]] || REMAINING=$((REMAINING + 1))
done
if [[ "$REMAINING" -gt 0 ]]; then
    {
        echo ""
        echo ">>> $REMAINING problem(s) still unfinished. Resume with the same command:"
        echo "    ./scripts/server/run_lsr_synth.sh --method $METHOD --domain $DOMAIN --seed $SEED --iterations $ITERATIONS"
    } | tee -a "$LOG_FILE"
    exit 3
fi
exit 0
