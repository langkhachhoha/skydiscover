#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# symbolic_accuracy.sh — score finished LSR-Synth runs on the paper's *other*
# axis: symbolic accuracy, judged by GPT-4o.
#
#   ./scripts/server/symbolic_accuracy.sh                       # everything under outputs/lsr_synth
#   ./scripts/server/symbolic_accuracy.sh --methods specevo,openevolve_native
#   ./scripts/server/symbolic_accuracy.sh --domain matsci --workers 16
#
# No search is re-run and no dataset is touched. Every results.jsonl record
# already carries the discovered program and the ground-truth equation, so this
# only asks GPT-4o, per problem: could any constant parameter values make the
# discovered equation equivalent to the ground truth? (LLM-SRBench App. B.2,
# Fig. 11.) The answer rate per domain is the paper's SA column.
#
# Judgements are appended to <out-dir>/judgments.jsonl and reused, so an
# interrupted sweep resumes for free and adding a new baseline later only pays
# for that baseline. Use --fresh to re-judge from scratch.
#
# Only methods that have actually run are scored. Domains that cover fewer
# problems than the dataset holds are marked with * — that number is not yet the
# full-dataset figure.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ===========================================================================
# Defaults
# ===========================================================================
PATHS=()
METHODS=""
DOMAINS=""
LIMIT="0"
MODEL="${SA_MODEL:-}"        # default: openai/gpt-4o (OpenRouter) or gpt-4o
BASE_URL=""
# Judge sampling temperature. Edit this line, export SA_TEMPERATURE, or pass
# --temperature. The paper never states the evaluator's temperature (its
# tau = 0.8 in Table 2 is the *discovery* methods' sampling), so 0.0 is this
# repo's choice: the same (ground truth, hypothesis) pair should not change
# verdict between runs, and judgements are cached per temperature.
TEMPERATURE="${SA_TEMPERATURE:-0.0}"
WORKERS="${SA_WORKERS:-8}"
MAX_RETRIES="4"
GT_CONSTANTS="symbol"
OUT_DIR="outputs/symbolic_accuracy"
CSV=""
JSON=""
CONDA_ENV="${CONDA_ENV:-minhhieu}"
USE_CONDA=1
USE_TMUX=0
SESSION="lsr_symbolic_accuracy"
FRESH=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  symbolic_accuracy.sh [PATHS...] [options]

  PATHS   Run directories, trees of them, or results.jsonl files.
          (default: outputs/lsr_synth — every method and domain found there)

Selection
  --methods LIST     Comma-separated methods to score, e.g. specevo,openevolve_native.
                     (default: every method present in PATHS)
  --domain NAME      Single domain: chem_react | bio_pop_growth | phys_osc | matsci.
  --domains LIST     Comma-separated domains instead.
  --limit N          Judge at most N problems per (method, domain). Smoke tests
                     only — the reported percentage is then not the benchmark.

Judge
  --model ID         Judge model. (default: openai/gpt-4o via OpenRouter, or
                     gpt-4o against api.openai.com — matches the paper)
  --base-url URL     OpenAI-compatible endpoint. (default: from .env / the key)
  --temperature N    Judge sampling temperature. (default: 0.0 — the verdict
                     should not wobble between runs. The paper does not state
                     the evaluator's temperature. Also settable with
                     SA_TEMPERATURE, or by editing the Defaults block.)
  --workers N        Parallel judge calls. (default: 8, or SA_WORKERS)
  --max-retries N    Retries per call. (default: 4)
  --gt-constants M   symbol (default) | shared | raw — how the ground truth's
                     fitted constants are placeholdered before judging.

Output
  --out-dir DIR      judgments.jsonl + symbolic_accuracy.json land here.
                     (default: outputs/symbolic_accuracy)
  --csv FILE         Also write the per-domain table as CSV.
  --json FILE        Write the summary JSON here instead of the default path.
  --fresh            Ignore cached judgements and judge everything again.

Run control
  --tmux             Run detached in tmux (survives SSH disconnect).
  --session NAME     tmux session name. (default: lsr_symbolic_accuracy)
  --conda-env NAME   Conda env to activate. (default: $CONDA_ENV or minhhieu)
  --no-conda         Use whatever python is already active.
  --dry-run          Show the task counts and one example prompt, call no API.
  -h, --help         This help.

Examples
  # Everything finished so far, full dataset, 16 calls in flight:
  ./scripts/server/symbolic_accuracy.sh --workers 16

  # Just the two methods that have run, writing a CSV for the paper table:
  ./scripts/server/symbolic_accuracy.sh --methods specevo,openevolve_native \
      --csv outputs/symbolic_accuracy/table.csv

  # Check what would be judged, and what the judge would see, for free:
  ./scripts/server/symbolic_accuracy.sh --dry-run
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ===========================================================================
# Arguments
# ===========================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)       METHODS="$2"; shift 2 ;;
        --domain)        DOMAINS="$2"; shift 2 ;;
        --domains)       DOMAINS="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --base-url)      BASE_URL="$2"; shift 2 ;;
        --temperature)   TEMPERATURE="$2"; shift 2 ;;
        --workers)       WORKERS="$2"; shift 2 ;;
        --max-retries)   MAX_RETRIES="$2"; shift 2 ;;
        --gt-constants)  GT_CONSTANTS="$2"; shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        --csv)           CSV="$2"; shift 2 ;;
        --json)          JSON="$2"; shift 2 ;;
        --fresh)         FRESH=1; shift ;;
        --tmux)          USE_TMUX=1; shift ;;
        --session)       SESSION="$2"; shift 2 ;;
        --conda-env)     CONDA_ENV="$2"; shift 2 ;;
        --no-conda)      USE_CONDA=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        -*)              die "unknown option '$1' (try --help)" ;;
        *)               PATHS+=("$1"); shift ;;
    esac
done

[[ ${#PATHS[@]} -gt 0 ]] || PATHS=("outputs/lsr_synth")
[[ "$WORKERS" =~ ^[0-9]+$ ]] || die "--workers must be a whole number (got '$WORKERS')"
[[ "$LIMIT" =~ ^[0-9]+$ ]] || die "--limit must be a whole number (got '$LIMIT')"
case "$GT_CONSTANTS" in
    symbol|shared|raw) ;;
    *) die "invalid --gt-constants '$GT_CONSTANTS' (one of: symbol shared raw)" ;;
esac

cd "$REPO_ROOT"
SESSION="${SESSION//./_}"; SESSION="${SESSION//:/_}"

# ===========================================================================
# tmux re-exec
# ===========================================================================
if [[ "$USE_TMUX" == "1" && "${SA_INSIDE_TMUX:-0}" != "1" ]]; then
    command -v tmux >/dev/null 2>&1 || die "tmux is not installed (sudo apt-get install -y tmux)"
    tmux has-session -t "$SESSION" 2>/dev/null \
        && die "a tmux session named '$SESSION' is already running (attach: tmux attach -t $SESSION)"

    INNER=("$REPO_ROOT/scripts/server/symbolic_accuracy.sh" "${PATHS[@]}"
           --methods "$METHODS" --domains "$DOMAINS" --limit "$LIMIT"
           --model "$MODEL" --base-url "$BASE_URL" --temperature "$TEMPERATURE"
           --workers "$WORKERS" --max-retries "$MAX_RETRIES"
           --gt-constants "$GT_CONSTANTS" --out-dir "$OUT_DIR"
           --csv "$CSV" --json "$JSON" --session "$SESSION" --conda-env "$CONDA_ENV")
    [[ "$USE_CONDA" == "0" ]] && INNER+=(--no-conda)
    [[ "$FRESH"     == "1" ]] && INNER+=(--fresh)
    [[ "$DRY_RUN"   == "1" ]] && INNER+=(--dry-run)

    mkdir -p "$OUT_DIR"
    QUOTED="$(printf '%q ' "${INNER[@]}")"
    TMUX_CMD="SA_INSIDE_TMUX=1 PATH=$(printf '%q' "$PATH") HOME=$(printf '%q' "$HOME") ${QUOTED}; rc=\$?; echo; echo \"=== judging finished (exit \$rc) — press Enter to close ===\"; read -r _"
    tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$TMUX_CMD"
    cat <<EOF

Started in tmux.
  session  : $SESSION
  log file : $OUT_DIR/judge.log

Watch it live  :  tmux attach -t $SESSION      (detach: Ctrl-b then d)
Follow the log :  tail -f $OUT_DIR/judge.log
Stop it        :  tmux kill-session -t $SESSION

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

# ===========================================================================
# Environment
# ===========================================================================
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi
if [[ -z "${OPENAI_API_KEY:-}" && -n "${OPENROUTER_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="$OPENROUTER_API_KEY"
fi
export OPENAI_API_KEY
export PYTHONUNBUFFERED=1

if [[ "$DRY_RUN" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
    die "no API key found. Create $REPO_ROOT/.env containing: OPENAI_API_KEY=sk-or-v1-..."
fi

PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || die "python not found on PATH"

mkdir -p "$OUT_DIR"
if [[ "$FRESH" == "1" && -f "$OUT_DIR/judgments.jsonl" ]]; then
    echo ">>> --fresh: discarding cached judgements in $OUT_DIR/judgments.jsonl"
    mv "$OUT_DIR/judgments.jsonl" "$OUT_DIR/judgments.jsonl.bak"
fi

# ===========================================================================
# Run
# ===========================================================================
CMD=("$PY" "$REPO_ROOT/scripts/lsr_symbolic_accuracy.py" "${PATHS[@]}"
     --workers "$WORKERS" --max-retries "$MAX_RETRIES" --temperature "$TEMPERATURE"
     --gt-constants "$GT_CONSTANTS" --out-dir "$OUT_DIR" --expect-full)
[[ -n "$METHODS"  ]] && CMD+=(--methods "$METHODS")
[[ -n "$DOMAINS"  ]] && CMD+=(--domains "$DOMAINS")
[[ "$LIMIT" != "0" ]] && CMD+=(--limit "$LIMIT")
[[ -n "$MODEL"    ]] && CMD+=(--model "$MODEL")
[[ -n "$BASE_URL" ]] && CMD+=(--base-url "$BASE_URL")
[[ -n "$CSV"      ]] && CMD+=(--csv "$CSV")
[[ -n "$JSON"     ]] && CMD+=(--json "$JSON")
[[ "$DRY_RUN" == "1" ]] && CMD+=(--dry-run)

{
    echo "============================================================"
    echo " LLM-SRBench / LSR-Synth — symbolic accuracy"
    echo " paths     : ${PATHS[*]}"
    echo " methods   : ${METHODS:-all present}"
    echo " domains   : ${DOMAINS:-all present}"
    echo " judge     : ${MODEL:-gpt-4o (default)}  temperature=$TEMPERATURE  workers=$WORKERS"
    echo " gt consts : $GT_CONSTANTS"
    echo " out dir   : $OUT_DIR"
    echo " started   : $(date +%Y-%m-%dT%H:%M:%S%z)"
    echo "============================================================"
} | tee -a "$OUT_DIR/judge.log"

"${CMD[@]}" 2>&1 | tee -a "$OUT_DIR/judge.log"
