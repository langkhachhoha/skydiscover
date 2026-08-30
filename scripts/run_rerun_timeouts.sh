#!/usr/bin/env bash
#
# Launch scripts/rerun_timeout_evals.py in a detached tmux session, so a run
# that takes hours survives the SSH connection dropping.
#
#   ./scripts/run_rerun_timeouts.sh                  # relay_cpr400, 8 jobs, 60s
#   JOBS=16 ./scripts/run_rerun_timeouts.sh
#   ./scripts/run_rerun_timeouts.sh relay_cpr400
#
# The interpreter is resolved *here*, in the shell you activated your
# environment in, and handed to the worker as an absolute path.  A tmux
# session started from a tmux server that predates `conda activate` would
# otherwise inherit a stale PATH and reach for the wrong python.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ROOT="${1:-relay_cpr400}"
SESSION="${SESSION:-rerun_timeouts}"
JOBS="${JOBS:-8}"
TIMEOUT="${TIMEOUT:-60}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/rerun_timeouts}"
LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

die() { echo "error: $*" >&2; exit 1; }

command -v tmux >/dev/null || die "tmux is not installed (apt-get install tmux)"
[ -n "$PYTHON" ] || die "no python found on PATH"
[ -d "$ROOT" ] || die "no such directory: $ROOT (run this from the repo, or pass the path)"

# Fail here rather than inside tmux, where you would have to go hunting for
# the reason all 1443 candidates failed identically.  The environment you
# activated is tried first; a project venv is only a fallback, and whichever
# one is picked gets printed below so the choice is never a surprise.
usable() { [ -x "$1" ] && "$1" -c 'import numpy, scipy' >/dev/null 2>&1; }

if ! usable "$PYTHON"; then
    FOUND=""
    for CAND in "${CONDA_PREFIX:-}/bin/python" "${VIRTUAL_ENV:-}/bin/python" \
                "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/venv/bin/python"; do
        if usable "$CAND"; then FOUND="$CAND"; break; fi
    done
    [ -n "$FOUND" ] || die \
"$PYTHON cannot import numpy/scipy, and no fallback environment could either.
Activate your environment first (e.g. conda activate minhhieu), or set PYTHON=/path/to/python."
    echo "note: $PYTHON lacks numpy/scipy, falling back to $FOUND" >&2
    PYTHON="$FOUND"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    die "tmux session '$SESSION' already exists.
Attach with:  tmux attach -t $SESSION
Or kill it:   tmux kill-session -t $SESSION
Or use a different name:  SESSION=other $0"
fi

mkdir -p "$LOG_DIR"

CMD="$PYTHON -u scripts/rerun_timeout_evals.py '$ROOT' --jobs $JOBS --timeout $TIMEOUT --python '$PYTHON'"

echo "repo    : $REPO_ROOT"
echo "python  : $PYTHON"
echo "target  : $ROOT"
echo "jobs    : $JOBS   (timeout ${TIMEOUT}s per candidate)"
echo "session : $SESSION"
echo "log     : $LOG"
echo

# `exec bash` keeps the pane alive after the run finishes, so you can attach
# later and read the summary instead of finding the session gone.
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
    "$CMD 2>&1 | tee '$LOG'; echo; echo '=== finished, exit code '\$?' ==='; exec bash"

echo "started."
echo
echo "  watch it     : tmux attach -t $SESSION      (detach again with Ctrl-b then d)"
echo "  tail the log : tail -f $LOG"
echo "  stop it      : tmux kill-session -t $SESSION"
