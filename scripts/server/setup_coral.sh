#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_coral.sh — install what run_coral_cp.sh and run_pk_queue.sh need, at
# pinned versions, without touching ~/.bashrc. Safe to re-run: every step is
# skipped when its result is already there.
#
#   ./scripts/server/setup_coral.sh
#
#   uv        $UV_VERSION        -> ~/.local/bin/uv           (CORAL builds its grader venvs with it)
#   CORAL     $CORAL_COMMIT      -> $CORAL_DIR (+ .venv from `uv sync`, which provides `coral`)
#   OpenCode  v$OPENCODE_VERSION -> ~/.opencode/bin/opencode  (the agent runtime CORAL drives)
#   Frontier-CS $FRONTIER_COMMIT -> $FRONTIER_DIR             (Polyominoes = algorithmic problem 0:
#                                                              statement, checker, 70 test cases)
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECTS="${PROJECTS:-/home/ubuntu/projects}"
CORAL_DIR="${CORAL_DIR:-$PROJECTS/CORAL}"
FRONTIER_DIR="${FRONTIER_DIR:-$PROJECTS/Frontier-CS}"
UV_VERSION="${UV_VERSION:-0.13.7}"
CORAL_COMMIT="${CORAL_COMMIT:-0123dfb}"
OPENCODE_VERSION="${OPENCODE_VERSION:-1.19.5}"
FRONTIER_COMMIT="${FRONTIER_COMMIT:-59687e54}"

log() { echo "[setup] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[[ "$(uname -s)-$(uname -m)" == Linux-x86_64 ]] || die "only Linux x86_64 is handled here"
command -v git >/dev/null && command -v curl >/dev/null && command -v tar >/dev/null || die "need git, curl, tar"
command -v g++ >/dev/null || die "need g++ (Polyominoes compiles C++17 solutions and the testlib checker)"
mkdir -p "$HOME/.local/bin" "$HOME/.opencode/bin"
export PATH="$HOME/.local/bin:$HOME/.opencode/bin:$PATH"

# --- uv ---------------------------------------------------------------------
if command -v uv >/dev/null; then
    log "uv: $(uv --version)"
else
    tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/uv.tgz" \
        "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
    tar -xzf "$tmp/uv.tgz" -C "$tmp"
    install -m 755 "$tmp"/uv-x86_64-unknown-linux-gnu/uv "$tmp"/uv-x86_64-unknown-linux-gnu/uvx "$HOME/.local/bin/"
    rm -rf "$tmp"
    log "uv: installed $(uv --version) into ~/.local/bin"
fi

# --- CORAL ------------------------------------------------------------------
pin_repo() {    # url dir commit
    local url="$1" dir="$2" commit="$3"
    if [[ ! -d "$dir/.git" ]]; then
        git clone --quiet "$url" "$dir"
    fi
    if ! git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
        git -C "$dir" fetch --quiet --unshallow 2>/dev/null || git -C "$dir" fetch --quiet origin
    fi
    git -C "$dir" -c advice.detachedHead=false checkout --quiet "$commit"
    log "$(basename "$dir"): $(git -C "$dir" log -1 --format='%h %ad %s' --date=short)"
}
pin_repo https://github.com/Human-Agent-Society/CORAL.git "$CORAL_DIR" "$CORAL_COMMIT"
if [[ ! -x "$CORAL_DIR/.venv/bin/coral" ]]; then
    (cd "$CORAL_DIR" && uv sync --quiet)
fi
log "coral: $("$CORAL_DIR/.venv/bin/coral" --version 2>/dev/null || echo '?') at $CORAL_DIR/.venv/bin/coral"

# --- OpenCode ---------------------------------------------------------------
if [[ -x "$HOME/.opencode/bin/opencode" ]] && "$HOME/.opencode/bin/opencode" --version 2>/dev/null | grep -q "$OPENCODE_VERSION"; then
    log "opencode: $("$HOME/.opencode/bin/opencode" --version)"
else
    tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/oc.tgz" \
        "https://github.com/anomalyco/opencode/releases/download/v$OPENCODE_VERSION/opencode-linux-x64.tar.gz"
    tar -xzf "$tmp/oc.tgz" -C "$tmp"
    install -m 755 "$(find "$tmp" -type f -name opencode | head -1)" "$HOME/.opencode/bin/opencode"
    rm -rf "$tmp"
    log "opencode: installed $("$HOME/.opencode/bin/opencode" --version) into ~/.opencode/bin"
fi

# --- Frontier-CS (Polyominoes data) -----------------------------------------
pin_repo https://github.com/FrontierCS/Frontier-CS.git "$FRONTIER_DIR" "$FRONTIER_COMMIT"
p0="$FRONTIER_DIR/algorithmic/problems/0"
[[ -f "$p0/chk.cc" && -f "$p0/statement.txt" ]] || die "Frontier-CS problem 0 not found at $p0"
log "polyominoes: $(ls "$p0/testdata" | grep -c '\.in$') test cases in $p0/testdata"

log "done. Next: ./scripts/server/run_coral_cp.sh check   and   ./scripts/server/run_pk_queue.sh check"
