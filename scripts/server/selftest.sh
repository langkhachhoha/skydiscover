#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# selftest.sh — verify the server setup without spending a cent.
#
#   bash scripts/server/selftest.sh
#
# Checks conda/tmux/python/.env, then exercises run_bench.sh in --dry-run mode
# (no LLM calls) and launches two throwaway tmux sessions to prove the
# detached-run machinery works end to end. Run it once after setup_env.sh, and
# again any time a run behaves strangely.
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RB="$REPO_ROOT/scripts/server/run_bench.sh"
CONDA_ENV_NAME="${1:-minhhieu}"
cd "$REPO_ROOT" || exit 1

PASS=0; FAIL=0; WARN=0; FAILED=""
ok()   { PASS=$((PASS+1)); printf '  [ OK ]  %s\n' "$1"; }
no()   { FAIL=$((FAIL+1)); FAILED="$FAILED
  - $1"; printf '  [FAIL]  %s\n' "$1"; [ -n "${2:-}" ] && printf '          %s\n' "$2"; }
warn() { WARN=$((WARN+1)); printf '  [WARN]  %s\n' "$1"; }
sec()  { printf '\n--- %s\n' "$1"; }

wait_for() { local f="$1" pat="$2" t="${3:-60}" i=0
  while [ "$i" -lt "$t" ]; do [ -f "$f" ] && grep -qE "$pat" "$f" 2>/dev/null && return 0; sleep 2; i=$((i+2)); done
  return 1; }

echo "============================================================"
echo " SkyDiscover server self-test  (env: $CONDA_ENV_NAME)"
echo "============================================================"

# ---------------------------------------------------------------------------
sec "1. Tools"
command -v conda >/dev/null 2>&1 && ok "conda found ($(conda --version 2>&1))" \
  || no "conda not found" "run scripts/server/setup_env.sh, then re-login"
command -v tmux  >/dev/null 2>&1 && ok "tmux found ($(tmux -V 2>&1))" \
  || no "tmux not found" "sudo apt-get install -y tmux   OR   conda install -y -c conda-forge tmux"
command -v git   >/dev/null 2>&1 && ok "git found" || warn "git not found (you cannot pull updates)"

# ---------------------------------------------------------------------------
sec "2. Conda environment '$CONDA_ENV_NAME'"
if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    ok "env '$CONDA_ENV_NAME' exists"
    ENVPY="$(conda run -n "$CONDA_ENV_NAME" python -c 'import sys;print(sys.executable)' 2>/dev/null)"
    PYVER="$(conda run -n "$CONDA_ENV_NAME" python -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
    case "$PYVER" in
      3.11|3.12) ok "python $PYVER (satisfies skydiscover AND levi)" ;;
      "")        no "could not run python inside the env" ;;
      *)         no "python $PYVER out of range" "levi needs >=3.11,<3.13 — rebuild with setup_env.sh --recreate" ;;
    esac
    MISSING="$(conda run -n "$CONDA_ENV_NAME" python - <<'PY' 2>/dev/null
import importlib, sys
bad = []
for m in ["skydiscover","openai","yaml","numpy","scipy","networkx","dotenv",
          "litellm","sklearn","colorama","dspy"]:
    try: importlib.import_module(m)
    except Exception: bad.append(m)
sys.path.insert(0, "levi")
try: import levi
except Exception: bad.append("levi")
print(",".join(bad))
PY
)"
    [ -z "$MISSING" ] && ok "all core imports work (skydiscover + levi)" \
      || no "missing modules: $MISSING" "re-run scripts/server/setup_env.sh"
    conda run -n "$CONDA_ENV_NAME" which skydiscover-run >/dev/null 2>&1 \
      && ok "skydiscover-run on PATH" || no "skydiscover-run missing" "pip install -e . inside the env"
  else
    no "env '$CONDA_ENV_NAME' does not exist" "run: bash scripts/server/setup_env.sh"
  fi
fi

# ---------------------------------------------------------------------------
sec "3. API key (.env)"
if [ -f "$REPO_ROOT/.env" ]; then
  ok ".env exists"
  KEY="$(grep -E '^(OPENAI_API_KEY|OPENROUTER_API_KEY)=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2-)"
  if [ -n "$KEY" ]; then
    ok "API key present (${KEY:0:10}...${KEY: -4})"
    case "$KEY" in sk-or-*) ok "key looks like an OpenRouter key" ;;
                   *) warn "key does not start with 'sk-or-' — expected an OpenRouter key" ;; esac
  else
    no ".env has no OPENAI_API_KEY / OPENROUTER_API_KEY line"
  fi
  PERM="$(ls -l "$REPO_ROOT/.env" | cut -c1-10)"
  case "$PERM" in -rw-------) ok ".env permissions are private (600)" ;;
                  *) warn ".env is world-readable ($PERM) — run: chmod 600 .env" ;; esac
else
  no ".env missing" "create it: see docs/SERVER_GUIDE.md section 4"
fi

# ---------------------------------------------------------------------------
sec "4. Repository layout"
for p in levi/pyproject.toml levi/levi/blade levi/levi/simple scripts/run_blade.py \
         benchmarks/math/circle_packing/config.yaml; do
  [ -e "$p" ] && ok "$p" || no "missing $p" "did you clone with --recurse-submodules?"
done

# ---------------------------------------------------------------------------
sec "5. run_bench.sh argument handling (no API calls)"
"$RB" --help >/dev/null 2>&1 && ok "--help works" || no "--help failed"
"$RB" blade --ablation bogus --dry-run --no-conda >/dev/null 2>&1 \
  && no "bad --ablation was accepted" || ok "invalid --ablation rejected"
"$RB" baseline --baseline bogus --dry-run --no-conda >/dev/null 2>&1 \
  && no "bad --baseline was accepted" || ok "invalid --baseline rejected"
"$RB" blade --advanced-options '{broken' --dry-run --no-conda >/dev/null 2>&1 \
  && no "malformed JSON was accepted" || ok "malformed --advanced-options rejected"

OUT="$("$RB" blade --dry-run --no-conda --ablation no_crossover 2>&1)"
grep -q -- "--no-crossover" <<<"$OUT" && ok "ablation maps to the right CLI flag" || no "ablation mapping broken"
OUT="$("$RB" blade --dry-run --no-conda --advanced-options '{"n_cells":50,"p_crossover":0.2}' 2>&1)"
grep -q -- "--n-cells 50" <<<"$OUT" && grep -q -- "--p-crossover 0.2" <<<"$OUT" \
  && ok "advanced_options reach the CLI" || no "advanced_options mapping broken"

# Combining ablation axes: --ablation takes one name, so the multi-axis
# configurations are only reachable through advanced_options. JSON booleans
# must survive as the lowercase literal the flags are matched against.
OUT="$("$RB" blade --dry-run --no-conda --advanced-options \
  '{"meta_advice_disabled":true,"paradigm_force_mode":"reframe","single_prompt_operators":true,"p_crossover":0,"targeted_mutate_disabled":true}' 2>&1)"
MISSING=""
for f in --no-meta-advice "--paradigm-force-mode shift" --single-prompt-operators \
         "--p-crossover 0" --no-targeted-mutate; do
  grep -q -- "$f" <<<"$OUT" || MISSING="$MISSING $f"
done
[ -z "$MISSING" ] && ok "combined ablation axes reach the CLI (JSON booleans included)" \
  || no "advanced_options ablation axes dropped:$MISSING"
OUT="$("$RB" baseline --dry-run --no-conda --benchmark-dir benchmarks/math/circle_packing 2>&1)"
grep -q "evaluator.py" <<<"$OUT" && ok "baseline evaluator resolution works" || no "evaluator resolution broken"

grep -q "timeout     : 10800s" <<<"$OUT" && ok "baseline defaults to a 3h timeout (as baseline.yml)" \
  || no "baseline timeout default wrong" "$(grep 'timeout' <<<"$OUT")"
grep -q "timeout     : none" <<<"$("$RB" blade --dry-run --no-conda 2>&1)" \
  && ok "blade defaults to no timeout (as blade.yml)" || no "blade timeout default wrong"
grep -q "timeout     : 10800s" <<<"$("$RB" blade --dry-run --no-conda --timeout 3h 2>&1)" \
  && ok "--timeout accepts h/m/s suffixes" || no "--timeout suffix parsing broken"
"$RB" blade --dry-run --no-conda --timeout 5d >/dev/null 2>&1 \
  && no "invalid --timeout unit accepted" || ok "invalid --timeout rejected"

OUT="$("$RB" baseline --dry-run --no-conda --dollars 5 2>&1)"
grep -q -- "--dollars 5" <<<"$OUT" && ok "baseline --dollars reaches skydiscover-run" \
  || no "baseline --dollars not passed through"
"$RB" baseline --dry-run --no-conda --dollars 5usd >/dev/null 2>&1 \
  && no "non-numeric --dollars accepted" || ok "invalid --dollars rejected"

# ---------------------------------------------------------------------------
sec "6. tmux detached execution (2 throwaway sessions, no API calls)"
if command -v tmux >/dev/null 2>&1; then
  S="selftest_$$"
  "$RB" blade --tmux --session "$S" --dry-run --conda-env "$CONDA_ENV_NAME" > /tmp/st_$$.txt 2>&1
  LOG="$REPO_ROOT/$(sed -n 's/^  log file    : //p' /tmp/st_$$.txt)"
  tmux has-session -t "$S" 2>/dev/null && ok "tmux session created" || no "tmux session was not created"
  if wait_for "$LOG" "dry-run: not executing" 45; then
    ok "job ran inside tmux and wrote its log"
    grep -q "conda env   : $CONDA_ENV_NAME" "$LOG" && ok "conda env activated inside the tmux pane" \
      || warn "conda env not active inside tmux (log says: $(grep 'conda env' "$LOG" | head -1))"
  else
    no "no log produced inside tmux" "$LOG"
  fi
  sleep 2
  tmux capture-pane -p -t "$S" 2>/dev/null | grep -q "run finished (exit 0)" \
    && ok "pane reports the exit status and stays open" || warn "pane did not show the finish banner"

  # a duplicate name must be refused rather than silently clobbering a live run
  "$RB" blade --tmux --session "$S" --dry-run --no-conda >/dev/null 2>&1 \
    && no "duplicate session name was accepted" || ok "duplicate session name refused"

  tmux kill-session -t "$S" 2>/dev/null
  rm -f /tmp/st_$$.txt
  [ -n "$LOG" ] && rm -rf "$(dirname "$LOG")" 2>/dev/null
else
  warn "tmux missing — skipped the detached-execution checks"
fi

# ---------------------------------------------------------------------------
sec "7. Disk space"
AVAIL="$(df -h . | awk 'NR==2{print $4}')"
echo "  available on this filesystem: $AVAIL"

echo ""
echo "============================================================"
echo " PASS: $PASS   FAIL: $FAIL   WARN: $WARN"
if [ "$FAIL" -gt 0 ]; then
  echo " Problems:$FAILED"
  echo ""
  echo " Not ready — fix the above, see docs/SERVER_GUIDE.md section 11."
else
  echo ""
  echo " Ready. Next: verify the key really works (costs ~\$0.0001):"
  echo "   conda activate $CONDA_ENV_NAME && python scripts/test_openrouter_key.py"
  echo " Then start a run:"
  echo "   ./scripts/server/run_bench.sh blade --tmux --benchmark levi/examples/circle_packing --evaluations 200"
fi
echo "============================================================"
[ "$FAIL" -eq 0 ]
