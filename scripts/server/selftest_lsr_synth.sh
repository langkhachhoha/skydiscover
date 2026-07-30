#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# selftest_lsr_synth.sh — verify the LLM-SRBench / LSR-Synth integration
# without spending a cent.
#
#   bash scripts/server/selftest_lsr_synth.sh [conda-env]
#
# Checks the dataset, then scores every generated problem through BOTH
# integration paths (the baseline evaluator and the SpecEvo problem module),
# proves the two agree, exercises the parameter fit's failure and timeout
# handling, and dry-runs the domain runner. No LLM calls anywhere.
#
# Run it after setup, after `generate_dirs.py`, and any time a run looks wrong.
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_NAME="${1:-minhhieu}"
cd "$REPO_ROOT" || exit 1

PASS=0; FAIL=0; WARN=0; FAILED=""
ok()   { PASS=$((PASS+1)); printf '  [ OK ]  %s\n' "$1"; }
no()   { FAIL=$((FAIL+1)); FAILED="$FAILED
  - $1"; printf '  [FAIL]  %s\n' "$1"; [ -n "${2:-}" ] && printf '          %s\n' "$2"; }
warn() { WARN=$((WARN+1)); printf '  [WARN]  %s\n' "$1"; }
sec()  { printf '\n--- %s\n' "$1"; }

# Prefer the conda env's python, fall back to whatever is active.
PY="python"
if command -v conda >/dev/null 2>&1; then
  CAND="$(conda run -n "$CONDA_ENV_NAME" python -c 'import sys;print(sys.executable)' 2>/dev/null)"
  [ -x "$CAND" ] && PY="$CAND"
fi

echo "============================================================"
echo " LSR-Synth integration self-test"
echo " python: $PY"
echo "============================================================"

# ---------------------------------------------------------------------------
sec "1. Packages"
for mod in numpy scipy; do
  "$PY" -c "import $mod" 2>/dev/null && ok "$mod importable (needed to score)" \
    || no "$mod missing" "pip install $mod"
done
for mod in huggingface_hub pyarrow; do
  "$PY" -c "import $mod" 2>/dev/null && ok "$mod importable (needed to fetch the dataset)" \
    || warn "$mod missing — prepare_data.py cannot download (pip install $mod)"
done

# ---------------------------------------------------------------------------
sec "2. Dataset"
if "$PY" benchmarks/llm_srbench/prepare_data.py --check >/tmp/lsr_check.$$ 2>&1; then
  ok "all four domains materialised and self-consistent"
  sed 's/^/          /' /tmp/lsr_check.$$
else
  no "dataset check failed" "run: $PY benchmarks/llm_srbench/prepare_data.py"
  sed 's/^/          /' /tmp/lsr_check.$$
fi
rm -f /tmp/lsr_check.$$

# ---------------------------------------------------------------------------
sec "3. Every generated problem, through both paths"
"$PY" - <<'PYEOF'
import importlib.util, json, sys, time
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "benchmarks" / "llm_srbench"))
import lsr_eval as L

fails, n = [], 0
t0 = time.time()
for domain in L.domains():
    for ip in sorted((REPO / "benchmarks" / "llm_srbench" / domain).glob("*/initial_program.py")):
        pid = ip.parent.name
        n += 1
        try:
            spec = importlib.util.spec_from_file_location(f"ev_{domain}_{pid}", ip.parent / "evaluator.py")
            ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
            assert (ev.DOMAIN, ev.PROBLEM) == (domain, pid), \
                f"evaluator names {ev.DOMAIN}/{ev.PROBLEM}"
            rb = ev.evaluate(str(ip))
            assert rb.get("valid") == 1.0, f"baseline seed rejected: {rb.get('error')}"
            json.dumps(rb, allow_nan=False)   # must survive strict JSON (checkpoints)

            ldir = REPO / "levi" / "examples" / "llm_srbench" / domain / pid
            lspec = importlib.util.spec_from_file_location(f"pr_{domain}_{pid}", ldir / "problem.py")
            pm = importlib.util.module_from_spec(lspec); lspec.loader.exec_module(pm)
            assert (pm.DOMAIN, pm.PROBLEM) == (domain, pid)
            ns = {}
            exec(pm.SEED_PROGRAM, ns)
            rl = pm.score_fn(ns["equation"])
            assert rl.get("valid") == 1.0, f"SpecEvo seed rejected: {rl.get('error')}"
            assert abs(rb["score"] - rl["score"]) < 1e-12, \
                f"paths disagree: {rb['score']} vs {rl['score']}"

            meta = L.problem_meta(domain, pid)
            for v in meta["symbols"][1:]:
                assert f"{v}: np.ndarray" in pm.FUNCTION_SIGNATURE, f"{v} missing from signature"
            p = L.load_problem(domain, pid)
            assert p["train"].shape[1] == len(meta["symbols"])
            assert p["id_test"].shape[0] > 0 and p["ood_test"].shape[0] > 0
            tr, oo = p["train"][:, 1:], p["ood_test"][:, 1:]
            assert any(oo[:, j].min() > tr[:, j].max() or oo[:, j].max() < tr[:, j].min()
                       for j in range(tr.shape[1])), "OOD overlaps the train range"
        except Exception as exc:  # noqa: BLE001
            fails.append(f"{domain}/{pid}: {type(exc).__name__}: {exc}")

print(f"          {n} problems, both paths, in {time.time()-t0:.1f}s")
for f in fails[:10]:
    print("          FAIL:", f)
sys.exit(0 if (n and not fails) else 1)
PYEOF
if [ $? -eq 0 ]; then
  ok "every generated problem scores identically on both paths"
else
  no "at least one problem failed" "regenerate with: $PY benchmarks/llm_srbench/generate_dirs.py --limit 10"
fi

# ---------------------------------------------------------------------------
sec "4. Scoring engine: recovery, rejection, timeout"
"$PY" - <<'PYEOF'
import sys, time
sys.path.insert(0, "benchmarks/llm_srbench")
import lsr_eval as L

bad = []

# The published ground-truth structure for bpg0 must recover its true parameters.
gt = ("import numpy as np\n"
      "def equation(t, P, params):\n"
      "    return params[0]*(1 - P/params[1])*P + params[2]*P**(1.0/3.0)\n")
r = L.evaluate_source("bio_pop_growth", "bpg0", gt)
if not (r.get("valid") and r["train_nmse"] < 1e-9):
    bad.append(f"ground-truth structure did not fit: train_nmse={r.get('train_nmse')}")
if not (abs(r["fitted_params"][1] - 96.90688) < 1e-3):
    bad.append(f"fitted carrying capacity {r['fitted_params'][1]} != 96.90688 (data/engine mismatch)")

# Every failure mode must come back as score 0 with a reason, never an exception.
cases = {
    "syntax error":  "def equation(t, P, params)\n  return 1",
    "no equation":   "def other(): pass",
    "raises":        "def equation(t,P,params):\n    raise ValueError('x')",
    "all NaN":       "import numpy as np\ndef equation(t,P,params):\n    return t*np.nan",
    "wrong length":  "def equation(t,P,params):\n    return t[:3]",
    "params out of range": "def equation(t,P,params):\n    return params[99]*t",
}
for name, src in cases.items():
    r = L.evaluate_source("bio_pop_growth", "bpg0", src)
    if r.get("score") != 0.0 or not r.get("error"):
        bad.append(f"{name}: expected score 0 with an error, got {r.get('score')} / {r.get('error')}")

# The per-hypothesis cap must actually fire, close to the stated limit.
import os
os.environ["LSR_EVAL_TIMEOUT"] = "5"
t0 = time.time()
r = L.evaluate_source("bio_pop_growth", "bpg0", "def equation(t,P,params):\n    while True: pass")
dt = time.time() - t0
if r.get("score") != 0.0 or "Timeout" not in str(r.get("error", "")):
    bad.append(f"infinite loop was not timed out: {r.get('error')}")
if dt > 20:
    bad.append(f"timeout took {dt:.1f}s for a 5s limit")

# Both score modes must rank hypotheses identically (they are monotone transforms
# of the same train NMSE), and neither may reach 0.0 for a hypothesis that ran —
# 0.0 is reserved for failures, which the cases above rely on.
nmses = [1e-14, 1e-9, 1e-6, 1e-3, 0.05, 0.5, 1.0, 3.0, 1e4]
for mode in L.SCORE_MODES:
    os.environ["LSR_SCORE_MODE"] = mode
    scores = [L.score_from_nmse(v) for v in nmses]
    if any(a < b for a, b in zip(scores, scores[1:])):
        bad.append(f"{mode}: score is not decreasing in NMSE: {scores}")
    if min(scores) <= 0.0:
        bad.append(f"{mode}: a hypothesis that ran scored 0.0 (reserved for failures): {scores}")
os.environ.pop("LSR_SCORE_MODE", None)
if L.score_mode() != L.DEFAULT_SCORE_MODE:
    bad.append(f"unset LSR_SCORE_MODE did not fall back to {L.DEFAULT_SCORE_MODE}")
try:
    os.environ["LSR_SCORE_MODE"] = "not_a_mode"
    L.score_mode()
    bad.append("an unknown LSR_SCORE_MODE was accepted silently")
except ValueError:
    pass
finally:
    os.environ.pop("LSR_SCORE_MODE", None)

# An in-place mutating candidate must not corrupt the dataset for the next one.
import numpy as np
p = L.load_problem("bio_pop_growth", "bpg0")
before = p["train"].copy()
L.evaluate_source("bio_pop_growth", "bpg0",
                  "def equation(t,P,params):\n    P[:] = 0.0\n    return params[0]*P")
if not np.array_equal(before, p["train"]):
    bad.append("a mutating candidate corrupted the cached dataset")

for b in bad:
    print("          FAIL:", b)
print(f"          engine checks done (timeout fired in {dt:.1f}s for a 5s limit)")
sys.exit(1 if bad else 0)
PYEOF
if [ $? -eq 0 ]; then
  ok "ground truth recovered; every failure mode rejected; timeout and mutation guards hold"
else
  no "scoring engine behaved unexpectedly"
fi

# ---------------------------------------------------------------------------
sec "5. Domain runner (dry-run, no API calls)"
for m in specevo evox; do
  for d in chem_react bio_pop_growth phys_osc matsci; do
    if ./scripts/server/run_lsr_synth.sh --method "$m" --domain "$d" --problems 2 \
        --no-conda --dry-run --output-dir "/tmp/lsr_selftest_$$/$m/$d" >/tmp/lsr_dry.$$ 2>&1; then
      ok "dry-run $m / $d"
    else
      no "dry-run $m / $d failed" "$(tail -2 /tmp/lsr_dry.$$)"
    fi
  done
done
rm -rf "/tmp/lsr_selftest_$$" /tmp/lsr_dry.$$

sec "6. Summarizer"
if "$PY" scripts/lsr_summarize.py outputs/lsr_synth >/dev/null 2>&1; then
  ok "lsr_summarize.py runs (no results yet is fine)"
else
  no "lsr_summarize.py failed"
fi

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
printf ' %d passed, %d failed, %d warnings\n' "$PASS" "$FAIL" "$WARN"
if [ "$FAIL" -gt 0 ]; then
  echo " Failures:$FAILED"
  echo "============================================================"
  exit 1
fi
echo " LSR-Synth integration looks healthy."
echo " Next: ./scripts/server/run_lsr_synth.sh --method specevo --domain chem_react --tmux"
echo "============================================================"
