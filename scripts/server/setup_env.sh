#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot conda environment setup for running SkyDiscover / BLADE on a server.
#
#   bash scripts/server/setup_env.sh                # core env (recommended)
#   bash scripts/server/setup_env.sh --extra math   # + heavy math deps (jax, torch...)
#   bash scripts/server/setup_env.sh --name other   # different env name
#   bash scripts/server/setup_env.sh --recreate     # delete and rebuild the env
#
# Creates a conda environment (default name: minhhieu, Python 3.12) holding
# everything needed by BOTH entry points used on the server:
#   * scripts/run_blade.py   (BLADE — needs the levi/ dependency set)
#   * skydiscover-run        (baselines: openevolve_native / gepa_native /
#                             adaevolve / evox — needs the skydiscover package)
#
# Python 3.12 is the only version satisfying both constraints:
#   skydiscover: >=3.10,<3.14      levi: >=3.11,<3.13
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="minhhieu"
PY_VERSION="3.12"
RECREATE=0
EXTRAS=()

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  -n, --name NAME     Conda environment name (default: minhhieu)
  -p, --python VER    Python version (default: 3.12)
  -e, --extra NAME    Extra dependency group, repeatable. One of:
                        math   -> jax, optax, torch, numba, cvxpy, pymoo, ...
                        adrs   -> torch, pandas, networkx<3.4
                        torch  -> torch only (ADRS/eplb, kernelbench)
                        dev    -> pytest, black, isort, mypy
      --recreate      Remove an existing env of the same name first
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)    ENV_NAME="$2"; shift 2 ;;
        -p|--python)  PY_VERSION="$2"; shift 2 ;;
        -e|--extra)   EXTRAS+=("$2"); shift 2 ;;
        --recreate)   RECREATE=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# --- locate conda and make `conda activate` usable from a script -------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' not found on PATH." >&2
    echo "Install Miniconda first, e.g.:" >&2
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" >&2
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3" >&2
    echo "  \$HOME/miniconda3/bin/conda init bash && exec \$SHELL -l" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ "$RECREATE" == "1" ]] && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Removing existing env '$ENV_NAME'"
    conda env remove -y -n "$ENV_NAME"
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Env '$ENV_NAME' already exists — reusing it (pass --recreate to rebuild)."
else
    echo ">>> Creating conda env '$ENV_NAME' (python $PY_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip
fi

conda activate "$ENV_NAME"
echo ">>> Using python: $(python -V) at $(command -v python)"

cd "$REPO_ROOT"
python -m pip install --upgrade pip

echo ">>> Installing skydiscover (editable) + server requirements"
python -m pip install -e .
python -m pip install -r scripts/server/requirements-server.txt

for extra in "${EXTRAS[@]:-}"; do
    [[ -z "$extra" ]] && continue
    case "$extra" in
        math)  echo ">>> Installing extra: math";  python -m pip install -e ".[math]" ;;
        adrs)  echo ">>> Installing extra: adrs";  python -m pip install -e ".[adrs]" ;;
        dev)   echo ">>> Installing extra: dev";   python -m pip install -e ".[dev]" ;;
        torch) echo ">>> Installing extra: torch"; python -m pip install "torch>=2.9.1" ;;
        *) echo "ERROR: unknown extra '$extra'" >&2; exit 2 ;;
    esac
done

echo ""
echo ">>> Verifying the install"
python - <<'PY'
import importlib, sys

core = ["skydiscover", "openai", "yaml", "numpy", "scipy", "networkx", "dotenv",
        "litellm", "sklearn", "colorama", "dspy",
        # LLM-SRBench / LSR-Synth dataset preparation.
        "huggingface_hub", "pyarrow"]
missing = []
for mod in core:
    try:
        importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{mod}: {exc}")

sys.path.insert(0, "levi")
try:
    import levi  # noqa: F401
except Exception as exc:  # noqa: BLE001
    missing.append(f"levi: {exc}")

if missing:
    print("MISSING / BROKEN:")
    for m in missing:
        print("  -", m)
    raise SystemExit(1)
print("All core imports OK (skydiscover + levi).")
PY

echo ""
echo "============================================================"
echo " Environment '$ENV_NAME' is ready."
echo ""
echo " Activate it with:      conda activate $ENV_NAME"
echo " Then run a benchmark:  ./scripts/server/run_bench.sh --help"
echo "============================================================"
