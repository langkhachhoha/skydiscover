"""Baseline evaluator for CO-Bench :: Euclidean Steiner problem.

Delegates to the shared CO-Bench engine (benchmarks/co_bench/cobench_eval.py),
which scores the candidate `solve` on the vendored test instances under a hard
per-instance time limit (10s, CO-Bench protocol) and normalizes against the
best-known objective. `combined_score` == overall_score (mean over the
instances actually evaluated); dev_score/test_score reproduce CO-Bench's
dev/test split and are meaningful only when the full set is run.

Runtime knobs (env): COBENCH_TIMEOUT, COBENCH_MAX_CASES, COBENCH_MAX_INSTANCES.
Default sample: ALL test-case files x 3 instances per file. Set MAX_INSTANCES=0
for the full set, or MAX_CASES to a positive number to use fewer files.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cobench_eval as ce  # noqa: E402

TASK = "Euclidean Steiner problem"


def _cap(name, default):
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    v = int(raw)
    return None if v == 0 else v


def evaluate(program_path, fast_mode=False, **_):
    source = ce.read_solve_source(program_path)
    r = ce.evaluate_source(
        TASK,
        source,
        max_cases=_cap("COBENCH_MAX_CASES", None),
        max_instances=_cap("COBENCH_MAX_INSTANCES", 3),
    )
    return {
        "combined_score": r["score"],
        "dev_score": r["dev_score"],
        "test_score": r["test_score"],
        "overall_score": r["overall_score"],
        "valid_rate": r["valid_rate"],
        "num_cases": r["num_cases"],
        "num_instances": r["num_instances"],
        "feedback": r["feedback"],
        **({"error": r["error"]} if "error" in r else {}),
    }


if __name__ == "__main__":
    try:
        from wrapper import run
        run(evaluate)
    except Exception:
        import json
        print(json.dumps(evaluate(sys.argv[1])))
