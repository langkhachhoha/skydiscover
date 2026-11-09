"""SkyDiscover evaluator for CORAL's Kernel Engineering task (see kernel_eval.py)."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_eval  # noqa: E402

try:
    from skydiscover.evaluation.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover - evaluator used outside skydiscover
    EvaluationResult = None

EVAL_TIMEOUT_S = float(os.environ.get("KERNEL_EVAL_TIMEOUT", "300"))


def evaluate(program_path: str):
    t0 = time.time()
    try:
        res = kernel_eval.run_program_file(program_path, timeout=EVAL_TIMEOUT_S)
    except Exception as e:  # timeout included
        res = {"error": f"{type(e).__name__}: {e}"}
    ok = "error" not in res and res.get("is_correct", False)
    cycles = int(res.get("cycles") or 0)
    metrics = {
        "combined_score": float(kernel_eval.score_of(cycles)) if ok else 0.0,
        "cycles": float(cycles) if ok else 0.0,
        "correct": 1.0 if ok else 0.0,
        "eval_time": float(time.time() - t0),
    }
    text = kernel_eval.feedback(res)
    print(text)
    if EvaluationResult is not None:
        return EvaluationResult(metrics=metrics, artifacts={"feedback": text})
    return metrics
