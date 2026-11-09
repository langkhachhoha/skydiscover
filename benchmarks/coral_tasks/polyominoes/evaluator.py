"""SkyDiscover evaluator for CORAL's Polyominoes task (Frontier-CS algorithmic #0).

Scores a C++17 solution with the local judge in judge.py: 0..100, the mean
of the checker's per-case ratio over the 70 official test cases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge  # noqa: E402

try:
    from skydiscover.evaluation.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover - evaluator used outside skydiscover
    EvaluationResult = None


def evaluate(program_path: str):
    try:
        res = judge.judge(Path(program_path).read_text())
        text = res.feedback()
        metrics = {
            "combined_score": float(res.score),
            "compiled": 1.0 if res.compiled else 0.0,
            "cases_ok": float(res.counts.get("OK", 0)),
            "cases_tle": float(res.counts.get("TLE", 0)),
            "cases_wa": float(res.counts.get("WA", 0)),
            "cases_re": float(res.counts.get("RE", 0) + res.counts.get("MLE", 0)),
            "mean_ratio_valid": float(res.mean_ratio_ok),
            "max_cpu_s": float(res.max_cpu_s),
            "eval_time": float(res.seconds),
        }
    except Exception as e:
        text = f"judge error: {type(e).__name__}: {e}"
        metrics = {"combined_score": 0.0, "compiled": 0.0}
    print(text.splitlines()[0] if text else "")
    if EvaluationResult is not None:
        return EvaluationResult(metrics=metrics, artifacts={"feedback": text})
    return metrics
