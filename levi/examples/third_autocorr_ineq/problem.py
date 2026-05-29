"""
Third autocorrelation inequality (C3 upper bound) — BLADE / LEVI example.

Task: find a function f : R -> R (positive and negative values allowed),
discretised on the interval [-1/4, 1/4], that minimises the C3 ratio

    C3 = max_{-1/2 <= t <= 1/2} |(f * f)(t)|  /  ( integral f(x) dx )^2

The smaller the achievable C3, the better the upper bound on the third
autocorrelation inequality constant.

Mirrors the benchmark at benchmarks/math/third_autocorr_ineq.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


BENCHMARK = 1.4556427953745406
TIMEOUT_SECONDS = 600
VERIFY_TOL = 1e-3
INTEGRAL_TOL = 1e-9


PROBLEM_DESCRIPTION = f"""
# Third Autocorrelation Inequality (C3 upper bound)

## Problem
Find a function ``f : R -> R`` (values may be positive **and** negative),
discretised on the interval ``[-1/4, 1/4]`` into ``N`` equally-spaced
samples ``f_values[0..N-1]`` with step ``dx = 0.5 / N``, that gives a
**low** upper bound for the third autocorrelation inequality constant C3.

The C3 ratio is

    C3 = max_{{-1/2 <= t <= 1/2}} |(f ⋆ f)(t)|  /  ( integral_{{-1/4}}^{{1/4}} f(x) dx )^2

so any feasible discretised ``f`` with non-zero integral yields an upper
bound ``C3 <= achieved_ratio``. The best known reference upper bound is
``C3 = {BENCHMARK}`` (AlphaEvolve). Smaller is better.

## Output Format
Implement ``run()`` which **must** return a 4-tuple in this exact order:

    (f_values, c3_achieved, loss, n_points)

- ``f_values``: NumPy-compatible 1-D array of shape ``(n_points,)``, the
  discretised samples of ``f`` on a uniform grid over ``[-1/4, 1/4]``.
- ``c3_achieved``: Python float, the C3 ratio you claim, equal to
  ``max(|conv(f, f)| * dx) / (sum(f) * dx)^2`` with ``dx = 0.5 / n_points``.
- ``loss``: Python float, your training loss (typically the same as
  ``c3_achieved``; only used for logging).
- ``n_points``: Python int, number of discretisation points ``N``.

## Feasibility Constraints
- ``f_values`` must have shape ``(n_points,)`` and contain only finite
  numeric values.
- The integral ``sum(f_values) * dx`` must satisfy
  ``( integral )^2 >= {INTEGRAL_TOL}`` (otherwise the ratio is unstable
  and the candidate is rejected).
- The grader **re-computes** the C3 ratio with NumPy's ``np.convolve`` and
  ``np.abs``; the reported ``c3_achieved`` must agree with this recomputed
  value to within ``|delta| <= {VERIFY_TOL}``.

## Objective
Minimise ``c3_achieved``. The grader reports
``combined_score = {BENCHMARK} / c3_achieved`` so that a value above 1.0
beats the reference upper bound.

## Runtime Constraint
Your ``run()`` call must complete within **{TIMEOUT_SECONDS} seconds**.
"""


FUNCTION_SIGNATURE = f"""
import numpy as np

def run() -> tuple[np.ndarray, float, float, int]:
    '''
    Discretise a function f : R -> R on [-1/4, 1/4] and minimise the
    third-autocorrelation-inequality C3 ratio
        C3 = max |conv(f, f)| * dx  /  (sum(f) * dx)^2
    where dx = 0.5 / n_points.

    Returns:
        f_values:    numpy array of shape (n_points,), the discretised f.
        c3_achieved: float, the achieved C3 ratio (lower is better).
        loss:        float, training loss (typically equal to c3_achieved).
        n_points:    int, number of discretisation points.
    '''
    pass
"""


def _verify_c3(f_values: np.ndarray, c3_achieved: float, n_points: int) -> tuple[bool, str]:
    """Re-compute C3 with NumPy and check the reported value is consistent."""
    if f_values.shape != (n_points,):
        return False, (
            f"f_values has shape {f_values.shape}, expected ({n_points},)"
        )

    dx = 0.5 / n_points
    integral_f_sq = float((np.sum(f_values) * dx) ** 2)
    if integral_f_sq < INTEGRAL_TOL:
        return False, "Function integral is close to zero; ratio is unstable"

    conv = np.convolve(f_values, f_values, mode="full")
    max_abs_conv = float(np.max(np.abs(conv * dx)))
    computed_c3 = max_abs_conv / integral_f_sq

    delta = abs(computed_c3 - c3_achieved)
    if delta > VERIFY_TOL:
        return False, (
            f"C3 mismatch: reported {c3_achieved:.6f}, recomputed "
            f"{computed_c3:.6f}, delta={delta:.6f}"
        )
    return True, ""


def score_fn(run, _inputs=None) -> dict:
    """Score a candidate ``run`` function for the third autocorr inequality."""
    try:
        start = time.perf_counter()
        out = run()
        exec_time = time.perf_counter() - start

        if not isinstance(out, tuple) or len(out) != 4:
            return {"error": "run() must return a 4-tuple (f_values, c3_achieved, loss, n_points)"}

        f_values_raw, c3_raw, loss_raw, n_points_raw = out

        f_values = np.asarray(f_values_raw, dtype=float)
        if f_values.ndim != 1:
            return {"error": f"f_values must be 1-D, got shape {f_values.shape}"}
        if not np.isfinite(f_values).all():
            return {"error": "f_values contains non-finite values"}

        n_points = int(n_points_raw)
        c3_achieved = float(c3_raw)
        loss = float(loss_raw)

        if not np.isfinite(c3_achieved) or c3_achieved <= 0.0:
            return {"error": f"c3_achieved must be a positive finite float, got {c3_achieved!r}"}

        valid, reason = _verify_c3(f_values, c3_achieved, n_points)
        if not valid:
            return {"error": reason}

        return {
            "score": float(BENCHMARK / c3_achieved),
            "valid": 1.0,
            "c3": c3_achieved,
            "loss": loss,
            "n_points": n_points,
            "combined_score": float(BENCHMARK / c3_achieved),
            "execution_time": exec_time,
        }
    except Exception as e:
        return {"error": str(e)}
