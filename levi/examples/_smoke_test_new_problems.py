"""Smoke test for the 4 new BLADE problem examples.

Imports each problem module, runs ``score_fn`` against a trivial reference
candidate, and asserts the result dict has the expected keys / shapes. The
test does NOT call any LLM.

Usage:
    uv run python levi/examples/_smoke_test_new_problems.py
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


EXAMPLES_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _run_case(label: str, fn: Any) -> bool:
    print(f"\n=== {label} ===")
    try:
        fn()
        print(f"OK  {label}")
        return True
    except Exception:
        print(f"FAIL {label}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# heilbronn_triangle
# ---------------------------------------------------------------------------


def case_heilbronn_triangle() -> None:
    mod = _load_module(
        "_smoke_heilbronn_triangle",
        EXAMPLES_DIR / "heilbronn_triangle" / "problem.py",
    )

    # Reference candidate: 11 random points strictly inside the equilateral
    # triangle (rejection sampling).
    def heilbronn_triangle11() -> np.ndarray:
        sqrt3 = np.sqrt(3.0)
        rng = np.random.default_rng(2024)
        pts = []
        while len(pts) < 11:
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.01, sqrt3 / 2.0 - 0.01)
            if y <= sqrt3 * x - 0.01 and sqrt3 * x <= sqrt3 - y - 0.01:
                pts.append((x, y))
        return np.array(pts, dtype=float)

    result = mod.score_fn(heilbronn_triangle11)
    print("result:", {k: v for k, v in result.items() if k != "error"})
    _assert("error" not in result, f"unexpected error: {result.get('error')}")
    _assert("score" in result, "missing 'score' key")
    _assert("combined_score" in result, "missing 'combined_score' key")
    _assert(result["score"] > 0.0, f"score should be > 0, got {result['score']}")

    # Invalid candidate: shape mismatch
    bad_result = mod.score_fn(lambda: np.zeros((5, 2)))
    _assert("error" in bad_result, "expected error for bad shape")

    # Invalid candidate: point outside triangle
    def outside() -> np.ndarray:
        pts = np.array([[2.0, 2.0]] + [[0.5, 0.2]] * 10)
        return pts

    bad2 = mod.score_fn(outside)
    _assert("error" in bad2, "expected error for outside-triangle point")


# ---------------------------------------------------------------------------
# heilbronn_convex_13
# ---------------------------------------------------------------------------


def case_heilbronn_convex_13() -> None:
    mod = _load_module(
        "_smoke_heilbronn_convex_13",
        EXAMPLES_DIR / "heilbronn_convex_13" / "problem.py",
    )

    def heilbronn_convex13() -> np.ndarray:
        rng = np.random.default_rng(123)
        return rng.random((13, 2))

    result = mod.score_fn(heilbronn_convex13)
    print("result:", {k: v for k, v in result.items() if k != "error"})
    _assert("error" not in result, f"unexpected error: {result.get('error')}")
    _assert("score" in result, "missing 'score' key")
    _assert("combined_score" in result, "missing 'combined_score' key")
    _assert(result["convex_hull_area"] > 0.0, "convex hull area must be positive")
    _assert(result["score"] > 0.0, f"score should be > 0, got {result['score']}")

    # Invalid: collinear (degenerate hull) — all on y = 0
    def collinear() -> np.ndarray:
        return np.column_stack([np.linspace(0, 1, 13), np.zeros(13)])

    bad = mod.score_fn(collinear)
    _assert("error" in bad, "expected error for collinear points")


# ---------------------------------------------------------------------------
# minmax_distance_3
# ---------------------------------------------------------------------------


def case_minmax_distance_3() -> None:
    mod = _load_module(
        "_smoke_minmax_distance_3",
        EXAMPLES_DIR / "minmax_distance_3" / "problem.py",
    )

    def min_max_dist_dim3_14() -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.standard_normal((14, 3))

    result = mod.score_fn(min_max_dist_dim3_14)
    print("result:", {k: v for k, v in result.items() if k != "error"})
    _assert("error" not in result, f"unexpected error: {result.get('error')}")
    _assert("score" in result, "missing 'score' key")
    _assert("combined_score" in result, "missing 'combined_score' key")
    _assert(0.0 <= result["min_max_ratio"] <= 1.0, "ratio must be in [0, 1]")

    # Invalid: coincident points
    def coincident() -> np.ndarray:
        return np.zeros((14, 3))

    bad = mod.score_fn(coincident)
    _assert("error" in bad, "expected error for coincident points")


# ---------------------------------------------------------------------------
# signal_processing
# ---------------------------------------------------------------------------


def case_signal_processing() -> None:
    mod = _load_module(
        "_smoke_signal_processing",
        EXAMPLES_DIR / "signal_processing" / "problem.py",
    )

    def run_signal_processing(noisy_signal, window_size):  # noqa: D401
        x = np.asarray(noisy_signal, dtype=float)
        out_len = len(x) - window_size + 1
        y = np.zeros(out_len)
        for i in range(out_len):
            y[i] = np.mean(x[i : i + window_size])
        return {"filtered_signal": y}

    result = mod.score_fn(run_signal_processing)
    print("result:", {k: v for k, v in result.items() if k != "error"})
    _assert("error" not in result, f"unexpected error: {result.get('error')}")
    _assert("score" in result, "missing 'score' key")
    _assert("combined_score" in result, "missing 'combined_score' key")
    _assert(result["success_rate"] > 0.0, "success rate must be > 0 for a working filter")

    # Invalid: returns wrong shape
    def bad_filter(noisy_signal, window_size):
        return {"filtered_signal": np.array([])}

    bad = mod.score_fn(bad_filter)
    _assert("error" in bad, "expected error for empty output across all signals")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    cases = [
        ("heilbronn_triangle", case_heilbronn_triangle),
        ("heilbronn_convex_13", case_heilbronn_convex_13),
        ("minmax_distance_3", case_minmax_distance_3),
        ("signal_processing", case_signal_processing),
    ]
    failures = 0
    for label, fn in cases:
        if not _run_case(label, fn):
            failures += 1

    print()
    if failures:
        print(f"FAILED: {failures}/{len(cases)} smoke tests failed")
        return 1
    print(f"PASSED: {len(cases)} smoke tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
