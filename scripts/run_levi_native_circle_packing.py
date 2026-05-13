#!/usr/bin/env python3
"""Run LEVI's *own* circle_packing example (no SkyDiscover adapter), via OpenRouter.

Loads ``levi/examples/circle_packing/problem.py`` directly — same
``PROBLEM_DESCRIPTION``, ``FUNCTION_SIGNATURE``, and ``score_fn`` LEVI's
authors used. Both model slots are routed through OpenRouter so a single
``OPENROUTER_API_KEY`` is sufficient. Every budget knob is a CLI flag so the
GitHub Actions workflow can expose them as inputs.

Example::

    export OPENROUTER_API_KEY=sk-or-...
    uv run python scripts/run_levi_native_circle_packing.py \\
        --evals 100 --dollars 1.0 --seconds 1800
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEVI_CP_DIR = REPO_ROOT / "levi" / "examples" / "circle_packing"


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "" or value.lower() == "none":
        return None
    return float(value)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "" or value.lower() == "none":
        return None
    return int(value)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--small-model",
        default="openrouter/qwen/qwen3-30b-a3b-instruct-2507",
        help="OpenRouter id of the cheap mutation model.",
    )
    p.add_argument(
        "--large-model",
        default="openrouter/openai/gpt-5",
        help="OpenRouter id of the stronger paradigm-shift model.",
    )
    # All budgets are optional — pass empty string / "none" to leave a slot
    # unset. At least one must end up populated or LEVI will refuse to run.
    p.add_argument("--evals", default="100", help="Max evaluations (default: 100; '' to disable).")
    p.add_argument("--dollars", default="", help="Max USD spend (default: unset).")
    p.add_argument("--seconds", default="", help="Wall-clock cap in seconds (default: unset).")
    p.add_argument(
        "--target-score",
        default="",
        help="Stop early once this score is reached (default: unset).",
    )

    p.add_argument("--workers", type=int, default=4, help="Concurrent LLM workers (default: 4).")
    p.add_argument(
        "--eval-processes", type=int, default=4, help="Concurrent evaluator processes (default: 4)."
    )
    p.add_argument(
        "--eval-timeout", type=float, default=600.0, help="Per-candidate evaluation timeout in seconds."
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to drop snapshot.json + summary.json (default: outputs/levi/native_circle_packing/<ts>).",
    )
    return p.parse_args()


def _load_problem_module():
    """Import levi/examples/circle_packing/problem.py with its dir on sys.path."""
    if not LEVI_CP_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected {LEVI_CP_DIR} to exist. "
            "Did you run `git submodule update --init --recursive` on the levi submodule?"
        )
    sys.path.insert(0, str(LEVI_CP_DIR))
    spec = importlib.util.spec_from_file_location(
        "levi_native_cp_problem", LEVI_CP_DIR / "problem.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = _parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 2

    sys.path.insert(0, str(REPO_ROOT / "levi"))  # vendored submodule
    import levi  # type: ignore

    problem = _load_problem_module()

    evals = _parse_optional_int(args.evals)
    dollars = _parse_optional_float(args.dollars)
    seconds = _parse_optional_float(args.seconds)
    target_score = _parse_optional_float(args.target_score)

    if evals is None and dollars is None and seconds is None:
        print(
            "ERROR: at least one of --evals / --dollars / --seconds must be set "
            "(LEVI refuses to run without a budget).",
            file=sys.stderr,
        )
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "outputs" / "levi" / "native_circle_packing" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[levi] benchmark      = levi/examples/circle_packing (native)")
    print(f"[levi] small model    = {args.small_model}")
    print(f"[levi] large model    = {args.large_model}")
    print(f"[levi] evaluations    = {evals}")
    print(f"[levi] dollars        = {dollars}")
    print(f"[levi] seconds        = {seconds}")
    print(f"[levi] target_score   = {target_score}")
    print(f"[levi] output_dir     = {output_dir}")

    result = levi.evolve_code(
        problem.PROBLEM_DESCRIPTION,
        function_signature=problem.FUNCTION_SIGNATURE,
        seed_program=getattr(problem, "SEED_PROGRAM", None),
        score_fn=problem.score_fn,
        inputs=getattr(problem, "INPUTS", None),
        paradigm_model=args.large_model,
        mutation_model=args.small_model,
        budget_evals=evals,
        budget_dollars=dollars,
        budget_seconds=seconds,
        target_score=target_score,
        pipeline=levi.PipelineConfig(
            n_llm_workers=args.workers,
            n_eval_processes=args.eval_processes,
            eval_timeout=args.eval_timeout,
        ),
        output_dir=str(output_dir),
    )

    summary = {
        "benchmark": "levi/examples/circle_packing",
        "best_score": result.best_score,
        "total_evaluations": result.total_evaluations,
        "total_cost": result.total_cost,
        "archive_size": result.archive_size,
        "runtime_seconds": result.runtime_seconds,
        "budget": {
            "evaluations": evals,
            "dollars": dollars,
            "seconds": seconds,
            "target_score": target_score,
        },
        "small_model": args.small_model,
        "large_model": args.large_model,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "best_program.py").write_text(result.best_program or "")

    print()
    print(f"Best score        : {result.best_score:.6f}")
    print(f"Evaluations used  : {result.total_evaluations}")
    print(f"Total cost        : ${result.total_cost:.4f}")
    print(f"Archive size      : {result.archive_size}")
    print(f"Runtime           : {result.runtime_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
