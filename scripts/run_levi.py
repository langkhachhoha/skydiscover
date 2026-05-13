#!/usr/bin/env python3
"""Generic driver for any self-contained LEVI example.

Every example under ``levi/examples/<name>/`` follows the same shape: there
is a ``problem.py`` that exports ``PROBLEM_DESCRIPTION``,
``FUNCTION_SIGNATURE``, ``score_fn``, and optionally ``SEED_PROGRAM`` /
``INPUTS`` / ``get_lazy_inputs``. This script picks one of those directories
and runs LEVI on it, exposing every budget and pipeline knob as a flag so
the GitHub Actions reusable workflow can wire them straight from inputs.

The trick that makes pickling work (LEVI ships ``score_fn`` to a process
pool): we insert the example directory on ``sys.path`` and then
``importlib.import_module("problem")``. The module name on disk is
``problem`` — not a synthetic ``spec_from_file_location`` name — so worker
processes can re-import it and unpickle the closure cleanly.

Usage::

    uv run python scripts/run_levi_example.py \\
        --example-dir levi/examples/circle_packing \\
        --evals 100
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _opt_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() == "none":
        return None
    return int(value)


def _opt_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() == "none":
        return None
    return float(value)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--example-dir",
        required=True,
        help="Path to a LEVI example directory (e.g. levi/examples/circle_packing).",
    )
    p.add_argument(
        "--problem-module",
        default="problem",
        help="Name of the Python module to import from --example-dir (default: problem).",
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
    # All budgets are optional strings — blank / "none" disables that knob.
    p.add_argument("--evals", default="100", help="Max evaluations (default: 100; '' to disable).")
    p.add_argument("--dollars", default="", help="Max USD spend (default: unset).")
    p.add_argument("--seconds", default="", help="Wall-clock cap in seconds (default: unset).")
    p.add_argument("--target-score", default="", help="Stop early at this score (default: unset).")
    p.add_argument("--workers", default="4", help="Concurrent LLM workers (default: 4).")
    p.add_argument(
        "--eval-processes", default="4", help="Concurrent evaluator processes (default: 4)."
    )
    p.add_argument(
        "--eval-timeout", default="600", help="Per-candidate evaluation timeout (seconds)."
    )
    p.add_argument(
        "--init-diverse-seeds",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Init phase: how many algorithmically diverse seeds to aim for (Levi default 4; "
            "often +1 when no seed_program). Omit to use Levi defaults."
        ),
    )
    p.add_argument(
        "--init-variants-per-seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Init phase: variants per seed after diversity phase (Levi default 20). "
            "Total variant LLM calls ≈ this × (number of diverse seeds). Omit for Levi default."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to drop snapshot.json + summary.json (default: outputs/levi/<example>/<ts>).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 2

    example_dir = Path(args.example_dir).resolve()
    if not example_dir.is_dir():
        print(f"ERROR: --example-dir {example_dir} is not a directory.", file=sys.stderr)
        return 2

    # Make `problem` importable by its real on-disk name. Workers will
    # inherit sys.path (fork on Linux, explicit transfer on spawn) and
    # therefore be able to unpickle score_fn.
    sys.path.insert(0, str(example_dir))
    # Also make the vendored LEVI submodule importable.
    sys.path.insert(0, str(REPO_ROOT / "levi"))

    problem = importlib.import_module(args.problem_module)
    import levi  # type: ignore

    evals = _opt_int(args.evals)
    dollars = _opt_float(args.dollars)
    seconds = _opt_float(args.seconds)
    target_score = _opt_float(args.target_score)
    if evals is None and dollars is None and seconds is None:
        print(
            "ERROR: at least one of --evals / --dollars / --seconds must be set.",
            file=sys.stderr,
        )
        return 2

    workers = _opt_int(args.workers) or 4
    eval_processes = _opt_int(args.eval_processes) or 4
    eval_timeout = _opt_float(args.eval_timeout) or 600.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "outputs" / "levi" / example_dir.name / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = None
    if hasattr(problem, "INPUTS"):
        inputs = problem.INPUTS
    elif hasattr(problem, "get_lazy_inputs"):
        inputs = problem.get_lazy_inputs()

    print(f"[levi] example       = {example_dir}")
    print(f"[levi] small model   = {args.small_model}")
    print(f"[levi] large model   = {args.large_model}")
    print(f"[levi] budget evals  = {evals}")
    print(f"[levi] budget $      = {dollars}")
    print(f"[levi] budget secs   = {seconds}")
    print(f"[levi] target_score  = {target_score}")
    print(f"[levi] output_dir    = {output_dir}")

    init_updates: dict = {}
    if args.init_diverse_seeds is not None:
        init_updates["n_diverse_seeds"] = args.init_diverse_seeds
    if args.init_variants_per_seed is not None:
        init_updates["n_variants_per_seed"] = args.init_variants_per_seed
    base_init = levi.InitConfig()
    effective_init = base_init.model_copy(update=init_updates) if init_updates else base_init
    if init_updates:
        evolve_init = effective_init
        print(f"[levi] init diverse seeds      = {effective_init.n_diverse_seeds}")
        print(f"[levi] init variants per seed  = {effective_init.n_variants_per_seed}")
    else:
        evolve_init = None

    evolve_kw: dict = {
        "problem_description": problem.PROBLEM_DESCRIPTION,
        "function_signature": problem.FUNCTION_SIGNATURE,
        "seed_program": getattr(problem, "SEED_PROGRAM", None),
        "score_fn": problem.score_fn,
        "inputs": inputs,
        "paradigm_model": args.large_model,
        "mutation_model": args.small_model,
        "budget_evals": evals,
        "budget_dollars": dollars,
        "budget_seconds": seconds,
        "target_score": target_score,
        "pipeline": levi.PipelineConfig(
            n_llm_workers=workers,
            n_eval_processes=eval_processes,
            eval_timeout=eval_timeout,
        ),
        "output_dir": str(output_dir),
    }
    if evolve_init is not None:
        evolve_kw["init"] = evolve_init

    result = levi.evolve_code(**evolve_kw)

    summary = {
        "example_dir": str(example_dir),
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
        "init": {
            "n_diverse_seeds": effective_init.n_diverse_seeds,
            "n_variants_per_seed": effective_init.n_variants_per_seed,
        },
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
