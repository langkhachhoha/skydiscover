#!/usr/bin/env python3
"""Run LEVI against a SkyDiscover benchmark directory via OpenRouter.

A benchmark directory is any subdir of ``benchmarks/`` that contains a
plain-Python ``initial_program.py`` + ``evaluator.py`` (+ optional
``config.yaml``). For example::

    benchmarks/math/circle_packing/
    benchmarks/ADRS/txn_scheduling/         # has both evaluator.py and evaluator/
    benchmarks/ADRS/llm_sql/                # same

The adapter (see ``scripts/skydiscover_levi_adapter.py``) loads the benchmark,
auto-detects the entry-point function, and builds a picklable score function
that calls the evaluator's ``evaluate(program_path)``.

Both LEVI model slots are pinned to OpenRouter so a single
``OPENROUTER_API_KEY`` is sufficient. Only ``--evals`` is wired by default;
the dollar and seconds budgets are intentionally left empty.

Usage::

    export OPENROUTER_API_KEY=sk-or-...
    uv run python scripts/run_levi_skydiscover.py \\
        --benchmark benchmarks/math/circle_packing \\
        --evals 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "levi"))  # vendored submodule

from skydiscover_levi_adapter import load_benchmark, make_score_fn  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--benchmark",
        required=True,
        help="Path to a SkyDiscover benchmark dir (e.g. benchmarks/math/circle_packing).",
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
    p.add_argument(
        "--evals",
        type=int,
        default=100,
        help="Maximum number of evaluations (default: 100). All other budgets stay empty.",
    )
    p.add_argument("--workers", type=int, default=4, help="Concurrent LLM workers (default: 4).")
    p.add_argument(
        "--eval-processes",
        type=int,
        default=4,
        help="Concurrent evaluator processes (default: 4).",
    )
    p.add_argument(
        "--eval-timeout",
        type=float,
        default=600.0,
        help="Per-candidate evaluation timeout in seconds (default: 600).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to drop snapshot.json + summary.json (default: outputs/levi/<bench>/<timestamp>).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 2

    bench_dir = Path(args.benchmark).resolve()
    if not bench_dir.is_dir():
        print(f"ERROR: --benchmark {bench_dir} is not a directory.", file=sys.stderr)
        return 2

    spec = load_benchmark(bench_dir)

    import levi  # type: ignore  # vendored submodule on sys.path above

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "outputs" / "levi" / spec.name / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[levi] benchmark      = {spec.name} ({spec.bench_dir})")
    print(f"[levi] function       = {spec.function_signature}")
    print(f"[levi] small model    = {args.small_model}")
    print(f"[levi] large model    = {args.large_model}")
    print(f"[levi] evaluations    = {args.evals}")
    print(f"[levi] output_dir     = {output_dir}")

    result = levi.evolve_code(
        spec.problem_description,
        function_signature=spec.function_signature,
        seed_program=spec.seed_program,
        score_fn=make_score_fn(spec.evaluator_path),
        paradigm_model=args.large_model,
        mutation_model=args.small_model,
        budget_evals=args.evals,
        # budget_dollars / budget_seconds intentionally left None — only
        # evaluation count gates the run.
        pipeline=levi.PipelineConfig(
            n_llm_workers=args.workers,
            n_eval_processes=args.eval_processes,
            eval_timeout=args.eval_timeout,
        ),
        output_dir=str(output_dir),
    )

    summary = {
        "benchmark": spec.name,
        "benchmark_dir": str(spec.bench_dir),
        "function": spec.function_signature,
        "best_score": result.best_score,
        "total_evaluations": result.total_evaluations,
        "total_cost": result.total_cost,
        "archive_size": result.archive_size,
        "runtime_seconds": result.runtime_seconds,
        "small_model": args.small_model,
        "large_model": args.large_model,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    best_program_path = output_dir / "best_program.py"
    best_program_path.write_text(result.best_program or "")

    print()
    print(f"Best score        : {result.best_score:.6f}")
    print(f"Evaluations used  : {result.total_evaluations}")
    print(f"Total cost        : ${result.total_cost:.4f}")
    print(f"Archive size      : {result.archive_size}")
    print(f"Runtime           : {result.runtime_seconds:.1f}s")
    print(f"Summary           : {summary_path}")
    print(f"Best program      : {best_program_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
