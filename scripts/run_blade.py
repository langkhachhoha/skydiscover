#!/usr/bin/env python3
"""Generic driver for running BLADE on any self-contained LEVI example.

Mirrors ``scripts/run_levi.py`` but invokes :func:`levi.evolve_code_blade`
instead of :func:`levi.evolve_code`. Every example under
``levi/examples/<name>/`` exporting ``PROBLEM_DESCRIPTION``,
``FUNCTION_SIGNATURE``, ``score_fn`` (and optionally ``SEED_PROGRAM`` /
``INPUTS``) plugs in unchanged.

Usage::

    uv run python scripts/run_blade.py \\
        --example-dir levi/examples/blade_demo \\
        --evals 50
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

# Make the in-tree LEVI package importable whether the user runs this from
# the repo root, from /scripts, or from within an example dir.
LEVI_PKG = REPO_ROOT / "levi"
if LEVI_PKG.is_dir() and str(LEVI_PKG) not in sys.path:
    sys.path.insert(0, str(LEVI_PKG))


def _opt_int(value: str | None) -> int | None:
    if value is None:
        return None
    s = value.strip()
    if not s or s.lower() == "none":
        return None
    return int(s)


def _opt_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = value.strip()
    if not s or s.lower() == "none":
        return None
    return float(s)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--example-dir",
        required=True,
        help="Path to a LEVI example directory (e.g. levi/examples/blade_demo).",
    )
    p.add_argument(
        "--problem-module",
        default="problem",
        help="Name of the Python module to import from --example-dir (default: problem).",
    )

    # Models — defaults match _levi.yml so BLADE and LEVI are budget-comparable
    # out of the box. Qwen handles the high-frequency mutation calls; GPT-5
    # is reserved for the (much rarer) frontier paradigm shifts.
    p.add_argument(
        "--mutation-model",
        default="openrouter/qwen/qwen3-30b-a3b-instruct-2507",
        help="OpenRouter id of the small mutation model (BLADE 'worker bee').",
    )
    p.add_argument(
        "--paradigm-model",
        default="openrouter/openai/gpt-5",
        help="OpenRouter id of the larger frontier model used for paradigm shifts.",
    )
    p.add_argument(
        "--embedding-model",
        default="openrouter/openai/text-embedding-3-small",
        help="OpenRouter id of the description-embedding model used by the pool.",
    )

    # Budget
    p.add_argument("--evals", default="", help="Max evaluations (default: unset / no cap).")
    p.add_argument("--dollars", default="", help="Max USD spend (default: unset).")
    p.add_argument("--seconds", default="", help="Wall-clock cap in seconds (default: unset).")
    p.add_argument("--target-score", default="", help="Stop early at this score (default: unset).")

    # Concurrency
    p.add_argument("--workers", default="4", help="Concurrent LLM workers (default: 4).")
    p.add_argument("--eval-processes", default="4", help="Concurrent evaluator processes (default: 4).")
    p.add_argument("--eval-timeout", default="600", help="Per-candidate evaluation timeout in seconds.")

    # BLADE-specific knobs
    p.add_argument(
        "--pe-interval",
        default="50",
        help="Frontier paradigm-shift fires every N evaluations (default: 50; '' to disable).",
    )
    p.add_argument(
        "--pool-k",
        type=int,
        default=None,
        metavar="K",
        help="Top-K size of the BLADE pool (default: 100).",
    )
    p.add_argument(
        "--niche-threshold",
        type=float,
        default=None,
        metavar="F",
        help="Description-embedding cosine threshold for near-duplicate dedup (default: 0.92).",
    )
    p.add_argument(
        "--family-threshold",
        type=float,
        default=None,
        metavar="F",
        help="Single-linkage cosine threshold for family clustering (default: 0.72).",
    )
    p.add_argument(
        "--max-per-family",
        type=int,
        default=None,
        metavar="N",
        help="Maximum programs per family before weakest-in-family eviction (default: 8).",
    )
    p.add_argument(
        "--no-repair",
        action="store_true",
        help="Disable the one-shot self-repair branch.",
    )

    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to drop snapshot.json + summary.json (default: outputs/blade/<example>/<ts>).",
    )

    return p.parse_args()


def _load_repo_env() -> None:
    """Best-effort ``.env`` loader for local runs (CI sets env directly)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)


def _ensure_openrouter_env() -> None:
    """litellm reads OPENROUTER_API_KEY for ``openrouter/...`` model IDs.

    Many setups only export OPENAI_API_KEY=sk-or-v1-... (the OpenRouter key
    reused under the OpenAI variable name). Mirror it across so the BLADE
    LLM clients authenticate cleanly without further env juggling.
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    if key.startswith("sk-or-") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = key


def main() -> int:
    args = _parse_args()
    _load_repo_env()
    _ensure_openrouter_env()

    example_dir = (REPO_ROOT / args.example_dir).resolve() if not Path(args.example_dir).is_absolute() else Path(args.example_dir)
    if not example_dir.is_dir():
        print(f"ERROR: --example-dir does not exist: {example_dir}", file=sys.stderr)
        return 2

    # Make the example dir importable as a package-level "problem" module.
    sys.path.insert(0, str(example_dir))
    problem = importlib.import_module(args.problem_module)

    # Required attributes.
    try:
        problem_description = problem.PROBLEM_DESCRIPTION
        function_signature = problem.FUNCTION_SIGNATURE
        score_fn = problem.score_fn
    except AttributeError as e:
        print(f"ERROR: example module is missing required attribute: {e}", file=sys.stderr)
        return 2

    seed_program = getattr(problem, "SEED_PROGRAM", None)
    inputs = getattr(problem, "INPUTS", None)

    # Resolve numeric budgets.
    evals = _opt_int(args.evals)
    dollars = _opt_float(args.dollars)
    seconds = _opt_float(args.seconds)
    target_score = _opt_float(args.target_score)
    workers = int(args.workers)
    eval_processes = int(args.eval_processes)
    eval_timeout = float(args.eval_timeout)
    pe_interval = _opt_int(args.pe_interval) or 50

    # Output dir. Resolve relative paths against REPO_ROOT so callers can
    # invoke the script from any working directory (e.g. CI runs it from
    # ``levi/`` but the workflow artifact step expects the repo-root path).
    if args.output_dir:
        out_arg = Path(args.output_dir)
        output_dir = out_arg if out_arg.is_absolute() else (REPO_ROOT / out_arg).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = REPO_ROOT / "outputs" / "blade" / example_dir.name / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pool/selector overrides.
    overrides: dict = {}
    from levi.simple import PoolConfig

    pool_kwargs = {}
    if args.pool_k is not None:
        pool_kwargs["K"] = args.pool_k
    if args.niche_threshold is not None:
        pool_kwargs["niche_cosine_threshold"] = args.niche_threshold
    if args.family_threshold is not None:
        pool_kwargs["family_cosine_threshold"] = args.family_threshold
    if args.max_per_family is not None:
        pool_kwargs["max_per_family"] = args.max_per_family
    if pool_kwargs:
        overrides["pool_config"] = PoolConfig(**pool_kwargs)

    if args.no_repair:
        overrides["enable_repair"] = False

    import levi  # imported lazily so import errors surface clearly above

    print(f"[blade] example_dir       = {example_dir}")
    print(f"[blade] mutation_model    = {args.mutation_model}")
    print(f"[blade] paradigm_model    = {args.paradigm_model}")
    print(f"[blade] embedding_model   = {args.embedding_model}")
    print(f"[blade] budget evals      = {evals}")
    print(f"[blade] budget dollars    = {dollars}")
    print(f"[blade] budget seconds    = {seconds}")
    print(f"[blade] pe_cron_interval  = {pe_interval}")
    print(f"[blade] workers           = {workers}")
    print(f"[blade] output_dir        = {output_dir}")

    result = levi.evolve_code_blade(
        problem_description,
        function_signature=function_signature,
        score_fn=score_fn,
        seed_program=seed_program,
        inputs=inputs,
        mutation_model=args.mutation_model,
        paradigm_model=args.paradigm_model,
        embedding_model=args.embedding_model,
        budget_evals=evals,
        budget_dollars=dollars,
        budget_seconds=seconds,
        target_score=target_score,
        n_workers=workers,
        n_eval_processes=eval_processes,
        eval_timeout=eval_timeout,
        pe_cron_interval=pe_interval,
        output_dir=output_dir,
        **overrides,
    )

    summary = {
        "method": "blade",
        "example_dir": str(example_dir),
        "mutation_model": args.mutation_model,
        "paradigm_model": args.paradigm_model,
        "embedding_model": args.embedding_model,
        "best_score": result.best_score,
        "total_evaluations": result.total_evaluations,
        "total_cost": result.total_cost,
        "pool_size": result.pool_size,
        "runtime_seconds": result.runtime_seconds,
        "n_paradigm_trials": len(result.paradigm_trials),
        "budget": {
            "evaluations": evals,
            "dollars": dollars,
            "seconds": seconds,
            "target_score": target_score,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "best_program.py").write_text(result.best_program or "")

    print()
    print(f"Best score        : {result.best_score:.6f}")
    print(f"Evaluations used  : {result.total_evaluations}")
    print(f"Total cost        : ${result.total_cost:.4f}")
    print(f"Pool size         : {result.pool_size}")
    print(f"Paradigm trials   : {len(result.paradigm_trials)}")
    print(f"Runtime           : {result.runtime_seconds:.1f}s")
    print(f"Output dir        : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
