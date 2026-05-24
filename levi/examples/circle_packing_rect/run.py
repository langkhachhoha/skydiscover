#!/usr/bin/env python3
"""Run BLADE on the n=21 rectangle circle-packing problem."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


# Load .env at the repository root if present.
LEVI_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = LEVI_ROOT.parent
ENV = REPO_ROOT / ".env"
if ENV.exists():
    for raw in ENV.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)

# litellm reads OPENROUTER_API_KEY for openrouter/... model IDs. Mirror an
# OpenRouter key stored as OPENAI_API_KEY for local convenience.
if os.environ.get("OPENAI_API_KEY", "").startswith("sk-or-") and not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.environ["OPENAI_API_KEY"]

# Make the in-tree LEVI package importable when invoked from anywhere.
sys.path.insert(0, str(LEVI_ROOT))

import levi  # noqa: E402

from problem import (  # noqa: E402
    FUNCTION_SIGNATURE,
    PROBLEM_DESCRIPTION,
    score_fn,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals", type=int, default=20, help="Max evaluations")
    parser.add_argument("--dollars", type=float, default=None, help="Max USD spend")
    parser.add_argument("--seconds", type=float, default=None, help="Max wall-clock seconds")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-processes", type=int, default=2)
    parser.add_argument("--eval-timeout", type=float, default=360.0)
    parser.add_argument("--pe-interval", type=int, default=10)
    parser.add_argument("--n-diverse-seeds", type=int, default=2)
    parser.add_argument("--n-variants-per-seed", type=int, default=4)
    parser.add_argument("--n-paradigm-variants", type=int, default=2)
    parser.add_argument(
        "--mutation-model",
        default=os.getenv("BLADE_MUTATION_MODEL", "openrouter/qwen/qwen3-30b-a3b-instruct-2507"),
    )
    parser.add_argument(
        "--paradigm-model",
        default=os.getenv("BLADE_PARADIGM_MODEL", "openrouter/openai/gpt-5"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("BLADE_EMBEDDING_MODEL", "openrouter/openai/text-embedding-3-small"),
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("OPENROUTER_API_KEY or OPENAI_API_KEY is not set; aborting", file=sys.stderr)
        return 1

    result = levi.evolve_code_blade(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        score_fn=score_fn,
        mutation_model=args.mutation_model,
        paradigm_model=args.paradigm_model,
        embedding_model=args.embedding_model,
        budget_evals=args.evals,
        budget_dollars=args.dollars,
        budget_seconds=args.seconds,
        n_workers=args.workers,
        n_eval_processes=args.eval_processes,
        eval_timeout=args.eval_timeout,
        pe_cron_interval=args.pe_interval,
        n_diverse_seeds=args.n_diverse_seeds,
        n_variants_per_seed=args.n_variants_per_seed,
        n_paradigm_variants=args.n_paradigm_variants,
        output_dir=args.output_dir,
    )

    print()
    print("=" * 72)
    print(f"BLADE finished - evals={result.total_evaluations} cost=${result.total_cost:.4f}")
    print(f"  best score:    {result.best_score:.6f}")
    print(f"  pool size:     {result.pool_size}")
    print(f"  runtime:       {result.runtime_seconds:.1f}s")
    print(f"  output dir:    {result.output_dir}")
    print(f"  paradigm runs: {len(result.paradigm_trials)}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
