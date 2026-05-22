"""Run BLADE on the coin-change demo problem.

    OPENAI_API_KEY=sk-or-v1-... python examples/blade_demo/run.py \\
        --evals 20 --workers 2

By default uses small budgets so this is a smoke test, not a real run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Load .env at repo root if present.
ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT.parent / ".env"
if ENV.exists():
    for raw in ENV.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

# litellm reads OPENROUTER_API_KEY for `openrouter/...` model IDs. When the
# repo ships only OPENAI_API_KEY=sk-or-v1-... (the OpenRouter key reused as
# the OpenAI key), mirror it across so both provider routes work.
if os.environ.get("OPENAI_API_KEY", "").startswith("sk-or-") and not os.environ.get(
    "OPENROUTER_API_KEY"
):
    os.environ["OPENROUTER_API_KEY"] = os.environ["OPENAI_API_KEY"]

# Make the in-tree LEVI importable when invoked from anywhere.
sys.path.insert(0, str(ROOT))

import levi  # noqa: E402

from problem import (  # noqa: E402
    FUNCTION_SIGNATURE,
    PROBLEM_DESCRIPTION,
    SEED_PROGRAM,
    score_fn,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals", type=int, default=20, help="Max evaluations")
    parser.add_argument("--dollars", type=float, default=None, help="Max USD spend")
    parser.add_argument("--seconds", type=float, default=None, help="Max wall-clock seconds")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-processes", type=int, default=2)
    parser.add_argument("--pe-interval", type=int, default=10)
    parser.add_argument(
        "--mutation-model",
        default=os.getenv("BLADE_MUTATION_MODEL", "openrouter/openai/gpt-4o-mini"),
    )
    parser.add_argument(
        "--paradigm-model",
        default=os.getenv("BLADE_PARADIGM_MODEL", "openrouter/openai/gpt-4o"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("BLADE_EMBEDDING_MODEL", "openrouter/openai/text-embedding-3-small"),
    )
    parser.add_argument("--no-seed", action="store_true", help="Bootstrap from scratch instead of using SEED_PROGRAM")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; aborting", file=sys.stderr)
        return 1

    result = levi.evolve_code_blade(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        score_fn=score_fn,
        seed_program=None if args.no_seed else SEED_PROGRAM,
        mutation_model=args.mutation_model,
        paradigm_model=args.paradigm_model,
        embedding_model=args.embedding_model,
        budget_evals=args.evals,
        budget_dollars=args.dollars,
        budget_seconds=args.seconds,
        n_workers=args.workers,
        n_eval_processes=args.eval_processes,
        pe_cron_interval=args.pe_interval,
        output_dir=args.output_dir,
    )

    print()
    print("=" * 72)
    print(f"BLADE finished — evals={result.total_evaluations}  cost=${result.total_cost:.4f}")
    print(f"  best score:    {result.best_score:.4f}")
    print(f"  pool size:     {result.pool_size}")
    print(f"  runtime:       {result.runtime_seconds:.1f}s")
    print(f"  output dir:    {result.output_dir}")
    print(f"  paradigm runs: {len(result.paradigm_trials)}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
