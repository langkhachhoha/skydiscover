#!/usr/bin/env python3
"""Generic driver for running BLADE Lite on any self-contained LEVI example.

Mirrors ``scripts/run_levi.py`` but invokes :func:`levi.evolve_code_blade`.
Every example under ``levi/examples/<name>/`` exporting
``PROBLEM_DESCRIPTION``, ``FUNCTION_SIGNATURE``, ``score_fn`` (and
optionally ``SEED_PROGRAM`` / ``INPUTS``) plugs in unchanged.

Usage::

    uv run python scripts/run_blade.py \\
        --example-dir levi/examples/blade_demo \\
        --evals 50

The script exposes only the few knobs that change run-to-run plus the
three paper-facing ablation toggles (--ast-only, --emb-only,
--static-cells). Lower-level knobs (e.g. number of cells, recluster
cadence, β bounds) are reachable through ``BladeConfig`` if needed.
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

    # Models
    p.add_argument(
        "--mutation-model",
        default="openrouter/qwen/qwen3-30b-a3b-instruct-2507",
        help="OpenRouter id of the small mutation model.",
    )
    p.add_argument(
        "--paradigm-model",
        default="openrouter/openai/gpt-5",
        help="OpenRouter id of the frontier model used for paradigm shifts.",
    )
    p.add_argument(
        "--embedding-model",
        default="openrouter/openai/text-embedding-3-small",
        help="OpenRouter id of the description-embedding model.",
    )

    # Budget
    p.add_argument("--evals", default="", help="Max evaluations (default: unset).")
    p.add_argument("--dollars", default="", help="Max USD spend (default: unset).")
    p.add_argument("--seconds", default="", help="Wall-clock cap in seconds (default: unset).")
    p.add_argument("--target-score", default="", help="Stop early at this score (default: unset).")

    # Concurrency
    p.add_argument("--workers", default="4", help="Concurrent LLM workers (default: 4).")
    p.add_argument("--eval-processes", default="4", help="Concurrent evaluator processes (default: 4).")
    p.add_argument("--eval-timeout", default="600", help="Per-candidate evaluation timeout in seconds.")

    # BLADE knobs
    p.add_argument(
        "--pe-interval", default="50",
        help="Frontier paradigm-shift cadence (every N evaluations; default 50).",
    )
    p.add_argument(
        "--n-diverse-seeds", type=int, default=5, metavar="N",
        help="Phase-1 diverse seeds (sequential, frontier model). Default 5.",
    )
    p.add_argument(
        "--n-variants-per-seed", type=int, default=20, metavar="N",
        help="Phase-2 variants per seed (parallel, mutation model). Default 20.",
    )
    p.add_argument(
        "--n-paradigm-variants", type=int, default=4, metavar="K",
        help="Paradigm-shift fanout (parallel mutation variants of the new seed).",
    )
    p.add_argument(
        "--paradigm-n-anchors", type=int, default=None, metavar="N",
        help="Cell representatives sent to the frontier each paradigm shift (default 4).",
    )
    p.add_argument(
        "--paradigm-n-inspirations", type=int, default=None, metavar="N",
        help="Description-only inspirations alongside the anchors (default 5).",
    )
    p.add_argument(
        "--no-repair", action="store_true",
        help="Disable the one-shot self-repair branch.",
    )
    p.add_argument(
        "--no-meta-advice", action="store_true",
        help="Disable the LEVI-style lessons-learnt advisor.",
    )
    p.add_argument(
        "--meta-advice-interval", type=int, default=50, metavar="N",
        help="Refresh meta-advice every N evaluations (default: 50).",
    )

    # ------------------------------------------------------------------
    # Targeted-mutate analyzer (Đề xuất 1) — review of top-ranked parents.
    # ------------------------------------------------------------------
    p.add_argument(
        "--no-targeted-mutate", action="store_true",
        help="Disable the LLM-generated parent analysis + TARGETED_MUTATE_PROMPT.",
    )
    p.add_argument(
        "--analyzer-interval", type=int, default=None, metavar="N",
        help="Refresh cached parent analyses every N evaluations (default 30).",
    )
    p.add_argument(
        "--analyzer-top-k", type=int, default=None, metavar="K",
        help="How many top-ranked programs to analyse each refresh (default 3).",
    )
    p.add_argument(
        "--p-targeted-mutate", type=float, default=None, metavar="P",
        help="Probability of using TARGETED_MUTATE_PROMPT when an analysis is "
        "cached for the chosen parent (default 0.5).",
    )

    # ------------------------------------------------------------------
    # Three-mode paradigm-shift thresholds (Đề xuất 8).
    # ------------------------------------------------------------------
    p.add_argument(
        "--paradigm-synthesis-max-stagnation", type=float, default=None, metavar="S",
        help="At or below this stagnation level the shift uses synthesis mode (default 0.4).",
    )
    p.add_argument(
        "--paradigm-shift-max-stagnation", type=float, default=None, metavar="S",
        help="Below this stagnation level (and above synthesis cap) the shift "
        "uses the 'new paradigm' mode (default 0.7); above this it goes surgical.",
    )
    p.add_argument(
        "--paradigm-synthesis-n-anchors", type=int, default=None, metavar="N",
        help="Number of anchors surfaced in synthesis mode (default 3).",
    )
    p.add_argument(
        "--paradigm-shift-n-anchors", type=int, default=None, metavar="N",
        help="Number of anchors surfaced in shift mode (default 2).",
    )
    p.add_argument(
        "--paradigm-surgical-n-inspirations", type=int, default=None, metavar="N",
        help="Description-only inspirations passed to surgical mode (default 5).",
    )

    # ------------------------------------------------------------------
    # Ablation toggles (the three paper-facing knobs)
    # ------------------------------------------------------------------
    p.add_argument(
        "--ast-only", action="store_true",
        help="Ablation A2: use AST features only for behavior signature "
        "(disable the description-embedding half of the hybrid).",
    )
    p.add_argument(
        "--emb-only", action="store_true",
        help="Ablation A1: use description embedding only for behavior "
        "signature (disable the AST half of the hybrid).",
    )
    p.add_argument(
        "--static-cells", action="store_true",
        help="Ablation A3: fit KMeans once and freeze cells thereafter "
        "(disable adaptive re-clustering).",
    )

    # Archive low-level overrides (rarely needed; exposed for sweeps)
    p.add_argument(
        "--n-cells", type=int, default=None, metavar="N",
        help="Target number of cells in the archive (default 32).",
    )
    p.add_argument(
        "--recluster-every", type=int, default=None, metavar="N",
        help="Re-fit KMeans every N admits (default 30).",
    )
    p.add_argument(
        "--embedding-dim", type=int, default=None, metavar="N",
        help="PCA target dimension for the description-embedding half (default 8).",
    )

    p.add_argument(
        "--output-dir", default=None,
        help="Where to drop snapshot.json + summary.json (default: outputs/blade/<example>/<ts>).",
    )

    return p.parse_args()


def _load_repo_env() -> None:
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

    sys.path.insert(0, str(example_dir))
    problem = importlib.import_module(args.problem_module)

    try:
        problem_description = problem.PROBLEM_DESCRIPTION
        function_signature = problem.FUNCTION_SIGNATURE
        score_fn = problem.score_fn
    except AttributeError as e:
        print(f"ERROR: example module is missing required attribute: {e}", file=sys.stderr)
        return 2

    seed_program = getattr(problem, "SEED_PROGRAM", None)
    inputs = getattr(problem, "INPUTS", None)

    evals = _opt_int(args.evals)
    dollars = _opt_float(args.dollars)
    seconds = _opt_float(args.seconds)
    target_score = _opt_float(args.target_score)
    workers = int(args.workers)
    eval_processes = int(args.eval_processes)
    eval_timeout = float(args.eval_timeout)
    pe_interval = _opt_int(args.pe_interval) or 50

    if args.output_dir:
        out_arg = Path(args.output_dir)
        output_dir = out_arg if out_arg.is_absolute() else (REPO_ROOT / out_arg).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = REPO_ROOT / "outputs" / "blade" / example_dir.name / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check on ablation flags.
    if args.ast_only and args.emb_only:
        print("ERROR: --ast-only and --emb-only are mutually exclusive", file=sys.stderr)
        return 2

    overrides: dict = {}
    from levi.simple import ArchiveConfig

    arch_kwargs = {}
    if args.ast_only:
        arch_kwargs["use_embedding"] = False
    if args.emb_only:
        arch_kwargs["use_ast"] = False
    if args.static_cells:
        arch_kwargs["adaptive_recluster"] = False
    if args.n_cells is not None:
        arch_kwargs["n_cells"] = args.n_cells
    if args.recluster_every is not None:
        arch_kwargs["recluster_every"] = args.recluster_every
    if args.embedding_dim is not None:
        arch_kwargs["embedding_dim"] = args.embedding_dim
    if arch_kwargs:
        overrides["archive_config"] = ArchiveConfig(**arch_kwargs)

    if args.no_repair:
        overrides["enable_repair"] = False
    if args.paradigm_n_anchors is not None:
        overrides["paradigm_n_anchors"] = args.paradigm_n_anchors
    if args.paradigm_n_inspirations is not None:
        overrides["paradigm_n_inspirations"] = args.paradigm_n_inspirations

    if args.no_targeted_mutate:
        overrides["enable_targeted_mutate"] = False
    if args.analyzer_interval is not None:
        overrides["analyzer_interval"] = args.analyzer_interval
    if args.analyzer_top_k is not None:
        overrides["analyzer_top_k"] = args.analyzer_top_k
    if args.p_targeted_mutate is not None:
        overrides["p_targeted_mutate"] = args.p_targeted_mutate

    if args.paradigm_synthesis_max_stagnation is not None:
        overrides["paradigm_synthesis_max_stagnation"] = args.paradigm_synthesis_max_stagnation
    if args.paradigm_shift_max_stagnation is not None:
        overrides["paradigm_shift_max_stagnation"] = args.paradigm_shift_max_stagnation
    if args.paradigm_synthesis_n_anchors is not None:
        overrides["paradigm_synthesis_n_anchors"] = args.paradigm_synthesis_n_anchors
    if args.paradigm_shift_n_anchors is not None:
        overrides["paradigm_shift_n_anchors"] = args.paradigm_shift_n_anchors
    if args.paradigm_surgical_n_inspirations is not None:
        overrides["paradigm_surgical_n_inspirations"] = args.paradigm_surgical_n_inspirations

    import levi

    print(f"[blade] example_dir       = {example_dir}")
    print(f"[blade] mutation_model    = {args.mutation_model}")
    print(f"[blade] paradigm_model    = {args.paradigm_model}")
    print(f"[blade] embedding_model   = {args.embedding_model}")
    print(f"[blade] budget evals      = {evals}")
    print(f"[blade] budget dollars    = {dollars}")
    print(f"[blade] budget seconds    = {seconds}")
    print(f"[blade] pe_cron_interval  = {pe_interval}")
    print(f"[blade] n_diverse_seeds   = {args.n_diverse_seeds}")
    print(f"[blade] n_variants/seed   = {args.n_variants_per_seed}")
    print(f"[blade] n_paradigm_var    = {args.n_paradigm_variants}")
    print(f"[blade] ablation_ast_only = {args.ast_only}")
    print(f"[blade] ablation_emb_only = {args.emb_only}")
    print(f"[blade] ablation_static   = {args.static_cells}")
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
        n_diverse_seeds=args.n_diverse_seeds,
        n_variants_per_seed=args.n_variants_per_seed,
        n_paradigm_variants=args.n_paradigm_variants,
        enable_meta_advice=not args.no_meta_advice,
        meta_advice_interval=args.meta_advice_interval,
        output_dir=output_dir,
        **overrides,
    )

    summary = {
        "method": "blade-lite",
        "example_dir": str(example_dir),
        "mutation_model": args.mutation_model,
        "paradigm_model": args.paradigm_model,
        "embedding_model": args.embedding_model,
        "best_score": result.best_score,
        "total_evaluations": result.total_evaluations,
        "total_cost": result.total_cost,
        "archive_size": result.archive_size,
        "runtime_seconds": result.runtime_seconds,
        "n_paradigm_trials": len(result.paradigm_trials),
        "ablation": {
            "ast_only": args.ast_only,
            "emb_only": args.emb_only,
            "static_cells": args.static_cells,
        },
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
    print(f"Archive size      : {result.archive_size}")
    print(f"Paradigm trials   : {len(result.paradigm_trials)}")
    print(f"Runtime           : {result.runtime_seconds:.1f}s")
    print(f"Output dir        : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
