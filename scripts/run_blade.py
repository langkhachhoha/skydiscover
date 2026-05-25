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
        "--n-diverse-seeds",
        type=int,
        default=5,
        metavar="N",
        help="Phase-0 diverse seeds generated sequentially by the frontier model (default: 5, LEVI parity).",
    )
    p.add_argument(
        "--n-variants-per-seed",
        type=int,
        default=20,
        metavar="N",
        help="Phase-0 mutation-model variants spun off per diverse seed in parallel (default: 20).",
    )
    p.add_argument(
        "--n-paradigm-variants",
        type=int,
        default=4,
        metavar="K",
        help="Paradigm-shift fanout: K mutation-model variants of each accepted paradigm seed (default: 4).",
    )
    p.add_argument(
        "--paradigm-n-anchors",
        type=int,
        default=None,
        metavar="N",
        help="Number of full-code anchor representatives shown to the frontier "
        "during a paradigm shift (default: 3).",
    )
    p.add_argument(
        "--paradigm-n-inspirations",
        type=int,
        default=None,
        metavar="N",
        help="Number of description-only inspiration sketches shown alongside the "
        "anchors during a paradigm shift (default: 5; set 0 to disable).",
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
        help="Description-embedding cosine threshold for near-duplicate flagging (default: 0.92).",
    )
    p.add_argument(
        "--structural-threshold",
        type=float,
        default=None,
        metavar="F",
        help="AST-signature cosine threshold for the second-pass near-duplicate "
        "filter. Two candidates are dropped only when BOTH the description AND "
        "the AST agree (default: 0.97). Pass >1.0 to disable the AST layer.",
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
        help="Maximum programs per family before weakest-in-family eviction "
        "(default: 10; family cap only fires once the pool reaches K).",
    )
    p.add_argument(
        "--no-repair",
        action="store_true",
        help="Disable the one-shot self-repair branch.",
    )
    p.add_argument(
        "--no-meta-advice",
        action="store_true",
        help="Disable the LEVI-style lessons-learnt advisor.",
    )
    p.add_argument(
        "--meta-advice-interval",
        type=int,
        default=50,
        metavar="N",
        help="Refresh meta-advice every N evaluations (default: 50).",
    )

    # ----------------------------------------------------------------------
    # Ablation toggles for the new architectural components. Defaults match
    # ``BladeConfig`` (= all components ON); pass --disable-* for ablations.
    # ----------------------------------------------------------------------

    p.add_argument(
        "--ast-mode",
        choices=("bigram", "count14"),
        default=None,
        help="Structural AST signature implementation. 'bigram' (default) is "
        "the production node-type bigram histogram; 'count14' reproduces the "
        "legacy 14-count log-vector for ablation.",
    )
    p.add_argument(
        "--disable-quota-niching",
        action="store_true",
        help="Ablate quota niching (component A). When set, the family cap "
        "only fires after the pool reaches K — the legacy behaviour that "
        "let the pool collapse into one family during the fill phase.",
    )
    p.add_argument(
        "--target-n-families",
        type=int,
        default=None,
        metavar="N",
        help="Quota niching: target number of families in the pool. Each "
        "family gets ceil(K/N) slots (default: 5).",
    )
    p.add_argument(
        "--disable-paradigm-grace",
        action="store_true",
        help="Ablate paradigm grace (component D). Paradigm-source programs "
        "lose their eviction-grace window and can be killed immediately by "
        "the family/K cap.",
    )
    p.add_argument(
        "--paradigm-grace-evals",
        type=int,
        default=None,
        metavar="N",
        help="Eviction-grace window length for paradigm-source programs "
        "(default: 30).",
    )
    p.add_argument(
        "--disable-hall-of-fame",
        action="store_true",
        help="Ablate the Hall of Fame (component D). Without HoF, "
        "paradigm-shift anchor-backfill is unavailable and the snapshot "
        "carries no hall_of_fame block.",
    )
    p.add_argument(
        "--hof-size",
        type=int,
        default=None,
        metavar="N",
        help="Hall of Fame capacity (default: 30).",
    )
    p.add_argument(
        "--disable-paradigm-boost",
        action="store_true",
        help="Ablate the selector's paradigm-source priority boost "
        "(component D). Paradigm programs compete on raw UCB priority.",
    )
    p.add_argument(
        "--paradigm-boost",
        type=float,
        default=None,
        metavar="F",
        help="Additive priority boost for paradigm-source programs inside "
        "the exploit window (default: 0.6).",
    )
    p.add_argument(
        "--paradigm-exploit-window",
        type=int,
        default=None,
        metavar="N",
        help="How long the paradigm boost stays active for a given program "
        "(default: 25 evaluations).",
    )
    p.add_argument(
        "--disable-cross-family-anchors",
        action="store_true",
        help="Ablate cross-family anchor selection (component C). Falls back "
        "to legacy Pool.representatives(stage) which can return anchors all "
        "from one family.",
    )
    p.add_argument(
        "--disable-force-early-on-collapse",
        action="store_true",
        help="Ablate the stuck/collapse → early-stage routing (component C). "
        "Lets budget_progress alone choose the paradigm stage.",
    )
    p.add_argument(
        "--paradigm-temperature",
        type=float,
        default=None,
        metavar="F",
        help="Frontier-model temperature for paradigm shifts in the healthy "
        "regime (default: 0.7).",
    )
    p.add_argument(
        "--paradigm-temperature-stuck",
        type=float,
        default=None,
        metavar="F",
        help="Frontier-model temperature when stuck/collapsing (default: 1.0).",
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
    from levi.simple.selector import SelectorConfig

    pool_kwargs = {}
    if args.pool_k is not None:
        pool_kwargs["K"] = args.pool_k
    if args.niche_threshold is not None:
        pool_kwargs["niche_cosine_threshold"] = args.niche_threshold
    if args.structural_threshold is not None:
        pool_kwargs["structural_cosine_threshold"] = args.structural_threshold
    if args.family_threshold is not None:
        pool_kwargs["family_cosine_threshold"] = args.family_threshold
    if args.max_per_family is not None:
        pool_kwargs["max_per_family"] = args.max_per_family
    if args.ast_mode is not None:
        pool_kwargs["ast_mode"] = args.ast_mode
    if args.disable_quota_niching:
        pool_kwargs["enable_quota_niching"] = False
    if args.target_n_families is not None:
        pool_kwargs["target_n_families"] = args.target_n_families
    if args.disable_paradigm_grace:
        pool_kwargs["enable_paradigm_grace"] = False
    if args.paradigm_grace_evals is not None:
        pool_kwargs["paradigm_grace_evals"] = args.paradigm_grace_evals
    if args.disable_hall_of_fame:
        pool_kwargs["enable_hall_of_fame"] = False
    if args.hof_size is not None:
        pool_kwargs["hof_size"] = args.hof_size
    if pool_kwargs:
        overrides["pool_config"] = PoolConfig(**pool_kwargs)

    selector_kwargs = {}
    if args.disable_paradigm_boost:
        selector_kwargs["enable_paradigm_boost"] = False
    if args.paradigm_boost is not None:
        selector_kwargs["paradigm_boost"] = args.paradigm_boost
    if args.paradigm_exploit_window is not None:
        selector_kwargs["paradigm_exploit_window"] = args.paradigm_exploit_window
    if selector_kwargs:
        overrides["selector_config"] = SelectorConfig(**selector_kwargs)

    if args.no_repair:
        overrides["enable_repair"] = False
    if args.paradigm_n_anchors is not None:
        overrides["paradigm_n_anchors"] = args.paradigm_n_anchors
    if args.paradigm_n_inspirations is not None:
        overrides["paradigm_n_inspirations"] = args.paradigm_n_inspirations
    if args.disable_cross_family_anchors:
        overrides["paradigm_cross_family_anchors"] = False
    if args.disable_force_early_on_collapse:
        overrides["paradigm_force_early_on_collapse"] = False
    if args.paradigm_temperature is not None:
        overrides["paradigm_temperature"] = args.paradigm_temperature
    if args.paradigm_temperature_stuck is not None:
        overrides["paradigm_temperature_stuck"] = args.paradigm_temperature_stuck

    import levi  # imported lazily so import errors surface clearly above

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
