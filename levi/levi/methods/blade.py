"""BLADE entry point — mirrors :func:`levi.evolve_code` so existing problem
definitions plug in without changes."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..blade.orchestrator import BladeConfig, BladeOrchestrator, BladeResult
from ..utils.code_extraction import extract_fn_name

__all__ = ["evolve_code_blade", "BladeConfig", "BladeResult"]


def _setup_logging() -> None:
    """Configure logging for BLADE (mirrors levi._setup_logging)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def evolve_code_blade(
    problem_description: str,
    *,
    function_signature: str,
    score_fn: Callable[..., dict],
    seed_program: str | None = None,
    inputs: list[Any] | None = None,
    mutation_model: str | None = None,
    paradigm_model: str | None = None,
    embedding_model: str | None = None,
    budget_dollars: float | None = None,
    budget_evals: int | None = None,
    budget_seconds: float | None = None,
    target_score: float | None = None,
    n_workers: int = 4,
    n_eval_processes: int = 4,
    eval_timeout: float = 120.0,
    pe_cron_interval: int = 50,
    n_diverse_seeds: int = 5,
    n_variants_per_seed: int = 20,
    n_paradigm_variants: int = 4,
    enable_meta_advice: bool = True,
    meta_advice_interval: int = 50,
    output_dir: str | Path | None = None,
    **overrides: Any,
) -> BladeResult:
    """Run BLADE evolutionary code optimization.

    Mirrors :func:`levi.evolve_code` for parameter compatibility — any
    benchmark wrapper that calls ``levi.evolve_code(...)`` can swap to
    BLADE with a one-line change.

    Parameters
    ----------
    problem_description, function_signature, score_fn, seed_program, inputs
        Same as ``levi.evolve_code``.
    mutation_model, paradigm_model, embedding_model
        OpenRouter / litellm model IDs. Defaults follow BLADE's standard
        small / large / embedding split.
    budget_dollars, budget_evals, budget_seconds, target_score
        Stop conditions; first to trigger wins.
    n_workers, n_eval_processes, eval_timeout
        Concurrency knobs.
    pe_cron_interval
        Fire the frontier (paradigm-shift) model every N evaluations.
    n_diverse_seeds, n_variants_per_seed
        Phase-0 (bootstrap) shape. The frontier model generates
        ``n_diverse_seeds`` diverse seeds sequentially; the mutation model
        spins off ``n_variants_per_seed`` parallel variants per seed.
        LEVI parity defaults: 5 × 20 = up to 100 candidates before the
        evolutionary loop starts.
    n_paradigm_variants
        How many parallel variants the mutation model produces after each
        paradigm shift completes. LEVI parity default: 4.
    enable_meta_advice, meta_advice_interval
        LEVI-style lessons-learnt advisor. When on, the mutation model
        writes a short prescriptive note every ``meta_advice_interval``
        evaluations (default 50). The note is then injected (at 80%
        probability) into subsequent mutate / crossover prompts.
    output_dir
        Where to write ``snapshot.json`` + ``best.py``. Defaults to
        ``runs/blade-<timestamp>``.
    **overrides
        Any extra ``BladeConfig`` field (``archive_config``, ``monitor_config``,
        ``sampler_config``, ``parser_config``, ``llm_temperature``, ...).
    """
    if output_dir is None:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"blade-{ts}"

    cfg_kwargs: dict[str, Any] = dict(
        problem_description=problem_description,
        function_signature=function_signature,
        score_fn=score_fn,
        fn_name=extract_fn_name(function_signature),
        inputs=inputs,
        seed_program=seed_program,
        budget_dollars=budget_dollars,
        budget_evals=budget_evals,
        budget_seconds=budget_seconds,
        target_score=target_score,
        n_workers=n_workers,
        n_eval_processes=n_eval_processes,
        eval_timeout=eval_timeout,
        pe_cron_interval=pe_cron_interval,
        n_diverse_seeds=n_diverse_seeds,
        n_variants_per_seed=n_variants_per_seed,
        n_paradigm_variants=n_paradigm_variants,
        enable_meta_advice=enable_meta_advice,
        meta_advice_interval=meta_advice_interval,
        output_dir=output_dir,
    )
    if mutation_model is not None:
        cfg_kwargs["mutation_model"] = mutation_model
    if paradigm_model is not None:
        cfg_kwargs["paradigm_model"] = paradigm_model
    if embedding_model is not None:
        cfg_kwargs["embedding_model"] = embedding_model

    # Pass through any advanced BladeConfig overrides.
    cfg_kwargs.update(overrides)

    _setup_logging()
    config = BladeConfig(**cfg_kwargs)
    orchestrator = BladeOrchestrator(config)
    return asyncio.run(orchestrator.run())
