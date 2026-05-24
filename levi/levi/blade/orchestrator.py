"""BLADE pipeline orchestrator.

A self-contained async event loop that drives a mutation-model pool plus
occasional frontier paradigm shifts. Reuses:

* :class:`levi.clients.LM` for LLM calls (costs tracked per call).
* :class:`levi.utils.ResilientProcessPool` for sandboxed code execution.
* :func:`levi.utils.evaluation.evaluate_code` for the actual scoring call.
* :func:`levi.equilibrium.prompts` 3-phase paradigm templates (via
  :mod:`levi.blade.prompts`).

Replaces:

* Pool       → :class:`levi.simple.Pool`            (description-embedding niching)
* Monitor    → :class:`levi.simple.Monitor`         (3 sliding-window signals)
* Selector   → :class:`levi.simple.Selector`        (UCB-style priority)
* Parser     → :class:`levi.simple.OutputParser`    (## Description + ## Code)
* Embedder   → :class:`levi.simple.DescriptionEmbedder`
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..clients import LM
from ..clients.base import BaseClient
from ..simple import (
    DescriptionEmbedder,
    EmbedderConfig,
    Monitor,
    MonitorConfig,
    OutputParser,
    ParserConfig,
    Pool,
    PoolConfig,
    Program,
    Selector,
    SelectorConfig,
)
from ..simple.parser import fallback_summarize
from ..utils.evaluation import evaluate_code
from ..utils.resilient_pool import ResilientProcessPool
from .prompts import (
    build_crossover_prompt,
    build_diverse_seed_prompt,
    build_init_variant_prompt,
    build_meta_advice_prompt,
    build_mutate_prompt,
    build_paradigm_prompt,
    build_paradigm_variant_prompt,
    build_repair_prompt,
    get_budget_stage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BladeConfig:
    """End-to-end configuration for one BLADE run."""

    # Problem
    problem_description: str
    function_signature: str
    score_fn: Callable[..., dict]
    fn_name: str
    inputs: list[Any] | None = None
    seed_program: str | None = None

    # Models — defaults align with _levi.yml so BLADE and LEVI are
    # budget-comparable out of the box. Qwen runs the high-frequency
    # mutation/repair calls; GPT-5 is reserved for the (rarer) frontier
    # paradigm shifts.
    mutation_model: str = "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
    paradigm_model: str = "openrouter/openai/gpt-5"
    embedding_model: str = "openrouter/openai/text-embedding-3-small"

    # Budget (any one triggers stop)
    budget_dollars: float | None = None
    budget_evals: int | None = None
    budget_seconds: float | None = None
    target_score: float | None = None

    # Concurrency
    n_workers: int = 4
    n_eval_processes: int = 4
    eval_timeout: float = 120.0
    llm_temperature: float = 0.8
    llm_temperature_stuck: float = 1.1
    llm_max_tokens: int | None = None
    """Token cap for mutation / crossover / repair calls.

    Default ``None`` = let the provider use its own default. Capping at a
    small number (e.g. 1200) caused parse_miss storms on the Qwen mutation
    model in production: a long seed-program prompt + 800-char strict
    OUTPUT_FORMAT_INSTRUCTION would push the model to spend most of its
    budget on description prose, leaving the response truncated *before*
    the opening ```python fence — so the parser saw no code block at all.
    Letting the provider pick the ceiling avoids that whole class of bug."""
    paradigm_max_tokens: int | None = None
    """Token cap for the frontier (paradigm-shift) call. **Leave at None.**
    Reasoning-heavy models (GPT-5, o1, …) consume most of a fixed budget
    on internal thinking before any visible text, returning an empty
    content block when capped — letting the model use its provider-side
    default is the only reliable path."""

    # Loop schedule
    pe_cron_interval: int = 50
    """Frontier paradigm-shift fires every N completed evaluations.

    Trigger semantics: a background monitor task (``_pe_monitor``) wakes
    every ~2 s and fires PE when
    ``eval_count >= last_pe_eval_count + pe_cron_interval``. After the
    shift finishes, ``last_pe_eval_count`` is snapped to the current
    ``eval_count`` so the K+1 evals it spent don't immediately re-trigger.

    This is a boundary-crossing gate (not exact modulo) because BLADE's
    ``asyncio.gather`` in bootstrap phase 2 and paradigm fanout makes
    ``eval_count`` jump in bursts — an exact-modulo gate would silently
    skip boundaries the bursts jump over.
    """
    paradigm_min_pool_size: int = 5
    """Skip paradigm shift if the pool has fewer than this many programs
    (not enough representatives to make the prompt useful)."""

    paradigm_n_anchors: int = 3
    """Number of anchor programs (full code + description + score) shown to
    the frontier model during a paradigm shift. The frontier uses these to
    read the actual mechanism, not just paraphrases of it."""

    paradigm_n_inspirations: int = 5
    """Number of *additional* programs shown as description-only sketches
    alongside the anchors. These widen the model's view of the archive
    without exploding the token count. Inspirations are picked diversely
    (MMR) from the pool, excluding the anchors. Set to 0 to disable."""

    # Initial population (LEVI-style 2-phase bootstrap)
    n_diverse_seeds: int = 5
    """Number of diverse seeds the frontier model generates SEQUENTIALLY
    in phase 0. Each prompt sees the previously accepted seeds and is
    pushed to design a fundamentally different paradigm. Mirrors LEVI's
    ``init.n_diverse_seeds``."""
    n_variants_per_seed: int = 20
    """How many variants the mutation model spins off PER accepted seed
    in phase 0 (parallel asyncio.gather). Total init candidates ≈
    n_diverse_seeds × n_variants_per_seed."""
    init_diversity_temperature: float = 0.8
    """Temperature for the frontier-model diverse-seed calls."""
    init_variant_temperature: float = 0.9
    """Temperature for the mutation-model init-variant calls."""

    # Paradigm shift fanout
    n_paradigm_variants: int = 4
    """K variants the mutation model spins off after each accepted
    paradigm-shift solution (parallel asyncio.gather). LEVI default."""
    paradigm_variant_temperature: float = 0.8
    """Temperature for the mutation-model paradigm-variant calls."""

    # Operator mix
    p_crossover_healthy: float = 0.30
    p_crossover_stuck: float = 0.70

    # Repair (one-shot per error candidate, mutation model only)
    enable_repair: bool = True

    # Meta-advisor (cron-fired post-mortem summary, injected into prompts)
    enable_meta_advice: bool = True
    """Toggle the LEVI-style meta-advisor. When on, a short lessons-learnt
    paragraph is generated every ``meta_advice_interval`` evaluations and
    injected (at probability ``meta_advice_inject_p``) into subsequent
    mutate / crossover prompts."""
    meta_advice_interval: int = 50
    """Generate fresh meta-advice every N completed evaluations. Mirrors
    LEVI's ``config.meta_advice.interval``."""
    meta_advice_inject_p: float = 0.8
    """Probability of injecting the current advice into any given mutate
    or crossover prompt (LEVI default)."""
    meta_advice_temperature: float = 0.4
    """Temperature for the mutation-model advisor calls (low — we want a
    crisp, factual summary)."""
    meta_advice_max_tokens: int = 400
    """Token cap for the advisor's output. The injected block is short by
    design; capping prevents the small model from running off."""

    # Output
    output_dir: str | Path = "runs/blade"

    # Subsystem overrides (advanced)
    pool_config: PoolConfig = field(default_factory=PoolConfig)
    monitor_config: MonitorConfig = field(default_factory=MonitorConfig)
    selector_config: SelectorConfig = field(default_factory=SelectorConfig)
    parser_config: ParserConfig = field(default_factory=ParserConfig)
    embedder_config: EmbedderConfig | None = None  # filled from embedding_model

    seed: int = 0


@dataclass
class ParadigmTrial:
    """One frontier paradigm-shift attempt."""

    trial_idx: int
    stage: str
    description: str
    accepted: bool
    score: float | None
    delta_vs_prev_best: float | None

    def render(self) -> str:
        score_str = f"{self.score:.4f}" if self.score is not None else "n/a"
        delta = self.delta_vs_prev_best
        delta_str = f"Δ={delta:+.4f}" if delta is not None else "Δ=n/a"
        accepted_str = "✓" if self.accepted else "✗"
        desc = self.description.strip().replace("\n", " ")
        if len(desc) > 160:
            desc = desc[:160] + "…"
        return f"[#{self.trial_idx} {self.stage}] {accepted_str} score={score_str} {delta_str} :: {desc}"


@dataclass
class BladeResult:
    best_program: str
    best_score: float
    total_evaluations: int
    total_cost: float
    pool_size: int
    runtime_seconds: float
    output_dir: str
    paradigm_trials: list[ParadigmTrial] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _CallLog:
    cost: float = 0.0
    calls: int = 0

    def record(self, cost: float) -> None:
        self.cost += float(cost or 0.0)
        self.calls += 1


class BladeOrchestrator:
    """One-shot orchestrator. Construct, ``await run()``, read attributes."""

    def __init__(self, config: BladeConfig) -> None:
        self.config = config
        random.seed(config.seed)

        # Subsystems
        embed_cfg = config.embedder_config or EmbedderConfig(model=config.embedding_model)
        self.embedder = DescriptionEmbedder(embed_cfg)
        self.parser = OutputParser(config.parser_config)
        self.pool = Pool(config.pool_config)
        self.monitor = Monitor(config.monitor_config)
        self.selector = Selector(config.selector_config)

        # LLM clients (cost-aware)
        self.mutation_lm: BaseClient = LM(config.mutation_model)
        self.paradigm_lm: BaseClient = LM(config.paradigm_model)

        # Bookkeeping
        self.cost = _CallLog()
        self.paradigm_trials: list[ParadigmTrial] = []
        self.recent_trials: deque[str] = deque(maxlen=5)
        self.error_buffer: deque[tuple[str, float, str]] = deque(maxlen=64)
        """Tail-recent failures: (broken_code, parent_score, error_msg).
        BLADE reuses LEVI's one-shot repair pattern: mutation model gets a
        chance to fix the most recent crash before we move on."""

        self.start_time: float = 0.0
        self.stop_event = asyncio.Event()

        # PE trigger gate — LEVI-style: modulo + freshness so the K+1 evals
        # spent inside a paradigm shift don't immediately re-trigger it on
        # another interval boundary.
        self.last_pe_eval_count: int = 0
        self.pe_trigger_count: int = 0
        # At most one paradigm shift in flight at any time.
        self._pe_lock = asyncio.Lock()

        # Meta-advisor state. ``current_meta_advice`` is the active block
        # injected into mutate/crossover prompts; ``last_meta_advice_eval_count``
        # gates the cron so we fire once per interval boundary.
        self.current_meta_advice: str | None = None
        self.last_meta_advice_eval_count: int = 0
        self.meta_advice_trigger_count: int = 0
        self._meta_advice_lock = asyncio.Lock()

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._eval_processes: ResilientProcessPool | None = None
        self._semaphore = asyncio.Semaphore(config.n_workers)

        # In-flight counters (LEVI-parity status line)
        self._client_in_flight: int = 0  # LLM calls currently awaiting a response
        self._eval_in_flight: int = 0  # evaluator subprocess slots currently busy

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def _budget_exhausted(self) -> bool:
        cfg = self.config
        if cfg.budget_dollars is not None and self.cost.cost >= cfg.budget_dollars:
            return True
        if cfg.budget_evals is not None and self.monitor.eval_count >= cfg.budget_evals:
            return True
        if cfg.budget_seconds is not None and (time.time() - self.start_time) >= cfg.budget_seconds:
            return True
        if cfg.target_score is not None and self.monitor.best_score >= cfg.target_score:
            return True
        return False

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    # Sentinel for "no max_tokens argument was passed" so callers can
    # distinguish from explicit ``max_tokens=None`` (= let the provider
    # decide, used for reasoning-heavy paradigm models).
    _DEFAULT_MAX_TOKENS = object()

    async def _call(
        self,
        client: BaseClient,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None | object = _DEFAULT_MAX_TOKENS,
    ) -> str:
        """Wrap the LLM client. ``max_tokens`` semantics:

        * omitted (default sentinel) → use ``config.llm_max_tokens``.
        * explicit integer           → use that value.
        * explicit ``None``          → DO NOT send max_tokens at all
          (LM client strips Nones), letting the provider use its own
          default. Required for reasoning-heavy models like GPT-5 that
          spend most of any fixed budget on internal thinking.
        """
        if max_tokens is self._DEFAULT_MAX_TOKENS:
            effective_max_tokens: int | None = self.config.llm_max_tokens
        else:
            effective_max_tokens = max_tokens  # type: ignore[assignment]
        self._client_in_flight += 1
        try:
            result = await client.acompletion(
                prompt,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )
        finally:
            self._client_in_flight -= 1
        self.cost.record(result.cost)
        # Defensive: some providers return ``None`` content when the model
        # exhausted its token budget on internal reasoning before emitting
        # any visible text. Treat that as empty so the parser can decide.
        return result.text or ""

    async def _summarize_if_needed(self, code: str, description: str) -> str:
        """Fallback summarizer: mutation model writes a paragraph when the
        primary call returned code-only."""
        if description and len(description) >= self.config.parser_config.min_description_chars:
            return description

        async def _runner(prompt: str) -> str:
            return await self._call(self.mutation_lm, prompt, temperature=0.3)

        try:
            return (await fallback_summarize(code, completion_fn=_runner)).strip()
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("[BLADE] fallback summary failed: %s", e)
            return description or ""

    # ------------------------------------------------------------------
    # Evaluate + admit
    # ------------------------------------------------------------------

    async def _evaluate_code(self, code: str) -> tuple[float, dict, str | None]:
        """Returns (score, scores_dict, error_msg)."""
        assert self._eval_processes is not None
        self._eval_in_flight += 1
        try:
            try:
                result = await self._eval_processes.run(
                    evaluate_code,
                    code,
                    self.config.score_fn,
                    self.config.inputs,
                    self.config.fn_name,
                    timeout=self.config.eval_timeout,
                )
            except Exception as e:
                return float("-inf"), {}, f"executor error: {e}"
        finally:
            self._eval_in_flight -= 1
        if not isinstance(result, dict):
            return float("-inf"), {}, f"non-dict result: {type(result).__name__}"
        if "error" in result and result.get("error"):
            return float("-inf"), result, str(result["error"])
        score = float(result.get("score", 0.0))
        return score, result, None

    def _model_label(self, source: str) -> str:
        """Map a candidate's source tag to the model name that produced its code.

        Used purely for the ``[Eval #N]`` log line so readers can tell at a glance
        which model an evaluation came from (LEVI prints the model id; BLADE has
        only two so we expose just the trailing component for brevity)."""
        paradigm_sources = {"paradigm"}
        # The frontier (paradigm) model only writes the seed itself; everything
        # else — paradigm variants, init variants, mutate, crossover, repair —
        # is written by the mutation model.
        model_id = (
            self.config.paradigm_model if source in paradigm_sources else self.config.mutation_model
        )
        # Strip the provider prefix so e.g. "openrouter/qwen/qwen3-30b-..." becomes
        # "qwen3-30b-..." (LEVI parity).
        return model_id.rsplit("/", 1)[-1]

    def _record_reject(
        self,
        *,
        source: str,
        score: float = float("-inf"),
        error_msg: str | None = None,
    ) -> None:
        """Single hook for the ~20 spots that increment ``eval_count`` without
        admitting a program (parse miss, eval crash, LLM error, etc.). Bumps the
        monitor and emits the LEVI-style log line in one go."""
        self.monitor.record_eval(score=score, accepted=False, embedding=None)
        self._log_eval(source=source, score=score, accepted=False, is_new_best=False, error_msg=error_msg)

    def _log_eval(
        self,
        *,
        source: str,
        score: float,
        accepted: bool,
        is_new_best: bool,
        error_msg: str | None = None,
    ) -> None:
        """Emit one LEVI-style ``[Eval #N]`` log line.

        Must be called *after* ``monitor.record_eval`` so ``eval_count`` already
        reflects this evaluation."""
        model = self._model_label(source)
        if error_msg is not None:
            logger.info(
                "[Eval #%d] %-27s ERROR (%s): %s",
                self.monitor.eval_count,
                model,
                source,
                error_msg[:80],
            )
            return
        if is_new_best:
            status = "NEW BEST ★"
        elif accepted:
            status = "accepted"
        else:
            status = "rejected"
        best = self.monitor.best_score
        best_str = f"{best:.6f}" if best != float("-inf") else "n/a"
        score_str = f"{score:.6f}" if score != float("-inf") else "n/a"
        logger.info(
            "[Eval #%d] %-27s %-12s | source: %-18s | score: %s | best: %s | $%.3f",
            self.monitor.eval_count,
            model,
            status,
            source,
            score_str,
            best_str,
            self.cost.cost,
        )

    async def _admit(
        self,
        *,
        code: str,
        description: str,
        score: float,
        source: str,
        parent_score: float | None,
    ) -> tuple[bool, str]:
        """Embed, push to pool, update monitor."""
        embedding = await asyncio.to_thread(self.embedder.embed, description)
        program = Program(
            code=code,
            description=description,
            score=score,
            embedding=embedding,
            source=source,  # type: ignore[arg-type]
            created_at_eval=self.monitor.eval_count + 1,
        )
        prev_best = self.monitor.best_score
        accepted, reason = self.pool.add(program)
        self.monitor.record_eval(
            score=score,
            accepted=accepted,
            embedding=embedding if accepted else None,
        )
        is_new_best = score > prev_best  # monitor.best_score updated above
        self._log_eval(source=source, score=score, accepted=accepted, is_new_best=is_new_best)
        if not accepted and reason == "dropped_duplicate" and parent_score is not None:
            # Surface near-duplicate non-improvements at debug level.
            logger.debug("[BLADE] drop near-duplicate score=%.4f parent=%.4f", score, parent_score)
        return accepted, reason

    # ------------------------------------------------------------------
    # Worker steps
    # ------------------------------------------------------------------

    def _pick_inspirations(self, exclude: list[Program]) -> list[tuple[str, float]]:
        insps = self.selector.select_inspirations(
            self.pool.programs(),
            exclude=exclude,
            n_total=self.monitor.eval_count,
            stuck=self.monitor.is_stuck(),
        )
        for p in insps:
            self.pool.mark_used(p)
        return [(p.description, p.score) for p in insps]

    def _operator(self) -> str:
        stuck = self.monitor.is_stuck()
        p_xover = self.config.p_crossover_stuck if stuck else self.config.p_crossover_healthy
        return "crossover" if random.random() < p_xover else "mutate"

    def _temperature(self) -> float:
        return (
            self.config.llm_temperature_stuck
            if self.monitor.is_stuck()
            else self.config.llm_temperature
        )

    async def _generate_one(self) -> None:
        """One end-to-end candidate generation. Holds the worker semaphore."""
        async with self._semaphore:
            if self._budget_exhausted():
                self.stop_event.set()
                return
            programs = self.pool.programs()
            if not programs:
                # Cold start: bootstrap should have populated the pool. If it
                # didn't (seed failed + draft LLM call failed), re-run it from
                # the worker so we don't loop forever returning empty-handed.
                logger.warning("[BLADE] empty pool inside worker; re-bootstrapping")
                await self._bootstrap_population()
                if not self.pool.programs():
                    # Still empty — count this as a reject so eval_count
                    # advances and the budget eventually triggers a clean exit.
                    self._record_reject(source="bootstrap_failed", error_msg="empty pool after rebootstrap")
                return

            stuck = self.monitor.is_stuck()
            op = self._operator()
            temp = self._temperature()
            try:
                if op == "crossover" and len(programs) >= 2:
                    pair = self.selector.select_two_parents(
                        programs, n_total=self.monitor.eval_count, stuck=stuck
                    )
                    assert pair is not None
                    p_a, p_b = pair
                    self.pool.mark_used(p_a)
                    self.pool.mark_used(p_b)
                    insps = self._pick_inspirations([p_a, p_b])
                    prompt = build_crossover_prompt(
                        problem_description=self.config.problem_description,
                        function_signature=self.config.function_signature,
                        parent_a_code=p_a.code,
                        parent_a_score=p_a.score,
                        parent_b_code=p_b.code,
                        parent_b_score=p_b.score,
                        inspirations=insps,
                        meta_advice=self._pick_meta_advice(),
                    )
                    parent_score = max(p_a.score, p_b.score)
                else:
                    parent = self.selector.select_parent(
                        programs, n_total=self.monitor.eval_count, stuck=stuck
                    )
                    assert parent is not None
                    self.pool.mark_used(parent)
                    insps = self._pick_inspirations([parent])
                    prompt = build_mutate_prompt(
                        problem_description=self.config.problem_description,
                        function_signature=self.config.function_signature,
                        parent_code=parent.code,
                        parent_score=parent.score,
                        inspirations=insps,
                        meta_advice=self._pick_meta_advice(),
                    )
                    parent_score = parent.score

                raw = await self._call(self.mutation_lm, prompt, temperature=temp)
                parsed = self.parser.parse(raw)
                if not parsed.has_code:
                    # Still count the attempt — otherwise a chronically
                    # malformed-output provider produces no eval progress and
                    # the loop appears to hang.
                    self._record_reject(source=op, error_msg="parse_miss (no code in output)")
                    return
                score, _scores_dict, err = await self._evaluate_code(parsed.code)
                if err is not None:
                    self.error_buffer.append((parsed.code, parent_score, err))
                    self._record_reject(source=op, score=score, error_msg=err)
                    return

                description = await self._summarize_if_needed(parsed.code, parsed.description)
                await self._admit(
                    code=parsed.code,
                    description=description,
                    score=score,
                    source=op,
                    parent_score=parent_score,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Count the failed worker as a reject so eval_count advances
                # and the main loop doesn't appear stuck. Common causes:
                # provider 5xx, rate-limit, transient network errors.
                logger.exception("[BLADE] worker step failed; counting as reject")
                self._record_reject(source=op, error_msg=f"worker exception: {e}")

    async def _repair_one(self) -> None:
        """One-shot self-repair on the freshest error in the buffer."""
        if not self.config.enable_repair or not self.error_buffer:
            return
        broken_code, parent_score, error_msg = self.error_buffer.popleft()
        prompt = build_repair_prompt(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            broken_code=broken_code,
            parent_score=parent_score,
            error_msg=error_msg,
        )
        try:
            raw = await self._call(self.mutation_lm, prompt, temperature=0.4)
        except Exception as e:
            logger.exception("[BLADE] repair LLM call failed; counting as reject")
            self._record_reject(source="repair", error_msg=f"LLM error: {e}")
            return
        parsed = self.parser.parse(raw)
        if not parsed.has_code:
            self._record_reject(source="repair", error_msg="parse_miss (no code in output)")
            return
        score, _scores, err = await self._evaluate_code(parsed.code)
        if err is not None:
            # Drop — one-shot only, per design (no infinite repair loops).
            self._record_reject(source="repair", score=score, error_msg=err)
            return
        description = await self._summarize_if_needed(parsed.code, parsed.description)
        await self._admit(
            code=parsed.code,
            description=description,
            score=score,
            source="repair",
            parent_score=parent_score,
        )

    def _budget_progress(self) -> float:
        """Fraction of the tightest budget consumed so far, in ``[0, 1]``.

        Used to route paradigm-shift stage (early/mid/late). Mirrors LEVI's
        ``state.budget_progress`` (min across the three caps; 0.0 when
        none configured)."""
        cfg = self.config
        progress = 0.0
        if cfg.budget_evals is not None and cfg.budget_evals > 0:
            progress = max(progress, min(1.0, self.monitor.eval_count / cfg.budget_evals))
        if cfg.budget_dollars is not None and cfg.budget_dollars > 0:
            progress = max(progress, min(1.0, self.cost.cost / cfg.budget_dollars))
        if cfg.budget_seconds is not None and cfg.budget_seconds > 0:
            elapsed = time.time() - self.start_time
            progress = max(progress, min(1.0, elapsed / cfg.budget_seconds))
        return progress

    async def _paradigm_shift(self) -> None:
        """Frontier paradigm shift, LEVI-style two-step:

        1. Frontier (paradigm) model generates ONE paradigm-shift solution
           via the three-stage prompt (early / mid / late). Evaluated and
           admitted as ``source="paradigm"``.
        2. Mutation (small) model fans out :attr:`config.n_paradigm_variants`
           variants of the accepted paradigm code in PARALLEL via
           :func:`asyncio.gather`. Each variant is admitted as
           ``source="paradigm_variant"``.

        If step 1 fails (LLM error, parse miss, eval error), step 2 is
        skipped — there's no fresh base code to fan out from.

        Total evals per call ≈ 1 + n_paradigm_variants (LEVI parity).
        """
        cfg = self.config
        if len(self.pool) < cfg.paradigm_min_pool_size:
            return
        stage = get_budget_stage(
            budget_progress=self._budget_progress(),
            stagnation=self.monitor.stagnation_level(),
        )
        anchors = self.pool.representatives(stage, n=cfg.paradigm_n_anchors)  # type: ignore[arg-type]
        anchor_triples = [(p.code, p.description, p.score) for p in anchors]
        for p in anchors:
            self.pool.mark_used(p)

        # Build a diverse pool of *additional* inspirations (description-only).
        # We over-fetch from ``representatives("early", …)`` (which uses MMR with
        # a low score weight, so the selection is diversity-biased) and skip
        # any program that already appears as an anchor.
        inspiration_pairs: list[tuple[str, float]] = []
        if cfg.paradigm_n_inspirations > 0:
            anchor_ids = {id(p) for p in anchors}
            wanted = cfg.paradigm_n_inspirations
            # Ask for n_anchors + n_inspirations so we still have ``wanted``
            # candidates after filtering out the anchors. ``representatives``
            # clamps to the pool size, so on small pools we just get fewer.
            fetch_n = len(anchors) + wanted
            candidates = self.pool.representatives("early", n=fetch_n)  # type: ignore[arg-type]
            for p in candidates:
                if id(p) in anchor_ids:
                    continue
                inspiration_pairs.append((p.description, p.score))
                if len(inspiration_pairs) >= wanted:
                    break

        prev_best = self.monitor.best_score
        prompt = build_paradigm_prompt(
            stage=stage,
            problem_description=cfg.problem_description,
            function_signature=cfg.function_signature,
            n_evaluations=self.monitor.eval_count,
            n_families=self.pool.num_families(),
            anchors=anchor_triples,
            inspirations=inspiration_pairs,
            recent_trials=list(self.recent_trials),
        )

        # ----- step 1: frontier paradigm seed -----
        try:
            raw = await self._call(
                self.paradigm_lm,
                prompt,
                temperature=0.7,
                max_tokens=cfg.paradigm_max_tokens,  # None — never cap frontier
            )
        except Exception as e:
            logger.exception("[BLADE PE] frontier call failed; counting as reject")
            self._record_reject(source="paradigm", error_msg=f"LLM error: {e}")
            return

        parsed = self.parser.parse(raw)
        if not parsed.has_code:
            trial = ParadigmTrial(
                trial_idx=len(self.paradigm_trials) + 1,
                stage=stage,
                description=parsed.description or "(no description)",
                accepted=False,
                score=None,
                delta_vs_prev_best=None,
            )
            self.paradigm_trials.append(trial)
            self.recent_trials.append(trial.render())
            self._record_reject(source="paradigm", error_msg="parse_miss (no code in output)")
            return

        score, _scores, err = await self._evaluate_code(parsed.code)
        description = await self._summarize_if_needed(parsed.code, parsed.description)
        accepted = False
        if err is None:
            accepted, _reason = await self._admit(
                code=parsed.code,
                description=description,
                score=score,
                source="paradigm",
                parent_score=prev_best if prev_best != float("-inf") else None,
            )
        else:
            self.error_buffer.append((parsed.code, prev_best, err))
            self._record_reject(source="paradigm", score=score, error_msg=err)

        delta = None
        if accepted and prev_best != float("-inf"):
            delta = score - prev_best
        elif accepted:
            delta = 0.0
        trial = ParadigmTrial(
            trial_idx=len(self.paradigm_trials) + 1,
            stage=stage,
            description=description if description else parsed.description,
            accepted=accepted,
            score=score if err is None else None,
            delta_vs_prev_best=delta,
        )
        self.paradigm_trials.append(trial)
        self.recent_trials.append(trial.render())
        if accepted:
            # Reset uses_count so the freshly-injected paradigm doesn't
            # immediately inherit a stale novelty penalty.
            self.pool.reset_uses_after_paradigm()

        # ----- step 2: mutation fanout (parallel) -----
        if err is not None or not parsed.has_code:
            logger.info(
                "[BLADE PE] frontier candidate unusable (err=%s) — skipping fanout",
                err if err else "parse_miss",
            )
            return
        if cfg.n_paradigm_variants <= 0:
            return
        if self._budget_exhausted():
            return

        base_code = parsed.code
        base_score = score
        logger.info(
            "[BLADE PE] fanout: %d variants from paradigm seed (score=%.4f)",
            cfg.n_paradigm_variants,
            base_score,
        )

        async def _one_paradigm_variant() -> None:
            if self._budget_exhausted():
                return
            v_prompt = build_paradigm_variant_prompt(
                problem_description=cfg.problem_description,
                function_signature=cfg.function_signature,
                base_code=base_code,
                base_score=base_score,
            )
            try:
                raw_v = await self._call(
                    self.mutation_lm,
                    v_prompt,
                    temperature=cfg.paradigm_variant_temperature,
                )
            except Exception as e:
                logger.exception("[BLADE PE] variant LLM call failed; counting as reject")
                self._record_reject(source="paradigm_variant", error_msg=f"LLM error: {e}")
                return
            parsed_v = self.parser.parse(raw_v)
            if not parsed_v.has_code:
                self._record_reject(source="paradigm_variant", error_msg="parse_miss (no code in output)")
                return
            v_score, _vscores, v_err = await self._evaluate_code(parsed_v.code)
            if v_err is not None:
                self.error_buffer.append((parsed_v.code, base_score, v_err))
                self._record_reject(source="paradigm_variant", score=v_score, error_msg=v_err)
                return
            v_description = await self._summarize_if_needed(parsed_v.code, parsed_v.description)
            await self._admit(
                code=parsed_v.code,
                description=v_description,
                score=v_score,
                source="paradigm_variant",
                parent_score=base_score,
            )

        await asyncio.gather(*(_one_paradigm_variant() for _ in range(cfg.n_paradigm_variants)))

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def _bootstrap_population(self) -> None:
        """LEVI-style two-phase initial population.

        Phase 1 — Diverse seeds (SEQUENTIAL, frontier model). Each call
        sees all previously accepted seeds, so the model is pushed toward
        algorithmic diversity. If the user supplied ``seed_program``, it
        is admitted first and counts as one of the seeds.

        Phase 2 — Variants (PARALLEL, mutation model). For each accepted
        seed, fan out ``n_variants_per_seed`` variants whose prompts each
        sample two seeds as inspirations. All variants are evaluated and
        admitted concurrently via :func:`asyncio.gather`.

        Mirrors :class:`levi.init.diversifier.Diversifier` and uses LEVI's
        DIVERSITY_SEED_PROMPT + VARIANT_GENERATION_PROMPT verbatim
        (wrapped to comply with BLADE's description-required parser).
        """
        cfg = self.config

        # ----- phase 1: diverse seeds (sequential) -----
        diverse_seeds: list[tuple[str, float, str]] = []  # (code, score, description)

        # If the user provided a seed_program, evaluate it first and treat it
        # as the first diverse seed.
        if cfg.seed_program:
            score, _scores, err = await self._evaluate_code(cfg.seed_program)
            if err is None:
                description = await self._summarize_if_needed(cfg.seed_program, "")
                await self._admit(
                    code=cfg.seed_program,
                    description=description,
                    score=score,
                    source="init",
                    parent_score=None,
                )
                diverse_seeds.append((cfg.seed_program, score, description))
                logger.info("[BLADE init] seed_program admitted (score=%.4f)", score)
            else:
                logger.warning("[BLADE init] seed_program failed to evaluate: %s", err)
                self._record_reject(source="init", score=score, error_msg=err)

        # Generate the remaining diverse seeds sequentially. If there's no
        # user seed, generate one extra to compensate (LEVI parity).
        n_seeds = cfg.n_diverse_seeds + (0 if cfg.seed_program else 1)
        max_retries = 3
        for i in range(n_seeds):
            if self._budget_exhausted():
                logger.info("[BLADE init] phase 1 stop (budget exhausted)")
                break

            success = False
            for attempt in range(max_retries):
                if self._budget_exhausted():
                    break
                tag = f"[seed {i + 1}]" if attempt == 0 else f"[seed {i + 1} retry {attempt}]"
                prompt = build_diverse_seed_prompt(
                    problem_description=cfg.problem_description,
                    function_signature=cfg.function_signature,
                    existing_seeds=[(c, s) for c, s, _ in diverse_seeds],
                )
                logger.info("[BLADE init] %s requesting diverse seed from %s",
                            tag, self.config.paradigm_model.rsplit("/", 1)[-1])
                try:
                    raw = await self._call(
                        self.paradigm_lm,
                        prompt,
                        temperature=cfg.init_diversity_temperature,
                        max_tokens=cfg.paradigm_max_tokens,  # None — never cap frontier
                    )
                except Exception as e:
                    logger.exception("[BLADE init] %s frontier call failed", tag)
                    self._record_reject(source="init", error_msg=f"LLM error: {e}")
                    continue

                parsed = self.parser.parse(raw)
                if not parsed.has_code:
                    logger.info("[BLADE init] %s no code in frontier output", tag)
                    self._record_reject(source="init", error_msg="parse_miss (no code in output)")
                    continue

                score, _scores, err = await self._evaluate_code(parsed.code)
                if err is not None:
                    logger.info("[BLADE init] %s eval failed: %s", tag, str(err)[:80])
                    self.error_buffer.append((parsed.code, float("-inf"), err))
                    self._record_reject(source="init", score=score, error_msg=err)
                    continue

                description = await self._summarize_if_needed(parsed.code, parsed.description)
                await self._admit(
                    code=parsed.code,
                    description=description,
                    score=score,
                    source="init",
                    parent_score=None,
                )
                diverse_seeds.append((parsed.code, score, description))
                logger.info("[BLADE init] %s OK (score=%.4f)", tag, score)
                success = True
                break

            if not success:
                logger.warning("[BLADE init] seed %d gave up after %d retries", i + 1, max_retries)

        logger.info("[BLADE init] phase 1 done: %d seeds admitted", len(diverse_seeds))

        if not diverse_seeds:
            logger.error("[BLADE init] phase 1 produced no usable seeds — skipping phase 2")
            return

        # ----- phase 2: variants (parallel) -----
        if cfg.n_variants_per_seed <= 0:
            return

        n_variants = cfg.n_variants_per_seed * len(diverse_seeds)
        logger.info(
            "[BLADE init] phase 2 generating %d variants (%d × %d) in parallel",
            n_variants,
            cfg.n_variants_per_seed,
            len(diverse_seeds),
        )

        # Build the prompts up front, sampling 2 seeds per variant as
        # inspirations (LEVI parity). When fewer than 2 seeds exist we
        # take what we have.
        n_inspirations = min(2, len(diverse_seeds))
        prompts: list[str] = []
        for _seed_code, _seed_score, _ in diverse_seeds:
            for _ in range(cfg.n_variants_per_seed):
                insps = random.sample(diverse_seeds, n_inspirations)
                prompts.append(
                    build_init_variant_prompt(
                        problem_description=cfg.problem_description,
                        function_signature=cfg.function_signature,
                        inspirations=[(c, s) for c, s, _ in insps],
                    )
                )

        async def _one_variant(prompt: str) -> None:
            if self._budget_exhausted():
                return
            try:
                raw = await self._call(
                    self.mutation_lm,
                    prompt,
                    temperature=cfg.init_variant_temperature,
                )
            except Exception as e:
                logger.exception("[BLADE init] variant LLM call failed; counting as reject")
                self._record_reject(source="init", error_msg=f"LLM error: {e}")
                return
            parsed = self.parser.parse(raw)
            if not parsed.has_code:
                self._record_reject(source="init", error_msg="parse_miss (no code in output)")
                return
            score, _scores, err = await self._evaluate_code(parsed.code)
            if err is not None:
                self.error_buffer.append((parsed.code, float("-inf"), err))
                self._record_reject(source="init", score=score, error_msg=err)
                return
            description = await self._summarize_if_needed(parsed.code, parsed.description)
            await self._admit(
                code=parsed.code,
                description=description,
                score=score,
                source="init",
                parent_score=None,
            )

        await asyncio.gather(*(_one_variant(p) for p in prompts))
        logger.info("[BLADE init] phase 2 done — pool size now %d", len(self.pool))

    # ------------------------------------------------------------------
    # Top-level run loop
    # ------------------------------------------------------------------

    async def run(self) -> BladeResult:
        self.start_time = time.time()
        # Spin up evaluator subprocess pool.
        self._eval_processes = ResilientProcessPool(max_workers=self.config.n_eval_processes)
        cfg = self.config
        logger.info(
            "[BLADE] starting — budget: $%s evals=%s seconds=%s target=%s",
            cfg.budget_dollars,
            cfg.budget_evals,
            cfg.budget_seconds,
            cfg.target_score,
        )
        logger.info(
            "[BLADE] models: mutation=%s | paradigm=%s | embedding=%s",
            cfg.mutation_model,
            cfg.paradigm_model,
            cfg.embedding_model,
        )
        logger.info(
            "[BLADE] knobs: workers=%d | pe_interval=%d | n_diverse_seeds=%d | n_variants/seed=%d | n_paradigm_var=%d",
            cfg.n_workers,
            cfg.pe_cron_interval,
            cfg.n_diverse_seeds,
            cfg.n_variants_per_seed,
            cfg.n_paradigm_variants,
        )
        # Background status monitor — emits a [Status] heartbeat every 30 s so the
        # log isn't silent during long bootstrap/eval stretches. Starts immediately
        # (alongside the bootstrap) so phase-1 progress is visible too.
        status_task = asyncio.create_task(self._status_monitor())
        try:
            logger.info("[BLADE init] phase 1 starting — %d diverse seeds (sequential, frontier model)",
                        cfg.n_diverse_seeds)
            await self._bootstrap_population()
            logger.info("[BLADE] bootstrap complete — pool=%d best=%.6f cost=$%.3f evals=%d",
                        len(self.pool),
                        self.monitor.best_score if self.monitor.best_score != float("-inf") else float("nan"),
                        self.cost.cost,
                        self.monitor.eval_count)
            # Background coroutines that run alongside the main loop:
            #   * _pe_monitor       fires paradigm shifts on cron boundaries
            #   * _meta_advice_monitor refreshes the lessons-learnt block
            pe_monitor_task = asyncio.create_task(self._pe_monitor())
            advisor_task = asyncio.create_task(self._meta_advice_monitor())
            try:
                logger.info("[BLADE] entering evolutionary main loop (n_workers=%d)", cfg.n_workers)
                await self._main_loop()
            finally:
                for t in (pe_monitor_task, advisor_task):
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
        finally:
            status_task.cancel()
            try:
                await status_task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                self._eval_processes.shutdown()
            except Exception:  # pragma: no cover — defensive
                logger.exception("[BLADE] failed to shut down process pool cleanly")

        elapsed = time.time() - self.start_time
        best = self.pool.best()
        result = BladeResult(
            best_program=best.code if best else (self.config.seed_program or ""),
            best_score=best.score if best else float("-inf"),
            total_evaluations=self.monitor.eval_count,
            total_cost=self.cost.cost,
            pool_size=len(self.pool),
            runtime_seconds=elapsed,
            output_dir=str(self.output_dir),
            paradigm_trials=list(self.paradigm_trials),
        )
        self._save_snapshot(result)
        return result

    async def _main_loop(self) -> None:
        """Run mutation workers and opportunistic repair until budget exhausts.

        Paradigm shifts are scheduled by the parallel ``_pe_monitor`` task
        (LEVI-style cron-modulo + freshness gate), NOT inline here — that
        keeps the trigger logic in one place and avoids the back-to-back
        firing problem the inline cron had at small ``pe_cron_interval``.
        Self-repair stays inline because it's cheap and event-driven (only
        fires when the error_buffer has fresh entries).
        """
        cfg = self.config

        in_flight: set[asyncio.Task] = set()
        repair_task: asyncio.Task | None = None

        while not self.stop_event.is_set() and not self._budget_exhausted():
            # Launch up to n_workers concurrent mutation generations.
            while len(in_flight) < cfg.n_workers and not self._budget_exhausted():
                in_flight.add(asyncio.create_task(self._generate_one()))

            # Opportunistic repair (non-blocking; one in flight at a time).
            if (repair_task is None or repair_task.done()) and self.error_buffer:
                repair_task = asyncio.create_task(self._repair_one())

            # Wait for any of the worker / repair tasks to finish.
            wait_set = set(in_flight)
            if repair_task is not None and not repair_task.done():
                wait_set.add(repair_task)
            if not wait_set:
                await asyncio.sleep(0.05)
                continue

            done, _pending = await asyncio.wait(
                wait_set, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                exc = t.exception()
                if exc is not None:
                    logger.error("[BLADE] background task error: %s", exc)
            in_flight = {t for t in in_flight if not t.done()}

        # Drain remaining tasks (workers + repair). PE monitor / paradigm
        # shifts are owned by run() and are cancelled there.
        leftovers: list[asyncio.Task] = list(in_flight)
        if repair_task is not None and not repair_task.done():
            leftovers.append(repair_task)
        for t in leftovers:
            t.cancel()
        if leftovers:
            await asyncio.gather(*leftovers, return_exceptions=True)

    # ------------------------------------------------------------------
    # Status monitor (LEVI-parity heartbeat)
    # ------------------------------------------------------------------

    async def _status_monitor(self) -> None:
        """Emit a one-line ``[Status]`` heartbeat every 30 s.

        Mirrors :meth:`levi.pipeline.runner.PipelineRunner._status_monitor` so
        long stretches with no admissions (bootstrap-phase evaluations, eval
        timeouts, PE in-progress) still show progress in the log."""
        try:
            while not self.stop_event.is_set():
                await asyncio.sleep(30.0)
                if self.stop_event.is_set():
                    break
                best = self.monitor.best_score
                best_str = f"{best:.6f}" if best != float("-inf") else "n/a"
                elapsed = time.time() - self.start_time
                logger.info(
                    "[Status] Cost: $%.3f | Evals: %d | Clients in-flight: %d | "
                    "Eval in-flight: %d | Pool: %d | Best: %s | Elapsed: %.0fs",
                    self.cost.cost,
                    self.monitor.eval_count,
                    self._client_in_flight,
                    self._eval_in_flight,
                    len(self.pool),
                    best_str,
                    elapsed,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[Status] monitor crashed")

    # ------------------------------------------------------------------
    # PE monitor (LEVI-style cron-modulo + freshness)
    # ------------------------------------------------------------------

    async def _pe_monitor(self) -> None:
        """Wake every ~2 s; fire paradigm shifts on cron boundary crossings.

        Adapted from :meth:`levi.pipeline.runner.PipelineRunner._pe_monitor`.

        BLADE differs from LEVI in one important respect: bootstrap phase 2
        and paradigm fanout submit many evaluations through
        :func:`asyncio.gather`, so ``monitor.eval_count`` jumps in bursts
        (e.g. 12 → 17 between two wake-ups). An exact-modulo gate
        (``ec % interval == 0``) silently skips boundaries that the bursts
        jump over. We use a **boundary-crossing** gate instead:

        * fire whenever
          ``eval_count >= last_pe_eval_count + pe_cron_interval``,
          regardless of whether the count landed exactly on a multiple of
          the interval.
        * after the shift completes, snap ``last_pe_eval_count`` forward
          to the current ``eval_count`` so the K+1 variants we just ran
          don't immediately re-trigger.
        * hold ``_pe_lock`` for the duration of the shift, so at most one
          paradigm task ever runs at a time.

        With ``pe_cron_interval=N`` you should see approximately
        ``(eval_count - phase0_evals) / N`` triggers per run.
        """
        cfg = self.config
        if cfg.pe_cron_interval <= 0:
            logger.info("[BLADE PE] disabled (pe_cron_interval <= 0)")
            return
        try:
            while not self.stop_event.is_set() and not self._budget_exhausted():
                await asyncio.sleep(2.0)
                if self.stop_event.is_set() or self._budget_exhausted():
                    break

                ec = self.monitor.eval_count
                if ec > 0 and ec >= self.last_pe_eval_count + cfg.pe_cron_interval:
                    self.last_pe_eval_count = ec
                    async with self._pe_lock:
                        self.pe_trigger_count += 1
                        stage_preview = get_budget_stage(
                            budget_progress=self._budget_progress(),
                            stagnation=self.monitor.stagnation_level(),
                        )
                        best = self.monitor.best_score
                        best_str = f"{best:.6f}" if best != float("-inf") else "n/a"
                        logger.info(
                            "[BLADE PE] trigger #%d at eval=%d | stage=%s | best=%s | pool=%d | families=%d",
                            self.pe_trigger_count,
                            ec,
                            stage_preview,
                            best_str,
                            len(self.pool),
                            self.pool.num_families(),
                        )
                        try:
                            await self._paradigm_shift()
                        except Exception:
                            logger.exception("[BLADE PE] paradigm shift errored")
                        # Snap forward so the variants we just ran don't
                        # immediately retrigger on another boundary.
                        self.last_pe_eval_count = max(
                            self.last_pe_eval_count, self.monitor.eval_count
                        )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[BLADE PE] monitor crashed")

    # ------------------------------------------------------------------
    # Meta-advisor (LEVI port — cron-fired summary injected into prompts)
    # ------------------------------------------------------------------

    def _pick_meta_advice(self) -> str | None:
        """Return the current meta-advice with probability
        :attr:`BladeConfig.meta_advice_inject_p`, else ``None``.

        Random gating mirrors LEVI so the prompts retain some variance
        across calls even while the advice is fresh."""
        if not self.config.enable_meta_advice:
            return None
        if not self.current_meta_advice:
            return None
        if random.random() >= self.config.meta_advice_inject_p:
            return None
        return self.current_meta_advice

    def _recent_error_messages(self, n: int = 5) -> list[str]:
        """Tail of the error buffer for the meta-advisor prompt.

        Returns at most ``n`` strings; oldest first. Each element is the
        error message portion of an ``error_buffer`` entry."""
        if not self.error_buffer:
            return []
        items = list(self.error_buffer)[-n:]
        return [msg for (_code, _parent_score, msg) in items]

    async def _generate_meta_advice(self) -> None:
        """Ask the mutation model to write the next lessons-learnt block.

        Inputs are the current period's stats (best score, accept rate,
        stagnation, top recent errors) plus the previous advice (so the
        model can refine rather than restart). Output replaces
        ``self.current_meta_advice`` verbatim. Failures are non-fatal —
        we just keep the previous advice."""
        cfg = self.config
        prompt = build_meta_advice_prompt(
            problem_description=cfg.problem_description,
            function_signature=cfg.function_signature,
            best_score=self.monitor.best_score,
            n_evaluations=self.monitor.eval_count,
            accept_rate=self.monitor.acceptance_rate(),
            stagnation_level=self.monitor.stagnation_level(),
            recent_errors=self._recent_error_messages(5),
            previous_advice=self.current_meta_advice,
        )
        try:
            raw = await self._call(
                self.mutation_lm,
                prompt,
                temperature=cfg.meta_advice_temperature,
                max_tokens=cfg.meta_advice_max_tokens,
            )
        except Exception:
            logger.exception("[BLADE advisor] LLM call failed; keeping previous advice")
            return
        # The advisor output is prose; we don't run it through OutputParser
        # because there is no code to extract. Trim and keep.
        text = (raw or "").strip()
        if not text:
            logger.info("[BLADE advisor] empty response; keeping previous advice")
            return
        # Defensive cap so a runaway response can't bloat downstream prompts.
        if len(text) > 1200:
            text = text[:1200].rstrip() + "…"
        self.current_meta_advice = text
        logger.info(
            "[BLADE advisor] new advice (%d chars) at eval=%d",
            len(text),
            self.monitor.eval_count,
        )

    async def _meta_advice_monitor(self) -> None:
        """Wake every ~2 s; refresh meta-advice on boundary crossings.

        Same boundary-crossing semantics as ``_pe_monitor`` (see its
        docstring) — required because BLADE's ``asyncio.gather`` makes
        ``eval_count`` jump in bursts, which would slip past an exact-
        modulo gate. Holds ``_meta_advice_lock`` so at most one advisor
        call ever runs at a time."""
        cfg = self.config
        if not cfg.enable_meta_advice or cfg.meta_advice_interval <= 0:
            logger.info(
                "[BLADE advisor] disabled (enable_meta_advice=%s, interval=%d)",
                cfg.enable_meta_advice,
                cfg.meta_advice_interval,
            )
            return
        try:
            while not self.stop_event.is_set() and not self._budget_exhausted():
                await asyncio.sleep(2.0)
                if self.stop_event.is_set() or self._budget_exhausted():
                    break

                ec = self.monitor.eval_count
                if ec > 0 and ec >= self.last_meta_advice_eval_count + cfg.meta_advice_interval:
                    self.last_meta_advice_eval_count = ec
                    async with self._meta_advice_lock:
                        self.meta_advice_trigger_count += 1
                        logger.info(
                            "[BLADE advisor] trigger #%d at eval=%d",
                            self.meta_advice_trigger_count,
                            ec,
                        )
                        await self._generate_meta_advice()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[BLADE advisor] monitor crashed")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save_snapshot(self, result: BladeResult) -> None:
        snap = {
            "method": "blade",
            "best_score": result.best_score,
            "total_evaluations": result.total_evaluations,
            "total_cost": result.total_cost,
            "pool_size": result.pool_size,
            "runtime_seconds": result.runtime_seconds,
            "monitor": self.monitor.snapshot(),
            "meta_advice": {
                "enabled": self.config.enable_meta_advice,
                "interval": self.config.meta_advice_interval,
                "trigger_count": self.meta_advice_trigger_count,
                "current": self.current_meta_advice,
            },
            "paradigm_trials": [
                {
                    "idx": t.trial_idx,
                    "stage": t.stage,
                    "accepted": t.accepted,
                    "score": t.score,
                    "delta": t.delta_vs_prev_best,
                    "description": t.description,
                }
                for t in result.paradigm_trials
            ],
            "elites": [
                {
                    "score": p.score,
                    "source": p.source,
                    "created_at_eval": p.created_at_eval,
                    "uses_count": p.uses_count,
                    "family_id": p.family_id,
                    "description": p.description,
                    "content": p.code,
                }
                for p in sorted(self.pool.programs(), key=lambda x: -x.score)
            ],
        }
        (self.output_dir / "snapshot.json").write_text(json.dumps(snap, indent=2))
        if result.best_program:
            (self.output_dir / "best.py").write_text(result.best_program)
