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
    build_mutate_prompt,
    build_paradigm_prompt,
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
    llm_max_tokens: int = 1200
    """Token cap for mutation / crossover / repair calls."""
    paradigm_max_tokens: int | None = None
    """Token cap for the frontier (paradigm-shift) call. **Leave at None.**
    Reasoning-heavy models (GPT-5, o1, …) consume most of a fixed budget
    on internal thinking before any visible text, returning an empty
    content block when capped — letting the model use its provider-side
    default is the only reliable path."""

    # Loop schedule
    pe_cron_interval: int = 50
    """Frontier paradigm-shift fires every N completed evaluations."""
    paradigm_min_pool_size: int = 5
    """Skip paradigm shift if the pool has fewer than this many programs
    (not enough representatives to make the prompt useful)."""

    # Operator mix
    p_crossover_healthy: float = 0.30
    p_crossover_stuck: float = 0.70

    # Repair (one-shot per error candidate, mutation model only)
    enable_repair: bool = True

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

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._eval_processes: ResilientProcessPool | None = None
        self._semaphore = asyncio.Semaphore(config.n_workers)

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
        result = await client.acompletion(
            prompt,
            temperature=temperature,
            max_tokens=effective_max_tokens,
        )
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
        if not isinstance(result, dict):
            return float("-inf"), {}, f"non-dict result: {type(result).__name__}"
        if "error" in result and result.get("error"):
            return float("-inf"), result, str(result["error"])
        score = float(result.get("score", 0.0))
        return score, result, None

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
        accepted, reason = self.pool.add(program)
        self.monitor.record_eval(
            score=score,
            accepted=accepted,
            embedding=embedding if accepted else None,
        )
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
                # Cold start: just sample raw from seed prompt — handled by
                # _bootstrap_seed before workers start.
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
                    )
                    parent_score = parent.score

                raw = await self._call(self.mutation_lm, prompt, temperature=temp)
                parsed = self.parser.parse(raw)
                if not parsed.has_code:
                    logger.debug("[BLADE] no code in mutation output; skipping")
                    return
                score, _scores_dict, err = await self._evaluate_code(parsed.code)
                if err is not None:
                    self.error_buffer.append((parsed.code, parent_score, err))
                    self.monitor.record_eval(score=score, accepted=False, embedding=None)
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
            except Exception:
                logger.exception("[BLADE] worker step failed")

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
        except Exception:
            logger.exception("[BLADE] repair LLM call failed")
            return
        parsed = self.parser.parse(raw)
        if not parsed.has_code:
            return
        score, _scores, err = await self._evaluate_code(parsed.code)
        if err is not None:
            # Drop — one-shot only, per design (no infinite repair loops).
            self.monitor.record_eval(score=score, accepted=False, embedding=None)
            return
        description = await self._summarize_if_needed(parsed.code, parsed.description)
        await self._admit(
            code=parsed.code,
            description=description,
            score=score,
            source="repair",
            parent_score=parent_score,
        )

    async def _paradigm_shift(self) -> None:
        """Frontier (paradigm-shift) LLM call. Picks 3 representatives by phase,
        feeds them as descriptions only, and records a ParadigmTrial."""
        if len(self.pool) < self.config.paradigm_min_pool_size:
            return
        stage = get_budget_stage(
            budget_progress=0.0,
            stagnation=self.monitor.stagnation_level(),
        )
        reps = self.pool.representatives(stage, n=3)  # type: ignore[arg-type]
        rep_pairs = [(p.description, p.score) for p in reps]
        for p in reps:
            self.pool.mark_used(p)

        prev_best = self.monitor.best_score
        prompt = build_paradigm_prompt(
            stage=stage,
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            n_evaluations=self.monitor.eval_count,
            n_regions=self.pool.num_families(),
            representatives=rep_pairs,
            recent_trials=list(self.recent_trials),
        )
        try:
            # Frontier call: never cap tokens — reasoning-heavy models
            # need their full provider-side budget to produce any visible
            # output. ``max_tokens=None`` makes the LM client drop the
            # field entirely. See BladeConfig.paradigm_max_tokens.
            raw = await self._call(
                self.paradigm_lm,
                prompt,
                temperature=0.7,
                max_tokens=self.config.paradigm_max_tokens,  # = None by default
            )
        except Exception:
            logger.exception("[BLADE] paradigm-shift LLM call failed")
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
            self.monitor.record_eval(score=score, accepted=False, embedding=None)

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

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def _bootstrap_seed(self) -> None:
        """Push the seed program (if any) into the pool. Otherwise ask the
        mutation model to draft an initial paradigm from the signature alone."""
        seed = self.config.seed_program
        if seed:
            score, _scores, err = await self._evaluate_code(seed)
            description = await self._summarize_if_needed(seed, "")
            if err is None:
                await self._admit(
                    code=seed,
                    description=description,
                    score=score,
                    source="init",
                    parent_score=None,
                )
                return
            logger.warning("[BLADE] seed program failed to evaluate: %s", err)

        # No usable seed — ask the mutation model for an initial draft.
        prompt = (
            f"# Problem\n{self.config.problem_description}\n\n"
            f"# Signature\n```python\n{self.config.function_signature}\n```\n\n"
            "Write a baseline solution. Pick any reasonable paradigm — we "
            "will diversify from here.\n\n"
        )
        from ..simple.parser import OUTPUT_FORMAT_INSTRUCTION

        prompt = prompt + OUTPUT_FORMAT_INSTRUCTION
        try:
            raw = await self._call(self.mutation_lm, prompt, temperature=0.6)
        except Exception:
            logger.exception("[BLADE] seed bootstrap LLM call failed")
            return
        parsed = self.parser.parse(raw)
        if not parsed.has_code:
            return
        score, _scores, err = await self._evaluate_code(parsed.code)
        if err is not None:
            return
        description = await self._summarize_if_needed(parsed.code, parsed.description)
        await self._admit(
            code=parsed.code,
            description=description,
            score=score,
            source="init",
            parent_score=None,
        )

    # ------------------------------------------------------------------
    # Top-level run loop
    # ------------------------------------------------------------------

    async def run(self) -> BladeResult:
        self.start_time = time.time()
        # Spin up evaluator subprocess pool.
        self._eval_processes = ResilientProcessPool(max_workers=self.config.n_eval_processes)
        try:
            await self._bootstrap_seed()
            await self._main_loop()
        finally:
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
        """Run mutation workers in waves; fire paradigm shifts on the cron.

        Frontier paradigm calls and self-repair are dispatched as background
        tasks (one of each at most) so they cannot block the mutation
        producers — a slow frontier model would otherwise stall the worker
        pool for tens of seconds.
        """
        cfg = self.config

        in_flight: set[asyncio.Task] = set()
        paradigm_task: asyncio.Task | None = None
        repair_task: asyncio.Task | None = None
        next_pe_at = cfg.pe_cron_interval
        last_eval_count = -1
        stall_ticks = 0

        while not self.stop_event.is_set() and not self._budget_exhausted():
            # Launch up to n_workers concurrent mutation generations.
            while len(in_flight) < cfg.n_workers and not self._budget_exhausted():
                in_flight.add(asyncio.create_task(self._generate_one()))

            # Trigger cron paradigm shift (non-blocking — single task at a time).
            if (
                paradigm_task is None or paradigm_task.done()
            ) and self.monitor.eval_count >= next_pe_at:
                next_pe_at = self.monitor.eval_count + cfg.pe_cron_interval
                paradigm_task = asyncio.create_task(self._paradigm_shift())

            # Opportunistic repair (non-blocking; one in flight at a time).
            if (repair_task is None or repair_task.done()) and self.error_buffer:
                repair_task = asyncio.create_task(self._repair_one())

            # Wait for any of the worker / background tasks to finish.
            wait_set = set(in_flight)
            if paradigm_task is not None and not paradigm_task.done():
                wait_set.add(paradigm_task)
            if repair_task is not None and not repair_task.done():
                wait_set.add(repair_task)
            if not wait_set:
                # Nothing scheduled — guard against busy-spin on edge cases.
                await asyncio.sleep(0.05)
                continue

            done, _pending = await asyncio.wait(
                wait_set, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                exc = t.exception()
                if exc is not None:
                    logger.error("[BLADE] background task error: %s", exc)
            # Only worker tasks live in `in_flight`; paradigm/repair tasks
            # are tracked separately so we don't accidentally drop them.
            in_flight = {t for t in in_flight if not t.done()}

            # Stall detection: if eval_count hasn't moved for ~40 wait-cycles,
            # something is wrong (provider down, all workers failing). Bail.
            if self.monitor.eval_count == last_eval_count:
                stall_ticks += 1
                if stall_ticks >= 40:
                    logger.warning("[BLADE] stall detected; stopping early")
                    break
            else:
                stall_ticks = 0
                last_eval_count = self.monitor.eval_count

        # Drain remaining tasks.
        leftovers: list[asyncio.Task] = list(in_flight)
        if paradigm_task is not None and not paradigm_task.done():
            leftovers.append(paradigm_task)
        if repair_task is not None and not repair_task.done():
            leftovers.append(repair_task)
        for t in leftovers:
            t.cancel()
        if leftovers:
            await asyncio.gather(*leftovers, return_exceptions=True)

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
