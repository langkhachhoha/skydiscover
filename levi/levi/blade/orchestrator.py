"""BLADE Lite orchestrator.

Async event loop that drives a mutation-model pool plus periodic
frontier paradigm shifts. The control flow is intentionally simple —
the only sophistication lives inside :class:`ClusterArchive` (adaptive
hybrid-behavior MAP-Elites) and :class:`RankSampler` (Zipfian rank).

Phases:

1. **Bootstrap phase 1** — frontier model writes ``n_diverse_seeds``
   paradigm seeds sequentially, each one shown the seeds already
   accepted so the model is pushed toward fresh paradigms.

2. **Bootstrap phase 2** — mutation model fans out
   ``n_variants_per_seed`` variants per seed in parallel.

3. **Main loop** — up to ``n_workers`` mutate/crossover workers running
   concurrently. Every ``pe_cron_interval`` evaluations a
   background task fires a paradigm shift: frontier writes one seed
   informed by the current cell representatives (one program per cell,
   ranked by score, capped at ``paradigm_n_anchors``), then the
   mutation model fans out ``n_paradigm_variants`` variants.

The orchestrator deliberately has **no** Hall-of-Fame, **no** grace
period, **no** selector boost, **no** quota cap. A program enters the
archive only if it beats the incumbent of its cell, and once in, it is
treated like any other program. The archive's cells are re-clustered
periodically so the niche structure follows the search rather than
being imposed up-front.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
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
    ArchiveConfig,
    ClusterArchive,
    DescriptionEmbedder,
    EmbedderConfig,
    Monitor,
    MonitorConfig,
    OutputParser,
    ParserConfig,
    Program,
    RankSampler,
    RankSamplerConfig,
)
from ..simple.parser import fallback_summarize
from ..utils.evaluation import evaluate_code
from ..utils.resilient_pool import ResilientProcessPool
from .prompts import (
    PromptSampler,
    build_analysis_prompt,
    build_crossover_prompt,
    build_diverse_seed_prompt,
    build_init_variant_prompt,
    build_meta_advice_prompt,
    build_mutate_prompt,
    build_paradigm_shift_prompt,
    build_paradigm_variant_prompt,
    build_surgical_exploit_prompt,
    build_synthesis_prompt,
    build_targeted_mutate_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BladeConfig:
    """End-to-end configuration for one BLADE Lite run.

    Compared to prior versions, this dataclass has shrunk by ~40 fields.
    The only paper-facing ablation knobs are the three components of
    :class:`ArchiveConfig` (``use_ast``, ``use_embedding``,
    ``adaptive_recluster``).
    """

    # Problem
    problem_description: str
    function_signature: str
    score_fn: Callable[..., dict]
    fn_name: str
    inputs: list[Any] | None = None
    seed_program: str | None = None

    # Models
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
    llm_max_tokens: int | None = None
    paradigm_max_tokens: int | None = None

    # Loop schedule
    pe_cron_interval: int = 50
    """Frontier paradigm-shift fires every N completed evaluations."""

    paradigm_min_archive_size: int = 5
    """Skip paradigm shift if the archive has fewer occupied cells than
    this — not enough representatives to make the prompt useful."""

    paradigm_n_anchors: int = 4
    """How many cell representatives (full code + description + score)
    to send to the frontier model. Capped by the number of occupied
    cells. 4 is the LEVI default and matches what GPT-5 handles
    comfortably in a single prompt without truncation."""

    paradigm_n_inspirations: int = 5
    """Description-only inspirations sent alongside the anchors. These
    widen the frontier's view of the archive without inflating tokens."""

    # Initial population
    n_diverse_seeds: int = 5
    n_variants_per_seed: int = 20
    init_diversity_temperature: float = 0.8
    init_variant_temperature: float = 0.9

    # Paradigm shift fanout
    n_paradigm_variants: int = 4
    paradigm_temperature: float = 0.8
    paradigm_variant_temperature: float = 0.85

    # Operator mix
    p_crossover: float = 0.35
    """Static probability of choosing crossover over mutate in the main
    loop. The rank sampler already adapts its β to stagnation, which
    covers the explore/exploit trade-off; an extra 'healthy vs stuck'
    switch produced no measurable benefit in prior versions."""

    # Operator-prompt ablation (A6).
    single_prompt_operators: bool = False
    """When True, the :class:`PromptSampler` stops drawing mutate /
    crossover templates at random and always uses the single simplest
    template per operator (``MUTATE_PROMPT_GENERAL`` for mutate,
    ``CROSSOVER_PROMPT_STRUCTURAL`` for crossover). This is the A6
    ablation — it isolates the contribution of prompt-template
    diversity from the rest of the search. The targeted-mutate prompt
    is orthogonal and controlled by ``enable_targeted_mutate`` (A5)."""

    # Targeted-mutate analyzer (Đề xuất 1 — analyse-then-mutate)
    enable_targeted_mutate: bool = True
    """Master toggle for the LLM-generated parent analysis pipeline.
    When on, the orchestrator periodically asks the mutation model for
    a short 'strengths / weaknesses / suggested changes' review of the
    top-ranked programs, caches it, and (with probability
    ``p_targeted_mutate``) injects it into a separate
    :data:`TARGETED_MUTATE_PROMPT` instead of the standard mutate
    prompt."""

    analyzer_interval: int = 30
    """Refresh cached analyses every N evaluations."""

    analyzer_top_k: int = 3
    """How many *new* programs to analyse on each refresh.

    Each cycle the analyser walks the archive in score-descending order
    and picks the first ``analyzer_top_k`` programs that do NOT yet
    have a cached analysis. Programs that already have a cache entry
    are skipped (their analysis is still valid). Cache entries are
    only evicted when the underlying program leaves the archive (cell
    incumbent kicked it out, or the program was never re-inserted by
    a re-cluster) — never just because the program slipped out of
    today's top-K.

    Behavioural consequence: when the top-K is frozen (best score
    plateaued, no churn), the cache grows monotonically — refresh #1
    analyses ranks 1-3, refresh #2 analyses ranks 4-6, and so on,
    until eventually every program in the archive has an analysis.
    From then on the analyser is a no-op until a fresh program is
    admitted, at which point exactly one new analysis is produced.

    This replaces the prior "top-K only, evict everything else"
    policy that re-analysed the same three frozen parents every
    cycle and starved targeted-mutate of analysis-level diversity."""

    analyzer_temperature: float = 0.3
    analyzer_max_tokens: int = 500

    p_targeted_mutate: float = 0.5
    """Probability of choosing :data:`TARGETED_MUTATE_PROMPT` when the
    chosen parent has a cached analysis. When no cached analysis is
    available the orchestrator falls back to the standard mutate
    sampler regardless of this probability. Lowered from 0.7 → 0.5
    because the prior value crowded out the prompt-template diversity
    provided by :class:`PromptSampler` whenever a top-K parent was
    picked, which happens most of the time under the Zipfian sampler."""

    # Paradigm-shift mode thresholds (three-mode shift).
    #
    # Mapping (rationale):
    #   stagnation ≤ paradigm_synthesis_max_stagnation  →  synthesis
    #   stagnation ≤ paradigm_surgical_max_stagnation   →  surgical
    #   stagnation  >  paradigm_surgical_max_stagnation →  shift
    #
    # The earlier version routed *high* stagnation to ``surgical`` and
    # mid stagnation to ``shift``. Empirically that left the search
    # trapped: when the champion has plateaued, asking the frontier for
    # another local fix to the same code rarely escapes. The fix that
    # actually broke prior plateaus in our runs was the ``shift`` mode
    # (a genuinely new paradigm class). We therefore route the deepest
    # stagnation to ``shift`` and reserve ``surgical`` for the
    # mid-stagnation band, where the champion still has some momentum
    # and a careful local polish can compound.
    paradigm_synthesis_max_stagnation: float = 0.4
    """At or below this stagnation level the paradigm shift picks the
    ``synthesis`` mode (combine 2-3 close-in-score anchors)."""

    paradigm_surgical_max_stagnation: float = 0.7
    """Above ``paradigm_synthesis_max_stagnation`` and at or below this
    threshold the paradigm shift picks ``surgical`` (deep tune of the
    champion). Above this threshold the shift flips to ``shift`` mode
    (propose a fundamentally new paradigm class) — when the search is
    deeply stuck, jumping paradigms is more likely to unlock progress
    than further sanding the same local optimum."""

    paradigm_synthesis_n_anchors: int = 3
    """Number of anchors surfaced in synthesis mode (2 or 3)."""

    paradigm_shift_n_anchors: int = 2
    """Number of anchors surfaced in shift mode."""

    paradigm_surgical_n_inspirations: int = 5
    """Description-only inspirations passed to surgical mode in
    addition to the single (champion) anchor."""

    # Paradigm-prompt ablation (A8).
    paradigm_force_mode: str | None = None
    """When set to one of ``"synthesis"``, ``"surgical"`` or
    ``"shift"``, every paradigm shift uses that single mode regardless
    of stagnation, instead of routing across the three modes via
    ``_pick_paradigm_mode``. This is the A8 ablation — it keeps the
    frontier paradigm-shift loop ON (unlike A7, which disables it
    entirely) but collapses the three-prompt repertoire down to one,
    isolating the contribution of mode diversity. ``None`` (default)
    keeps the adaptive three-mode routing."""

    # Meta-advisor — periodically writes a short prescriptive note that
    # future mutation prompts include verbatim.
    enable_meta_advice: bool = True
    meta_advice_interval: int = 50
    meta_advice_inject_p: float = 0.35
    """Probability of injecting the current advice block into a given
    mutation / crossover prompt.

    Lowered from 0.7 → 0.35. At 0.7 nearly every mutation prompt saw
    the same advice text, which heavily biased the mutation model
    toward whichever direction the last advisor cycle endorsed and
    blunted the diversity provided by the prompt-template sampler.
    0.35 means ~1 in 3 prompts is advice-guided, which still gives the
    advisor a meaningful nudge across a paradigm-shift window (50
    evals × 0.35 ≈ 17 advice-guided attempts) without saturating it.
    """

    meta_advice_temperature: float = 0.4
    meta_advice_max_tokens: int = 400
    meta_advice_mode: str = "rich"
    """Either ``"rich"`` (default) or ``"errors_only"``.

    ``rich`` feeds the advisor with the top-K archived descriptions,
    recent admits with score-delta vs parent, a saturation signal
    (admits whose score-delta vs parent is ≈ 0, indicating an operator
    family that is converting but no longer improving), and a typed
    error taxonomy — so the advice block can name what's working,
    what's saturated, what to try next, and what to avoid. The
    saturated-bucket signal is inspired by SeaEvo's Strategic
    Landscape Navigation, which tracks effective / saturated /
    underexplored strategy families at the population level rather
    than reasoning about individual programs in isolation.
    ``errors_only`` deliberately drops the success-side signals and is
    the ablation flag for "old-style" defensive advice.
    """

    # Output
    output_dir: str | Path = "runs/blade"

    # Safety
    llm_call_timeout: float = 600.0
    shutdown_grace_seconds: float = 15.0

    # Subsystem overrides (advanced)
    archive_config: ArchiveConfig = field(default_factory=ArchiveConfig)
    monitor_config: MonitorConfig = field(default_factory=MonitorConfig)
    sampler_config: RankSamplerConfig = field(default_factory=RankSamplerConfig)
    parser_config: ParserConfig = field(default_factory=ParserConfig)
    embedder_config: EmbedderConfig | None = None

    seed: int = 0


@dataclass
class ParadigmTrial:
    trial_idx: int
    description: str
    accepted: bool
    score: float | None
    delta_vs_prev_best: float | None

    def render(self, *, max_desc_chars: int = 280) -> str:
        score_str = f"{self.score:.4f}" if self.score is not None else "n/a"
        delta = self.delta_vs_prev_best
        delta_str = f"Δ={delta:+.4f}" if delta is not None else "Δ=n/a"
        accepted_str = "✓" if self.accepted else "✗"
        desc = self.description.strip().replace("\n", " ")
        if len(desc) > max_desc_chars:
            desc = desc[: max_desc_chars - 1] + "…"
        return f"[#{self.trial_idx}] {accepted_str} score={score_str} {delta_str} :: {desc}"


@dataclass
class BladeResult:
    best_program: str
    best_score: float
    total_evaluations: int
    total_cost: float
    archive_size: int
    runtime_seconds: float
    output_dir: str
    paradigm_trials: list[ParadigmTrial] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cost log
# ---------------------------------------------------------------------------


@dataclass
class _CallLog:
    cost: float = 0.0
    calls: int = 0

    def record(self, cost: float) -> None:
        self.cost += float(cost or 0.0)
        self.calls += 1


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class BladeOrchestrator:
    def __init__(self, config: BladeConfig) -> None:
        self.config = config
        random.seed(config.seed)
        self._rng = random.Random(config.seed)

        embed_cfg = config.embedder_config or EmbedderConfig(model=config.embedding_model)
        self.embedder = DescriptionEmbedder(embed_cfg)
        self.parser = OutputParser(config.parser_config)
        self.archive = ClusterArchive(config.archive_config)
        self.monitor = Monitor(config.monitor_config)
        self.sampler = RankSampler(config.sampler_config)

        self.mutation_lm: BaseClient = LM(config.mutation_model)
        self.paradigm_lm: BaseClient = LM(config.paradigm_model)

        self.cost = _CallLog()
        self.paradigm_trials: list[ParadigmTrial] = []
        self.recent_trials: deque[str] = deque(maxlen=6)
        # Rolling signal streams the meta-advisor reads each cycle.
        # Failed candidates simply count as rejects — there is no longer
        # a one-shot repair operator that drains an error queue.
        self._advisor_admits: deque[tuple[str, float, float | None]] = deque(maxlen=12)
        self._advisor_errors: deque[tuple[str, str]] = deque(maxlen=64)

        self.start_time: float = 0.0
        self.stop_event = asyncio.Event()

        self.last_pe_eval_count: int = 0
        self.pe_trigger_count: int = 0
        self._pe_lock = asyncio.Lock()

        self.current_meta_advice: str | None = None
        self.last_meta_advice_eval_count: int = 0
        self.meta_advice_trigger_count: int = 0
        self._meta_advice_lock = asyncio.Lock()

        # Targeted-mutate analyzer state (Đề xuất 1).
        self.prompt_sampler = PromptSampler(
            single_prompt=config.single_prompt_operators
        )
        self._analysis_cache: dict[int, str] = {}
        self._analysis_lock = asyncio.Lock()
        self.last_analyzer_eval_count: int = 0
        self.analyzer_trigger_count: int = 0
        self.targeted_mutate_count: int = 0

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._eval_processes: ResilientProcessPool | None = None
        self._semaphore = asyncio.Semaphore(config.n_workers)

        self._client_in_flight: int = 0
        self._eval_in_flight: int = 0

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

    _DEFAULT_MAX_TOKENS = object()

    async def _call(
        self,
        client: BaseClient,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None | object = _DEFAULT_MAX_TOKENS,
    ) -> str:
        if max_tokens is self._DEFAULT_MAX_TOKENS:
            effective_max_tokens: int | None = self.config.llm_max_tokens
        else:
            effective_max_tokens = max_tokens  # type: ignore[assignment]
        self._client_in_flight += 1
        try:
            call_coro = client.acompletion(
                prompt, temperature=temperature, max_tokens=effective_max_tokens,
            )
            timeout = self.config.llm_call_timeout
            if timeout and timeout > 0:
                result = await asyncio.wait_for(call_coro, timeout=timeout)
            else:
                result = await call_coro
        finally:
            self._client_in_flight -= 1
        self.cost.record(result.cost)
        return result.text or ""

    async def _summarize_if_needed(self, code: str, description: str) -> str:
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
        # Anything that came out of the frontier paradigm call uses the
        # paradigm model. Currently those sources are: "paradigm" (the
        # canonical Program.source for an admitted paradigm seed) and
        # "paradigm_<mode>" (the per-mode label we record on rejects so
        # the run log keeps each mode's failure rate separable).
        is_paradigm = source == "paradigm" or source.startswith("paradigm_")
        # Variant fanout runs on the mutation model — exclude that path.
        if source.startswith("paradigm_variant"):
            is_paradigm = False
        model_id = (
            self.config.paradigm_model if is_paradigm else self.config.mutation_model
        )
        return model_id.rsplit("/", 1)[-1]

    def _record_reject(self, *, source: str, score: float = float("-inf"), error_msg: str | None = None) -> None:
        self.monitor.record_eval(score=score, accepted=False)
        if error_msg:
            self._advisor_errors.append((source, error_msg))
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
        model = self._model_label(source)
        if error_msg is not None:
            logger.info(
                "[Eval #%d] %-27s ERROR (%s): %s",
                self.monitor.eval_count, model, source, error_msg[:80],
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
            self.monitor.eval_count, model, status, source, score_str, best_str, self.cost.cost,
        )

    async def _admit(
        self,
        *,
        code: str,
        description: str,
        score: float,
        source: str,
        parent_score: float | None,
        force_on_drop: bool = False,
    ) -> tuple[bool, str]:
        """Admit a candidate into the archive.

        ``force_on_drop=True`` is the paradigm-shift escape valve: if the
        archive would otherwise reject the program with ``dropped_worse``
        (its score didn't beat the cell incumbent), we evict the single
        lowest-score program currently in the archive and retry. Used only
        for frontier ``paradigm`` seeds in mode=``shift`` so a deeply
        stalled search can still inject a fresh paradigm.
        """
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
        if force_on_drop:
            accepted, reason = self.archive.force_add(program)
        else:
            accepted, reason = self.archive.add(program)
        self.monitor.record_eval(score=score, accepted=accepted)
        is_new_best = accepted and score > prev_best
        if accepted:
            self._advisor_admits.append((source, score, parent_score))
        self._log_eval(source=source, score=score, accepted=accepted, is_new_best=is_new_best)
        if not accepted and reason == "dropped_worse":
            logger.debug("[BLADE] dropped score=%.4f did not beat cell incumbent", score)
        if accepted and reason == "forced":
            logger.info(
                "[BLADE PE] force-admitted paradigm seed (score=%.4f) by evicting weakest program",
                score,
            )
        return accepted, reason

    # ------------------------------------------------------------------
    # Worker steps
    # ------------------------------------------------------------------

    def _pick_inspirations(self, exclude: list[Program]) -> list[tuple[str, float]]:
        insps = self.sampler.select_inspirations(
            self.archive.programs(),
            exclude=exclude,
            stagnation=self.monitor.stagnation_level(),
            rng=self._rng,
        )
        for p in insps:
            self.archive.mark_used(p)
        return [(p.description, p.score) for p in insps]

    def _pick_meta_advice(self) -> str | None:
        if not self.config.enable_meta_advice or not self.current_meta_advice:
            return None
        if self._rng.random() >= self.config.meta_advice_inject_p:
            return None
        return self.current_meta_advice

    async def _generate_one(self) -> None:
        async with self._semaphore:
            if self._budget_exhausted():
                self.stop_event.set()
                return
            programs = self.archive.programs()
            if not programs:
                logger.warning("[BLADE] empty archive inside worker; re-bootstrapping")
                await self._bootstrap_population()
                if not self.archive.programs():
                    self._record_reject(source="bootstrap_failed", error_msg="empty archive after rebootstrap")
                return

            stagnation = self.monitor.stagnation_level()
            op = "crossover" if self._rng.random() < self.config.p_crossover and len(programs) >= 2 else "mutate"
            try:
                if op == "crossover":
                    pair = self.sampler.select_two_parents(programs, stagnation=stagnation, rng=self._rng)
                    assert pair is not None
                    p_a, p_b = pair
                    self.archive.mark_used(p_a)
                    self.archive.mark_used(p_b)
                    insps = self._pick_inspirations([p_a, p_b])
                    _label, template = self.prompt_sampler.pick_crossover(self._rng)
                    op = f"crossover_{_label}"
                    prompt = build_crossover_prompt(
                        problem_description=self.config.problem_description,
                        function_signature=self.config.function_signature,
                        parent_a_code=p_a.code, parent_a_score=p_a.score,
                        parent_b_code=p_b.code, parent_b_score=p_b.score,
                        inspirations=insps,
                        meta_advice=self._pick_meta_advice(),
                        template=template,
                    )
                    parent_score = max(p_a.score, p_b.score)
                else:
                    parent = self.sampler.select_parent(programs, stagnation=stagnation, rng=self._rng)
                    assert parent is not None
                    self.archive.mark_used(parent)
                    insps = self._pick_inspirations([parent])
                    # Targeted-mutate path: only if we have a cached
                    # analysis for this parent AND the coin flip lands.
                    cached_analysis = self._analysis_cache.get(id(parent))
                    use_targeted = (
                        self.config.enable_targeted_mutate
                        and cached_analysis is not None
                        and self._rng.random() < self.config.p_targeted_mutate
                    )
                    if use_targeted:
                        self.targeted_mutate_count += 1
                        op = "mutate_targeted"
                        prompt = build_targeted_mutate_prompt(
                            problem_description=self.config.problem_description,
                            function_signature=self.config.function_signature,
                            parent_code=parent.code, parent_score=parent.score,
                            analysis=cached_analysis,
                            inspirations=insps,
                            meta_advice=self._pick_meta_advice(),
                        )
                    else:
                        _label, template = self.prompt_sampler.pick_mutate(self._rng)
                        op = f"mutate_{_label}"
                        prompt = build_mutate_prompt(
                            problem_description=self.config.problem_description,
                            function_signature=self.config.function_signature,
                            parent_code=parent.code, parent_score=parent.score,
                            inspirations=insps,
                            meta_advice=self._pick_meta_advice(),
                            template=template,
                        )
                    parent_score = parent.score

                raw = await self._call(self.mutation_lm, prompt, temperature=self.config.llm_temperature)
                parsed = self.parser.parse(raw)
                if not parsed.has_code:
                    self._record_reject(source=op, error_msg="parse_miss (no code in output)")
                    return
                score, _scores_dict, err = await self._evaluate_code(parsed.code)
                if err is not None:
                    self._record_reject(source=op, score=score, error_msg=err)
                    return
                description = await self._summarize_if_needed(parsed.code, parsed.description)
                await self._admit(
                    code=parsed.code, description=description, score=score,
                    source=op, parent_score=parent_score,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[BLADE] worker step failed; counting as reject")
                self._record_reject(source=op, error_msg=f"worker exception: {e}")

    # ------------------------------------------------------------------
    # Analyzer — accumulating review of as-yet-unanalysed parents
    # ------------------------------------------------------------------

    async def _analyze_parent(self, parent: Program) -> str | None:
        """Generate (or fetch from cache) a short LLM review of *parent*.

        Cached by ``id(parent)`` — re-runs the analysis only when the
        parent object is brand new in memory. Failed calls return
        ``None`` and are not cached so a future refresh can retry."""
        key = id(parent)
        async with self._analysis_lock:
            cached = self._analysis_cache.get(key)
            if cached is not None:
                return cached
        prompt = build_analysis_prompt(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            parent_code=parent.code,
            parent_score=parent.score,
            parent_description=parent.description,
        )
        try:
            text = await self._call(
                self.mutation_lm, prompt,
                temperature=self.config.analyzer_temperature,
                max_tokens=self.config.analyzer_max_tokens,
            )
        except Exception:
            logger.exception("[BLADE analyzer] LLM call failed; skipping this parent")
            return None
        text = (text or "").strip()
        if not text:
            return None
        async with self._analysis_lock:
            self._analysis_cache[key] = text
        # No per-entry cap: the cache is bounded above by the archive
        # size (cells × in-flight). Eviction is driven exclusively by
        # ``_refresh_analyses`` when a program leaves the archive, so
        # we never silently drop a still-valid analysis for an arbitrary
        # "oldest entry" — that's the bug the prior cap could cause.
        return text

    async def _refresh_analyses(self) -> None:
        """Refresh cached analyses with an accumulating, churn-driven policy.

        Each refresh:
          1. Evicts cache entries for programs that no longer live in
             the archive (kicked out by a better cell incumbent or
             dropped during re-cluster). The cache is keyed by
             ``id(program)`` so the only valid entries are ones whose
             program object is still reachable from the archive.
          2. Walks the archive in score-descending order and analyses
             the first ``analyzer_top_k`` programs that do *not* yet
             have a cache entry. Programs already in the cache are
             skipped — their analysis is unchanged.

        Concretely: refresh #1 analyses ranks 1-3; refresh #2, given a
        frozen top-3, analyses ranks 4-6; refresh #3 analyses ranks
        7-9; … until every program in the archive has an analysis,
        after which refreshes are no-ops until a new program enters
        the archive (cell churn → exactly one new analysis next
        cycle). Top-K analyses survive across cycles unless their
        parent program is actually displaced.
        """
        cfg = self.config
        if not cfg.enable_targeted_mutate:
            return
        programs = self.archive.programs()
        if not programs:
            return

        live_ids = {id(p) for p in programs}
        async with self._analysis_lock:
            stale = [k for k in self._analysis_cache if k not in live_ids]
            for k in stale:
                self._analysis_cache.pop(k, None)
            cached_ids = set(self._analysis_cache.keys())

        ranked = sorted(programs, key=lambda p: -p.score)
        to_analyze: list[Program] = []
        for p in ranked:
            if len(to_analyze) >= cfg.analyzer_top_k:
                break
            if id(p) not in cached_ids:
                to_analyze.append(p)

        if not to_analyze:
            return
        await asyncio.gather(*(self._analyze_parent(p) for p in to_analyze))

    async def _analyzer_monitor(self) -> None:
        cfg = self.config
        if not cfg.enable_targeted_mutate or cfg.analyzer_interval <= 0:
            logger.info(
                "[BLADE analyzer] disabled (enable=%s, interval=%d)",
                cfg.enable_targeted_mutate, cfg.analyzer_interval,
            )
            return
        try:
            while not self.stop_event.is_set() and not self._budget_exhausted():
                await asyncio.sleep(2.0)
                if self.stop_event.is_set() or self._budget_exhausted():
                    break
                ec = self.monitor.eval_count
                if ec > 0 and ec >= self.last_analyzer_eval_count + cfg.analyzer_interval:
                    self.last_analyzer_eval_count = ec
                    self.analyzer_trigger_count += 1
                    logger.info(
                        "[BLADE analyzer] trigger #%d at eval=%d "
                        "(analysing up to %d uncached programs; cache=%d)",
                        self.analyzer_trigger_count, ec,
                        cfg.analyzer_top_k, len(self._analysis_cache),
                    )
                    try:
                        await self._refresh_analyses()
                    except Exception:
                        logger.exception("[BLADE analyzer] refresh errored")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[BLADE analyzer] monitor crashed")

    # ------------------------------------------------------------------
    # Paradigm shift
    # ------------------------------------------------------------------

    def _paradigm_anchors(self, n: int) -> list[Program]:
        """Top-*n* cell representatives by score, descending."""
        cells = self.archive.cells()
        return sorted(cells.values(), key=lambda p: -p.score)[: max(0, n)]

    def _pick_paradigm_mode(self) -> str:
        """Decide which paradigm-shift mode to fire.

        - synthesis: low stagnation, many close-in-score anchors.
        - surgical:  mid stagnation, champion still has momentum to polish.
        - shift:     high stagnation, the current paradigm family is
                     exhausted and a fresh paradigm is the most likely
                     way out.

        A8 ablation: when ``cfg.paradigm_force_mode`` is set, that mode
        is returned unconditionally (no stagnation routing).
        """
        cfg = self.config
        if cfg.paradigm_force_mode is not None:
            return cfg.paradigm_force_mode
        s = self.monitor.stagnation_level()
        if s <= cfg.paradigm_synthesis_max_stagnation:
            return "synthesis"
        if s <= cfg.paradigm_surgical_max_stagnation:
            return "surgical"
        return "shift"

    def _build_paradigm_prompt_for_mode(
        self, mode: str
    ) -> tuple[str, list[Program], list[tuple[str, float]]]:
        """Assemble (prompt, anchors_used, inspiration_pairs) for *mode*."""
        cfg = self.config
        all_cells = list(self.archive.cells().values())
        all_cells.sort(key=lambda p: -p.score)

        if mode == "synthesis":
            n_anchors = cfg.paradigm_synthesis_n_anchors
            anchors = all_cells[:n_anchors]
            anchor_triples = [(p.code, p.description, p.score) for p in anchors]
            anchor_ids = {id(p) for p in anchors}
            insps = [
                (p.description, p.score)
                for p in all_cells if id(p) not in anchor_ids
            ][: cfg.paradigm_n_inspirations]
            prompt = build_synthesis_prompt(
                problem_description=cfg.problem_description,
                function_signature=cfg.function_signature,
                n_evaluations=self.monitor.eval_count,
                n_cells=self.archive.num_occupied_cells(),
                anchors=anchor_triples,
                inspirations=insps,
                recent_trials=list(self.recent_trials),
                stagnation=self.monitor.stagnation_level(),
            )
            return prompt, anchors, insps

        if mode == "shift":
            n_anchors = cfg.paradigm_shift_n_anchors
            anchors = all_cells[:n_anchors]
            anchor_triples = [(p.code, p.description, p.score) for p in anchors]
            anchor_ids = {id(p) for p in anchors}
            insps = [
                (p.description, p.score)
                for p in all_cells if id(p) not in anchor_ids
            ][: cfg.paradigm_n_inspirations]
            prompt = build_paradigm_shift_prompt(
                problem_description=cfg.problem_description,
                function_signature=cfg.function_signature,
                n_evaluations=self.monitor.eval_count,
                n_cells=self.archive.num_occupied_cells(),
                anchors=anchor_triples,
                inspirations=insps,
                recent_trials=list(self.recent_trials),
                stagnation=self.monitor.stagnation_level(),
            )
            return prompt, anchors, insps

        # surgical: 1 anchor (champion), top descriptions as inspiration.
        anchors = all_cells[:1]
        anchor_triples = [(p.code, p.description, p.score) for p in anchors]
        anchor_ids = {id(p) for p in anchors}
        insps = [
            (p.description, p.score)
            for p in all_cells if id(p) not in anchor_ids
        ][: cfg.paradigm_surgical_n_inspirations]
        prompt = build_surgical_exploit_prompt(
            problem_description=cfg.problem_description,
            function_signature=cfg.function_signature,
            n_evaluations=self.monitor.eval_count,
            n_cells=self.archive.num_occupied_cells(),
            anchors=anchor_triples,
            inspirations=insps,
            recent_trials=list(self.recent_trials),
            stagnation=self.monitor.stagnation_level(),
        )
        return prompt, anchors, insps

    async def _paradigm_shift(self) -> None:
        cfg = self.config
        if self.archive.num_occupied_cells() < cfg.paradigm_min_archive_size:
            return

        mode = self._pick_paradigm_mode()
        prompt, anchors, _insps = self._build_paradigm_prompt_for_mode(mode)
        for p in anchors:
            self.archive.mark_used(p)

        prev_best = self.monitor.best_score
        logger.info(
            "[BLADE PE] mode=%s anchors=%d stagnation=%.2f best=%.4f",
            mode, len(anchors), self.monitor.stagnation_level(),
            prev_best if prev_best != float("-inf") else float("nan"),
        )

        source_label = f"paradigm_{mode}"

        # ---- frontier paradigm seed ----
        try:
            raw = await self._call(
                self.paradigm_lm, prompt,
                temperature=cfg.paradigm_temperature,
                max_tokens=cfg.paradigm_max_tokens,
            )
        except Exception as e:
            logger.exception("[BLADE PE] frontier call failed; counting as reject")
            self._record_reject(source=source_label, error_msg=f"LLM error: {e}")
            return

        parsed = self.parser.parse(raw)
        if not parsed.has_code:
            trial = ParadigmTrial(
                trial_idx=len(self.paradigm_trials) + 1,
                description=parsed.description or "(no description)",
                accepted=False, score=None, delta_vs_prev_best=None,
            )
            self.paradigm_trials.append(trial)
            self.recent_trials.append(trial.render())
            self._record_reject(source=source_label, error_msg="parse_miss (no code in output)")
            return

        score, _scores, err = await self._evaluate_code(parsed.code)
        description = await self._summarize_if_needed(parsed.code, parsed.description)
        accepted = False
        if err is None:
            # mode=="shift" fires only when stagnation is high; the
            # frontier seed is the search's escape attempt. If it would
            # otherwise be dropped by its cell incumbent, evict the
            # weakest program in the archive and inject the seed anyway —
            # losing one weak slot is cheaper than missing the chance to
            # break out of the stuck paradigm family.
            force_on_drop = (mode == "shift")
            accepted, _reason = await self._admit(
                code=parsed.code, description=description, score=score,
                source="paradigm",
                parent_score=prev_best if prev_best != float("-inf") else None,
                force_on_drop=force_on_drop,
            )
        else:
            self._record_reject(source=source_label, score=score, error_msg=err)

        delta = None
        if accepted and prev_best != float("-inf"):
            delta = score - prev_best
        elif accepted:
            delta = 0.0
        trial = ParadigmTrial(
            trial_idx=len(self.paradigm_trials) + 1,
            description=description or parsed.description,
            accepted=accepted, score=score if err is None else None,
            delta_vs_prev_best=delta,
        )
        self.paradigm_trials.append(trial)
        self.recent_trials.append(trial.render())

        # ---- mutation fanout ----
        if err is not None or not parsed.has_code or cfg.n_paradigm_variants <= 0:
            return
        if self._budget_exhausted():
            return

        base_code = parsed.code
        base_score = score
        n_variants = cfg.n_paradigm_variants
        logger.info(
            "[BLADE PE] fanout: %d variants from paradigm seed (score=%.4f, accepted=%s)",
            n_variants, base_score, accepted,
        )

        # Stagger sampling temperature across the sibling fanout so the set
        # spans exploit → explore in one pass instead of all variants
        # sampling at the same temperature with the same prompt (which was
        # the previous failure mode — every sibling tweaked the same few
        # constants).
        center_temp = cfg.paradigm_variant_temperature
        if n_variants <= 1:
            variant_temps = [center_temp]
        else:
            # Spread evenly across [center - 0.25, center + 0.15], clipped
            # to a safe LLM range. Asymmetric range biases slightly toward
            # exploration.
            lo, hi = max(0.2, center_temp - 0.25), min(1.4, center_temp + 0.15)
            step = (hi - lo) / (n_variants - 1)
            variant_temps = [lo + step * i for i in range(n_variants)]

        async def _one_paradigm_variant(v_idx: int) -> None:
            if self._budget_exhausted():
                return
            v_temp = variant_temps[v_idx]
            v_prompt = build_paradigm_variant_prompt(
                problem_description=cfg.problem_description,
                function_signature=cfg.function_signature,
                base_code=base_code, base_score=base_score,
            )
            try:
                raw_v = await self._call(self.mutation_lm, v_prompt, temperature=v_temp)
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
                self._record_reject(source="paradigm_variant", score=v_score, error_msg=v_err)
                return
            v_description = await self._summarize_if_needed(parsed_v.code, parsed_v.description)
            await self._admit(
                code=parsed_v.code, description=v_description, score=v_score,
                source="paradigm_variant", parent_score=base_score,
            )

        await asyncio.gather(*(_one_paradigm_variant(i) for i in range(n_variants)))

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def _bootstrap_population(self) -> None:
        cfg = self.config
        diverse_seeds: list[tuple[str, float, str]] = []

        if cfg.seed_program:
            score, _scores, err = await self._evaluate_code(cfg.seed_program)
            if err is None:
                description = await self._summarize_if_needed(cfg.seed_program, "")
                await self._admit(
                    code=cfg.seed_program, description=description, score=score,
                    source="init", parent_score=None,
                )
                diverse_seeds.append((cfg.seed_program, score, description))
                logger.info("[BLADE init] seed_program admitted (score=%.4f)", score)
            else:
                self._record_reject(source="init", score=score, error_msg=err)

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
                        self.paradigm_lm, prompt,
                        temperature=cfg.init_diversity_temperature,
                        max_tokens=cfg.paradigm_max_tokens,
                    )
                except Exception as e:
                    logger.exception("[BLADE init] %s frontier call failed", tag)
                    self._record_reject(source="init", error_msg=f"LLM error: {e}")
                    continue
                parsed = self.parser.parse(raw)
                if not parsed.has_code:
                    self._record_reject(source="init", error_msg="parse_miss (no code in output)")
                    continue
                score, _scores, err = await self._evaluate_code(parsed.code)
                if err is not None:
                    self._record_reject(source="init", score=score, error_msg=err)
                    continue
                description = await self._summarize_if_needed(parsed.code, parsed.description)
                await self._admit(
                    code=parsed.code, description=description, score=score,
                    source="init", parent_score=None,
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

        if cfg.n_variants_per_seed <= 0:
            return

        n_variants = cfg.n_variants_per_seed * len(diverse_seeds)
        logger.info(
            "[BLADE init] phase 2 generating %d variants (%d × %d) in parallel",
            n_variants, cfg.n_variants_per_seed, len(diverse_seeds),
        )
        n_inspirations = min(2, len(diverse_seeds))
        prompts: list[str] = []
        for _seed_code, _seed_score, _ in diverse_seeds:
            for _ in range(cfg.n_variants_per_seed):
                insps = self._rng.sample(diverse_seeds, n_inspirations)
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
                raw = await self._call(self.mutation_lm, prompt, temperature=cfg.init_variant_temperature)
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
                self._record_reject(source="init", score=score, error_msg=err)
                return
            description = await self._summarize_if_needed(parsed.code, parsed.description)
            await self._admit(
                code=parsed.code, description=description, score=score,
                source="init", parent_score=None,
            )

        await asyncio.gather(*(_one_variant(p) for p in prompts))
        logger.info("[BLADE init] phase 2 done — archive size now %d", len(self.archive))

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self) -> BladeResult:
        self.start_time = time.time()
        self._eval_processes = ResilientProcessPool(max_workers=self.config.n_eval_processes)
        cfg = self.config
        logger.info(
            "[BLADE] starting — budget: $%s evals=%s seconds=%s target=%s",
            cfg.budget_dollars, cfg.budget_evals, cfg.budget_seconds, cfg.target_score,
        )
        logger.info(
            "[BLADE] models: mutation=%s | paradigm=%s | embedding=%s",
            cfg.mutation_model, cfg.paradigm_model, cfg.embedding_model,
        )
        logger.info(
            "[BLADE] archive: n_cells=%d use_ast=%s use_embedding=%s adaptive_recluster=%s",
            cfg.archive_config.n_cells,
            cfg.archive_config.use_ast,
            cfg.archive_config.use_embedding,
            cfg.archive_config.adaptive_recluster,
        )
        logger.info(
            "[BLADE] knobs: workers=%d pe_interval=%d n_diverse_seeds=%d n_variants/seed=%d n_paradigm_var=%d",
            cfg.n_workers, cfg.pe_cron_interval, cfg.n_diverse_seeds,
            cfg.n_variants_per_seed, cfg.n_paradigm_variants,
        )
        status_task = asyncio.create_task(self._status_monitor())
        try:
            logger.info("[BLADE init] phase 1 starting — %d diverse seeds (sequential, frontier model)",
                        cfg.n_diverse_seeds)
            await self._bootstrap_population()
            logger.info("[BLADE] bootstrap complete — archive=%d best=%.6f cost=$%.3f evals=%d",
                        len(self.archive),
                        self.monitor.best_score if self.monitor.best_score != float("-inf") else float("nan"),
                        self.cost.cost, self.monitor.eval_count)
            # BLADE PE only counts eval-cadence from the end of the init
            # phase. Init evals (phase 1 frontier seeds + phase 2 mutation
            # variants) must not push PE toward its first trigger.
            self.last_pe_eval_count = self.monitor.eval_count
            pe_monitor_task = asyncio.create_task(self._pe_monitor())
            advisor_task = asyncio.create_task(self._meta_advice_monitor())
            analyzer_task = asyncio.create_task(self._analyzer_monitor())
            try:
                logger.info("[BLADE] entering evolutionary main loop (n_workers=%d)", cfg.n_workers)
                await self._main_loop()
            finally:
                self.stop_event.set()
                await self._cancel_with_grace(
                    [pe_monitor_task, advisor_task, analyzer_task],
                    label="pe/advisor/analyzer",
                )
        finally:
            self.stop_event.set()
            await self._cancel_with_grace([status_task], label="status")
            try:
                self._eval_processes.shutdown()
            except Exception:  # pragma: no cover — defensive
                logger.exception("[BLADE] failed to shut down process pool cleanly")

        elapsed = time.time() - self.start_time
        best = self.archive.best()
        result = BladeResult(
            best_program=best.code if best else (self.config.seed_program or ""),
            best_score=best.score if best else float("-inf"),
            total_evaluations=self.monitor.eval_count,
            total_cost=self.cost.cost,
            archive_size=len(self.archive),
            runtime_seconds=elapsed,
            output_dir=str(self.output_dir),
            paradigm_trials=list(self.paradigm_trials),
        )
        self._save_snapshot(result)
        return result

    async def _cancel_with_grace(self, tasks: list[asyncio.Task], *, label: str) -> None:
        live = [t for t in tasks if t is not None and not t.done()]
        for t in live:
            t.cancel()
        if not live:
            return
        grace = self.config.shutdown_grace_seconds
        try:
            await asyncio.wait_for(asyncio.gather(*live, return_exceptions=True), timeout=grace)
        except (TimeoutError, asyncio.TimeoutError):
            stuck = [t for t in live if not t.done()]
            logger.warning(
                "[BLADE] %s task(s) did not honor cancel within %.1fs (%d stuck); abandoning",
                label, grace, len(stuck),
            )

    async def _main_loop(self) -> None:
        cfg = self.config
        in_flight: set[asyncio.Task] = set()
        while not self.stop_event.is_set() and not self._budget_exhausted():
            while len(in_flight) < cfg.n_workers and not self._budget_exhausted():
                in_flight.add(asyncio.create_task(self._generate_one()))
            if not in_flight:
                await asyncio.sleep(0.05)
                continue
            done, _pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED, timeout=2.0)
            for t in done:
                exc = t.exception()
                if exc is not None:
                    logger.error("[BLADE] background task error: %s", exc)
            in_flight = {t for t in in_flight if not t.done()}

        leftovers: list[asyncio.Task] = list(in_flight)
        for t in leftovers:
            t.cancel()
        if leftovers:
            grace = self.config.shutdown_grace_seconds
            try:
                await asyncio.wait_for(asyncio.gather(*leftovers, return_exceptions=True), timeout=grace)
            except (TimeoutError, asyncio.TimeoutError):
                stuck = [t for t in leftovers if not t.done()]
                logger.warning(
                    "[BLADE] %d worker task(s) did not honor cancel within %.1fs; abandoning",
                    len(stuck), grace,
                )

    # ------------------------------------------------------------------
    # Status + cron monitors
    # ------------------------------------------------------------------

    async def _status_monitor(self) -> None:
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
                    "Eval in-flight: %d | Archive: %d (cells %d) | Best: %s | Elapsed: %.0fs",
                    self.cost.cost, self.monitor.eval_count,
                    self._client_in_flight, self._eval_in_flight,
                    len(self.archive), self.archive.num_occupied_cells(),
                    best_str, elapsed,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[Status] monitor crashed")

    async def _pe_monitor(self) -> None:
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
                        best = self.monitor.best_score
                        best_str = f"{best:.6f}" if best != float("-inf") else "n/a"
                        logger.info(
                            "[BLADE PE] trigger #%d at eval=%d | stagnation=%.2f "
                            "(global=%.2f local=%.2f) | best=%s | cells=%d",
                            self.pe_trigger_count, ec,
                            self.monitor.stagnation_level(),
                            self.monitor.global_stagnation(),
                            self.monitor.local_stagnation(),
                            best_str, self.archive.num_occupied_cells(),
                        )
                        try:
                            await self._paradigm_shift()
                        except Exception:
                            logger.exception("[BLADE PE] paradigm shift errored")
                        self.last_pe_eval_count = max(self.last_pe_eval_count, self.monitor.eval_count)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[BLADE PE] monitor crashed")

    # ------------------------------------------------------------------
    # Meta-advisor
    # ------------------------------------------------------------------

    def _top_descriptions_for_advisor(self, k: int = 3) -> list[tuple[str, float]]:
        """Top-K archived programs by score — descriptions + scores only."""
        programs = list(self.archive.programs())
        programs.sort(key=lambda p: p.score, reverse=True)
        out: list[tuple[str, float]] = []
        for p in programs[:k]:
            desc = (p.description or "").strip()
            if desc:
                out.append((desc, float(p.score)))
        return out

    def _recent_admits_for_advisor(self, n: int = 8) -> list[tuple[str, float, float | None]]:
        """Last ``n`` admits as ``(source, score, delta_vs_parent or None)``."""
        items = list(self._advisor_admits)[-n:]
        rows: list[tuple[str, float, float | None]] = []
        for source, score, parent_score in items:
            if parent_score is None or not math.isfinite(parent_score):
                delta: float | None = None
            else:
                delta = float(score - parent_score)
            rows.append((source, float(score), delta))
        return rows

    def _errors_by_source_for_advisor(self, n: int = 20) -> list[tuple[str, str]]:
        """Last ``n`` ``(source, error_message)`` pairs since the previous
        advisor trigger, used to build the typed error taxonomy."""
        return list(self._advisor_errors)[-n:]

    async def _generate_meta_advice(self) -> None:
        cfg = self.config
        # If the advisor mode is "errors_only" we deliberately drop the
        # success-side signal (top descriptions + recent admits). This
        # reproduces the pre-improvement behavior and is what the
        # ``--meta-advice-mode errors_only`` ablation flips on.
        if getattr(cfg, "meta_advice_mode", "rich") == "errors_only":
            top_desc: list[tuple[str, float]] = []
            admits: list[tuple[str, float, float | None]] = []
        else:
            top_desc = self._top_descriptions_for_advisor(k=3)
            admits = self._recent_admits_for_advisor(n=8)
        errors = self._errors_by_source_for_advisor(n=20)
        prompt = build_meta_advice_prompt(
            problem_description=cfg.problem_description,
            function_signature=cfg.function_signature,
            best_score=self.monitor.best_score,
            n_evaluations=self.monitor.eval_count,
            accept_rate=self.monitor.acceptance_rate(),
            stagnation_level=self.monitor.stagnation_level(),
            top_descriptions=top_desc,
            recent_admits=admits,
            errors_by_source=errors,
            previous_advice=self.current_meta_advice,
        )
        try:
            raw = await self._call(
                self.mutation_lm, prompt,
                temperature=cfg.meta_advice_temperature,
                max_tokens=cfg.meta_advice_max_tokens,
            )
        except Exception:
            logger.exception("[BLADE advisor] LLM call failed; keeping previous advice")
            return
        text = (raw or "").strip()
        if not text:
            logger.info("[BLADE advisor] empty response; keeping previous advice")
            return
        if len(text) > 1200:
            text = text[:1200].rstrip() + "…"
        self.current_meta_advice = text
        # Clear the per-window error queue so the next advisor cycle sees
        # a fresh taxonomy. Admits intentionally keep their rolling window
        # so the advisor can spot operators that are *consistently*
        # productive across cycles.
        self._advisor_errors.clear()
        logger.info("[BLADE advisor] new advice (%d chars) at eval=%d", len(text), self.monitor.eval_count)

    async def _meta_advice_monitor(self) -> None:
        cfg = self.config
        if not cfg.enable_meta_advice or cfg.meta_advice_interval <= 0:
            logger.info(
                "[BLADE advisor] disabled (enable_meta_advice=%s, interval=%d)",
                cfg.enable_meta_advice, cfg.meta_advice_interval,
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
                            self.meta_advice_trigger_count, ec,
                        )
                        await self._generate_meta_advice()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[BLADE advisor] monitor crashed")

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _save_snapshot(self, result: BladeResult) -> None:
        cells_dump = []
        for cell_id, prog in self.archive.cells().items():
            cells_dump.append({
                "cell_id": cell_id,
                "score": prog.score,
                "source": prog.source,
                "created_at_eval": prog.created_at_eval,
                "uses_count": prog.uses_count,
                "description": prog.description,
                "content": prog.code,
            })
        cells_dump.sort(key=lambda d: -d["score"])

        snap = {
            "method": "blade-lite",
            "best_score": result.best_score,
            "total_evaluations": result.total_evaluations,
            "total_cost": result.total_cost,
            "archive_size": result.archive_size,
            "num_occupied_cells": self.archive.num_occupied_cells(),
            "runtime_seconds": result.runtime_seconds,
            "monitor": self.monitor.snapshot(),
            "meta_advice": {
                "enabled": self.config.enable_meta_advice,
                "interval": self.config.meta_advice_interval,
                "mode": self.config.meta_advice_mode,
                "inject_p": self.config.meta_advice_inject_p,
                "trigger_count": self.meta_advice_trigger_count,
                "current": self.current_meta_advice,
            },
            "analyzer": {
                "enabled": self.config.enable_targeted_mutate,
                "interval": self.config.analyzer_interval,
                "top_k": self.config.analyzer_top_k,
                "trigger_count": self.analyzer_trigger_count,
                "targeted_mutate_count": self.targeted_mutate_count,
                "p_targeted_mutate": self.config.p_targeted_mutate,
                "cache_size": len(self._analysis_cache),
            },
            "paradigm_modes": {
                "synthesis_max_stagnation": self.config.paradigm_synthesis_max_stagnation,
                "surgical_max_stagnation": self.config.paradigm_surgical_max_stagnation,
                "synthesis_n_anchors": self.config.paradigm_synthesis_n_anchors,
                "shift_n_anchors": self.config.paradigm_shift_n_anchors,
                "surgical_n_inspirations": self.config.paradigm_surgical_n_inspirations,
                "force_mode": self.config.paradigm_force_mode,
            },
            "paradigm_trials": [
                {
                    "idx": t.trial_idx,
                    "accepted": t.accepted,
                    "score": t.score,
                    "delta": t.delta_vs_prev_best,
                    "description": t.description,
                }
                for t in result.paradigm_trials
            ],
            "cells": cells_dump,
            "ablation": {
                "use_ast": self.config.archive_config.use_ast,
                "use_embedding": self.config.archive_config.use_embedding,
                "adaptive_recluster": self.config.archive_config.adaptive_recluster,
                "n_cells": self.config.archive_config.n_cells,
                "embedding_dim": self.config.archive_config.embedding_dim,
                "recluster_every": self.config.archive_config.recluster_every,
                "single_prompt_operators": self.config.single_prompt_operators,
                "paradigm_force_mode": self.config.paradigm_force_mode,
            },
        }
        (self.output_dir / "snapshot.json").write_text(json.dumps(snap, indent=2))
        if result.best_program:
            (self.output_dir / "best.py").write_text(result.best_program)
