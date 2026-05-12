"""
FOREController — discovery controller for FORE.

Subclasses ``DiscoveryController`` so the heavy lifting (LLM call, eval,
diff parsing, retry, parallel orchestration, checkpoints, monitor callback)
is inherited. FORE only customises three small hooks:

1. Before each iteration, if the database reports stagnation and the cooldown
   has elapsed, ask the ``ReflectiveReviewer`` for a new ``FertilityReview``
   and store it on the database.
2. Make the active review available to the context builder via
   ``self._prompt_context``.
3. After the LLM responds, parse the ``<fore_meta>`` block from the raw
   response and attach it to the child program's metadata BEFORE the base
   class hands the result to ``database.add()`` (so the database can use the
   strategy description for clustering, verdict computation, and POV stats).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Optional

from skydiscover.context_builder.fore.builder import FOREContextBuilder
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.fore.descriptions import StrategyDescription, parse_strategy_block
from skydiscover.search.fore.review import ReflectiveReviewer
from skydiscover.search.utils.discovery_utils import SerializableResult, load_evaluator_code
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)


class FOREController(DiscoveryController):
    """FORE-aware discovery controller."""

    # Fraction of total iterations used to auto-scale review_cooldown and
    # review_window when the config leaves them at 0 (the "auto" sentinel).
    # Mirrors evox's DEFAULT_SWITCH_RATIO so stagnation handling tracks the
    # length of the run instead of a magic constant.
    DEFAULT_REVIEW_RATIO = 0.10

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        # Override the context builder set up by the base class.
        self.context_builder = FOREContextBuilder(self.config)

        # The Reflective Reviewer borrows the guide LLM pool (lighter / cheaper
        # than the main mutation pool by convention) if configured, otherwise
        # falls back to the main pool.
        try:
            reviewer_pool = self.guide_llms if self.guide_llms.models else self.llms
        except Exception:
            reviewer_pool = self.llms

        self.reviewer = ReflectiveReviewer(
            llm_pool=reviewer_pool,
            system_message=self.config.context_builder.system_message or "",
            evaluator_code=load_evaluator_code(self.evaluation_file),
        )

        # Optional JSONL log of FORE-specific signals.
        self._fore_log_path: Optional[str] = None
        self._setup_fore_logging()

        logger.info(
            "FOREController initialized (review_cooldown=%d, review_window=%d, k_remaining=%d — may be auto-scaled at run_discovery)",
            getattr(self.database, "review_cooldown", 0),
            getattr(self.database, "review_window", 0),
            getattr(self.database, "k_remaining_init", 0),
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_fore_logging(self) -> None:
        output_dir = self.output_dir or getattr(self.config.search.database, "db_path", None) or "."
        try:
            os.makedirs(output_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._fore_log_path = os.path.join(output_dir, f"fore_stats_{stamp}.jsonl")
            logger.info("FORE stats log: %s", self._fore_log_path)
        except Exception as e:  # noqa: BLE001
            logger.debug("FORE: could not init stats log: %s", e)
            self._fore_log_path = None

    def _log_event(self, event: dict) -> None:
        if not self._fore_log_path:
            return
        try:
            event.setdefault("timestamp", datetime.now().isoformat())
            with open(self._fore_log_path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.debug("FORE: failed to write log event: %s", e)

    # ------------------------------------------------------------------
    # Run hook — wire actual max_iterations into review/POV horizons
    # ------------------------------------------------------------------

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
    ):
        """Resolve review/POV horizons from the actual run length, then run.

        ``max_iterations`` may differ from ``self.config.max_iterations``
        (the runner / CLI can override it). We propagate any per-database
        ratio overrides too, falling back to ``DEFAULT_REVIEW_RATIO``.
        """
        try:
            db = self.database
            # Allow controller-level default to seed both ratios when the
            # config dataclass leaves them unset (e.g. older configs).
            if getattr(db, "review_cooldown_ratio", 0.0) <= 0.0:
                db.review_cooldown_ratio = self.DEFAULT_REVIEW_RATIO
            if getattr(db, "review_window_ratio", 0.0) <= 0.0:
                db.review_window_ratio = self.DEFAULT_REVIEW_RATIO
            if hasattr(db, "configure_for_max_iterations"):
                db.configure_for_max_iterations(max_iterations)
        except Exception as e:  # noqa: BLE001
            logger.warning("FORE: could not auto-scale review horizons: %s", e)

        return await super().run_discovery(
            start_iteration=start_iteration,
            max_iterations=max_iterations,
            checkpoint_callback=checkpoint_callback,
            post_process_result=post_process_result,
            retry_times=retry_times,
        )

    # ------------------------------------------------------------------
    # Per-iteration hook
    # ------------------------------------------------------------------

    async def _maybe_run_review(self, iteration: int) -> None:
        db = self.database
        if not hasattr(db, "detect_stagnation") or not hasattr(db, "can_run_review"):
            return
        if not db.can_run_review(iteration):
            return
        should, reason = db.detect_stagnation()
        if not should:
            return

        best = db.get_best_program() if hasattr(db, "get_best_program") else None
        best_score = get_score(best.metrics) if best and best.metrics else None

        review = await self.reviewer.generate(
            fertility_summary=db.get_fertility_summary(),
            recent_attempts=db.get_recent_attempts(10),
            global_best_score=best_score,
            trigger_reason=reason,
            iteration=iteration,
        )
        if review is None:
            self._log_event({"event": "review_failed", "iteration": iteration, "reason": reason})
            return

        db.set_active_review(review)
        self._log_event(
            {
                "event": "review_generated",
                "iteration": iteration,
                "reason": reason,
                "next_steps": review.next_steps,
                "effective": [x.get("label") for x in review.effective_lineages],
                "exhausted": [x.get("label") for x in review.exhausted_lineages],
                "embryonic": [x.get("label") for x in review.embryonic_lineages],
            }
        )

    async def _run_iteration(self, iteration: int, retry_times: int = 1) -> SerializableResult:
        # 1) Maybe trigger a reflective review BEFORE generating this iter.
        try:
            await self._maybe_run_review(iteration)
        except Exception as e:  # noqa: BLE001
            logger.warning("FORE: review attempt failed (continuing): %s", e)

        # 2) Refresh the prompt context shared with the context builder. The
        #    builder reads ``self._prompt_context`` (already supported by the
        #    base controller for arbitrary extras) to inject the active review.
        review = None
        if hasattr(self.database, "consume_review_for_prompt"):
            review = self.database.consume_review_for_prompt()
        if review is not None:
            self._prompt_context["fore_review"] = review.render_markdown()
        else:
            self._prompt_context.pop("fore_review", None)

        if hasattr(self.database, "get_pov_diagnostics"):
            try:
                self._prompt_context["fore_diagnostics"] = self.database.get_pov_diagnostics(top_k=3)
            except Exception:
                self._prompt_context.pop("fore_diagnostics", None)

        # 3) Run the underlying iteration via the base implementation.
        result = await super()._run_iteration(iteration, retry_times=retry_times)

        # 4) Parse <fore_meta> from the LLM response and attach to the child.
        if result and not result.error and result.child_program_dict:
            sd = parse_strategy_block(result.llm_response or "")
            metadata = result.child_program_dict.setdefault("metadata", {})
            # Don't overwrite if an earlier hook already populated FORE meta.
            metadata.setdefault("fore", sd.to_dict())

            self._log_event(
                {
                    "event": "child_evaluated",
                    "iteration": iteration,
                    "child_id": result.child_program_dict.get("id"),
                    "parent_id": result.parent_id,
                    "fitness": get_score(result.child_program_dict.get("metrics") or {}),
                    "strategy_label": sd.strategy_label,
                    "has_fore_meta": bool(sd.description or sd.hypothesis),
                }
            )

        return result
