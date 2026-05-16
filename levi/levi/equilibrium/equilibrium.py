"""
Punctuated Equilibrium: Periodic paradigm shifts in evolutionary search.

Inspired by the biological theory that evolution proceeds in bursts of rapid
change separated by long periods of stasis. This module implements periodic
"paradigm shift" events that inject fundamentally new solutions into the archive.
"""

import asyncio
import logging
import random

import numpy as np
from sklearn.cluster import KMeans

from ..artifacts import ArtifactAdapter
from ..clients.base import client_name
from ..config import LeviConfig
from ..core import EvaluationResult
from ..pipeline.state import (
    BudgetLimitReached,
    PipelineState,
    StrategicBlueprint,
    coerce_finite_float,
)
from ..pool import CVTMAPElitesPool
from ..pool.cvt_map_elites import Elite
from ..prompts import PromptBundle
from ..selection import ComponentSelector, make_component_selector
from ..utils import ResilientProcessPool, coerce_score
from .prompts import parse_blueprint

logger = logging.getLogger(__name__)


class PunctuatedEquilibrium:
    """
    Implements punctuated equilibrium for CVT-MAP-Elites.

    Periodically:
    1. Clusters occupied centroids into behavioral regions
    2. Selects best elite from each cluster as representative
    3. Generates paradigm shift solution using heavy model
    4. Generates variants using lighter models
    5. Inserts solutions into the archive
    """

    def __init__(
        self,
        config: LeviConfig,
        pool: CVTMAPElitesPool,
        executor: ResilientProcessPool,
        artifact_adapter: ArtifactAdapter,
        archive_lock: asyncio.Lock,
        state: PipelineState,
        main_component_selector: ComponentSelector | None = None,
    ):
        self.config = config
        self.pool = pool
        self.executor = executor
        self.artifact_adapter = artifact_adapter
        self.archive_lock = archive_lock
        self.state = state
        self.pe_config = config.punctuated_equilibrium
        self.main_component_selector = main_component_selector

        self.pe_component_selector: ComponentSelector | None = None
        self._is_bundle = getattr(artifact_adapter, "is_bundle_artifact", False)
        if self._is_bundle:
            self.pe_component_selector = make_component_selector(self.pe_config.component_selector)

    def _cluster_occupied_centroids(self, n_clusters_override: int | None = None) -> dict[int, list[int]]:
        """
        Cluster occupied centroids and return mapping of cluster_id -> cell_indices.

        Returns empty dict if not enough occupied cells for clustering.

        When `n_clusters_override` is provided (SAL Cơ chế E — Hard-PE) we use
        that value instead of `self.pe_config.n_clusters`, allowing a heavier
        partition of the archive on the rare "we're really stuck" trigger.
        """
        elites = self.pool.get_elites()
        target_n = n_clusters_override if n_clusters_override is not None else self.pe_config.n_clusters
        if len(elites) < target_n:
            logger.info(f"[PE] Not enough elites ({len(elites)}) for {target_n} clusters")
            return {}

        cell_indices = list(elites.keys())
        centroids = self.pool._centroids

        if centroids is None:
            logger.warning("[PE] Centroids not initialized")
            return {}

        # Get centroid vectors for occupied cells only
        occupied_centroids = centroids[cell_indices]

        n_clusters = min(target_n, len(cell_indices))
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=3,
            random_state=None,  # Different clustering each time
        )
        labels = kmeans.fit_predict(occupied_centroids)

        # Build cluster -> cell_indices mapping
        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            label_int = int(label)
            if label_int not in clusters:
                clusters[label_int] = []
            clusters[label_int].append(cell_indices[idx])

        return clusters

    def _select_cluster_representatives(
        self,
        clusters: dict[int, list[int]],
        *,
        farthest_first: bool = False,
    ) -> list[tuple[int, Elite]]:
        """
        Select a representative elite from each cluster.

        Default: per-cluster max-score (the legacy behaviour).

        SAL Cơ chế E `farthest_first=True`: from each cluster pick the elite
        whose behaviour vector is farthest from the centroid of all elites
        already picked. That diversifies the prompt context — the heavy
        model sees structurally distinct anchors, not score-similar ones.
        """
        representatives: list[tuple[int, Elite]] = []
        elites = self.pool.get_elites()

        if not farthest_first:
            for cluster_id, cell_indices in clusters.items():
                cluster_elites = [(idx, elites[idx]) for idx in cell_indices]
                _best_idx, best_elite = max(
                    cluster_elites, key=lambda x: x[1].result.primary_score
                )
                representatives.append((cluster_id, best_elite))
            return representatives

        # Farthest-first per-cluster selection (SAL Hard-PE).
        picked_vecs: list[np.ndarray] = []
        feature_names = self.pool._feature_names

        def _vec(elite: Elite) -> np.ndarray:
            try:
                return np.array([elite.behavior[f] for f in feature_names], dtype=float)
            except Exception:
                return np.zeros(len(feature_names), dtype=float)

        for cluster_id, cell_indices in clusters.items():
            cluster_elites = [(idx, elites[idx]) for idx in cell_indices]
            if not picked_vecs:
                # First cluster — fall back to max-score so we keep a strong
                # anchor in the prompt.
                _, best_elite = max(cluster_elites, key=lambda x: x[1].result.primary_score)
                representatives.append((cluster_id, best_elite))
                picked_vecs.append(_vec(best_elite))
                continue

            anchor_centroid = np.mean(np.stack(picked_vecs, axis=0), axis=0)
            best_dist = -1.0
            best_elite_here = cluster_elites[0][1]
            for _, elite in cluster_elites:
                v = _vec(elite)
                d = float(np.linalg.norm(v - anchor_centroid))
                if d > best_dist:
                    best_dist = d
                    best_elite_here = elite
            representatives.append((cluster_id, best_elite_here))
            picked_vecs.append(_vec(best_elite_here))

        return representatives

    def _should_fire_hard_pe(self) -> bool:
        """SAL Cơ chế E gate — fire a heavier PE when *really* stuck.

        Conditions (all must hold):
          - SAL enabled, mechanism E enabled.
          - We haven't burned the per-run hard-PE budget yet.
          - The last 2 PE triggers in a row produced no NEW BEST.
          - Stagnation depth s(t) ≥ hard_pe_threshold.
        """
        sal = self.config.sal
        if not (sal.enabled and sal.enable_e_hard_pe):
            return False
        if self.state.hard_pe_count >= sal.hard_pe_max_per_run:
            return False
        if self.state.consecutive_pe_no_best < 2:
            return False
        s = self.state.stagnation_depth(sal.tau)
        return s >= sal.hard_pe_threshold

    def _build_paradigm_shift_prompt(
        self,
        representatives: list[tuple[int, Elite]],
        n_evaluations: int,
        budget_progress: float = 0.0,
        target: str | None = None,
    ) -> str:
        if target is not None and self._is_bundle:
            return self.artifact_adapter.build_component_paradigm_shift_prompt(
                target,
                representatives,
                n_evaluations=n_evaluations,
                budget_progress=budget_progress,
            )

        # SAL Cơ chế A: pass stagnation + trajectory context to the code adapter.
        sal = self.config.sal
        stagnation: float | None = None
        best_score: float | None = None
        evals_since_best: int | None = None
        sal_thresholds: tuple[float, float] | None = None
        if sal.enabled and sal.enable_a_pe_staging:
            stagnation = self.state.stagnation_depth(sal.tau)
            evals_since_best = self.state.evals_since_best()
            best_score = (
                self.state.best_score_so_far
                if self.state.best_score_so_far != float("-inf")
                else None
            )
            sal_thresholds = (sal.pe_staging_mid_threshold, sal.pe_staging_late_threshold)

        return self.artifact_adapter.build_paradigm_shift_prompt(
            representatives,
            n_evaluations=n_evaluations,
            budget_progress=budget_progress,
            stagnation=stagnation,
            best_score=best_score,
            evals_since_best=evals_since_best,
            sal_thresholds=sal_thresholds,
        )

    def _build_variant_prompt(
        self,
        base_code: str,
        base_score: float,
        target: str | None = None,
        blueprint_text: str | None = None,
    ) -> str:
        if target is not None and self._is_bundle:
            base_bundle = PromptBundle.deserialize_loose(base_code)
            return self.artifact_adapter.build_component_variant_prompt(target, base_bundle, base_score)
        # HLS — when a blueprint is in scope, condition the variant on it
        # so light models keep producing the same algorithmic family
        # instead of regressing to the parent's prior strategy.
        if blueprint_text and hasattr(self.artifact_adapter, "build_blueprint_variant_prompt"):
            return self.artifact_adapter.build_blueprint_variant_prompt(blueprint_text, base_code, base_score)
        return self.artifact_adapter.build_variant_prompt(base_code, base_score)

    # ------------------------------------------------------------------
    # HLS — Strategic Blueprint generation
    # ------------------------------------------------------------------

    def _hls_enabled(self) -> bool:
        if self._is_bundle:
            return False
        if not self.pe_config.use_blueprint:
            return False
        return hasattr(self.artifact_adapter, "build_blueprint_prompt")

    async def _generate_blueprint(
        self,
        heavy_model,
        representatives: list[tuple[int, Elite]],
        n_evaluations: int,
        *,
        stagnation: float | None,
        best_score: float | None,
        evals_since_best: int | None,
        max_tokens: int,
        reasoning_effort: str | None,
    ) -> tuple[StrategicBlueprint | None, float]:
        """Ask the heavy model for a short, structured strategic blueprint.

        Returns ``(blueprint, cost)``. ``blueprint`` is None when generation
        failed or the response could not be parsed; the caller should then
        fall back to the legacy full-code paradigm-shift path so the PE
        event is not wasted.
        """
        top_failures = [
            err for err, _ in sorted(
                self.state.all_error_counts.items(), key=lambda x: -x[1]
            )[:3]
        ] or None

        prompt = self.artifact_adapter.build_blueprint_prompt(
            representatives,
            n_evaluations=n_evaluations,
            stagnation=stagnation,
            best_score=best_score,
            evals_since_best=evals_since_best,
            top_failures=top_failures,
            max_words=350,
        )

        extras: dict = {}
        if reasoning_effort:
            if reasoning_effort == "disabled":
                extras["extra_body"] = {"reasoning": {"enabled": False}}
            else:
                extras["reasoning_effort"] = reasoning_effort

        try:
            response = await self.state.acompletion(
                heavy_model,
                prompt=[{"role": "user", "content": prompt}],
                temperature=self.pe_config.temperature,
                max_tokens=max_tokens,
                timeout=300,
                **extras,
            )
        except BudgetLimitReached:
            raise
        except Exception as e:
            logger.warning(f"[PE] Blueprint generation call failed: {e}")
            return None, 0.0

        raw = response.text or ""
        cost = float(getattr(response, "cost", 0.0) or 0.0)

        parsed = parse_blueprint(raw)
        if not parsed:
            logger.warning("[PE] Blueprint parse failed (no sections matched); falling back to legacy paradigm shift")
            return None, cost

        blueprint = StrategicBlueprint(
            diagnosis=parsed.get("diagnosis", ""),
            approach=parsed.get("approach", ""),
            invariants=parsed.get("invariants", ""),
            pseudocode=parsed.get("pseudocode", ""),
            raw=raw.strip(),
            pe_event_id=self.state.pe_trigger_count,
        )

        # Soft validation: a blueprint without an APPROACH section is
        # useless to implementers. Reject and fall back.
        if not blueprint.approach.strip():
            logger.warning("[PE] Blueprint had no APPROACH section; falling back to legacy paradigm shift")
            return None, cost

        return blueprint, cost

    def _pick_pe_component(self) -> str | None:
        if not self._is_bundle or self.pe_component_selector is None:
            return None
        seed_bundle = getattr(self.artifact_adapter, "seed_bundle", None)
        if seed_bundle is None:
            return None
        context: dict = {}
        if self.pe_config.share_main_selector_stats and self.main_component_selector is not None:
            context["main_stats"] = self.main_component_selector.stats()
        return self.pe_component_selector.select(
            list(seed_bundle.editable_targets), context=context or None
        )

    async def _evaluate(self, code: str) -> dict:
        """Evaluate code using the executor."""
        return await self.artifact_adapter.evaluate(self.executor, code)

    async def trigger(self, n_evaluations: int, budget_progress: float = 0.0) -> dict:
        """
        Trigger a punctuated equilibrium event.

        Args:
            n_evaluations: Current evaluation count (for prompt context)
            budget_progress: Fraction of budget consumed (0-1)

        Returns:
            Dict with statistics about the PE event:
            {
                "triggered": bool,
                "paradigm_generated": bool,
                "paradigm_score": Optional[float],
                "paradigm_accepted": bool,
                "paradigm_cell": Optional[int],
                "variants_generated": int,
                "variants_accepted": int,
                "total_cost": float,
            }
        """
        stats = {
            "triggered": True,
            "paradigm_generated": False,
            "paradigm_score": None,
            "paradigm_accepted": False,
            "paradigm_cell": None,
            "variants_generated": 0,
            "variants_accepted": 0,
            "variant_cells": [],
            "total_cost": 0.0,
            "evaluations": [],
        }
        pe_evals_started = 0

        def can_start_pe_eval() -> bool:
            if self.state.budget_exhausted:
                return False
            if self.state.budget.evaluations is None:
                return True
            eval_limit = int(coerce_finite_float(self.state.budget.evaluations, default=0.0))
            if eval_limit <= 0:
                return False
            return (self.state.eval_count + pe_evals_started) < eval_limit

        # SAL Cơ chế E — check whether this trigger should be a Hard-PE
        # (heavier clustering, farthest-first reps, forced reasoning effort).
        sal = self.config.sal
        is_hard_pe = self._should_fire_hard_pe()
        hard_n_clusters = sal.hard_pe_n_clusters if is_hard_pe else None
        if is_hard_pe:
            self.state.hard_pe_count += 1
            logger.info(
                f"[PE] HARD-PE #{self.state.hard_pe_count}/{sal.hard_pe_max_per_run} "
                f"(s={self.state.stagnation_depth(sal.tau):.2f}, "
                f"consecutive_no_best={self.state.consecutive_pe_no_best})"
            )
            stats["hard_pe"] = True

        # Step 1: Cluster occupied centroids
        async with self.archive_lock:
            clusters = self._cluster_occupied_centroids(n_clusters_override=hard_n_clusters)

        if not clusters:
            stats["triggered"] = False
            return stats

        logger.info(f"[PE] Clustered {len(self.pool.get_elites())} elites into {len(clusters)} clusters")

        # Step 2: Select cluster representatives
        async with self.archive_lock:
            representatives = self._select_cluster_representatives(
                clusters, farthest_first=is_hard_pe
            )

        for cluster_id, elite in representatives:
            logger.info(f"[PE] Cluster {cluster_id} rep: score={elite.result.primary_score:.17g}")

        # Step 3: Generate paradigm shift solution
        heavy_models = self.pe_config.heavy_models
        if not heavy_models:
            heavy_models = [self.config.sampler_model_pairs[0].model]
        heavy_model = random.choice(heavy_models)

        pe_target = self._pick_pe_component()
        if pe_target is not None:
            logger.info(f"[PE] Selected component for paradigm shift: {pe_target}")
            stats["pe_target"] = pe_target

        # HLS — try the Strategic Blueprint path first. On failure we fall
        # through to the legacy "heavy generates full code" pathway so a
        # parser hiccup never wastes the PE trigger.
        active_blueprint: StrategicBlueprint | None = None
        if pe_target is None and self._hls_enabled():
            sal = self.config.sal
            try:
                blueprint, blueprint_cost = await self._generate_blueprint(
                    heavy_model,
                    representatives,
                    n_evaluations,
                    stagnation=(
                        self.state.stagnation_depth(sal.tau)
                        if sal.enabled else None
                    ),
                    best_score=(
                        self.state.best_score_so_far
                        if self.state.best_score_so_far != float("-inf")
                        else None
                    ),
                    evals_since_best=self.state.evals_since_best(),
                    max_tokens=self.config.blueprint.max_tokens,
                    reasoning_effort=(
                        sal.hard_pe_reasoning_effort
                        if is_hard_pe else self.pe_config.reasoning_effort
                    ),
                )
            except BudgetLimitReached:
                logger.info("[PE] Budget exhausted before blueprint generation")
                return stats
            stats["total_cost"] += blueprint_cost
            stats["blueprint_cost"] = blueprint_cost
            if blueprint is not None:
                active_blueprint = blueprint
                stats["blueprint_generated"] = True
                logger.info(
                    f"[PE] Blueprint generated ({len(blueprint.raw)} chars, "
                    f"approach={blueprint.approach[:80]!r}...)"
                )

        # Select which model writes the reference implementation. Under
        # HLS the heavy model already produced the blueprint, so the
        # reference implementation is delegated to a *light* model
        # (cost reduction) when ``light_only_implementations`` is True.
        impl_models: list = []
        if active_blueprint is not None and self.pe_config.light_only_implementations:
            impl_models = list(self.pe_config.variant_models or [])
            if not impl_models:
                impl_models = [p.model for p in self.config.sampler_model_pairs[:3]]
            if not impl_models:
                impl_models = [heavy_model]
            ref_model = random.choice(impl_models)
            prompt = self.artifact_adapter.build_blueprint_implementation_prompt(
                active_blueprint.raw, representatives
            )
            ref_role = "blueprint_impl"
        else:
            ref_model = heavy_model
            prompt = self._build_paradigm_shift_prompt(
                representatives, n_evaluations, budget_progress, target=pe_target
            )
            ref_role = "paradigm_shift"

        try:
            extras = {}
            # SAL Cơ chế E — force higher reasoning effort on Hard-PE.
            effective_reasoning_effort = self.pe_config.reasoning_effort
            if is_hard_pe:
                effective_reasoning_effort = sal.hard_pe_reasoning_effort
            if ref_role == "blueprint_impl":
                # Light implementer doesn't need expensive reasoning — it
                # is just rendering the blueprint as code.
                effective_reasoning_effort = None

            # Add reasoning_effort for DeepSeek models if configured
            if effective_reasoning_effort:
                if effective_reasoning_effort == "disabled":
                    # Disable reasoning entirely (e.g., for GLM models)
                    extras["extra_body"] = {"reasoning": {"enabled": False}}
                    logger.info(f"[PE] Reasoning disabled for {ref_role}")
                else:
                    extras["reasoning_effort"] = effective_reasoning_effort
                    logger.info(f"[PE] Using reasoning_effort={effective_reasoning_effort} for {ref_role}")

            response = await self.state.acompletion(
                ref_model,
                prompt=[{"role": "user", "content": prompt}],
                temperature=self.pe_config.temperature,
                # max_tokens=4096,
                timeout=300,
                **extras,
            )
            content = response.text
            cost = response.cost
            stats["total_cost"] += cost
        except BudgetLimitReached:
            logger.info("[PE] Budget exhausted before paradigm shift generation")
            return stats
        except Exception as e:
            logger.warning(f"[PE] {ref_role} generation failed: {e}")
            return stats

        if pe_target is not None and representatives:
            anchor_elite = max(representatives, key=lambda item: item[1].result.primary_score)[1]
            paradigm_code = self.artifact_adapter.extract_candidate(
                content,
                parent_content=anchor_elite.program.content,
                target=pe_target,
            )
        else:
            paradigm_code = self.artifact_adapter.extract_candidate(content)
        if not paradigm_code:
            logger.warning("[PE] Failed to extract paradigm shift code")
            return stats

        stats["paradigm_generated"] = True
        logger.debug("[PE] Paradigm shift code generated (%d chars)", len(paradigm_code))

        # Step 4: Evaluate paradigm shift solution
        if not can_start_pe_eval():
            logger.info("[PE] Skipping paradigm shift evaluation (budget exhausted)")
            stats["evaluations"].append(
                {
                    "source": "paradigm_shift",
                    "model": client_name(ref_model),
                    "error": "Budget exhausted",
                    "archive_size": self.pool.size(),
                }
            )
            return stats

        try:
            pe_evals_started += 1
            result = await self._evaluate(paradigm_code)
        except Exception as e:
            logger.warning(f"[PE] Paradigm shift evaluation failed: {e}")
            result = {"error": str(e)}

        if "error" not in result:
            score, score_error = coerce_score(result)
            if score_error is not None:
                logger.warning(f"[PE] Paradigm shift invalid score: {score_error}")
                result = {"error": score_error}
            else:
                result = dict(result)
                result["score"] = score

        if "error" not in result:
            stats["paradigm_score"] = score

            program = self.artifact_adapter.make_program(
                paradigm_code,
                metadata={
                    "source": "punctuated_equilibrium",
                    "pe_type": "paradigm_shift",
                },
            )
            eval_result = EvaluationResult(
                scores=result,
                is_valid=True,
            )

            async with self.archive_lock:
                accepted, cell_idx = self.pool.add(program, eval_result)

            stats["paradigm_accepted"] = accepted
            stats["paradigm_cell"] = cell_idx
            if pe_target is not None and self.pe_component_selector is not None:
                self.pe_component_selector.update(pe_target, accepted=accepted)
            stats["evaluations"].append(
                {
                    "source": "paradigm_shift",
                    "model": client_name(ref_model),
                    "score": score,
                    "accepted": accepted,
                    "cell_index": cell_idx,
                    "archive_size": self.pool.size(),
                }
            )

            logger.info(f"[PE] Paradigm shift: score={score:.17g}, accepted={accepted}, cell={cell_idx}")
        else:
            error_message = str(result.get("error", "unknown"))
            logger.info(f"[PE] Paradigm shift eval error: {error_message[:100]}")
            stats["evaluations"].append(
                {
                    "source": "paradigm_shift",
                    "model": client_name(ref_model),
                    "error": error_message,
                    "archive_size": self.pool.size(),
                }
            )
            paradigm_code = None  # Can't generate variants

        # Step 5: Generate variants (only if paradigm was valid)
        if paradigm_code and stats["paradigm_score"] is not None:
            variant_models = self.pe_config.variant_models
            if not variant_models:
                # Use lighter models from config, cycle through sampler models
                variant_models = [p.model for p in self.config.sampler_model_pairs[:3]]
                if not variant_models:
                    variant_models = [heavy_model]

            logger.info(
                f"[PE] Generating {self.pe_config.n_variants} variants using models: "
                f"{[client_name(model) for model in variant_models[:3]]}..."
            )

            blueprint_text_for_variants = (
                active_blueprint.raw if active_blueprint is not None else None
            )
            variant_prompt = self._build_variant_prompt(
                paradigm_code,
                stats["paradigm_score"],
                target=pe_target,
                blueprint_text=blueprint_text_for_variants,
            )

            async def generate_variant(model, idx: int):
                try:
                    response = await self.state.acompletion(
                        model,
                        prompt=[{"role": "user", "content": variant_prompt}],
                        temperature=self.pe_config.temperature,
                        # max_tokens=4096,
                        timeout=300,
                    )
                    return {
                        "idx": idx,
                        "content": response.text,
                        "cost": response.cost,
                        "model": client_name(model),
                    }
                except BudgetLimitReached:
                    return {"idx": idx, "error": "Budget exhausted", "model": client_name(model)}
                except Exception as e:
                    return {"idx": idx, "error": str(e), "model": client_name(model)}

            # Generate variants in parallel
            tasks = [
                generate_variant(variant_models[i % len(variant_models)], i) for i in range(self.pe_config.n_variants)
            ]
            variant_results = await asyncio.gather(*tasks)

            # Evaluate and insert variants
            for vr in variant_results:
                if "error" in vr:
                    logger.warning(
                        f"[PE] Variant {vr['idx']} generation failed ({vr.get('model', '?')}): {vr['error'][:100]}"
                    )
                    continue

                stats["total_cost"] += vr["cost"]

                if pe_target is not None:
                    variant_code = self.artifact_adapter.extract_candidate(
                        vr["content"],
                        parent_content=paradigm_code,
                        target=pe_target,
                    )
                else:
                    variant_code = self.artifact_adapter.extract_candidate(vr["content"])
                if not variant_code:
                    logger.warning(f"[PE] Variant {vr['idx']} code extraction failed ({vr.get('model', '?')})")
                    continue

                stats["variants_generated"] += 1

                if not can_start_pe_eval():
                    logger.info(f"[PE] Skipping variant {vr['idx']} evaluation (budget exhausted)")
                    stats["evaluations"].append(
                        {
                            "source": "variant",
                            "model": vr.get("model", "unknown"),
                            "error": "Budget exhausted",
                            "archive_size": self.pool.size(),
                        }
                    )
                    continue

                try:
                    pe_evals_started += 1
                    result = await self._evaluate(variant_code)
                except Exception as e:
                    logger.warning(f"[PE] Variant {vr['idx']} evaluation exception: {e}")
                    result = {"error": str(e)}

                if "error" not in result:
                    score, score_error = coerce_score(result)
                    if score_error is not None:
                        logger.warning(f"[PE] Variant {vr['idx']} invalid score: {score_error}")
                        result = {"error": score_error}
                    else:
                        result = dict(result)
                        result["score"] = score

                if "error" not in result:
                    program = self.artifact_adapter.make_program(
                        variant_code,
                        metadata={
                            "source": "punctuated_equilibrium",
                            "pe_type": "variant",
                        },
                    )
                    eval_result = EvaluationResult(
                        scores=result,
                        is_valid=True,
                    )

                    async with self.archive_lock:
                        accepted, cell_idx = self.pool.add(program, eval_result)

                    if accepted:
                        stats["variants_accepted"] += 1
                        stats["variant_cells"].append(cell_idx)
                    if pe_target is not None and self.pe_component_selector is not None:
                        self.pe_component_selector.update(pe_target, accepted=accepted)
                    stats["evaluations"].append(
                        {
                            "source": "variant",
                            "model": vr.get("model", "unknown"),
                            "score": score,
                            "accepted": accepted,
                            "cell_index": cell_idx,
                            "archive_size": self.pool.size(),
                        }
                    )

                    logger.info(f"[PE] Variant {vr['idx']}: score={score:.17g}, accepted={accepted}")
                else:
                    error_message = str(result.get("error", "unknown"))
                    logger.warning(f"[PE] Variant {vr['idx']} eval error: {error_message[:100]}")
                    stats["evaluations"].append(
                        {
                            "source": "variant",
                            "model": vr.get("model", "unknown"),
                            "error": error_message,
                            "archive_size": self.pool.size(),
                        }
                    )

        # HLS — install the blueprint into pipeline state with a TTL so
        # main-loop mutations can read its APPROACH section as a strategic
        # directive. We only install when at least one implementation
        # entered the archive (unless the operator disabled that guard);
        # otherwise the blueprint already failed once and shouldn't be
        # spread further.
        if active_blueprint is not None:
            bp_cfg = self.config.blueprint
            any_accepted = bool(stats["paradigm_accepted"]) or stats["variants_accepted"] > 0
            install_ok = any_accepted or not bp_cfg.require_accepted
            if bp_cfg.enabled and install_ok:
                ttl = max(1, int(round(bp_cfg.ttl_multiplier * max(self.pe_config.interval, 1))))
                active_blueprint.accepted = any_accepted
                self.state.install_blueprint(active_blueprint, ttl_evals=ttl)
                stats["blueprint_installed"] = True
                stats["blueprint_ttl"] = ttl
                logger.info(
                    f"[PE] Blueprint installed (TTL={ttl} evals, accepted={any_accepted})"
                )
            else:
                stats["blueprint_installed"] = False

        logger.info(
            f"[PE] Complete: paradigm_accepted={stats['paradigm_accepted']}, "
            f"variants={stats['variants_accepted']}/{stats['variants_generated']}, "
            f"cost=${stats['total_cost']:.3f}"
        )

        return stats
