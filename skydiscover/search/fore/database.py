"""
FOREDatabase — Fertility-Oriented Reflective Evolution population database.

Key responsibilities:
- Store programs with attached FertilityStats and StrategyDescription metadata.
- Cluster programs by Jaccard distance over their strategy description tokens
  (with a code-token backstop when the description is missing).
- Sample parents via Thompson sampling over Posterior Offspring Value (POV).
- Detect stagnation triggers for the Reflective Review.
- Persist all FORE-specific state to disk via the base CheckpointManager.
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.adaevolve.archive.diversity import CodeDiversity
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.fore.descriptions import (
    StrategyDescription,
    compute_verdict,
    jaccard,
    tokenize_strategy,
)
from skydiscover.search.fore.fertility import (
    FertilityStats,
    NIGPrior,
    expected_pov,
    pov_score,
)
from skydiscover.search.fore.review import FertilityReview
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)


_CODE_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]+")


def _code_tokens(solution: str, max_tokens: int = 200) -> set:
    if not solution:
        return set()
    toks = [t.lower() for t in _CODE_TOKEN_RE.findall(solution) if len(t) >= 3]
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
    return set(toks)


class FOREDatabase(ProgramDatabase):
    """Fertility-aware program database with Thompson-sampling parent selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)

        # --- Prior ---
        self.prior = NIGPrior(
            mu_0=getattr(config, "prior_mu_0", 0.0),
            kappa_0=getattr(config, "prior_kappa_0", 2.0),
            alpha_0=getattr(config, "prior_alpha_0", 2.0),
            beta_0=getattr(config, "prior_beta_0", 0.5),
        )

        # --- Sizes ---
        self.population_size = int(getattr(config, "population_size", 80))
        self.cluster_similarity_threshold = float(
            getattr(config, "cluster_similarity_threshold", 0.55)
        )
        self.k_remaining_init = int(getattr(config, "k_remaining", 100))
        self.fertility_alpha = float(getattr(config, "fertility_alpha", 0.7))
        self.fertility_k_max = int(getattr(config, "fertility_k_max", 20))

        # --- POV weights ---
        self.w_novelty = float(getattr(config, "w_novelty", 0.3))
        self.w_rarity = float(getattr(config, "w_rarity", 0.2))
        self.w_age_penalty = float(getattr(config, "w_age_penalty", 0.01))
        self.w_negative_penalty = float(getattr(config, "w_negative_penalty", 0.2))
        self.delta_normalization = float(getattr(config, "delta_normalization", 1.0))

        # --- Review triggers ---
        self.review_rate_threshold = float(getattr(config, "review_rate_threshold", 0.1))
        self.review_window = int(getattr(config, "review_window", 12))
        self.pov_floor = float(getattr(config, "pov_floor", 0.0))
        self.review_cooldown = int(getattr(config, "review_cooldown", 20))
        self.review_uses = int(getattr(config, "review_uses", 3))

        # --- RNG ---
        seed = getattr(config, "random_seed", 42)
        self._rng = random.Random(seed if seed is not None else 42)

        # --- State ---
        # Per-program fertility stats; created at add() time.
        self.fertility: Dict[str, FertilityStats] = {}
        # Per-program strategy description.
        self.strategy: Dict[str, StrategyDescription] = {}
        # Cluster bookkeeping: cluster_id -> [program_ids]; per-program inverse.
        self.clusters: Dict[int, List[str]] = {}
        self.program_cluster: Dict[str, int] = {}
        self._next_cluster_id: int = 0
        # Cached cluster signatures (token sets) for fast Jaccard.
        self._cluster_tokens: Dict[int, set] = {}

        # Code-based novelty (reused from AdaEvolve).
        self.diversity = CodeDiversity()

        # Improvement history for stagnation detection (1.0 if global best
        # advanced this iteration, else 0.0).
        self._recent_improvements: List[float] = []
        self._iteration: int = 0

        # Active review and history.
        self._active_review: Optional[FertilityReview] = None
        self._tried_reviews: List[Dict[str, Any]] = []
        self._last_review_iteration: int = -10**9

        logger.info(
            "FOREDatabase initialized: pop=%d, cluster_th=%.2f, k_remaining=%d",
            self.population_size,
            self.cluster_similarity_threshold,
            self.k_remaining_init,
        )

    # ==================================================================
    # ProgramDatabase interface
    # ==================================================================

    def add(self, program: Program, iteration: Optional[int] = None, **kwargs: Any) -> str:
        if iteration is not None:
            program.iteration_found = iteration
            self.last_iteration = max(self.last_iteration, iteration)
            self._iteration = max(self._iteration, iteration)

        if iteration == 0 or program.iteration_found == 0:
            if self.initial_program_id is None:
                self.initial_program_id = program.id
                self.initial_program_score = get_score(program.metrics or {})

        # --- 1. Pull / parse strategy description ---
        fore_meta = (program.metadata or {}).get("fore") if program.metadata else None
        strategy = StrategyDescription.from_dict(fore_meta) if fore_meta else StrategyDescription()

        # --- 2. Assign cluster ---
        cluster_id = self._assign_cluster(program, strategy)
        strategy.cluster_id = cluster_id

        # --- 3. Initial structural prior inputs for FertilityStats ---
        novelty = self._compute_novelty(program)
        rarity = self._compute_cluster_rarity(cluster_id)
        stats = FertilityStats(
            novelty_score=novelty,
            cluster_rarity=rarity,
            age_at_birth=self._iteration,
        )

        # --- 4. Update parent's FertilityStats with the observed Delta ---
        parent_id = program.parent_id
        parent_fitness: Optional[float] = None
        if parent_id and parent_id in self.programs:
            parent_program = self.programs[parent_id]
            parent_fitness = get_score(parent_program.metrics or {})
            child_fitness = get_score(program.metrics or {})
            delta = (child_fitness - parent_fitness) / max(self.delta_normalization, 1e-9)
            parent_stats = self.fertility.get(parent_id)
            if parent_stats is None:
                # Defensive: parent was added before FORE existed in this DB.
                parent_stats = FertilityStats()
                self.fertility[parent_id] = parent_stats
            parent_stats.update_with_child(delta)

            # --- 5. Verdict on this child ---
            strategy.verdict = compute_verdict(
                parent_fitness=parent_fitness,
                child_fitness=child_fitness,
                parent_mean_delta_plus=parent_stats.mean_delta_plus(),
            )
        else:
            # Seed/initial program: no verdict.
            strategy.verdict = strategy.verdict or "seed"

        # --- 6. Store ---
        self.programs[program.id] = program
        self.fertility[program.id] = stats
        self.strategy[program.id] = strategy
        # Make sure metadata reflects final values (cluster_id, verdict).
        program.metadata = dict(program.metadata or {})
        program.metadata["fore"] = strategy.to_dict()

        # --- 7. Update best + improvement history ---
        prev_best_id = self.best_program_id
        self._update_best_program(program)
        improved = self.best_program_id != prev_best_id
        self._recent_improvements.append(1.0 if improved else 0.0)
        if len(self._recent_improvements) > max(self.review_window * 4, 50):
            self._recent_improvements = self._recent_improvements[-self.review_window * 4 :]

        # --- 8. Enforce population cap ---
        self._enforce_population_limit()

        # --- 9. Save to disk if configured ---
        if self.config.db_path:
            self._save_program(program)

        logger.debug(
            "FORE add: %s cluster=%d novelty=%.3f rarity=%.3f verdict=%s",
            program.id[:8],
            cluster_id,
            novelty,
            rarity,
            strategy.verdict,
        )
        return program.id

    def sample(
        self,
        num_context_programs: Optional[int] = 4,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        if not self.programs:
            raise ValueError("FOREDatabase.sample called on empty database")

        num_context_programs = num_context_programs or 4
        pool = list(self.programs.values())
        k_remaining = max(1, self.k_remaining_init - self._iteration)

        # Thompson-sample one POV per program; pick top.
        scored: List[Tuple[float, Program]] = []
        for p in pool:
            stats = self.fertility.get(p.id) or FertilityStats()
            fit = get_score(p.metrics or {})
            s = pov_score(
                fitness=fit,
                stats=stats,
                prior=self.prior,
                k_remaining=k_remaining,
                rng=self._rng,
                iteration=self._iteration,
                w_novelty=self.w_novelty,
                w_rarity=self.w_rarity,
                w_age_penalty=self.w_age_penalty,
                w_negative_penalty=self.w_negative_penalty,
                alpha=self.fertility_alpha,
                k_max=self.fertility_k_max,
            )
            scored.append((s, p))
        scored.sort(key=lambda t: t[0], reverse=True)

        parent_score, parent = scored[0]
        parent_cluster = self.program_cluster.get(parent.id, -1)
        parent_stats = self.fertility.get(parent.id) or FertilityStats()
        parent_fit = get_score(parent.metrics or {})

        # Context: 1 sibling (same cluster, complementary verdict) + remainder
        # from distinct other clusters.
        wanted = max(0, num_context_programs)
        context: List[Program] = []
        seen_ids = {parent.id}
        seen_clusters = {parent_cluster}

        sibling = self._find_complementary_sibling(parent, scored, seen_ids)
        if sibling is not None and wanted > 0:
            context.append(sibling)
            seen_ids.add(sibling.id)

        for _s, candidate in scored:
            if len(context) >= wanted:
                break
            if candidate.id in seen_ids:
                continue
            cluster_id = self.program_cluster.get(candidate.id, -1)
            if cluster_id in seen_clusters and cluster_id != -1:
                continue
            context.append(candidate)
            seen_ids.add(candidate.id)
            seen_clusters.add(cluster_id)

        # If we still need more, fall back to highest-POV remaining.
        if len(context) < wanted:
            for _s, candidate in scored:
                if len(context) >= wanted:
                    break
                if candidate.id in seen_ids:
                    continue
                context.append(candidate)
                seen_ids.add(candidate.id)

        parent_label = self._build_parent_label(
            parent=parent,
            parent_score=parent_score,
            parent_fitness=parent_fit,
            parent_stats=parent_stats,
            parent_cluster=parent_cluster,
        )

        return {parent_label: parent}, {"FORE cross-cluster references": context}

    # ==================================================================
    # FORE-specific helpers
    # ==================================================================

    def detect_stagnation(self) -> Tuple[bool, str]:
        """Return ``(should_review, reason)``.

        Three triggers (OR'd together):
        1. Rate trigger: rolling improvement rate over ``review_window`` is
           below ``review_rate_threshold``.
        2. POV-floor trigger: median POV of the top-10 parents is below
           ``pov_floor`` after a small batch of Thompson samples.
        3. All-cluster exhausted: max cluster mean Delta+ falls under an
           epsilon.
        """
        if len(self.programs) < 2:
            return False, "warming-up"

        # Trigger 1 -- improvement rate
        window = self._recent_improvements[-self.review_window :]
        if len(window) >= self.review_window:
            rate = sum(window) / len(window)
            if rate < self.review_rate_threshold:
                return True, f"low improvement rate ({rate:.2f} < {self.review_rate_threshold})"

        # Trigger 2 -- POV-floor
        try:
            top = sorted(
                self.programs.values(),
                key=lambda p: get_score(p.metrics or {}),
                reverse=True,
            )[:10]
            k_remaining = max(1, self.k_remaining_init - self._iteration)
            povs: List[float] = []
            for p in top:
                stats = self.fertility.get(p.id) or FertilityStats()
                fit = get_score(p.metrics or {})
                # Average a few Thompson samples for a stable estimate.
                samples = [
                    pov_score(
                        fitness=fit,
                        stats=stats,
                        prior=self.prior,
                        k_remaining=k_remaining,
                        rng=self._rng,
                        iteration=self._iteration,
                        w_novelty=self.w_novelty,
                        w_rarity=self.w_rarity,
                        w_age_penalty=self.w_age_penalty,
                        w_negative_penalty=self.w_negative_penalty,
                        alpha=self.fertility_alpha,
                        k_max=self.fertility_k_max,
                    )
                    for _ in range(3)
                ]
                povs.append(sum(samples) / len(samples))
            if povs:
                povs.sort()
                median = povs[len(povs) // 2]
                if median < self.pov_floor and self._iteration > self.review_window:
                    return True, f"median POV below floor ({median:.3f} < {self.pov_floor})"
        except Exception as e:  # noqa: BLE001
            logger.debug("FORE: POV-floor check skipped: %s", e)

        # Trigger 3 -- all-cluster exhausted
        max_cluster_mean = 0.0
        for cid in self.clusters.keys():
            mean = self._cluster_mean_delta_plus(cid)
            if mean > max_cluster_mean:
                max_cluster_mean = mean
        if (
            self._iteration > self.review_window
            and len(self.clusters) >= 2
            and max_cluster_mean < 1e-4
        ):
            return True, f"all clusters exhausted (max mean Delta+ = {max_cluster_mean:.5f})"

        return False, ""

    def set_active_review(self, review: FertilityReview) -> None:
        review.uses_remaining = self.review_uses
        self._active_review = review
        self._last_review_iteration = self._iteration
        logger.info(
            "FORE: active review set (next_steps=%d, uses=%d)",
            len(review.next_steps),
            review.uses_remaining,
        )

    def consume_review_for_prompt(self) -> Optional[FertilityReview]:
        """Return the active review and decrement remaining uses."""
        if self._active_review is None:
            return None
        review = self._active_review
        review.uses_remaining -= 1
        if review.uses_remaining <= 0:
            self._tried_reviews.append(review.to_dict())
            self._active_review = None
        return review

    def can_run_review(self, iteration: int) -> bool:
        return iteration - self._last_review_iteration >= self.review_cooldown

    def get_fertility_summary(self) -> List[Dict[str, Any]]:
        """One row per cluster, used as input to the Reflective Reviewer."""
        out: List[Dict[str, Any]] = []
        for cid, member_ids in self.clusters.items():
            if not member_ids:
                continue
            fits = []
            mdps = []
            negs = 0
            tot = 0
            labels: List[str] = []
            for pid in member_ids:
                p = self.programs.get(pid)
                if p is None:
                    continue
                fits.append(get_score(p.metrics or {}))
                stats = self.fertility.get(pid)
                if stats is not None:
                    mdps.append(stats.mean_delta_plus())
                    negs += stats.negative_count
                    tot += stats.n + stats.negative_count
                desc = self.strategy.get(pid)
                if desc is not None and desc.strategy_label and desc.strategy_label != "unspecified":
                    labels.append(desc.strategy_label)

            if not fits:
                continue

            out.append(
                {
                    "cluster_id": cid,
                    "label": labels[0] if labels else "unlabeled",
                    "size": len(member_ids),
                    "mean_fitness": sum(fits) / len(fits),
                    "mean_delta_plus": (sum(mdps) / len(mdps)) if mdps else 0.0,
                    "negative_frac": negs / tot if tot else 0.0,
                }
            )
        out.sort(key=lambda r: r["mean_fitness"], reverse=True)
        return out

    def get_recent_attempts(self, n: int = 10) -> List[Dict[str, Any]]:
        recents = sorted(
            self.programs.values(), key=lambda p: getattr(p, "iteration_found", 0)
        )[-n:]
        out: List[Dict[str, Any]] = []
        for p in recents:
            desc = self.strategy.get(p.id) or StrategyDescription()
            out.append(
                {
                    "iteration": getattr(p, "iteration_found", 0),
                    "strategy_label": desc.strategy_label,
                    "description": desc.description,
                    "verdict": desc.verdict,
                    "fitness": get_score(p.metrics or {}),
                }
            )
        return out

    def get_pov_diagnostics(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Deterministic POV diagnostics for logging."""
        k_remaining = max(1, self.k_remaining_init - self._iteration)
        rows = []
        for p in self.programs.values():
            stats = self.fertility.get(p.id) or FertilityStats()
            fit = get_score(p.metrics or {})
            mean_pov = expected_pov(
                fitness=fit,
                stats=stats,
                prior=self.prior,
                k_remaining=k_remaining,
                iteration=self._iteration,
                w_novelty=self.w_novelty,
                w_rarity=self.w_rarity,
                w_age_penalty=self.w_age_penalty,
                w_negative_penalty=self.w_negative_penalty,
                alpha=self.fertility_alpha,
                k_max=self.fertility_k_max,
            )
            rows.append(
                {
                    "program_id": p.id,
                    "fitness": fit,
                    "expected_pov": mean_pov,
                    "n": stats.n,
                    "neg": stats.negative_count,
                    "mean_delta_plus": stats.mean_delta_plus(),
                    "cluster": self.program_cluster.get(p.id, -1),
                }
            )
        rows.sort(key=lambda r: r["expected_pov"], reverse=True)
        return rows[:top_k]

    # ==================================================================
    # Internals
    # ==================================================================

    def _assign_cluster(self, program: Program, strategy: StrategyDescription) -> int:
        """Assign program to an existing cluster or spawn a new one."""
        tokens = tokenize_strategy(strategy)
        # Backstop: if the description was empty, use code tokens. This keeps
        # initial-program / no-meta candidates from all collapsing into one
        # phony cluster.
        if not tokens:
            tokens = _code_tokens(program.solution or "")

        if not tokens:
            # Truly featureless — give it its own cluster.
            cid = self._next_cluster_id
            self._next_cluster_id += 1
            self.clusters[cid] = [program.id]
            self.program_cluster[program.id] = cid
            self._cluster_tokens[cid] = set()
            return cid

        best_cid = -1
        best_sim = 0.0
        for cid, sig in self._cluster_tokens.items():
            sim = jaccard(tokens, sig)
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        if best_cid != -1 and best_sim >= self.cluster_similarity_threshold:
            self.clusters[best_cid].append(program.id)
            self.program_cluster[program.id] = best_cid
            # Refresh signature lazily by mixing the new tokens in.
            self._cluster_tokens[best_cid] = self._cluster_tokens[best_cid] | tokens
            return best_cid

        cid = self._next_cluster_id
        self._next_cluster_id += 1
        self.clusters[cid] = [program.id]
        self.program_cluster[program.id] = cid
        self._cluster_tokens[cid] = set(tokens)
        return cid

    def _compute_novelty(self, program: Program) -> float:
        """Normalised code novelty in [0, 1]: mean distance to all neighbors."""
        if not self.programs:
            return 1.0
        others = [p for p in self.programs.values() if p.id != program.id]
        if not others:
            return 1.0
        # Sample at most 30 neighbors for speed.
        sample = others if len(others) <= 30 else self._rng.sample(others, 30)
        dists: List[float] = []
        for p in sample:
            try:
                d = self.diversity.distance(program, p)
            except Exception:
                d = 0.5
            dists.append(d)
        if not dists:
            return 1.0
        mean_d = sum(dists) / len(dists)
        # ``distance`` returns 0 (identical) .. 1 (different) for CodeDiversity.
        return max(0.0, min(1.0, mean_d))

    def _compute_cluster_rarity(self, cluster_id: int) -> float:
        size = len(self.clusters.get(cluster_id, []))
        total = max(len(self.programs), 1)
        # 1.0 when the cluster is brand new; → 0 when the cluster dominates.
        return 1.0 - (size / (total + 1))

    def _cluster_mean_delta_plus(self, cluster_id: int) -> float:
        members = self.clusters.get(cluster_id, [])
        vals: List[float] = []
        for pid in members:
            stats = self.fertility.get(pid)
            if stats is None or stats.n == 0:
                continue
            vals.append(stats.mean_delta_plus())
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def _find_complementary_sibling(
        self,
        parent: Program,
        scored: List[Tuple[float, Program]],
        seen_ids: set,
    ) -> Optional[Program]:
        parent_cluster = self.program_cluster.get(parent.id, -1)
        parent_strategy = self.strategy.get(parent.id) or StrategyDescription()
        parent_verdict = parent_strategy.verdict

        # First pass: same cluster, complementary verdict (different bucket).
        for _s, p in scored:
            if p.id in seen_ids or p.id == parent.id:
                continue
            if self.program_cluster.get(p.id, -1) != parent_cluster:
                continue
            cand_strategy = self.strategy.get(p.id) or StrategyDescription()
            if (cand_strategy.verdict or "?") != (parent_verdict or "?"):
                return p

        # Fallback: any sibling in same cluster.
        for _s, p in scored:
            if p.id in seen_ids or p.id == parent.id:
                continue
            if self.program_cluster.get(p.id, -1) == parent_cluster:
                return p

        return None

    def _build_parent_label(
        self,
        parent: Program,
        parent_score: float,
        parent_fitness: float,
        parent_stats: FertilityStats,
        parent_cluster: int,
    ) -> str:
        """Build the prompt-injected parent-selection label.

        Follows the AdaEvolve EXPLORE_LABEL / EXPLOIT_LABEL pattern: this
        string is the dict key returned alongside the parent and is shown
        verbatim to the LLM right before the parent's code.
        """
        strategy = self.strategy.get(parent.id) or StrategyDescription()
        cluster_size = len(self.clusters.get(parent_cluster, []))

        lines = [
            "## PARENT SELECTION (FORE — Thompson sampling on Posterior Offspring Value)",
            f"This parent was chosen because its sampled POV score was {parent_score:.4f}.",
            f"- Current fitness: {parent_fitness:.4f}",
            f"- Children evaluated so far: {parent_stats.n} positive, {parent_stats.negative_count} non-positive",
            f"- Empirical mean positive improvement (Δ+): {parent_stats.mean_delta_plus():.4f}",
            f"- Strategy cluster: id={parent_cluster}, size={cluster_size}, label='{strategy.strategy_label}'",
        ]
        if parent_stats.n == 0 and parent_stats.negative_count == 0:
            lines.append(
                "- This parent has no offspring history yet — Thompson sampling is exploring its potential."
            )
        elif parent_stats.mean_delta_plus() > 0.0 and parent_stats.n >= parent_stats.negative_count:
            lines.append(
                "- This parent is productive: previous mutations have improved on it."
            )
        elif parent_stats.negative_count > parent_stats.n:
            lines.append(
                "- This parent has many regressions; consider a structurally different change."
            )

        if strategy.hypothesis:
            lines.append(f"- Original hypothesis: {strategy.hypothesis[:240]}")
        return "\n".join(lines)

    def _enforce_population_limit(self) -> None:
        if self.population_size <= 0 or len(self.programs) <= self.population_size:
            return
        # Protect: best program, top fertility (highest expected POV).
        diagnostics = self.get_pov_diagnostics(top_k=max(self.population_size // 2, 5))
        protected = {row["program_id"] for row in diagnostics}
        if self.best_program_id:
            protected.add(self.best_program_id)
        if self.initial_program_id:
            protected.add(self.initial_program_id)

        evictable = [
            p
            for p in self.programs.values()
            if p.id not in protected
        ]
        # Evict lowest expected-POV first.
        evictable.sort(key=lambda p: get_score(p.metrics or {}))

        while len(self.programs) > self.population_size and evictable:
            victim = evictable.pop(0)
            self._remove_program(victim.id)

    def _remove_program(self, pid: str) -> None:
        if pid not in self.programs:
            return
        self.programs.pop(pid, None)
        self.fertility.pop(pid, None)
        self.strategy.pop(pid, None)
        cid = self.program_cluster.pop(pid, None)
        if cid is not None and cid in self.clusters:
            self.clusters[cid] = [p for p in self.clusters[cid] if p != pid]
            if not self.clusters[cid]:
                self.clusters.pop(cid, None)
                self._cluster_tokens.pop(cid, None)
