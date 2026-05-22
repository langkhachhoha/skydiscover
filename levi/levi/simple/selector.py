"""UCB-style parent and inspiration selector.

Priority for an unpicked candidate ``p`` given an already-picked set ``S``:

    priority(p; S) = score_norm(p)                                       # exploit
                   + α · sqrt( log(1 + N_total) / (1 + uses_count(p)) )  # UCB novelty
                   + β · exp( -(N_total - created_at_eval(p)) / τ )      # recency boost
                   − γ · max_{q ∈ S} cosine(p, q)                        # diversity penalty

α, β, γ swap to "stuck" mode when ``Monitor.is_stuck()`` so exploration
pressure rises automatically.

For *parents* (single or pair) and *inspirations* (k), selection is greedy
batched: pick the top-priority program given the currently selected set,
then update S and repeat. This keeps batches diverse without solving an
NP-hard subset-selection.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .embedder import cosine
from .pool import Program

logger = logging.getLogger(__name__)


@dataclass
class SelectorConfig:
    # explore/exploit weights, healthy ↔ stuck
    alpha_healthy: float = 0.5
    alpha_stuck: float = 0.8
    beta_healthy: float = 0.3
    beta_stuck: float = 0.5
    gamma_healthy: float = 0.4
    gamma_stuck: float = 0.7

    recency_tau: float = 30.0
    """Half-life-ish constant for recency exp decay (in eval units)."""

    n_inspirations: int = 3

    crossover_min_family_separation: float = 0.65
    """When picking the second parent for crossover, prefer one whose
    family-mean cosine to the first parent is BELOW this value, to push
    the LLM toward cross-family hybridization."""


@dataclass
class Selector:
    config: SelectorConfig = field(default_factory=SelectorConfig)

    def weights(self, *, stuck: bool) -> tuple[float, float, float]:
        if stuck:
            return self.config.alpha_stuck, self.config.beta_stuck, self.config.gamma_stuck
        return self.config.alpha_healthy, self.config.beta_healthy, self.config.gamma_healthy

    # ------------------------------------------------------------------
    # Core priority
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_scores(programs: Sequence[Program]) -> dict[int, float]:
        if not programs:
            return {}
        arr = np.array([p.score for p in programs], dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        spread = max(hi - lo, 1e-9)
        return {id(p): (p.score - lo) / spread for p in programs}

    def _priority(
        self,
        p: Program,
        *,
        already_picked: Sequence[Program],
        n_total: int,
        norm_scores: dict[int, float],
        alpha: float,
        beta: float,
        gamma: float,
    ) -> float:
        score = norm_scores.get(id(p), 0.0)
        novelty = math.sqrt(math.log(1 + max(n_total, 1)) / (1 + p.uses_count))
        age = max(0, n_total - p.created_at_eval)
        recency = math.exp(-age / max(self.config.recency_tau, 1e-6))
        if already_picked:
            max_sim = max(cosine(p.embedding, q.embedding) for q in already_picked)
        else:
            max_sim = 0.0
        return score + alpha * novelty + beta * recency - gamma * max_sim

    # ------------------------------------------------------------------
    # Public selection API
    # ------------------------------------------------------------------

    def select_parent(
        self,
        programs: Sequence[Program],
        *,
        n_total: int,
        stuck: bool,
    ) -> Program | None:
        if not programs:
            return None
        alpha, beta, gamma = self.weights(stuck=stuck)
        norm = self._normalized_scores(programs)
        return max(
            programs,
            key=lambda p: self._priority(
                p,
                already_picked=(),
                n_total=n_total,
                norm_scores=norm,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            ),
        )

    def select_two_parents(
        self,
        programs: Sequence[Program],
        *,
        n_total: int,
        stuck: bool,
    ) -> tuple[Program, Program] | None:
        """Pick two parents biased toward cross-family hybridization."""
        if len(programs) < 2:
            return None
        p1 = self.select_parent(programs, n_total=n_total, stuck=stuck)
        assert p1 is not None
        others = [p for p in programs if p is not p1]
        # Prefer those whose cosine to p1 is below the separation threshold.
        far = [p for p in others if cosine(p.embedding, p1.embedding) < self.config.crossover_min_family_separation]
        candidates = far if far else others
        alpha, beta, gamma = self.weights(stuck=stuck)
        norm = self._normalized_scores(candidates)
        p2 = max(
            candidates,
            key=lambda p: self._priority(
                p,
                already_picked=[p1],
                n_total=n_total,
                norm_scores=norm,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            ),
        )
        return p1, p2

    def select_inspirations(
        self,
        programs: Sequence[Program],
        *,
        exclude: Sequence[Program],
        n_total: int,
        stuck: bool,
        k: int | None = None,
    ) -> list[Program]:
        """Greedy batched pick of *k* inspirations."""
        k = self.config.n_inspirations if k is None else k
        if k <= 0:
            return []
        exclude_ids = {id(p) for p in exclude}
        pool = [p for p in programs if id(p) not in exclude_ids]
        if not pool:
            return []
        alpha, beta, gamma = self.weights(stuck=stuck)
        norm = self._normalized_scores(pool)
        picked: list[Program] = []
        seen = set(exclude_ids)
        while pool and len(picked) < k:
            best = None
            best_val = -math.inf
            for c in pool:
                if id(c) in seen:
                    continue
                val = self._priority(
                    c,
                    already_picked=picked + list(exclude),
                    n_total=n_total,
                    norm_scores=norm,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                if val > best_val:
                    best_val = val
                    best = c
            if best is None:
                break
            picked.append(best)
            seen.add(id(best))
            pool = [c for c in pool if id(c) not in seen]
        return picked
