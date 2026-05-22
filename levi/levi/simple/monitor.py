"""Search-state monitor: three dense sliding-window signals.

Replaces the PPS stagnation formula. Monitor only *routes* (frontier phase,
operator mix, advisor mode) — it does **not** trigger frontier paradigm
shifts (those stay on a cron schedule, exactly as in LEVI).

Three signals:

- ``plateau_steps``: evaluations since the global best score increased.
  Drives ``stagnation_level() ∈ [0,1]`` for phase routing.
- ``accept_window``: deque of bool, recent pool-accept outcomes. A drop in
  acceptance rate flags ``is_stuck()``.
- ``diversity_window``: deque of pairwise cosine of recently added
  embeddings. A high mean flags ``is_collapsing()`` — pool is converging on
  one family.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .embedder import cosine

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    plateau_max: int = 100
    """Denominator for ``stagnation_level()`` — at plateau_max, level=1.0."""

    accept_window_size: int = 50
    diversity_window_size: int = 20

    stuck_plateau_threshold: int = 80
    stuck_accept_threshold: float = 0.08
    collapse_diversity_threshold: float = 0.78
    """Mean pairwise cosine above this ⇒ recent additions are too similar.
    Tuned from live runs: cross-paradigm pairs average ~0.6, same-paradigm
    variants average ~0.78-0.85, so 0.78 fires once recent accepts are
    almost exclusively same-paradigm."""

    score_eps: float = 1e-9
    """Minimum score improvement to count as a 'new best'."""


@dataclass
class Monitor:
    config: MonitorConfig = field(default_factory=MonitorConfig)

    eval_count: int = 0
    best_score: float = -math.inf
    last_best_eval: int = 0

    accept_window: deque[bool] = field(default_factory=lambda: deque(maxlen=50))
    diversity_window: deque[float] = field(default_factory=lambda: deque(maxlen=20))

    # Buffer of recent embeddings used to compute pairwise diversity. We keep
    # them separately from accept_window because diversity tracks *added*
    # programs (winners), not all attempted evaluations.
    _recent_embeddings: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self) -> None:
        if self.accept_window.maxlen != self.config.accept_window_size:
            self.accept_window = deque(self.accept_window, maxlen=self.config.accept_window_size)
        if self.diversity_window.maxlen != self.config.diversity_window_size:
            self.diversity_window = deque(self.diversity_window, maxlen=self.config.diversity_window_size)
        if self._recent_embeddings.maxlen != self.config.diversity_window_size:
            self._recent_embeddings = deque(self._recent_embeddings, maxlen=self.config.diversity_window_size)

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def record_eval(self, *, score: float, accepted: bool, embedding: np.ndarray | None) -> None:
        """Record one evaluation outcome. Call once per finished candidate."""
        self.eval_count += 1
        self.accept_window.append(bool(accepted))

        # Only count as new best if this candidate was actually accepted into
        # the pool — rejects can't be the current incumbent, even if their
        # raw score looks high (e.g. crashed-after-accept paths).
        if accepted and score > self.best_score + self.config.score_eps:
            self.best_score = score
            self.last_best_eval = self.eval_count

        if accepted and embedding is not None and embedding.size > 0:
            # Compute mean pairwise cosine of the new embedding vs the
            # existing buffer, then push it. Single scalar per accept keeps
            # the window cheap.
            if self._recent_embeddings:
                sims = [cosine(embedding, e) for e in self._recent_embeddings]
                self.diversity_window.append(float(np.mean(sims)))
            self._recent_embeddings.append(embedding.astype(np.float32, copy=False))

    # ------------------------------------------------------------------
    # Read-only signals
    # ------------------------------------------------------------------

    @property
    def plateau_steps(self) -> int:
        return self.eval_count - self.last_best_eval

    def stagnation_level(self) -> float:
        """Float in [0,1]: 0 means just improved, 1 means stuck ≥ plateau_max.

        Feeds ``get_budget_stage`` to choose early / mid / late frontier prompt.
        """
        return min(1.0, self.plateau_steps / max(1, self.config.plateau_max))

    def acceptance_rate(self) -> float:
        if not self.accept_window:
            return 1.0  # Avoid early false-positive "stuck" before window fills.
        return sum(self.accept_window) / len(self.accept_window)

    def mean_recent_diversity(self) -> float:
        if not self.diversity_window:
            return 0.0
        return float(np.mean(self.diversity_window))

    def is_stuck(self) -> bool:
        """Plateau too long OR acceptance rate collapsed."""
        if self.plateau_steps > self.config.stuck_plateau_threshold:
            return True
        if len(self.accept_window) >= max(10, self.config.accept_window_size // 2):
            if self.acceptance_rate() < self.config.stuck_accept_threshold:
                return True
        return False

    def is_collapsing(self) -> bool:
        """Recent accepted programs are too similar — pool converging on one
        family. Distinct from ``is_stuck``: progress may still be happening,
        but only within a single basin.
        """
        if len(self.diversity_window) < max(3, self.config.diversity_window_size // 4):
            return False
        return self.mean_recent_diversity() > self.config.collapse_diversity_threshold

    # ------------------------------------------------------------------
    # Aggregate state (cheap snapshot for logging / advisor)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, float | int | bool]:
        return {
            "eval_count": self.eval_count,
            "best_score": self.best_score if math.isfinite(self.best_score) else float("nan"),
            "plateau_steps": self.plateau_steps,
            "stagnation_level": self.stagnation_level(),
            "accept_rate": self.acceptance_rate(),
            "mean_recent_diversity": self.mean_recent_diversity(),
            "is_stuck": self.is_stuck(),
            "is_collapsing": self.is_collapsing(),
        }
