"""
Posterior Offspring Value (POV) — math core for FORE.

Implements Normal-Inverse-Gamma conjugate updates on the per-parent
distribution of positive child improvements Delta+, plus Thompson sampling
over Student-t posteriors. See FORE_METHOD_PLAN.md Section 2 for the full
derivation.

This module has no I/O and no LLM dependencies; it is meant to be unit-tested
in isolation.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class NIGPrior:
    """Normal-Inverse-Gamma prior on (mu, sigma^2)."""

    mu_0: float = 0.0
    kappa_0: float = 2.0
    alpha_0: float = 2.0
    beta_0: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class FertilityStats:
    """Per-parent statistics for Posterior Offspring Value estimation.

    Maintains running sums to compute the posterior in O(1) per update.
    Only the positive part Delta+ = max(Delta, 0) is accumulated for the
    Normal model; non-positive Deltas are counted separately and used to
    discount the structural prior.
    """

    n: int = 0
    sum_delta_plus: float = 0.0
    sum_sq_delta_plus: float = 0.0
    negative_count: int = 0

    # Structural prior inputs, set once at insertion time.
    novelty_score: float = 0.0
    cluster_rarity: float = 0.0
    age_at_birth: int = 0

    # Clip used by update_with_child to keep posterior numerically stable
    # under heavy-tailed Delta distributions.
    delta_clip: float = 5.0

    def update_with_child(self, child_delta: float) -> None:
        """Record one child improvement Delta = f(child) - f(parent).

        Negative or zero Deltas only increment ``negative_count``; positive
        Deltas (clipped at ``delta_clip``) update the Normal sufficient
        statistics.
        """
        if child_delta is None or not math.isfinite(child_delta):
            return
        if child_delta <= 0.0:
            self.negative_count += 1
            return
        # Clip to keep heavy-tail observations from destabilising the
        # posterior. The clip is large enough that real signals survive.
        clipped = min(child_delta, self.delta_clip)
        self.n += 1
        self.sum_delta_plus += clipped
        self.sum_sq_delta_plus += clipped * clipped

    def mean_delta_plus(self) -> float:
        if self.n == 0:
            return 0.0
        return self.sum_delta_plus / self.n

    def positive_fraction(self) -> float:
        total = self.n + self.negative_count
        if total == 0:
            return 0.0
        return self.n / total

    def posterior_t(self, prior: NIGPrior) -> Tuple[float, float, float]:
        """Return ``(loc, scale, df)`` of the Student-t posterior on mu.

        With Normal-Inverse-Gamma conjugacy on Normal-likelihood data, the
        marginal posterior of mu is Student-t with the parameters below.
        """
        n = self.n
        if n == 0:
            # Prior predictive: t with df = 2*alpha_0, location mu_0,
            # scale^2 = beta_0 / (alpha_0 * kappa_0).
            loc = prior.mu_0
            df = max(2.0 * prior.alpha_0, 2.001)
            scale_sq = prior.beta_0 / (prior.alpha_0 * prior.kappa_0)
            return loc, math.sqrt(max(scale_sq, 1e-12)), df

        mean = self.sum_delta_plus / n
        # Sum of squared deviations.
        s_sq = max(self.sum_sq_delta_plus - n * mean * mean, 0.0)

        kappa_n = prior.kappa_0 + n
        loc = (prior.kappa_0 * prior.mu_0 + n * mean) / kappa_n
        alpha_n = prior.alpha_0 + 0.5 * n
        beta_n = (
            prior.beta_0
            + 0.5 * s_sq
            + 0.5 * (prior.kappa_0 * n / kappa_n) * (mean - prior.mu_0) ** 2
        )
        df = 2.0 * alpha_n
        scale_sq = beta_n / (alpha_n * kappa_n)
        return loc, math.sqrt(max(scale_sq, 1e-12)), df

    def sample_mu(self, prior: NIGPrior, rng: random.Random) -> float:
        """Thompson-sample mu from the Student-t posterior.

        Uses the standard trick t = Z / sqrt(W / df) with Z ~ Normal(0,1),
        W ~ ChiSquared(df). ``random.gauss`` and ``random.gammavariate``
        give us both without scipy.
        """
        loc, scale, df = self.posterior_t(prior)
        z = rng.gauss(0.0, 1.0)
        # ChiSquared(df) = Gamma(df/2, 2). Guard df very close to 0.
        df_safe = max(df, 1.001)
        w = rng.gammavariate(df_safe / 2.0, 2.0)
        # Avoid division blow-up when w is tiny.
        denom = math.sqrt(max(w / df_safe, 1e-9))
        return loc + scale * (z / denom)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FertilityStats":
        keep = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in keep})


def fertility_multiplier(
    k_remaining: int, alpha: float = 0.7, k_max: int = 20
) -> float:
    """Compute eff(K, b) = 1 + sum_{k=1..K-1} alpha^k with K clamped at k_max.

    See Lemma 1 in FORE_METHOD_PLAN.md Section 2.2.
    """
    K = min(max(k_remaining, 1), k_max)
    if K == 1:
        return 1.0
    if abs(alpha - 1.0) < 1e-9:
        return float(K)
    # Closed-form geometric series for stability.
    return (1.0 - alpha ** K) / (1.0 - alpha)


def pov_score(
    fitness: float,
    stats: FertilityStats,
    prior: NIGPrior,
    k_remaining: int,
    rng: random.Random,
    iteration: int = 0,
    w_novelty: float = 0.3,
    w_rarity: float = 0.2,
    w_age_penalty: float = 0.01,
    w_negative_penalty: float = 0.2,
    alpha: float = 0.7,
    k_max: int = 20,
) -> float:
    """Thompson-sample one Posterior Offspring Value for a parent.

    POV = fitness + eff(K_remaining) * max(mu_sample, 0) + structural_bonus

    The structural bonus is added directly to the prior mean before sampling,
    so it both shifts the mode and decays as more children are observed
    (kappa_n grows).
    """
    structural_bonus = (
        w_novelty * stats.novelty_score
        + w_rarity * stats.cluster_rarity
        - w_age_penalty * max(0, iteration - stats.age_at_birth)
        - w_negative_penalty * (stats.negative_count / max(stats.n + stats.negative_count, 1))
    )

    effective_prior = NIGPrior(
        mu_0=prior.mu_0 + structural_bonus,
        kappa_0=prior.kappa_0,
        alpha_0=prior.alpha_0,
        beta_0=prior.beta_0,
    )

    mu_sample = stats.sample_mu(effective_prior, rng)
    mult = fertility_multiplier(k_remaining, alpha=alpha, k_max=k_max)
    return fitness + mult * max(mu_sample, 0.0)


def expected_pov(
    fitness: float,
    stats: FertilityStats,
    prior: NIGPrior,
    k_remaining: int,
    iteration: int = 0,
    w_novelty: float = 0.3,
    w_rarity: float = 0.2,
    w_age_penalty: float = 0.01,
    w_negative_penalty: float = 0.2,
    alpha: float = 0.7,
    k_max: int = 20,
) -> float:
    """Deterministic mean POV (for logging / diagnostics)."""
    structural_bonus = (
        w_novelty * stats.novelty_score
        + w_rarity * stats.cluster_rarity
        - w_age_penalty * max(0, iteration - stats.age_at_birth)
        - w_negative_penalty * (stats.negative_count / max(stats.n + stats.negative_count, 1))
    )
    effective_prior = NIGPrior(
        mu_0=prior.mu_0 + structural_bonus,
        kappa_0=prior.kappa_0,
        alpha_0=prior.alpha_0,
        beta_0=prior.beta_0,
    )
    loc, _scale, _df = stats.posterior_t(effective_prior)
    mult = fertility_multiplier(k_remaining, alpha=alpha, k_max=k_max)
    return fitness + mult * max(loc, 0.0)
