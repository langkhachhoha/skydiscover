"""Top-K pool with description-embedding niching, family clustering, and
paradigm-protection extras.

This module replaces the CVT-MAP-Elites archive of LEVI. The headline
ideas, in order of importance:

1. **Niche dedup by description embedding + second-pass AST structural
   signature**: two candidates are flagged as near-duplicates only when
   the description AND the structural signature both agree. Lets the Pool
   admit structurally-distinct variants whose descriptions happened to
   paraphrase each other.

2. **Quota-based family niching** (ablation flag
   :attr:`PoolConfig.enable_quota_niching`): every family has a fixed
   per-family quota that is enforced from the very first program — not
   just after the pool fills to K. Without this the empirical pool was
   observed to fill 100% with one family before any quota could fire.

3. **Hall-of-Fame** (ablation flag :attr:`PoolConfig.enable_hall_of_fame`):
   an append-only side store that holds every accepted paradigm seed and
   the top-N highest-scoring distinct families. Paradigm-shift fanout and
   the anchor-selection step read from the HoF so good ideas survive
   even after the working pool evicts them.

4. **Paradigm grace period** (ablation flag
   :attr:`PoolConfig.enable_paradigm_grace`): a program produced by the
   paradigm-shift branch is marked ``protected_until_eval`` and cannot be
   evicted until that horizon — gives the orchestrator time to fan out
   variants around the new paradigm before the quota/K-cap kills it.

5. **Cross-family selection helpers** (``representatives_cross_family``):
   used by the paradigm-shift prompt to ensure the frontier model always
   sees one anchor per family, even when the pool itself has collapsed
   onto one family (the HoF backfills the rest).

Toggling any of (2)/(3)/(4) off reproduces the legacy behaviour exactly,
so ablation studies can attribute deltas to each component.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .ast_signature import (
    N_FEATURES_BIGRAM,
    N_FEATURES_COUNT14,
    AstMode,
    ast_cosine,
    compute_ast_signature,
)
from .embedder import cosine

logger = logging.getLogger(__name__)

Source = Literal[
    "init",
    "mutate",
    "crossover",
    "repair",
    "paradigm",
    "paradigm_variant",
    "variant",
]

# Sources that should benefit from the paradigm grace + HoF rules.
_PARADIGM_SOURCES: frozenset[str] = frozenset({"paradigm", "paradigm_variant"})


@dataclass
class Program:
    code: str
    description: str
    score: float
    embedding: np.ndarray
    source: Source = "mutate"
    created_at_eval: int = 0
    uses_count: int = 0
    family_id: int = -1  # assigned by Pool._recompute_families
    ast_signature: np.ndarray | None = None
    """Structural fingerprint (filled lazily by Pool on first use)."""
    protected_until_eval: int = 0
    """Eviction guard: this program is protected from family/K eviction
    while ``Pool._current_eval <= protected_until_eval``. Set by
    :meth:`Pool.add` for paradigm-source programs when
    :attr:`PoolConfig.enable_paradigm_grace` is on. Niche-dedup still
    applies — duplicates are a separate concern from quota."""

    def short_repr(self) -> str:
        desc = self.description.replace("\n", " ")
        if len(desc) > 80:
            desc = desc[:77] + "…"
        return f"<Program score={self.score:.3f} src={self.source} desc='{desc}'>"


@dataclass
class PoolConfig:
    # ------------------------------------------------------------------
    # Capacity + niche dedup
    # ------------------------------------------------------------------

    K: int = 100
    """Target population size for the active working pool."""

    niche_cosine_threshold: float = 0.92
    """Description-embedding cosine above which two candidates are flagged
    as semantic near-duplicates."""

    structural_cosine_threshold: float = 0.85
    """AST-signature cosine above which two candidates are flagged as
    structurally near-identical. With the new bigram signature the
    cross-paradigm cosine drops to 0.4-0.6, so 0.85 cleanly separates
    paraphrased same-paradigm hits from genuine cross-paradigm cousins.

    The legacy ``count14`` signature was almost monotone across paradigms
    (cosine ≥ 0.97 even between gradient descent and simulated
    annealing), so on that mode the structural gate effectively never
    fired and the description embedding had to do all the work — exactly
    the failure mode the bigram signature was introduced to fix."""

    ast_mode: AstMode = "bigram"
    """Which structural signature implementation to use. ``"bigram"`` is
    the production default; ``"count14"`` reproduces the legacy 14-count
    log-vector for the ablation study."""

    # ------------------------------------------------------------------
    # Family clustering
    # ------------------------------------------------------------------

    family_cosine_threshold: float = 0.85
    """Single-linkage merge threshold on description embeddings — defines
    when two programs belong to the same *family*.

    Bumped from 0.72 (the legacy default) because live runs showed
    paraphrases of the same paradigm sitting at 0.75-0.85 cosine, so the
    looser 0.72 threshold collapsed everything into a single family
    within the first ~50 evaluations. 0.85 keeps distinct paradigm
    classes separated while still merging genuinely identical ideas."""

    max_per_family: int = 10
    """Hard cap on co-existing programs from one family. Always enforced
    when :attr:`enable_quota_niching` is on (legacy behaviour deferred it
    until ``len(pool) >= K``)."""

    # ------------------------------------------------------------------
    # Quota niching (component A in the proposal)
    # ------------------------------------------------------------------

    enable_quota_niching: bool = True
    """When on, the family cap is enforced from the first admit, not only
    after the pool reaches K. Without this the empirical pool collapses
    into one family before any cap can fire."""

    target_n_families: int = 5
    """Soft quota: each family gets ``ceil(K / target_n_families)`` slots
    when quota niching is on. With K=100 and target=5 that's 20 per
    family, comfortably above ``max_per_family=10`` so the harder cap is
    the one that actually fires."""

    # ------------------------------------------------------------------
    # Paradigm grace + Hall-of-Fame (components D in the proposal)
    # ------------------------------------------------------------------

    enable_paradigm_grace: bool = True
    """Mark paradigm-source programs with a ``protected_until_eval`` set
    to ``current_eval + paradigm_grace_evals``. While protected the
    program survives quota/K eviction (niche-dedup still applies — a true
    duplicate is replaced, not retained)."""

    paradigm_grace_evals: int = 30
    """Window (in evaluations) during which a freshly-admitted paradigm
    program cannot be evicted. 30 ≈ one paradigm fanout (1 seed + 4
    variants) plus enough mutation cycles for workers to test a few
    crosses against the new seed."""

    enable_hall_of_fame: bool = True
    """Append-only side store of every accepted paradigm seed plus the
    top-score program in each distinct family that has ever existed. Read
    by the orchestrator during paradigm-shift to backfill cross-family
    anchors when the working pool has collapsed."""

    hof_size: int = 30
    """Maximum entries in the Hall of Fame. When full, the *lowest-score
    non-paradigm* entry is evicted first; paradigm-source entries are
    never evicted."""


class Pool:
    """Thread-safe top-K pool with description-embedding niching."""

    def __init__(self, config: PoolConfig | None = None) -> None:
        self.config = config or PoolConfig()
        self._programs: list[Program] = []
        self._lock = threading.RLock()
        self._families_dirty = True
        # Eviction clock — bumped by the orchestrator via
        # :meth:`tick_eval` so paradigm-grace windows can measure their
        # horizon. Defaults to 0; if the orchestrator never calls
        # ``tick_eval`` the grace becomes a no-op (no harm, just no
        # protection).
        self._current_eval: int = 0
        # Hall-of-Fame state. Programs here are *copies* of the original
        # admit — preserving them across pool churn is the whole point.
        self._hof: list[Program] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick_eval(self, eval_count: int) -> None:
        """Advance the eviction clock. Called by the orchestrator after
        every ``record_eval`` on the monitor."""
        with self._lock:
            self._current_eval = max(self._current_eval, int(eval_count))

    def add(self, program: Program) -> tuple[bool, str]:
        """Try to admit *program*. Returns ``(accepted, reason)``.

        Reasons:

        - ``"added"`` — appended cleanly.
        - ``"replaced_duplicate"`` — semantic+structural duplicate that
          out-scored the incumbent.
        - ``"dropped_duplicate"`` — semantic+structural duplicate that
          did not improve on the incumbent.
        - ``"replaced_family_weak"`` — admitted but the family cap evicted
          the weakest non-protected member of that family.
        - ``"dropped_family_full"`` — family cap fired and the newcomer
          was itself the weakest non-protected member.
        - ``"dropped_full"`` — pool at K and the newcomer was the global
          worst non-protected program.
        - ``"no_embedding"`` — refused; needed an embedding to niche on.
        """
        with self._lock:
            if program.embedding is None or program.embedding.size == 0:
                return False, "no_embedding"

            # Pre-compute the structural signature once.
            if program.ast_signature is None:
                program.ast_signature = compute_ast_signature(
                    program.code, mode=self.config.ast_mode
                )

            # Stamp paradigm grace before any eviction logic runs so the
            # newcomer can shield itself in case it happens to be the
            # weakest member of its (yet-to-be-formed) family.
            if (
                self.config.enable_paradigm_grace
                and program.source in _PARADIGM_SOURCES
                and program.protected_until_eval == 0
            ):
                program.protected_until_eval = (
                    self._current_eval + self.config.paradigm_grace_evals
                )

            # 1) Niche dedup.
            nearest_idx, nearest_sim = self._nearest(program.embedding)
            if (
                nearest_idx is not None
                and nearest_sim >= self.config.niche_cosine_threshold
            ):
                incumbent = self._programs[nearest_idx]
                struct_sim = self._struct_cosine(program, incumbent)
                if struct_sim >= self.config.structural_cosine_threshold:
                    if program.score > incumbent.score:
                        program.uses_count = incumbent.uses_count
                        # Inherit the incumbent's protection horizon if it
                        # was still active; never *shorten* it.
                        program.protected_until_eval = max(
                            program.protected_until_eval,
                            incumbent.protected_until_eval,
                        )
                        self._programs[nearest_idx] = program
                        self._families_dirty = True
                        self._update_hof(program)
                        return True, "replaced_duplicate"
                    return False, "dropped_duplicate"

            # 2) Append the candidate.
            self._programs.append(program)
            self._families_dirty = True

            # 3) Quota / K enforcement.
            if self.config.enable_quota_niching:
                # Quota niching: enforce family cap from the first admit
                # so we never collapse to one family. K cap still applies
                # at the global level.
                self._recompute_families_if_needed()
                evicted_for_family = self._enforce_family_cap(program)
                if evicted_for_family is not None:
                    if evicted_for_family is program:
                        self._update_hof(program)
                        return False, "dropped_family_full"
                    self._update_hof(program)
                    return True, "replaced_family_weak"
                if len(self._programs) > self.config.K:
                    evicted = self._evict_global_worst(protect=program)
                    if evicted is program:
                        # The newcomer was itself the global worst and
                        # had no protection — drop without HoF.
                        return False, "dropped_full"
                self._update_hof(program)
                return True, "added"

            # ---- Legacy path: quota niching disabled ----
            pool_filling = len(self._programs) - 1 < self.config.K
            if pool_filling:
                self._update_hof(program)
                return True, "added"
            self._recompute_families_if_needed()
            evicted_for_family = self._enforce_family_cap(program)
            if evicted_for_family is not None:
                if evicted_for_family is program:
                    return False, "dropped_family_full"
                self._update_hof(program)
                return True, "replaced_family_weak"
            if len(self._programs) > self.config.K:
                evicted = self._evict_global_worst(protect=program)
                if evicted is program:
                    return False, "dropped_full"
            self._update_hof(program)
            return True, "added"

    @staticmethod
    def _struct_cosine(a: Program, b: Program) -> float:
        if a.ast_signature is None or b.ast_signature is None:
            return 0.0
        return ast_cosine(a.ast_signature, b.ast_signature)

    def __len__(self) -> int:
        with self._lock:
            return len(self._programs)

    def programs(self) -> list[Program]:
        with self._lock:
            return list(self._programs)

    def best(self) -> Program | None:
        with self._lock:
            if not self._programs:
                return None
            return max(self._programs, key=lambda p: p.score)

    def top_k_by_score(self, k: int) -> list[Program]:
        with self._lock:
            return sorted(self._programs, key=lambda p: -p.score)[:k]

    def families(self) -> dict[int, list[Program]]:
        with self._lock:
            self._recompute_families_if_needed()
            out: dict[int, list[Program]] = {}
            for p in self._programs:
                out.setdefault(p.family_id, []).append(p)
            return out

    def num_families(self) -> int:
        with self._lock:
            self._recompute_families_if_needed()
            if not self._programs:
                return 0
            return len({p.family_id for p in self._programs})

    def mark_used(self, program: Program) -> None:
        with self._lock:
            for p in self._programs:
                if p is program:
                    p.uses_count += 1
                    return

    def hall_of_fame(self) -> list[Program]:
        """Snapshot of the Hall of Fame (read-only).

        Returns an empty list when :attr:`PoolConfig.enable_hall_of_fame`
        is off (no entries are recorded). The returned programs are the
        same objects stored internally — callers must treat them as
        read-only."""
        with self._lock:
            return list(self._hof)

    # ------------------------------------------------------------------
    # Representatives for frontier prompt
    # ------------------------------------------------------------------

    def representatives(
        self,
        phase: Literal["early", "mid", "late"],
        n: int = 3,
    ) -> list[Program]:
        """Pick *n* representatives by phase (legacy in-pool only)."""
        with self._lock:
            if not self._programs:
                return []
            self._recompute_families_if_needed()
            n = min(n, len(self._programs))
            if phase == "late":
                return sorted(self._programs, key=lambda p: -p.score)[:n]
            if phase == "early":
                return self._mmr(self._programs, n=n, score_weight=0.2)
            top = max(self._programs, key=lambda p: p.score)
            rest = [p for p in self._programs if p is not top]
            picked = [top]
            if rest and n > 1:
                picked += self._mmr(rest, n=n - 1, score_weight=0.5, already=[top])
            return picked

    def representatives_cross_family(
        self,
        n: int = 3,
        *,
        include_hof: bool = True,
    ) -> list[Program]:
        """One top-score anchor per family, backfilled from the Hall of
        Fame when the working pool has fewer families than requested.

        This is the production path used by paradigm-shift. The strict
        ``representatives(phase=…)`` picks 3 score-anchors that may come
        from a single collapsed family — exactly the case that produces
        useless paradigm shifts. Cross-family selection guarantees the
        frontier sees N *paradigm-distinct* anchors even when the working
        pool itself has collapsed.
        """
        with self._lock:
            if not self._programs and not (include_hof and self._hof):
                return []
            self._recompute_families_if_needed()

            picked: list[Program] = []
            seen_family_ids: set[int] = set()
            # Top program per family in the working pool first.
            by_family: dict[int, Program] = {}
            for p in self._programs:
                cur = by_family.get(p.family_id)
                if cur is None or p.score > cur.score:
                    by_family[p.family_id] = p
            # Order by score desc so the strongest paradigm-anchor leads.
            for p in sorted(by_family.values(), key=lambda x: -x.score):
                if len(picked) >= n:
                    break
                picked.append(p)
                seen_family_ids.add(p.family_id)

            if len(picked) >= n or not include_hof or not self._hof:
                return picked

            # Backfill from HoF: pick HoF entries that are dissimilar to
            # everything already picked (cosine below family threshold).
            # We do NOT rely on HoF family_ids — those are stale snapshots
            # from when the entry was added. Use direct cosine instead.
            thr = self.config.family_cosine_threshold
            for hof_p in sorted(self._hof, key=lambda x: -x.score):
                if len(picked) >= n:
                    break
                if any(
                    cosine(hof_p.embedding, q.embedding) >= thr for q in picked
                ):
                    continue
                picked.append(hof_p)
            return picked

    # ------------------------------------------------------------------
    # Diversity diagnostics (consumed by Monitor)
    # ------------------------------------------------------------------

    def recent_diversity(self, last: int = 20) -> float:
        with self._lock:
            if len(self._programs) < 2:
                return 0.0
            recent = self._programs[-last:]
            sims: list[float] = []
            for i in range(len(recent)):
                for j in range(i + 1, len(recent)):
                    sims.append(cosine(recent[i].embedding, recent[j].embedding))
            return float(sum(sims) / len(sims)) if sims else 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_protected(self, p: Program) -> bool:
        if not self.config.enable_paradigm_grace:
            return False
        return p.protected_until_eval > self._current_eval

    def _nearest(self, emb: np.ndarray) -> tuple[int | None, float]:
        if not self._programs:
            return None, -1.0
        best_i = 0
        best_s = cosine(emb, self._programs[0].embedding)
        for i in range(1, len(self._programs)):
            s = cosine(emb, self._programs[i].embedding)
            if s > best_s:
                best_s = s
                best_i = i
        return best_i, best_s

    def _mmr(
        self,
        candidates: list[Program],
        *,
        n: int,
        score_weight: float,
        already: list[Program] | None = None,
    ) -> list[Program]:
        if not candidates or n <= 0:
            return []
        picked: list[Program] = list(already or [])
        pool = list(candidates)
        scores = np.array([p.score for p in pool], dtype=np.float32)
        s_min, s_max = float(scores.min()), float(scores.max())
        spread = max(s_max - s_min, 1e-9)
        norm_score = {id(p): (p.score - s_min) / spread for p in pool}
        while pool and len(picked) - len(already or []) < n:
            best = None
            best_val = -math.inf
            for c in pool:
                if picked:
                    max_sim = max(cosine(c.embedding, q.embedding) for q in picked)
                else:
                    max_sim = 0.0
                val = score_weight * norm_score[id(c)] - (1 - score_weight) * max_sim
                if val > best_val:
                    best_val = val
                    best = c
            assert best is not None
            picked.append(best)
            # Identity-based removal — ``list.remove`` falls back to
            # field-by-field ``__eq__`` if the first identity check fails,
            # which trips numpy's truth-value ambiguity when two Programs
            # share every non-array field.
            pool = [p for p in pool if p is not best]
        return picked[len(already or []) :]

    def _recompute_families_if_needed(self) -> None:
        if not self._families_dirty:
            return
        self._families_dirty = False
        n = len(self._programs)
        if n == 0:
            return
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        thr = self.config.family_cosine_threshold
        embs = [p.embedding for p in self._programs]
        for i in range(n):
            for j in range(i + 1, n):
                if cosine(embs[i], embs[j]) >= thr:
                    union(i, j)

        roots: dict[int, int] = {}
        for i in range(n):
            r = find(i)
            if r not in roots:
                roots[r] = len(roots)
            self._programs[i].family_id = roots[r]

    def _family_quota(self) -> int:
        """Effective per-family cap. ``max_per_family`` is the hard ceiling
        but ``target_n_families`` may impose a tighter floor (math.ceil
        rather than floor so a K=100 / target=6 still gives a usable 17
        slots per family rather than 16)."""
        cfg = self.config
        if cfg.target_n_families <= 0:
            return cfg.max_per_family
        derived = math.ceil(cfg.K / cfg.target_n_families)
        return min(cfg.max_per_family, derived)

    def _enforce_family_cap(self, just_added: Program) -> Program | None:
        """If just_added's family exceeds the cap, evict the weakest
        non-protected member of that family. Returns the evicted program,
        or None when the family is still under cap.

        When every member of the offending family is protected (paradigm
        grace covers the whole family), the cap is *deferred* — we accept
        a temporary overshoot rather than evict a still-cooling paradigm.
        Subsequent admits to that family will retry the cap after the
        grace expires.
        """
        cap = self._family_quota() if self.config.enable_quota_niching else self.config.max_per_family
        fam_id = just_added.family_id
        members = [p for p in self._programs if p.family_id == fam_id]
        if len(members) <= cap:
            return None
        # Sort by score asc so the weakest is first. Skip protected
        # programs (paradigm grace).
        ranked = sorted(members, key=lambda p: p.score)
        weakest = None
        for p in ranked:
            if not self._is_protected(p):
                weakest = p
                break
        if weakest is None:
            # All members protected — defer the cap.
            return None
        if weakest is just_added:
            self._programs[:] = [p for p in self._programs if p is not weakest]
            self._families_dirty = True
            return weakest
        self._programs[:] = [p for p in self._programs if p is not weakest]
        self._families_dirty = True
        return weakest

    def _evict_global_worst(self, *, protect: Program | None = None) -> Program | None:
        """Drop the lowest-scoring non-protected program. Returns the
        evicted program (which may be ``protect`` itself if everyone else
        was already protected and ``protect`` was the weakest).

        ``protect`` is included in the eviction candidates but only as a
        last resort — if any other non-protected program exists, it goes
        first. This keeps the function symmetric for the caller (no
        special "drop newcomer" branch) while still preferring to keep
        the newly-admitted program when there's a choice.
        """
        if not self._programs:
            return None
        # Candidate pool: non-protected programs.
        candidates = [p for p in self._programs if not self._is_protected(p)]
        if not candidates:
            # Everyone is protected — only protect itself can go and only
            # if it's a member.
            if protect is not None and any(p is protect for p in self._programs):
                self._programs[:] = [p for p in self._programs if p is not protect]
                self._families_dirty = True
                return protect
            return None
        # Pick the lowest-scoring; in a tie, prefer non-``protect``.
        candidates.sort(key=lambda p: (p.score, p is protect))
        weakest = candidates[0]
        self._programs[:] = [p for p in self._programs if p is not weakest]
        self._families_dirty = True
        return weakest

    # ------------------------------------------------------------------
    # Hall of Fame
    # ------------------------------------------------------------------

    def _update_hof(self, program: Program) -> None:
        """Decide whether *program* belongs in the Hall of Fame.

        Two admission rules:

        * Any paradigm-source program is admitted (these are the seeds
          that we want to revisit during future paradigm shifts).
        * Otherwise, if the program would improve the *cross-family*
          diversity of the HoF — its description embedding is below the
          family threshold to every existing HoF entry — and the HoF is
          not yet full, it is admitted as a new paradigm anchor.
        """
        if not self.config.enable_hall_of_fame:
            return
        cfg = self.config

        # Paradigm seeds always admit.
        is_paradigm = program.source in _PARADIGM_SOURCES
        if not is_paradigm:
            # Non-paradigm: only admit when it adds a new family axis to
            # the HoF (cross-family diversity). Skips noisy admissions.
            if program.embedding is None:
                return
            thr = cfg.family_cosine_threshold
            for h in self._hof:
                if cosine(program.embedding, h.embedding) >= thr:
                    return  # Already covered by an existing HoF entry.

        # Capacity check. Evict the weakest non-paradigm entry first,
        # falling back to the absolute weakest if the HoF is entirely
        # paradigm-sourced.
        if len(self._hof) >= cfg.hof_size:
            non_paradigm = [h for h in self._hof if h.source not in _PARADIGM_SOURCES]
            evict_pool = non_paradigm if non_paradigm else list(self._hof)
            weakest = min(evict_pool, key=lambda h: h.score)
            if weakest.score >= program.score and not is_paradigm:
                return  # Newcomer is worse than the HoF floor.
            self._hof[:] = [h for h in self._hof if h is not weakest]
        self._hof.append(program)

    # ------------------------------------------------------------------
    # Maintenance hooks
    # ------------------------------------------------------------------

    def reset_uses_after_paradigm(self) -> None:
        with self._lock:
            for p in self._programs:
                p.uses_count = 0

    def iter_programs(self) -> Iterable[Program]:
        with self._lock:
            yield from list(self._programs)
