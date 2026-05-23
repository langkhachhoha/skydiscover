"""Top-K pool with description-embedding niching and family clustering.

Replaces CVT-MAP-Elites. Key ideas:

- Programs are stored as ``Program`` records with their description embedding.
- ``add`` rejects/merges near-duplicates by cosine similarity over the
  *description* embedding (semantic dedup, not syntactic).
- A lightweight family clustering layer enforces diversity: no single family
  may exceed ``max_per_family`` slots in the top-K. Families are computed by
  agglomerative single-linkage at cosine threshold ``family_threshold``,
  recomputed lazily when the pool changes.
- ``representatives`` provides three selection modes for the frontier prompt
  (early / mid / late), matching the SIMPLE-EVO design.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .ast_signature import N_FEATURES, ast_cosine, compute_ast_signature
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
    """Structural fingerprint (filled lazily by Pool on first use). Used as
    the second pass of the niche-dedup check: two candidates with very close
    description embeddings but very different AST shapes are kept as
    *structurally distinct* and both admitted."""

    def short_repr(self) -> str:
        desc = self.description.replace("\n", " ")
        if len(desc) > 80:
            desc = desc[:77] + "…"
        return f"<Program score={self.score:.3f} src={self.source} desc='{desc}'>"


@dataclass
class PoolConfig:
    K: int = 100
    """Target population size. Until the pool reaches K, the niching rules
    are relaxed so we *fill* the pool fast (drop only on exact-match
    structural+semantic duplicates, ignore family cap). Once the pool hits
    K, the niching and family cap kick in to maintain quality + diversity.
    Excess admissions evict lowest-score-first (within the offending family
    when family cap fires, else globally)."""

    niche_cosine_threshold: float = 0.92
    """Description-embedding cosine above which two candidates are flagged
    as *semantic* near-duplicates. Tuned on text-embedding-3-small live
    runs: distinct paradigms sit at 0.55-0.70 cosine, paraphrases of the
    same idea cluster above 0.92.

    A flag from this layer alone is no longer enough to drop a candidate —
    the second-pass AST check has to agree (see ``structural_cosine_threshold``)."""

    structural_cosine_threshold: float = 0.97
    """AST-signature cosine above which two candidates are flagged as
    *structurally* near-identical. A candidate is treated as a true
    near-duplicate (and either replaces the incumbent or is dropped)
    only when BOTH description cosine ≥ ``niche_cosine_threshold`` AND
    AST cosine ≥ ``structural_cosine_threshold``. The 0.97 default is
    deliberately tight: small structural edits (an extra branch, a swapped
    data structure) easily fall below 0.97 and earn a slot in the pool,
    even when their description paraphrases an existing entry.

    Set to a value > 1.0 to disable the AST layer (description-only dedup)."""

    family_cosine_threshold: float = 0.72
    """Single-linkage merge threshold on description embeddings — defines
    when two programs belong to the same *family*. Used only once the pool
    has filled to K. Tuned on live data: distinct paradigm classes (DP vs
    BFS vs greedy) average ~0.62, while variants of the same paradigm
    average ~0.75-0.85, so 0.72 separates them well."""

    max_per_family: int = 10
    """Hard cap on how many programs from one family may co-exist in the
    pool, applied *only when len(pool) ≥ K*. While the pool is still
    filling we want as many distinct candidates as possible, so the cap is
    deferred. Once at capacity, exceeding the cap evicts the *lowest-scoring*
    member of that family (not the global worst) so no single family squats
    the top-K."""


class Pool:
    """Thread-safe top-K pool with description-embedding niching."""

    def __init__(self, config: PoolConfig | None = None) -> None:
        self.config = config or PoolConfig()
        self._programs: list[Program] = []
        self._lock = threading.RLock()
        self._families_dirty = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, program: Program) -> tuple[bool, str]:
        """Try to admit *program*. Returns ``(accepted, reason)``.

        Two-phase semantics:

        * **While the pool is filling (len < K)**: maximise raw diversity.
          Only flag a candidate as a near-duplicate when BOTH the
          description embedding AND the AST signature say so — that catches
          identical re-emissions while letting genuine structural variants
          through. Family cap is **not** enforced; we want the pool packed.

        * **Once at capacity (len ≥ K)**: enforce niche + family cap + the
          global K cap so quality stays high.

        Reasons returned:

        - ``"added"`` — appended cleanly.
        - ``"replaced_duplicate"`` — semantic+structural duplicate that
          out-scored the incumbent.
        - ``"dropped_duplicate"`` — semantic+structural duplicate that did
          not improve on the incumbent.
        - ``"replaced_family_weak"`` — admitted but family cap evicted the
          weakest of that family.
        - ``"dropped_full"`` — pool at K and the newcomer was the global
          worst.
        - ``"no_embedding"`` — refused; needed an embedding to niche on.
        """
        with self._lock:
            if program.embedding is None or program.embedding.size == 0:
                # Pool requires an embedding to dedup; reject silently.
                return False, "no_embedding"
            # Pre-compute the structural signature once so we can pass it
            # into the same comparison the family cap will use later.
            if program.ast_signature is None:
                program.ast_signature = compute_ast_signature(program.code)

            pool_filling = len(self._programs) < self.config.K

            # 1) Niche dedup — embedding flags the candidate as semantically
            #    near a neighbour. We then ask the AST whether they are
            #    structurally close too. Only the "and" of both layers
            #    counts as a true near-duplicate.
            nearest_idx, nearest_sim = self._nearest(program.embedding)
            if nearest_idx is not None and nearest_sim >= self.config.niche_cosine_threshold:
                incumbent = self._programs[nearest_idx]
                struct_sim = self._struct_cosine(program, incumbent)
                structural_dup = struct_sim >= self.config.structural_cosine_threshold
                if structural_dup:
                    if program.score > incumbent.score:
                        # Preserve uses_count on replacement so a duplicate
                        # idea cannot dodge the novelty penalty by being
                        # re-emitted.
                        program.uses_count = incumbent.uses_count
                        self._programs[nearest_idx] = program
                        self._families_dirty = True
                        return True, "replaced_duplicate"
                    return False, "dropped_duplicate"
                # Structurally distinct despite a near-identical
                # description: keep the newcomer as a genuinely new variant.

            # 2) Append the candidate.
            self._programs.append(program)
            self._families_dirty = True

            # 3) While the pool is still filling, skip family cap and the
            #    global K cap entirely. The goal is to reach K as fast as
            #    possible with as many distinct ideas as possible.
            if pool_filling:
                return True, "added"

            # 4) Pool is at or past K: enforce family cap first (so a
            #    crowded family evicts its own weakest), then the global
            #    K cap.
            self._recompute_families_if_needed()
            evicted_for_family = self._enforce_family_cap(program)
            if evicted_for_family is not None:
                if evicted_for_family is program:
                    return False, "dropped_family_full"
                return True, "replaced_family_weak"

            if len(self._programs) > self.config.K:
                # Drop the lowest-scoring program (could be the new one).
                worst_idx = min(range(len(self._programs)), key=lambda i: self._programs[i].score)
                worst = self._programs.pop(worst_idx)
                self._families_dirty = True
                if worst is program:
                    return False, "dropped_full"
            return True, "added"

    @staticmethod
    def _struct_cosine(a: Program, b: Program) -> float:
        """Cosine over AST signatures; 0.0 if either is missing.

        When the signature is missing on either side we cannot reason about
        structural similarity, so we return 0.0. The Pool then needs the
        description cosine alone to be above the niche threshold AND the
        structural threshold — i.e. the AST layer effectively never fires,
        and behavior falls back to description-only dedup."""
        if a.ast_signature is None or b.ast_signature is None:
            return 0.0
        return ast_cosine(a.ast_signature, b.ast_signature)

    def __len__(self) -> int:
        with self._lock:
            return len(self._programs)

    def programs(self) -> list[Program]:
        """Snapshot (shallow copy) of programs."""
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
        """Programs grouped by family id."""
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
        """Increment uses_count on an identity match."""
        with self._lock:
            for p in self._programs:
                if p is program:
                    p.uses_count += 1
                    return

    # ------------------------------------------------------------------
    # Representatives for frontier prompt (3 phases)
    # ------------------------------------------------------------------

    def representatives(
        self,
        phase: Literal["early", "mid", "late"],
        n: int = 3,
    ) -> list[Program]:
        """Pick *n* representatives by phase.

        - early: diversity-bias. Greedy MMR with low score weight.
        - mid: top-score anchor + diverse complements.
        - late: top-n by score; surgical context for the heaviest incumbent.
        """
        with self._lock:
            if not self._programs:
                return []
            # Keep family_id current for callers that read it (e.g. log
            # lines, the paradigm prompt builder). The family cap itself
            # only fires once the pool is at K (see ``add``), but family_id
            # is otherwise inspected even before the pool fills.
            self._recompute_families_if_needed()
            n = min(n, len(self._programs))
            if phase == "late":
                return sorted(self._programs, key=lambda p: -p.score)[:n]
            if phase == "early":
                return self._mmr(self._programs, n=n, score_weight=0.2)
            # mid
            top = max(self._programs, key=lambda p: p.score)
            rest = [p for p in self._programs if p is not top]
            picked = [top]
            if rest and n > 1:
                picked += self._mmr(rest, n=n - 1, score_weight=0.5, already=[top])
            return picked

    # ------------------------------------------------------------------
    # Diversity diagnostics (consumed by Monitor)
    # ------------------------------------------------------------------

    def recent_diversity(self, last: int = 20) -> float:
        """Mean pairwise cosine of the last *last* programs added.

        Higher value ⇒ pool converging on one idea (low diversity).
        Returns 0 when fewer than 2 programs are available.
        """
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
        """Greedy MMR. ``score_weight`` (λ) trades score vs diversity."""
        if not candidates or n <= 0:
            return []
        picked: list[Program] = list(already or [])
        pool = list(candidates)
        # Normalize scores to [0,1] for comparable scaling against cosine.
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
            pool.remove(best)
        return picked[len(already or []) :]

    def _recompute_families_if_needed(self) -> None:
        if not self._families_dirty:
            return
        self._families_dirty = False
        n = len(self._programs)
        if n == 0:
            return
        # Union-find over pairs with cosine >= family_threshold.
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

        # Assign compact family ids.
        roots: dict[int, int] = {}
        for i in range(n):
            r = find(i)
            if r not in roots:
                roots[r] = len(roots)
            self._programs[i].family_id = roots[r]

    def _enforce_family_cap(self, just_added: Program) -> Program | None:
        """If just_added's family exceeds the cap, evict the weakest in it.

        Returns the evicted program (or None).
        """
        cap = self.config.max_per_family
        fam_id = just_added.family_id
        members = [p for p in self._programs if p.family_id == fam_id]
        if len(members) <= cap:
            return None
        weakest = min(members, key=lambda p: p.score)
        if weakest is just_added:
            # The newcomer is the worst in an already-full family — drop it.
            self._programs.remove(weakest)
            self._families_dirty = True
            return weakest
        self._programs.remove(weakest)
        self._families_dirty = True
        return weakest

    # ------------------------------------------------------------------
    # Maintenance hooks
    # ------------------------------------------------------------------

    def reset_uses_after_paradigm(self) -> None:
        """Zero all uses_count. Call after frontier accepts a paradigm shift
        so existing programs do not start the next epoch disadvantaged."""
        with self._lock:
            for p in self._programs:
                p.uses_count = 0

    def iter_programs(self) -> Iterable[Program]:
        with self._lock:
            yield from list(self._programs)
