"""Tests for the SIMPLE-EVO pool."""

from __future__ import annotations

import pytest

from levi.simple.pool import Pool, PoolConfig, Program

from ._fake_embeddings import family, vec


def _mk(score: float, embedding, desc: str = "p", src: str = "mutate", ts: int = 0) -> Program:
    return Program(
        code=f"# {desc}",
        description=desc,
        score=score,
        embedding=embedding,
        source=src,  # type: ignore[arg-type]
        created_at_eval=ts,
    )


def test_admits_distinct_programs() -> None:
    pool = Pool(PoolConfig(K=10, niche_cosine_threshold=0.95, family_cosine_threshold=0.6))
    accepted, reason = pool.add(_mk(0.5, vec(1, 0, 0)))
    assert accepted and reason == "added"
    accepted, reason = pool.add(_mk(0.6, vec(0, 1, 0)))
    assert accepted and reason == "added"
    assert len(pool) == 2


def test_near_duplicate_replaces_when_better() -> None:
    pool = Pool(PoolConfig(K=10, niche_cosine_threshold=0.9))
    pool.add(_mk(0.5, vec(1, 0, 0), desc="A"))
    # Same direction → cosine ≈ 1.0
    accepted, reason = pool.add(_mk(0.7, vec(1, 0.01, 0), desc="A'"))
    assert accepted and reason == "replaced_duplicate"
    assert len(pool) == 1
    assert pool.programs()[0].score == 0.7


def test_near_duplicate_drops_when_worse() -> None:
    pool = Pool(PoolConfig(K=10, niche_cosine_threshold=0.9))
    pool.add(_mk(0.5, vec(1, 0, 0)))
    accepted, reason = pool.add(_mk(0.3, vec(1, 0.01, 0)))
    assert not accepted and reason == "dropped_duplicate"


def test_family_cap_evicts_weakest_in_family() -> None:
    # All four vectors are within the family threshold (jitter small enough
    # that all pairwise cosines exceed 0.7) but below the dedup threshold
    # (no pair > 0.999). The family cap should evict the weakest member each
    # time a new program joins the (already-full) family.
    pool = Pool(
        PoolConfig(
            K=10,
            niche_cosine_threshold=0.999,
            family_cosine_threshold=0.85,
            max_per_family=2,
        )
    )
    fam0 = family(seed=0, jitter=0.10, n=4)
    for i, e in enumerate(fam0):
        pool.add(_mk(0.1 + 0.1 * i, e, desc=f"f0-{i}", ts=i))
    fams = pool.families()
    sizes = {fid: len(members) for fid, members in fams.items()}
    assert max(sizes.values()) <= 2, sizes
    scores = sorted(p.score for p in pool.programs())
    assert scores == pytest.approx([0.3, 0.4])


def test_top_k_eviction_when_full() -> None:
    pool = Pool(
        PoolConfig(
            K=3,
            niche_cosine_threshold=0.99,
            family_cosine_threshold=0.0,  # no family clustering
            max_per_family=100,
        )
    )
    # 5 mutually orthogonal vectors so no dedup, no family interference.
    components = [
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    ]
    for i, c in enumerate(components):
        pool.add(_mk(score=float(i), embedding=vec(*c), desc=f"p{i}"))
    assert len(pool) <= 3
    scores = sorted(p.score for p in pool.programs())
    assert scores == pytest.approx([2.0, 3.0, 4.0])


def test_representatives_early_diverse() -> None:
    pool = Pool(PoolConfig(K=20, niche_cosine_threshold=0.99, family_cosine_threshold=0.1))
    # Three distinct families with varying scores.
    for fid in range(3):
        for v in family(seed=fid, jitter=0.02, n=3):
            pool.add(_mk(0.5 + 0.01 * fid, v, desc=f"fam{fid}"))
    reps = pool.representatives("early", n=3)
    assert len(reps) == 3
    # Reps should come from different families.
    fams = {r.family_id for r in reps}
    assert len(fams) >= 2  # at minimum 2 of 3 distinct families


def test_representatives_late_top_score() -> None:
    pool = Pool(PoolConfig(K=20, niche_cosine_threshold=0.99, family_cosine_threshold=0.0, max_per_family=100))
    # Orthogonal axes so nothing dedups or family-merges.
    axes = [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)]
    for i, a in enumerate(axes):
        pool.add(_mk(score=float(i), embedding=vec(*a), desc=f"p{i}"))
    reps = pool.representatives("late", n=3)
    assert [r.score for r in reps] == [4.0, 3.0, 2.0]


def test_recent_diversity_high_when_same_family() -> None:
    # Same family but jitter large enough that none of the pairs hit the
    # near-duplicate threshold, so all 6 land in the pool.
    pool = Pool(
        PoolConfig(
            K=20,
            niche_cosine_threshold=0.9999,
            family_cosine_threshold=0.0,
            max_per_family=100,
        )
    )
    for i, v in enumerate(family(seed=0, jitter=0.10, n=6)):
        pool.add(_mk(0.5 + 0.01 * i, v, desc=f"d{i}"))
    assert len(pool) == 6
    div = pool.recent_diversity(last=6)
    assert div > 0.8, f"expected high cosine, got {div}"


def test_recent_diversity_low_when_diverse() -> None:
    pool = Pool(PoolConfig(K=20, niche_cosine_threshold=0.999, family_cosine_threshold=0.0, max_per_family=100))
    # Orthogonal axes ⇒ pairwise cosine ≈ 0
    for axis in range(6):
        comp = [0.0] * 8
        comp[axis] = 1.0
        pool.add(_mk(0.5, vec(*comp), desc=f"ax{axis}"))
    assert len(pool) == 6
    div = pool.recent_diversity(last=6)
    assert div < 0.3, f"expected low cosine, got {div}"


def test_reset_uses_after_paradigm() -> None:
    pool = Pool()
    p = _mk(0.5, vec(1, 0, 0))
    pool.add(p)
    pool.mark_used(p)
    pool.mark_used(p)
    assert pool.programs()[0].uses_count == 2
    pool.reset_uses_after_paradigm()
    assert pool.programs()[0].uses_count == 0
