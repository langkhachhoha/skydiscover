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


def test_family_cap_deferred_until_pool_full() -> None:
    # While the pool is still filling (len < K) the family cap must NOT
    # fire — we want to admit as much raw diversity as possible. K=3 so
    # we can verify the deferred behaviour in a single test.
    pool = Pool(
        PoolConfig(
            K=3,
            niche_cosine_threshold=0.999,
            structural_cosine_threshold=1.5,  # disable AST layer for this test
            family_cosine_threshold=0.85,
            max_per_family=2,
        )
    )
    fam0 = family(seed=0, jitter=0.10, n=3)
    pool.add(_mk(0.1, fam0[0], desc="f0-0"))
    pool.add(_mk(0.2, fam0[1], desc="f0-1"))
    # Two members of one family, max_per_family=2, but pool not at K yet →
    # cap deferred, both still present.
    assert len(pool) == 2
    assert max(len(m) for m in pool.families().values()) == 2

    # Third add fills the pool to K=3 — still under the pool-filling rule
    # at the moment of the call, so the cap is deferred for this one too.
    pool.add(_mk(0.3, fam0[2], desc="f0-2"))
    assert len(pool) == 3
    assert max(len(m) for m in pool.families().values()) == 3

    # Now pool is at K. The 4th add triggers the family cap path: the
    # weakest of the over-sized family is evicted to keep size ≤ K.
    fam0_more = family(seed=0, jitter=0.10, n=5)
    pool.add(_mk(0.9, fam0_more[3], desc="f0-3"))
    assert len(pool) == 3
    scores = sorted(p.score for p in pool.programs())
    assert pytest.approx(0.1) not in scores  # weakest of the family was evicted
    assert pytest.approx(0.9) in scores


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


def test_ast_layer_keeps_structurally_distinct_variants() -> None:
    # Two candidates with near-identical descriptions (high embedding
    # cosine) but very different code shapes should BOTH be admitted —
    # this is the whole point of the AST second pass.
    pool = Pool(
        PoolConfig(
            K=10,
            niche_cosine_threshold=0.9,  # both candidates trip this
            structural_cosine_threshold=0.97,  # AST must agree to drop
        )
    )
    emb = vec(1, 0, 0)
    # First candidate: a tight loop with a single branch.
    code_a = (
        "def f(x):\n"
        "    total = 0\n"
        "    for i in range(len(x)):\n"
        "        if x[i] > 0:\n"
        "            total += x[i]\n"
        "    return total\n"
    )
    # Second candidate: same description but a comprehension-based body
    # — fundamentally different AST shape (no for+if, has GeneratorExp).
    code_b = "def f(x):\n    return sum(v for v in x if v > 0)\n"

    p_a = Program(code=code_a, description="positive-sum", score=0.5, embedding=emb)
    p_b = Program(code=code_b, description="positive-sum", score=0.4, embedding=vec(1, 0.01, 0))
    accepted_a, _ = pool.add(p_a)
    accepted_b, reason_b = pool.add(p_b)
    assert accepted_a
    assert accepted_b, f"AST layer should have kept structurally distinct variant, got {reason_b}"
    assert len(pool) == 2


def test_ast_layer_drops_true_duplicate() -> None:
    # Same code (or near-identical code) + near-identical embedding → the
    # second pass agrees and the lower-scoring one is dropped.
    pool = Pool(
        PoolConfig(
            K=10,
            niche_cosine_threshold=0.9,
            structural_cosine_threshold=0.97,
        )
    )
    code = "def f(x):\n    return sum(x)\n"
    p_a = Program(code=code, description="sum", score=0.7, embedding=vec(1, 0, 0))
    p_b = Program(code=code, description="sum", score=0.4, embedding=vec(1, 0.01, 0))
    pool.add(p_a)
    accepted, reason = pool.add(p_b)
    assert not accepted
    assert reason == "dropped_duplicate"


def test_reset_uses_after_paradigm() -> None:
    pool = Pool()
    p = _mk(0.5, vec(1, 0, 0))
    pool.add(p)
    pool.mark_used(p)
    pool.mark_used(p)
    assert pool.programs()[0].uses_count == 2
    pool.reset_uses_after_paradigm()
    assert pool.programs()[0].uses_count == 0
