"""Tests for the SIMPLE-EVO selector."""

from __future__ import annotations

from levi.simple.pool import Program
from levi.simple.selector import Selector, SelectorConfig

from ._fake_embeddings import family, vec


def _mk(score: float, embedding, *, uses: int = 0, ts: int = 0, desc: str = "p") -> Program:
    return Program(
        code="",
        description=desc,
        score=score,
        embedding=embedding,
        source="mutate",
        created_at_eval=ts,
        uses_count=uses,
    )


def test_high_score_low_uses_wins_when_healthy() -> None:
    sel = Selector()
    programs = [
        _mk(0.9, vec(1, 0), uses=0, ts=100),
        _mk(0.4, vec(0, 1), uses=0, ts=100),
    ]
    parent = sel.select_parent(programs, n_total=100, stuck=False)
    assert parent is not None
    assert parent.score == 0.9


def test_unused_program_gets_novelty_bonus() -> None:
    sel = Selector(SelectorConfig(alpha_healthy=2.0, beta_healthy=0.0, gamma_healthy=0.0))
    programs = [
        _mk(0.9, vec(1, 0), uses=100),
        _mk(0.7, vec(0, 1), uses=0),
    ]
    parent = sel.select_parent(programs, n_total=200, stuck=False)
    # With heavy novelty weight, the unused program should win despite lower score.
    assert parent is not None
    assert parent.score == 0.7


def test_recency_boost_favors_new_programs() -> None:
    sel = Selector(SelectorConfig(alpha_healthy=0.0, beta_healthy=2.0, gamma_healthy=0.0, recency_tau=5.0))
    programs = [
        _mk(0.5, vec(1, 0), ts=0),   # old
        _mk(0.5, vec(0, 1), ts=95),  # fresh
    ]
    parent = sel.select_parent(programs, n_total=100, stuck=False)
    assert parent is not None
    assert parent.created_at_eval == 95


def test_inspirations_are_distinct_from_parent_and_diverse() -> None:
    sel = Selector(SelectorConfig(n_inspirations=3, gamma_healthy=0.9))
    # 6 programs in 3 distinct families
    progs = []
    for fid in range(3):
        for e in family(seed=fid, jitter=0.02, n=2):
            progs.append(_mk(0.5, e, desc=f"f{fid}"))
    parent = progs[0]
    insp = sel.select_inspirations(progs, exclude=[parent], n_total=100, stuck=False, k=3)
    assert len(insp) == 3
    # Parent should be excluded.
    assert parent not in insp


def test_two_parents_prefer_cross_family() -> None:
    sel = Selector(SelectorConfig(crossover_min_family_separation=0.5))
    progs = []
    # family 0
    for e in family(seed=0, jitter=0.02, n=3):
        progs.append(_mk(0.8, e, desc="f0"))
    # family 1 (far)
    for e in family(seed=1, jitter=0.02, n=3):
        progs.append(_mk(0.7, e, desc="f1"))
    result = sel.select_two_parents(progs, n_total=100, stuck=False)
    assert result is not None
    p1, p2 = result
    # Cross-family pick should land on different descriptions.
    assert p1.description != p2.description


def test_stuck_mode_increases_exploration_weights() -> None:
    sel = Selector(
        SelectorConfig(
            alpha_healthy=0.1, alpha_stuck=2.0,
            beta_healthy=0.0, beta_stuck=0.0,
            gamma_healthy=0.0, gamma_stuck=0.0,
        )
    )
    progs = [
        _mk(0.9, vec(1, 0), uses=50),
        _mk(0.5, vec(0, 1), uses=0),
    ]
    # Healthy: exploit wins.
    assert sel.select_parent(progs, n_total=100, stuck=False).score == 0.9
    # Stuck: novelty bonus large enough to flip the pick.
    assert sel.select_parent(progs, n_total=100, stuck=True).score == 0.5
