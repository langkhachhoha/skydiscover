"""Tests for the SIMPLE-EVO monitor."""

from __future__ import annotations

from levi.simple.monitor import Monitor, MonitorConfig

from ._fake_embeddings import family, vec


def test_plateau_grows_until_new_best() -> None:
    m = Monitor()
    for _ in range(5):
        m.record_eval(score=0.1, accepted=False, embedding=None)
    assert m.plateau_steps == 5
    m.record_eval(score=1.0, accepted=True, embedding=vec(1, 0))
    assert m.plateau_steps == 0
    assert m.best_score == 1.0


def test_stagnation_level_caps_at_one() -> None:
    m = Monitor(MonitorConfig(plateau_max=10))
    for _ in range(25):
        m.record_eval(score=-1.0, accepted=False, embedding=None)
    assert m.stagnation_level() == 1.0


def test_is_stuck_via_plateau() -> None:
    m = Monitor(MonitorConfig(plateau_max=100, stuck_plateau_threshold=20))
    for _ in range(25):
        m.record_eval(score=-1.0, accepted=False, embedding=None)
    assert m.is_stuck()


def test_is_stuck_via_accept_rate() -> None:
    m = Monitor(MonitorConfig(stuck_plateau_threshold=999, stuck_accept_threshold=0.1))
    # Fill the window with rejects.
    for _ in range(50):
        m.record_eval(score=-1.0, accepted=False, embedding=None)
    assert m.is_stuck()


def test_not_stuck_when_progressing() -> None:
    m = Monitor(MonitorConfig(stuck_plateau_threshold=20, stuck_accept_threshold=0.05))
    for i in range(15):
        m.record_eval(score=float(i), accepted=True, embedding=vec(0, 0, i + 1))
    assert not m.is_stuck()


def test_is_collapsing_when_recent_too_similar() -> None:
    m = Monitor(MonitorConfig(collapse_diversity_threshold=0.7, diversity_window_size=10))
    for e in family(seed=0, jitter=0.01, n=10):
        m.record_eval(score=1.0, accepted=True, embedding=e)
    assert m.is_collapsing()


def test_is_not_collapsing_when_recent_diverse() -> None:
    m = Monitor(MonitorConfig(collapse_diversity_threshold=0.7, diversity_window_size=10))
    for axis in range(8):
        e = vec(*([0] * axis + [1]))
        m.record_eval(score=1.0, accepted=True, embedding=e)
    assert not m.is_collapsing()


def test_snapshot_contains_all_signals() -> None:
    m = Monitor()
    m.record_eval(score=0.5, accepted=True, embedding=vec(1, 0))
    snap = m.snapshot()
    for key in (
        "eval_count",
        "best_score",
        "plateau_steps",
        "stagnation_level",
        "accept_rate",
        "mean_recent_diversity",
        "is_stuck",
        "is_collapsing",
    ):
        assert key in snap, key
