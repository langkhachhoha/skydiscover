"""GEPA native: merge-candidate selection must not get stuck on one pair."""

from skydiscover.config import GEPANativeDatabaseConfig
from skydiscover.search.base_database import Program
from skydiscover.search.gepa_native.database import GEPANativeDatabase


def _db_like_cp26_run() -> tuple[GEPANativeDatabase, list[str]]:
    """The pool of the 2026-11-08 circle_packing run: the seed leads on validity and
    eval_time (non-fitness metrics), the best program on every score metric."""
    db = GEPANativeDatabase("gepa_native", GEPANativeDatabaseConfig())
    rows = [  # id, combined_score, eval_time
        ("seed", 0.3642, 0.13), ("best", 0.9645, 0.10), ("b", 0.9645, 0.08),
        ("c", 0.9583, 0.12), ("d", 0.8681, 0.12),
    ]
    for pid, score, t in rows:
        db.add(Program(id=pid, solution=f"# {pid}", metrics={
            "validity": 1.0, "combined_score": score, "target_ratio": score, "eval_time": t}))
    return db, [r[0] for r in rows]


def _key(a: Program, b: Program) -> tuple[str, str]:
    return tuple(sorted((a.id, b.id)))


def test_first_pick_is_the_metric_leaders():
    db, _ = _db_like_cp26_run()
    assert _key(*db.get_merge_candidates()) == ("best", "seed")


def test_tried_pairs_are_skipped_until_none_is_left():
    db, ids = _db_like_cp26_run()
    tried: set[tuple[str, str]] = set()
    n_pairs = len(ids) * (len(ids) - 1) // 2
    for _ in range(n_pairs):
        a, b = db.get_merge_candidates(exclude=tried)
        assert a.id != b.id
        k = _key(a, b)
        assert k not in tried, f"pair {k} offered twice"
        tried.add(k)
    assert len(tried) == n_pairs
    a, b = db.get_merge_candidates(exclude=tried)
    assert a.id == b.id   # every pair tried: the caller's self-merge guard skips it
