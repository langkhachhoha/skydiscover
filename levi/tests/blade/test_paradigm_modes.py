"""Tests for the 3-mode paradigm-shift dispatch in BladeOrchestrator.

We do not call any LLMs here — we only verify the routing logic
inside :meth:`BladeOrchestrator._pick_paradigm_mode` and the anchor
configuration inside :meth:`_build_paradigm_prompt_for_mode`.
"""

from __future__ import annotations

import numpy as np

from levi.blade.orchestrator import BladeConfig, BladeOrchestrator
from levi.simple import EmbedderConfig
from levi.simple.archive import Program


def _stub_orch() -> BladeOrchestrator:
    cfg = BladeConfig(
        problem_description="test",
        function_signature="def solve(x):",
        score_fn=lambda fn, _i=None: {"score": 0.0},
        fn_name="solve",
        budget_evals=1,
        n_workers=1,
        n_eval_processes=1,
        eval_timeout=1.0,
        pe_cron_interval=999,
        output_dir="/tmp/blade_test_paradigm_modes",
        embedder_config=EmbedderConfig(model="fake/embed", dim=8),
    )
    return BladeOrchestrator(cfg)


def _add(orch: BladeOrchestrator, n: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        emb = rng.normal(size=8).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        p = Program(
            code=f"def solve(x):\n    return x + {i}\n",
            description=f"paradigm {i % 4} variant {i}",
            score=float(2.0 - 0.01 * i),
            embedding=emb,
        )
        orch.archive.add(p)


def test_mode_dispatch_by_stagnation() -> None:
    orch = _stub_orch()
    cfg = orch.config

    # synthesis at low stagnation.
    orch.monitor.eval_count = 10
    orch.monitor.last_best_eval = 10
    orch.monitor.last_admit_eval = 10
    assert orch._pick_paradigm_mode() == "synthesis"

    # shift at mid stagnation: push the admit gap so local stagnation is
    # mid-range. admit_gap_max default is 20; we want stagnation around
    # 0.5-0.6 → admit_gap=12.
    orch.monitor.eval_count = 22
    orch.monitor.last_admit_eval = 10
    orch.monitor.last_best_eval = 22
    s = orch.monitor.stagnation_level()
    assert cfg.paradigm_synthesis_max_stagnation < s <= cfg.paradigm_shift_max_stagnation
    assert orch._pick_paradigm_mode() == "shift"

    # surgical at high stagnation: admit_gap >> admit_gap_max.
    orch.monitor.eval_count = 200
    orch.monitor.last_admit_eval = 10
    orch.monitor.last_best_eval = 10
    assert orch.monitor.stagnation_level() > cfg.paradigm_shift_max_stagnation
    assert orch._pick_paradigm_mode() == "surgical"


def test_synthesis_mode_surfaces_top_3_anchors() -> None:
    orch = _stub_orch()
    _add(orch, n=10)
    prompt, anchors, _insps = orch._build_paradigm_prompt_for_mode("synthesis")
    assert "MOVE: SYNTHESIS" in prompt
    assert len(anchors) == orch.config.paradigm_synthesis_n_anchors


def test_shift_mode_surfaces_top_2_anchors() -> None:
    orch = _stub_orch()
    _add(orch, n=10)
    prompt, anchors, _insps = orch._build_paradigm_prompt_for_mode("shift")
    assert "MOVE: SHIFT" in prompt
    assert len(anchors) == orch.config.paradigm_shift_n_anchors


def test_surgical_mode_surfaces_only_champion() -> None:
    orch = _stub_orch()
    _add(orch, n=10)
    prompt, anchors, insps = orch._build_paradigm_prompt_for_mode("surgical")
    assert "MOVE: SURGICAL" in prompt
    # Exactly one anchor regardless of how many programs are in the
    # archive — surgical mode focuses on the champion only.
    assert len(anchors) == 1
    # Inspirations (description-only) should be non-empty when the
    # archive has more programs.
    assert len(insps) >= 1
