"""Tests for the three contributions of the refactor:

1. PPS — Posterior-Plateau Stagnation
2. AdaptiveRankSampler — parameter-free Zipfian rank sampling
3. HLS Strategic Blueprint — parser, TTL, and producer injection guard

These are tight, focused unit tests; integration with the full evolutionary
loop is covered by ``test_integration.py`` and ``test_sal.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from levi.behavior.extractor import BehaviorExtractor
from levi.config.models import BlueprintConfig, BudgetConfig
from levi.core import EvaluationResult, Program
from levi.equilibrium.prompts import parse_blueprint
from levi.pipeline.state import PipelineState, StrategicBlueprint
from levi.pool.cvt_map_elites import AdaptiveRankSampler, Elite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(budget: BudgetConfig | None = None) -> PipelineState:
    return PipelineState(budget or BudgetConfig(evaluations=200))


def _mock_elites(scores: list[float]) -> dict[int, Elite]:
    elites = {}
    for i, s in enumerate(scores):
        elites[i] = Elite(
            program=Program(content=f"# {i}"),
            result=EvaluationResult(scores={"score": s}, is_valid=True),
            behavior={},  # type: ignore[arg-type]
        )
    return elites


# ---------------------------------------------------------------------------
# PPS — Posterior-Plateau Stagnation
# ---------------------------------------------------------------------------


class TestPPS:
    def test_alpha_is_zero_at_t_zero_so_signal_is_plateau(self):
        state = _make_state(BudgetConfig(evaluations=100))
        state.budget_tracker.eval_count = 40
        state.eval_count_at_last_best = 0  # plateau p=40/80 = 0.5
        # b ≈ 40/100 = 0.4, α = 0.16 — still mostly plateau-driven.
        s = state.stagnation_depth(tau=80)
        assert 0.40 < s < 0.55

    def test_record_new_best_resets_plateau_and_seeds_hazard(self):
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 5.0
        state.budget_tracker.eval_count = 100
        state.eval_count_at_last_best = 50  # plateau in progress

        state.record_new_best()

        assert state.eval_count_at_last_best == 100
        assert len(state.new_best_history) == 1
        eval_count, cost = state.new_best_history[-1]
        assert eval_count == 100
        assert cost == pytest.approx(5.0)

    def test_hazard_estimate_uses_window_since_first_history_entry(self):
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 9.0  # b = 0.9
        state.budget_tracker.eval_count = 100
        state.eval_count_at_last_best = 20  # plateau p = 1
        # 5 NEW BEST events, the first at cost=1.0, last at cost=5.0.
        for i in range(5):
            state.new_best_history.append((10 * (i + 1), 1.0 + i))
        s = state.stagnation_depth(tau=80)
        # B_W = 9 - 1 = 8, λ̂ = 6/8 = 0.75; B_rem = 1; exp(-0.75) ≈ 0.47.
        # posterior = 1.0 * 0.47 = 0.47.  α = 0.81 → s = 0.19·1 + 0.81·0.47 ≈ 0.57.
        assert 0.45 < s < 0.65

    def test_pps_clamped_to_0_1(self):
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 100.0  # overshoot
        state.budget_tracker.eval_count = 1000
        state.eval_count_at_last_best = 0
        s = state.stagnation_depth(tau=10)
        assert 0.0 <= s <= 1.0

    def test_no_budget_falls_back_to_pure_plateau(self):
        state = _make_state(BudgetConfig(target_score=10.0))
        state.budget_tracker.eval_count = 100
        state.eval_count_at_last_best = 60  # p = 40/80 = 0.5
        s = state.stagnation_depth(tau=80)
        assert 0.49 < s < 0.51

    def test_freshly_improving_returns_zero_regardless_of_budget(self):
        """Behavioural guarantee: if every recent eval is a NEW BEST, PPS
        cannot panic-trigger no matter how late in the run we are."""
        state = _make_state(BudgetConfig(dollars=10.0))
        state.budget_tracker.total_cost = 9.5  # b ≈ 1.0
        state.budget_tracker.eval_count = 200
        state.eval_count_at_last_best = 200  # plateau = 0
        for i in range(20):
            state.new_best_history.append((i * 10, 0.4 * i))
        s = state.stagnation_depth(tau=80)
        assert s == 0.0


# ---------------------------------------------------------------------------
# AdaptiveRankSampler — parameter-free rank-based selection
# ---------------------------------------------------------------------------


class TestAdaptiveRankSampler:
    def test_low_stagnation_prefers_top_ranked(self):
        np.random.seed(0)
        sampler = AdaptiveRankSampler()
        elites = _mock_elites([10 - i for i in range(10)])

        counts = np.zeros(10)
        for _ in range(3000):
            cells = sampler.select_cells(elites, n=1, context={"stagnation": 0.0})
            counts[cells[0]] += 1
        # Top-ranked cell should dominate (Zipfian with β=2).
        assert counts[0] > counts[3] > counts[9]

    def test_high_stagnation_approaches_uniform(self):
        np.random.seed(1)
        sampler = AdaptiveRankSampler()
        elites = _mock_elites([10 - i for i in range(10)])

        counts = np.zeros(10)
        for _ in range(3000):
            cells = sampler.select_cells(elites, n=1, context={"stagnation": 1.0})
            counts[cells[0]] += 1
        # With β=0.2 the worst rank should still get a meaningful share.
        assert counts[9] / counts[0] > 0.4

    def test_no_replacement(self):
        np.random.seed(2)
        sampler = AdaptiveRankSampler()
        elites = _mock_elites([10 - i for i in range(10)])
        chosen = sampler.select_cells(elites, n=5, context={"stagnation": 0.5})
        assert len(set(chosen)) == 5

    def test_default_context_does_not_crash(self):
        sampler = AdaptiveRankSampler()
        elites = _mock_elites([1.0, 2.0, 3.0])
        cells = sampler.select_cells(elites, n=2, context=None)
        assert len(cells) == 2

    def test_beta_floor(self):
        sampler = AdaptiveRankSampler(beta_max=2.0, beta_min=0.5)
        assert sampler._compute_beta(stagnation=1.0) == 0.5
        assert sampler._compute_beta(stagnation=0.0) == 2.0


# ---------------------------------------------------------------------------
# HLS — Strategic Blueprint
# ---------------------------------------------------------------------------


SAMPLE_BLUEPRINT = """DIAGNOSIS:
The archive is dominated by greedy nearest-neighbour placements that fail
near the boundary.

APPROACH:
Switch to simulated annealing on a global energy that penalises overlap and
rewards radius. Perturb one circle per step and accept by Metropolis.

INVARIANTS:
- output is a list of (x, y, r) tuples
- circles fit in the unit square
- no pair overlaps

PSEUDOCODE:
state = random_init(n)
for step in range(N):
    cand = perturb(state, T)
    if accept(E(cand) - E(state), T):
        state = cand
    T *= cooling
return state
"""


class TestBlueprintParser:
    def test_parses_all_four_sections(self):
        parsed = parse_blueprint(SAMPLE_BLUEPRINT)
        assert parsed is not None
        assert "greedy nearest-neighbour" in parsed["diagnosis"]
        assert "simulated annealing" in parsed["approach"]
        assert "(x, y, r)" in parsed["invariants"]
        assert "random_init" in parsed["pseudocode"]

    def test_returns_none_on_unstructured_text(self):
        assert parse_blueprint("just prose, no headers") is None
        assert parse_blueprint("") is None

    def test_missing_section_left_empty(self):
        partial = "DIAGNOSIS:\nfoo\n\nAPPROACH:\nbar\n"
        parsed = parse_blueprint(partial)
        assert parsed is not None
        assert parsed["diagnosis"] == "foo"
        assert parsed["approach"] == "bar"
        assert parsed["invariants"] == ""
        assert parsed["pseudocode"] == ""

    def test_lowercase_and_extra_whitespace_tolerated(self):
        text = "diagnosis :\nA\napproach:\nB\ninvariants:\nC\npseudocode:\nD"
        parsed = parse_blueprint(text)
        assert parsed is not None
        assert parsed["approach"] == "B"

    def test_strips_pseudocode_code_fences(self):
        text = SAMPLE_BLUEPRINT.replace("PSEUDOCODE:\n", "PSEUDOCODE:\n```python\n") + "```"
        parsed = parse_blueprint(text)
        assert parsed is not None
        assert "```" not in parsed["pseudocode"]


class TestBlueprintTTL:
    def test_install_and_directive_text(self):
        state = _make_state()
        bp = StrategicBlueprint(
            approach="Try simulated annealing.",
            pseudocode="state = init()\nfor t: ...",
            raw="full text",
            pe_event_id=1,
        )
        state.install_blueprint(bp, ttl_evals=5)
        assert state.current_blueprint is bp
        assert bp.is_active
        directive = bp.directive_text()
        assert "simulated annealing" in directive
        assert "Pseudocode sketch:" in directive

    def test_tick_decrements_and_drops_at_zero(self):
        state = _make_state()
        bp = StrategicBlueprint(approach="A", raw="A", pe_event_id=1)
        state.install_blueprint(bp, ttl_evals=2)
        state.consume_blueprint_tick()
        assert state.current_blueprint is bp
        assert bp.ttl_evals == 1
        state.consume_blueprint_tick()
        assert state.current_blueprint is None
        assert bp.ttl_evals == 0

    def test_inactive_blueprint_returns_empty_directive(self):
        bp = StrategicBlueprint(approach="", pseudocode="", raw="", pe_event_id=0)
        assert bp.directive_text() == ""
        assert not bp.is_active

    def test_install_clamps_negative_ttl(self):
        state = _make_state()
        bp = StrategicBlueprint(approach="A", raw="A", pe_event_id=1)
        state.install_blueprint(bp, ttl_evals=-5)
        assert bp.ttl_evals == 0
        assert not bp.is_active


# ---------------------------------------------------------------------------
# BlueprintConfig defaults
# ---------------------------------------------------------------------------


class TestBlueprintConfigDefaults:
    def test_defaults_are_conservative(self):
        c = BlueprintConfig()
        assert c.enabled is True
        # TTL multiplier is >1 so a blueprint outlives one PE interval, but
        # not so large that an outdated directive lingers forever.
        assert 1.0 <= c.ttl_multiplier <= 3.0
        assert 0.0 < c.inject_probability < 1.0
        assert 0.0 < c.stagnation_gate < 1.0
        assert c.max_tokens > 0
