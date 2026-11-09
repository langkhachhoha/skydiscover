"""cp_report.py: a baseline's eval count is every LLM-written program evaluated."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "server"))
import cp_report as R  # noqa: E402

# Trimmed from outputs/server/cp_exp/t3_baselines/circle_packing__gepa_native/run.log
GEPA_LOG = """\
12:08:23 [evaluator] Evaluated program 4549 in 0.27s: validity=1.0000, combined_score=0.3642 [cost=$0.0000/$40.00, llm_calls=0]
12:10:05 [evaluator] Evaluated program dced in 0.13s: validity=0.0000, combined_score=0.0000 [cost=$0.0953/$40.00, llm_calls=1]
12:10:05 WARNING [default_discovery_controller] Evaluation failed (attempt 1/3): validity=0.0, error=Evaluation failed (validity=0)
12:12:24 [evaluator] Evaluated program b4a6 in 0.25s: validity=1.0000, combined_score=0.8673 [cost=$0.2399/$40.00, llm_calls=2]
12:12:24 [default_discovery_controller] Iteration 1: Program b4a6 (parent: 4549) completed in 241.05s (llm: 138.54s, eval: 0.25s) [cost=$0.2399/$40.00, llm_calls=2]
12:12:50 [evaluator] Evaluated program 30f4 in 0.26s: validity=1.0000, combined_score=0.8681 [cost=$0.2824/$40.00, llm_calls=3]
12:12:50 [search.gepa_native] Iteration 2: Merge completed (llm: 26.06s, eval: 0.26s) [cost=$0.2824/$40.00, llm_calls=3]
12:24:18 [evaluator] Evaluated program 9e2f in 0.16s: validity=1.0000, combined_score=0.9645 [cost=$0.9353/$40.00, llm_calls=8]
12:24:18 [search.gepa_native] Iteration 4: REJECTED child (child_score=0.9645 <= parent_score=0.9645) [cost=$0.9353/$40.00, llm_calls=8]
11:39:37 ERROR [evaluator] Stage 1 timed out (360s)
11:40:00 ERROR [evaluator] Program 77aa timed out after 600s (limit: 600s)
11:41:00 ERROR [evaluator] All attempts failed 88bb: boom
15:09:08 [evaluator] Evaluated program b4a6 in 0.17s: validity=1.0000, combined_score=0.8673 [cost=$10.98/$40.00, llm_calls=82]
"""


def test_rejected_merged_invalid_and_timed_out_programs_count_but_the_seed_does_not():
    assert len(R.ITER_RE.findall(GEPA_LOG)) == 1           # the old count
    assert R._baseline_candidates(GEPA_LOG) == 6            # dced, b4a6, 30f4, 9e2f, 77aa, 88bb; b4a6's re-score once


def test_log_without_cost_suffix_drops_one_seed():
    log = "\n".join(f"10:00:0{i} [evaluator] Evaluated program p{i} in 0.1s: combined_score=0.5" for i in range(4))
    assert R._baseline_candidates(log) == 3
    assert R._baseline_candidates("") == 0


# Trimmed from outputs/server/pk_exp/kernel_builder/kernel_builder__openevolve_native/run.log
KERNEL_LOG = """\
01:17:20 [evaluator] Evaluated program 492c in 1.07s: combined_score=12.4042, cycles=11910.0000, correct=1.0000 [cost=$0.0000/$20.00, llm_calls=0]
01:20:21 [evaluator] Evaluated program 5eb6 in 0.05s: combined_score=0.0000, cycles=0.0000, correct=0.0000 [cost=$0.2386/$20.00, llm_calls=1]
01:21:52 [evaluator] Evaluated program c854 in 1.04s: combined_score=15.6697, cycles=9428.0000, correct=1.0000 [cost=$0.3259/$20.00, llm_calls=2]
01:21:52 [default_discovery_controller] Metrics: combined_score=99.0000, cycles=9428.0000 [cost=$0.3259/$20.00, llm_calls=2]
"""


def test_live_best_before_the_first_checkpoint_uses_the_tasks_own_score(tmp_path):
    # circle packing: sum_radii, not the combined_score ratio
    assert R._baseline_scores_seen(GEPA_LOG, "circle_packing") == []
    assert max(R._baseline_scores_seen("x [evaluator] Evaluated program a in 1s: sum_radii=2.6100, combined_score=0.99", "circle_packing")) == 2.61
    # any other task: combined_score from evaluator lines only (not the Metrics echo)
    assert max(R._baseline_scores_seen(KERNEL_LOG, "kernel_builder")) == 15.6697
    d = tmp_path / "kernel_builder__openevolve_native"
    d.mkdir()
    (d / "run.log").write_text(KERNEL_LOG)
    assert R.collect_baseline(d)["best"] == 15.6697
