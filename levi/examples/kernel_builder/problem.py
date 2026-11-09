"""
Kernel Engineering (Anthropic's VLIW SIMD take-home) -- CORAL's stress-test task, for SpecEvo.

The candidate defines ``KernelBuilder`` (as in CORAL's seed) plus the entry
point ``kernel_builder()`` returning that class. Scoring is shared with the
baselines' evaluator: benchmarks/coral_tasks/kernel_builder/kernel_eval.py
(forest_height=10, rounds=16, batch_size=256, 8 correctness runs; score =
147734 / cycles, wrong output = error).
"""

from __future__ import annotations

import sys
from pathlib import Path

_TASKS = Path(__file__).resolve().parents[3] / "benchmarks" / "coral_tasks"
sys.path.insert(0, str(_TASKS / "kernel_builder"))
sys.path.insert(0, str(_TASKS))

import kernel_eval  # noqa: E402
from make_configs import KERNEL_TASK  # noqa: E402

PROBLEM_DESCRIPTION = f"""
# Kernel Engineering: VLIW SIMD tree traversal

{KERNEL_TASK}
## Program shape
Keep the whole module self-contained (it is exec'd on its own): the `KernelBuilder` class, any
helpers and constants it needs, and the entry point `kernel_builder()` that returns the class.
The starting program shows the instruction format the simulator accepts.
"""

# The entry point must be the first ``def`` here: SpecEvo takes the function
# name from it (levi.utils.code_extraction.extract_fn_name).
FUNCTION_SIGNATURE = '''
def kernel_builder() -> type:
    """Return the KernelBuilder class defined in this module. The class must provide
    build_kernel(self, forest_height, n_nodes, batch_size, rounds), which fills self.instrs
    with VLIW instruction bundles, and debug_info(self)."""
    return KernelBuilder
'''

SEED_PROGRAM = (
    (_TASKS / "kernel_builder" / "initial_program.py").read_text().rstrip()
    + "\n\n\ndef kernel_builder() -> type:\n"
    '    """Return the KernelBuilder class defined in this module."""\n'
    "    return KernelBuilder\n"
)


def score_fn(kernel_builder, _inputs=None) -> dict:
    try:
        cls = kernel_builder()
        res = kernel_eval.run_builder(cls)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:2000]}
    if not res.get("is_correct"):
        return {"error": kernel_eval.feedback(res)}
    cycles = int(res["cycles"])
    return {"score": kernel_eval.score_of(cycles), "cycles": float(cycles)}
