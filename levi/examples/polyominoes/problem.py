"""
Polyominoes packing (Frontier-CS algorithmic #0) -- CORAL's stress-test task, for SpecEvo.

SpecEvo evolves Python, so the candidate is a Python module whose
``solution()`` returns the full C++17 source; that source is judged exactly
like the baselines' ``initial_program.cpp`` candidates, by
benchmarks/coral_tasks/polyominoes/judge.py (score 0..100, the mean
per-case ratio over the 70 official tests, 2 s CPU / 256 MB per case).
"""

from __future__ import annotations

import sys
from pathlib import Path

_TASKS = Path(__file__).resolve().parents[3] / "benchmarks" / "coral_tasks"
sys.path.insert(0, str(_TASKS / "polyominoes"))
sys.path.insert(0, str(_TASKS))

import judge  # noqa: E402
from make_configs import POLYOMINO_TASK, polyomino_statement  # noqa: E402

PROBLEM_DESCRIPTION = f"""
# Pack the Polyominoes (Frontier-CS algorithmic #0)

{POLYOMINO_TASK}
## How your answer is delivered
Write the C++17 program, then deliver it as ONE Python function `solution()` whose body is a
single `return r\"\"\"...\"\"\"` holding the complete C++ source (see the signature below). Put the
whole program inside that raw string -- no other Python, no separate variables. Only the C++
is compiled and judged. Do not use triple double-quotes inside the C++ code.

# Problem statement (statement.txt)

{polyomino_statement()}
"""

# The C++ lives inside the function's own return statement: in a smoke run,
# a module-level CPP_SOURCE plus a separate `def solution` lost one half or the
# other in 6 of 14 mutation-model candidates.
FUNCTION_SIGNATURE = '''
def solution() -> str:
    """Complete C++17 solution (g++ -O2 -std=gnu++17, 2 s CPU / 256 MB per test)."""
    return r"""
#include <bits/stdc++.h>
using namespace std;
int main() { /* read stdin, write the packing to stdout */ }
"""
'''

# The same empty skeleton CORAL and the baselines start from.
SEED_PROGRAM = (
    "def solution() -> str:\n"
    '    """Complete C++17 solution (g++ -O2 -std=gnu++17, 2 s CPU / 256 MB per test)."""\n'
    '    return r"""\n'
    + (_TASKS / "polyominoes" / "initial_program.cpp").read_text()
    + '"""\n'
)


def score_fn(solution, _inputs=None) -> dict:
    try:
        src = solution()
    except Exception as e:
        return {"error": f"solution() raised {type(e).__name__}: {e}"}
    if not isinstance(src, str):
        return {"error": f"solution() must return the C++ source as str, got {type(src).__name__}"}
    try:
        res = judge.judge_isolated(src)   # this process also runs the candidate's Python
    except Exception as e:
        return {"error": f"judge error: {type(e).__name__}: {e}"}
    metrics = {
        "score": res.score,
        "cases_ok": float(res.counts.get("OK", 0)),
        "cases_tle": float(res.counts.get("TLE", 0)),
        "cases_wa": float(res.counts.get("WA", 0)),
        "cases_re": float(res.counts.get("RE", 0) + res.counts.get("MLE", 0)),
        "mean_ratio_valid": res.mean_ratio_ok,
        "max_cpu_s": res.max_cpu_s,
        "execution_time": res.seconds,
    }
    if not res.compiled:
        return {"error": res.error[:2000]}
    if res.counts.get("OK", 0) == 0:
        # Nothing valid: a failure, so its cause reaches the Advisor's error log.
        return {**metrics, "score": 0.0, "error": res.feedback(max_lines=5)[:2000]}
    return metrics
