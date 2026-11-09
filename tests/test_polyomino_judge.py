"""Mechanics of the local Polyominoes judge, on a made-up problem.

The real problem's checker and data live in a Frontier-CS checkout; these
tests swap in a tiny problem (sum two numbers) and a Python stand-in for the
testlib checker, so they pin the judge's own rules -- limits, verdicts, the
``Ratio:`` parsing and the 0..100 average -- without that checkout.
"""

from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "coral_tasks" / "polyominoes"))
import judge as J  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")

# Stand-in checker: full marks for the right sum, half for "close", testlib
# exit codes 0 (_ok), 7 (_points) and 1 (_wa).
CHECKER = textwrap.dedent("""\
    #!/usr/bin/env python3
    import sys
    inp, out, ans = sys.argv[1:4]
    a, b = map(int, open(inp).read().split())
    try:
        got = int(open(out).read().split()[0])
    except Exception:
        sys.stderr.write("wrong answer: no number\\n"); sys.exit(1)
    if got == a + b:
        sys.stderr.write("ok Ratio: 1.000000\\n"); sys.exit(0)
    if abs(got - (a + b)) == 1:
        sys.stderr.write("points Ratio: 0.500000\\n"); sys.exit(7)
    sys.stderr.write("wrong answer\\n"); sys.exit(1)
""")

GOOD = "#include <cstdio>\nint main(){long long a,b;scanf(\"%lld %lld\",&a,&b);printf(\"%lld\\n\",a+b);}\n"
OFF_BY_ONE = "#include <cstdio>\nint main(){long long a,b;scanf(\"%lld %lld\",&a,&b);printf(\"%lld\\n\",a+b+1);}\n"
SPIN = "int main(){volatile unsigned long long x=0;for(;;)x++;}\n"
CRASH = "#include <cstdlib>\nint main(){std::abort();}\n"
HOG = ("#include <vector>\n#include <cstdio>\nint main(){std::vector<char> v(300u<<20,1);"
       "long s=0;for(size_t i=0;i<v.size();i+=4096)s+=v[i];printf(\"%ld\\n\",s);}\n")


@pytest.fixture
def problem(tmp_path: Path) -> tuple[Path, Path]:
    pdir = tmp_path / "problem"
    (pdir / "testdata").mkdir(parents=True)
    for i, (a, b) in enumerate([(1, 2), (10, 20), (-5, 5), (7, 8)], start=1):
        (pdir / "testdata" / f"{i}.in").write_text(f"{a} {b}\n")
        (pdir / "testdata" / f"{i}.ans").write_text(f"{a + b}\n")
    chk = tmp_path / "chk.py"
    chk.write_text(CHECKER)
    chk.chmod(0o755)
    return pdir, chk


def _judge(src: str, problem: tuple[Path, Path]) -> J.JudgeResult:
    pdir, chk = problem
    return J.judge(src, problem_dir=pdir, checker=chk, workers=2)


def test_correct_solution_scores_100(problem):
    r = _judge(GOOD, problem)
    assert r.compiled and r.counts == {"OK": 4}
    assert r.score == pytest.approx(100.0)


def test_partial_points_are_averaged(problem):
    r = _judge(OFF_BY_ONE, problem)
    assert r.counts == {"OK": 4}
    assert r.score == pytest.approx(50.0)
    assert r.mean_ratio_ok == pytest.approx(0.5)


def test_compile_error_scores_zero(problem):
    r = _judge("int main( {", problem)
    assert not r.compiled and r.score == 0.0
    assert "compilation error" in r.error


def test_empty_source_scores_zero(problem):
    assert _judge("   \n", problem).score == 0.0


def test_cpu_limit_is_a_tle(problem):
    r = _judge(SPIN, problem)
    assert r.counts == {"TLE": 4} and r.score == 0.0


def test_crash_is_a_runtime_error(problem):
    r = _judge(CRASH, problem)
    assert r.counts == {"RE": 4} and r.score == 0.0


def test_memory_over_256mb_scores_zero(problem):
    r = _judge(HOG, problem)
    assert r.score == 0.0
    assert set(r.counts) <= {"MLE", "RE"}


def test_isolated_judge_matches_and_ignores_monkeypatching(problem, monkeypatch):
    pdir, chk = problem
    # A candidate's Python running next to the judge could do this; the
    # isolated judge runs in its own interpreter and never sees it.
    monkeypatch.setattr(J, "judge", lambda *a, **k: J.JudgeResult(100.0, True, 4))
    r = J.judge_isolated(OFF_BY_ONE, problem_dir=pdir, checker=chk)
    assert r.score == pytest.approx(50.0) and r.counts == {"OK": 4}
    assert all(isinstance(c, J.CaseResult) for c in r.cases)


def test_runaway_output_is_capped(problem):
    flood = ("#include <cstdio>\nint main(){static char b[1<<20];for(int i=0;i<(1<<20);i++)b[i]='7';"
             "for(int k=0;k<200;k++)fwrite(b,1,sizeof b,stdout);}\n")   # 200 MB
    r = _judge(flood, problem)
    assert r.score == 0.0
    assert set(r.counts) <= {"RE", "WA", "TLE"}


def test_no_output_is_wrong_answer(problem):
    r = _judge("int main(){}\n", problem)
    assert r.counts == {"WA": 4} and r.score == 0.0
    assert "WA" in r.feedback()
