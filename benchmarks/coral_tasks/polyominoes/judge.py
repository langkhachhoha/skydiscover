"""Local judge for Frontier-CS algorithmic problem 0, "Pack the Polyominoes".

The official judge runs go-judge inside Docker. This module reproduces what
that judge does, with plain g++ and subprocesses, so every method in the
comparison is scored by the same code without Docker:

  * compile the solution with ``g++ -O2 -pipe -std=gnu++17`` (judge_engine.js /
    gojudge.js flags) and the testlib checker ``chk.cc`` with the same flags;
  * run every test case under the problem's limits (config.yaml: 2 s CPU,
    256 MB), counting a case as 0 on TLE, MLE, a crash or a non-scoring
    checker verdict;
  * read the case score from the checker's ``Ratio: <x>`` line, only when the
    checker exited with testlib's ``_ok`` (0) or ``_points`` (7) code -- the
    same rule as ``applyScoreRatio`` in judge_engine.js;
  * the final score is ``100 * mean(case ratios)``, 0..100, higher is better.

Limits are enforced on CPU time (RLIMIT_CPU plus the child's rusage), not
wall-clock, so a busy machine does not turn into spurious TLEs; the wall
clock only backstops a solution that sleeps or blocks.

Data comes from a Frontier-CS checkout: ``$FRONTIER_CS_DIR`` (default
``/home/ubuntu/projects/Frontier-CS``, see scripts/server/setup_coral.sh).
"""

from __future__ import annotations

import hashlib
import os
import re
import resource
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

FRONTIER_CS_DIR = Path(os.environ.get("FRONTIER_CS_DIR", "/home/ubuntu/projects/Frontier-CS"))
PROBLEM_ID = "0"
PROBLEM_DIR = FRONTIER_CS_DIR / "algorithmic" / "problems" / PROBLEM_ID
TESTLIB_DIR = FRONTIER_CS_DIR / "algorithmic" / "judge" / "include"

TIME_LIMIT_S = 2.0          # problems/0/config.yaml: time: 2s
MEMORY_LIMIT_MB = 256       # problems/0/config.yaml: memory: 256m
WALL_LIMIT_S = 20.0         # backstop only; the CPU limit is the real one
COMPILE_TIMEOUT_S = 120.0
CHECKER_TIMEOUT_S = 60.0
GXX = ["g++", "-O2", "-pipe", "-std=gnu++17"]
OUTPUT_LIMIT_BYTES = 64 << 20

TESTLIB_OK = 0
TESTLIB_POINTS = 7
_RATIO = re.compile(r"Ratio: ([\d.]+)")

# Parallel test cases per evaluation. Each is single-threaded and capped at
# 2 s of CPU, so 70 cases take ~70*t/JUDGE_WORKERS seconds.
JUDGE_WORKERS = int(os.environ.get("POLYOMINO_JUDGE_WORKERS", "4"))


def statement(problem_dir: Path = PROBLEM_DIR) -> str:
    return (problem_dir / "statement.txt").read_text()


def test_cases(problem_dir: Path = PROBLEM_DIR) -> list[tuple[Path, Path]]:
    tdir = problem_dir / "testdata"
    ins = sorted(tdir.glob("*.in"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    return [(p, p.with_suffix(".ans")) for p in ins]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _checker_binary(problem_dir: Path = PROBLEM_DIR, testlib_dir: Path = TESTLIB_DIR) -> Path:
    """Compile chk.cc once per checker version; cached under ~/.cache."""
    src = problem_dir / "chk.cc"
    digest = hashlib.sha256(src.read_bytes() + (testlib_dir / "testlib.h").read_bytes()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "polyomino_judge"
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / f"chk_{digest}"
    if binary.exists():
        return binary
    tmp = cache / f"chk_{digest}.{os.getpid()}.tmp"
    proc = subprocess.run(
        [*GXX, str(src), "-I", str(testlib_dir), "-o", str(tmp)],
        capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S * 2,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"checker failed to compile:\n{proc.stderr[-2000:]}")
    os.replace(tmp, binary)  # atomic: concurrent evaluations may race here
    return binary


def _compile_solution(source: str, workdir: Path) -> tuple[Path | None, str]:
    src = workdir / "solution.cpp"
    src.write_text(source)
    exe = workdir / "solution"
    try:
        proc = subprocess.run(
            [*GXX, str(src), "-o", str(exe)],
            capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, f"compilation timed out after {COMPILE_TIMEOUT_S:.0f}s"
    if proc.returncode != 0:
        return None, "compilation error:\n" + proc.stderr[-3000:]
    return exe, ""


# ---------------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------------


def _limits() -> None:  # runs in the child between fork and exec
    cpu = int(TIME_LIMIT_S) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    # Address-space cap well above the RSS limit: the RSS check after the run
    # is the judge's rule; this only stops a runaway allocation early.
    vmem = (MEMORY_LIMIT_MB * 4) << 20
    resource.setrlimit(resource.RLIMIT_AS, (vmem, vmem))
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # A packing of 10^4 pieces is well under 1 MB; stop a runaway writer
    # before it fills the disk (the write fails -> RE / WA on that case).
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT_BYTES, OUTPUT_LIMIT_BYTES))
    os.setsid()


@dataclass
class CaseResult:
    name: str
    status: str            # OK | TLE | MLE | RE | WA | CHECKER_ERROR
    ratio: float = 0.0
    cpu_s: float = 0.0
    rss_mb: float = 0.0
    message: str = ""


def _run_case(exe: Path, checker: Path, inp: Path, ans: Path, workdir: Path) -> CaseResult:
    name = inp.stem
    out = workdir / f"{name}.out"
    with inp.open("rb") as fin, out.open("wb") as fout:
        proc = subprocess.Popen(
            [str(exe)], stdin=fin, stdout=fout, stderr=subprocess.DEVNULL,
            preexec_fn=_limits, cwd=workdir,
        )
        deadline = time.monotonic() + WALL_LIMIT_S
        status_code, rusage = None, None
        while True:
            pid, status_code, rusage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() > deadline:
                try:
                    os.killpg(proc.pid, 9)
                except ProcessLookupError:
                    pass
                pid, status_code, rusage = os.wait4(proc.pid, 0)
                proc.returncode = -9
                return CaseResult(name, "TLE", message=f"wall clock > {WALL_LIMIT_S:.0f}s")
            time.sleep(0.01)
        proc.returncode = os.waitstatus_to_exitcode(status_code)
    cpu = rusage.ru_utime + rusage.ru_stime
    rss_mb = rusage.ru_maxrss / 1024.0
    if cpu > TIME_LIMIT_S or proc.returncode in (-24, -9):   # SIGXCPU / SIGKILL from the CPU cap
        return CaseResult(name, "TLE", cpu_s=cpu, rss_mb=rss_mb, message=f"cpu {cpu:.2f}s")
    if rss_mb > MEMORY_LIMIT_MB:
        return CaseResult(name, "MLE", cpu_s=cpu, rss_mb=rss_mb, message=f"rss {rss_mb:.0f}MB")
    if proc.returncode != 0:
        kind = "MLE" if proc.returncode == -6 and rss_mb > MEMORY_LIMIT_MB * 0.9 else "RE"
        return CaseResult(name, kind, cpu_s=cpu, rss_mb=rss_mb, message=f"exit {proc.returncode}")
    try:
        chk = subprocess.run(
            [str(checker), str(inp), str(out), str(ans)],
            capture_output=True, text=True, timeout=CHECKER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return CaseResult(name, "CHECKER_ERROR", cpu_s=cpu, rss_mb=rss_mb, message="checker timed out")
    finally:
        try:
            out.unlink()
        except OSError:
            pass
    msg = (chk.stderr or "") + (chk.stdout or "")
    if chk.returncode in (TESTLIB_OK, TESTLIB_POINTS):
        m = _RATIO.search(msg)
        ratio = float(m.group(1)) if m else (1.0 if chk.returncode == TESTLIB_OK else 0.0)
        return CaseResult(name, "OK", ratio=ratio, cpu_s=cpu, rss_mb=rss_mb, message=msg.strip()[:200])
    return CaseResult(name, "WA", cpu_s=cpu, rss_mb=rss_mb, message=msg.strip()[:300])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    score: float                       # 0..100
    compiled: bool
    n_cases: int
    counts: dict[str, int] = field(default_factory=dict)
    mean_ratio_ok: float = 0.0         # mean ratio over cases that produced a valid packing
    max_cpu_s: float = 0.0
    seconds: float = 0.0
    error: str = ""
    cases: list[CaseResult] = field(default_factory=list)

    def feedback(self, max_lines: int = 12) -> str:
        if not self.compiled:
            return self.error
        head = (f"score {self.score:.3f}/100 over {self.n_cases} cases | "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
                + f" | mean ratio of valid cases {self.mean_ratio_ok:.4f} | max cpu {self.max_cpu_s:.2f}s"
                f" (limit {TIME_LIMIT_S:.0f}s)")
        bad = [c for c in self.cases if c.status != "OK"][:max_lines]
        lines = [f"  case {c.name}: {c.status} {c.message}".rstrip() for c in bad]
        return "\n".join([head, *lines])


def judge(
    source: str,
    *,
    workers: int | None = None,
    problem_dir: Path = PROBLEM_DIR,
    testlib_dir: Path = TESTLIB_DIR,
    checker: Path | None = None,
) -> JudgeResult:
    """Judge one C++ source. ``checker`` overrides compiling ``chk.cc``."""
    t0 = time.monotonic()
    cases = test_cases(problem_dir)
    if not cases:
        raise RuntimeError(f"no test cases under {problem_dir}/testdata (run scripts/server/setup_coral.sh)")
    if not source or not source.strip():
        return JudgeResult(0.0, False, len(cases), error="empty solution", seconds=0.0)
    checker = checker or _checker_binary(problem_dir, testlib_dir)
    workdir = Path(tempfile.mkdtemp(prefix="polyomino_"))
    try:
        exe, err = _compile_solution(source, workdir)
        if exe is None:
            return JudgeResult(0.0, False, len(cases), error=err, seconds=time.monotonic() - t0)
        with ThreadPoolExecutor(max_workers=max(1, workers or JUDGE_WORKERS)) as pool:
            results = list(pool.map(lambda c: _run_case(exe, checker, c[0], c[1], workdir), cases))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    ok = [r.ratio for r in results if r.status == "OK"]
    return JudgeResult(
        score=100.0 * sum(r.ratio for r in results) / max(1, len(results)),
        compiled=True,
        n_cases=len(results),
        counts=counts,
        mean_ratio_ok=(sum(ok) / len(ok)) if ok else 0.0,
        max_cpu_s=max((r.cpu_s for r in results), default=0.0),
        seconds=time.monotonic() - t0,
        cases=results,
    )


def judge_isolated(source: str, *, timeout: float = 900.0, problem_dir: Path | None = None,
                   checker: Path | None = None) -> JudgeResult:
    """``judge`` in a fresh interpreter, for callers that run untrusted Python
    (SpecEvo's candidate wrapper lives in the same process as this module and
    could otherwise monkeypatch it)."""
    import json
    import sys

    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as fh:
        fh.write(source)
        src = fh.name
    kwargs = ""
    if problem_dir is not None:
        kwargs += f", problem_dir=Path({str(problem_dir)!r})"
    if checker is not None:
        kwargs += f", checker=Path({str(checker)!r})"
    script = (
        "import dataclasses, json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "import judge\n"
        f"r = judge.judge(open({src!r}).read(){kwargs})\n"
        "print('JUDGE_RESULT ' + json.dumps(dataclasses.asdict(r)))\n"
    )
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True,
                              timeout=timeout, env={**os.environ})
    except subprocess.TimeoutExpired:
        return JudgeResult(0.0, False, 0, error=f"judge timed out after {timeout:.0f}s")
    finally:
        os.unlink(src)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("JUDGE_RESULT "):
            d = json.loads(line[len("JUDGE_RESULT "):])
            d["cases"] = [CaseResult(**c) for c in d.get("cases", [])]
            return JudgeResult(**d)
    raise RuntimeError(f"judge process failed (exit {proc.returncode}): {(proc.stderr or '').strip()[-1500:]}")


if __name__ == "__main__":  # python judge.py solution.cpp
    import sys

    res = judge(Path(sys.argv[1]).read_text())
    print(res.feedback(max_lines=70))
    print(f"judged in {res.seconds:.1f}s")
