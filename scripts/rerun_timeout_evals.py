#!/usr/bin/env python3
"""Re-run the candidates that only failed because the evaluator timed out.

A relay run records every candidate in ``eval_code_log.jsonl``.  When the
containerized evaluator hits its wall-clock limit it returns
``metrics={"error": 0.0, "timeout": True}`` (see
``skydiscover/evaluation/container_evaluator.py``), and the controller turns
that into the generic ``Evaluator failed after 1 attempts: Evaluation failed
(validity=0)``.  Those records say nothing about the code — only that the box
was busy.  Code that genuinely raised, overlapped or fell outside the
rectangle carries the real message instead, and is left alone here.

This script picks out *only* the timeout records, evaluates each one again
with the real ``circle_packing_rect`` evaluator under a shorter limit
(default 60s), and writes the outcome back over the same record, in the shape
the original writer would have produced:

``ok``
    ``metrics={"radii_sum": .., "combined_score": .., "eval_time": ..}``,
    ``score=combined_score``, ``error=None``
``evaluation_failed`` (the code raised or the packing was invalid)
    ``metrics={"combined_score": 0.0, "error": <msg>}``,
    ``error="Evaluator failed after 1 attempts: <msg>"``
``evaluation_failed`` (still too slow)
    ``metrics={"error": 0.0, "timeout": True}`` — unchanged in shape, so a
    record that times out again is indistinguishable from before.

``eval_time_s`` is refreshed to the measured wall time.  Every other field
(``llm_time_s``, ``wall_clock_s``, ``iteration``, ids, ``code``,
``llm_response``) belongs to the original run and is preserved untouched.

Each program runs in its own process group in a private temp dir, with BLAS
threading pinned to one thread so that ``--jobs`` workers stay comparable to
each other and to a serial run.

Usage::

    python scripts/rerun_timeout_evals.py relay_cpr400 --jobs 8
    python scripts/rerun_timeout_evals.py relay_cpr400 --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATOR = REPO_ROOT / "benchmarks/math/circle_packing_rect/evaluator/evaluator.py"


def can_evaluate(python: str) -> bool:
    """True if *python* can import what the evaluator needs."""
    try:
        return subprocess.run(
            [python, "-c", "import numpy, scipy"],
            capture_output=True,
            timeout=60,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def default_python() -> str:
    """The interpreter to evaluate candidates with.

    The evaluator needs numpy and scipy, which a bare system python often
    lacks.  The environment the user activated wins -- conda names it
    CONDA_PREFIX and virtualenv names it VIRTUAL_ENV -- and a project venv is
    only a fallback for when none of those can do the job.  Each candidate is
    probed rather than assumed, so an env that merely exists but is missing
    scipy does not silently fail 1443 times.
    """
    candidates = [sys.executable]
    for var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        prefix = os.environ.get(var)
        if prefix:
            candidates.append(str(Path(prefix) / "bin/python"))
    candidates += [str(REPO_ROOT / ".venv/bin/python"), str(REPO_ROOT / "venv/bin/python")]

    seen = set()
    for candidate in candidates:
        if candidate in seen or not Path(candidate).is_file():
            continue
        seen.add(candidate)
        if can_evaluate(candidate):
            return candidate
    return sys.executable

# The message the controller substitutes when the evaluator came back with no
# error string of its own -- which is what a timeout looks like.
GENERIC_FAILURE = "Evaluation failed (validity=0)"

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ----------------------------------------------------------------------------
# Selecting the records to re-run
# ----------------------------------------------------------------------------


def is_timeout_record(rec: dict) -> bool:
    """True only for a record whose evaluation hit the wall-clock limit.

    Deliberately narrow: a code that raised, produced overlapping circles or
    escaped the rectangle has a real error message and must not be re-run.
    """
    metrics = rec.get("metrics") or {}
    return metrics.get("timeout") is True and bool(rec.get("code"))


def find_logs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("eval_code_log.jsonl"))


def label_for(path: Path, root: Path) -> str:
    """`relayevolve/seed1` -- which baseline and seed this log belongs to."""
    try:
        rel = path.relative_to(root if root.is_dir() else root.parent)
    except ValueError:
        rel = path
    return str(rel.parent) if str(rel.parent) not in (".", "") else path.parent.name


# ----------------------------------------------------------------------------
# Running one program through the real evaluator
# ----------------------------------------------------------------------------


def run_one(code: str, evaluator: Path, timeout: float, python: str) -> tuple[str, dict, float]:
    """Evaluate *code*, returning ``(outcome, payload, wall_time)``.

    ``outcome`` is ``"ok"``, ``"error"`` or ``"timeout"``.  ``payload`` is the
    evaluator's parsed JSON for the first two and ``{}`` for a timeout.
    """
    workdir = Path(tempfile.mkdtemp(prefix="rerun_eval_"))
    try:
        program = workdir / "program.py"
        program.write_text(code)

        env = dict(os.environ)
        # One thread per evaluation: numpy/scipy would otherwise fan out over
        # every core and make parallel workers race each other for CPU, which
        # would show up as spurious timeouts.
        env.update(
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
        )

        start = time.time()
        proc = subprocess.Popen(
            [python, str(evaluator), str(program)],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            start_new_session=True,  # its own process group, so we can kill children too
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return "timeout", {}, time.time() - start
        elapsed = time.time() - start

        payload = _parse_payload(stdout)
        if payload is None:
            msg = f"evaluator produced no JSON (exit code {proc.returncode})"
            return "error", {"artifacts": {"error": msg}}, elapsed

        artifacts = payload.get("artifacts") or {}
        metrics = payload.get("metrics") or {}
        if payload.get("status") == "error" or "error" in artifacts:
            return "error", payload, elapsed
        if "radii_sum" not in metrics:
            payload.setdefault("artifacts", {})["error"] = "evaluator returned no radii_sum"
            return "error", payload, elapsed
        return "ok", payload, elapsed
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the whole process group, then reap, so nothing is left running."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.communicate(timeout=10)
    except Exception:
        pass


def _parse_payload(stdout: str) -> dict | None:
    """The wrapper prints one JSON object on stdout; take the last valid one."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ----------------------------------------------------------------------------
# Writing the outcome back into the record
# ----------------------------------------------------------------------------


def apply_outcome(rec: dict, outcome: str, payload: dict, elapsed: float, timeout: float) -> None:
    """Overwrite *rec* in place with the re-run result, in the original shape."""
    if outcome == "ok":
        metrics = payload.get("metrics") or {}
        combined = float(metrics.get("combined_score", 0.0))
        rec["status"] = "ok"
        rec["score"] = combined
        rec["metrics"] = {
            "radii_sum": float(metrics["radii_sum"]),
            "combined_score": combined,
            "eval_time": float(metrics.get("eval_time", 0.0)),
        }
        rec["error"] = None
    elif outcome == "error":
        artifacts = payload.get("artifacts") or {}
        msg = artifacts.get("error") or (payload.get("metrics") or {}).get("error")
        msg = str(msg) if msg else GENERIC_FAILURE
        rec["status"] = "evaluation_failed"
        rec["score"] = 0.0
        rec["metrics"] = {"combined_score": 0.0, "error": msg}
        rec["error"] = f"Evaluator failed after 1 attempts: {msg}"
    else:  # still too slow -- leave it looking exactly like a timeout
        rec["status"] = "evaluation_failed"
        rec["score"] = 0.0
        rec["metrics"] = {"error": 0.0, "timeout": True}
        rec["error"] = f"Evaluator failed after 1 attempts: {GENERIC_FAILURE}"

    rec["eval_time_s"] = round(elapsed, 3)


def mark(rec: dict, outcome: str, timeout: float, previous: float | None) -> None:
    """Record where this line's numbers came from, so a re-run is traceable."""
    rec["rerun"] = {
        "reason": "timeout",
        "timeout_s": timeout,
        "outcome": outcome,
        "previous_eval_time_s": previous,
    }


# ----------------------------------------------------------------------------
# Driving one file
# ----------------------------------------------------------------------------


def process_file(path: Path, args: argparse.Namespace, budget: list[int]) -> dict:
    label = label_for(path, Path(args.root))
    lines = path.read_text().splitlines()

    records: list[dict] = []
    for line in lines:
        line = line.strip()
        if line:
            records.append(json.loads(line))

    targets = [i for i, r in enumerate(records) if is_timeout_record(r)]
    if args.limit is not None:
        room = max(0, args.limit - budget[0])
        targets = targets[:room]
        budget[0] += len(targets)

    stats = {"label": label, "path": str(path), "timeouts": len(targets),
             "ok": 0, "error": 0, "timeout": 0}
    if not targets:
        log(f"[{label}] no timeout records to re-run")
        return stats

    log(f"[{label}] re-running {len(targets)} timed-out candidates "
        f"(limit {args.timeout:.0f}s each, {args.jobs} in parallel)")

    done = [0]

    def work(idx: int) -> tuple[int, str, dict, float]:
        outcome, payload, elapsed = run_one(
            records[idx]["code"], args.evaluator, args.timeout, args.python
        )
        with _print_lock:
            done[0] += 1
            n = done[0]
        score = ""
        if outcome == "ok":
            score = f" score={(payload.get('metrics') or {}).get('combined_score', 0):.4f}"
        log(f"[{label}] {n}/{len(targets)} line {idx + 1}: {outcome}"
            f"{score} ({elapsed:.1f}s)")
        return idx, outcome, payload, elapsed

    if args.backup and not args.dry_run:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    # Checkpoint as we go: a full pass over every log takes hours, and a
    # record already rewritten is no longer a timeout, so an interrupted run
    # picks up where it stopped instead of starting the file again.
    since_flush = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(work, idx) for idx in targets]
        try:
            for future in as_completed(futures):
                idx, outcome, payload, elapsed = future.result()
                previous = records[idx].get("eval_time_s")
                apply_outcome(records[idx], outcome, payload, elapsed, args.timeout)
                if not args.no_mark:
                    mark(records[idx], outcome, args.timeout, previous)
                stats[outcome] += 1
                since_flush += 1
                if not args.dry_run and args.flush_every and since_flush >= args.flush_every:
                    write_records(path, records)
                    since_flush = 0
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            if not args.dry_run and since_flush:
                write_records(path, records)
                log(f"[{label}] interrupted -- progress saved to {path}")
            raise

    if args.dry_run:
        log(f"[{label}] dry run -- {path} not modified")
        return stats

    write_records(path, records)
    log(f"[{label}] wrote {path}")
    return stats


def write_records(path: Path, records: list[dict]) -> None:
    """Rewrite the log atomically: temp file in the same dir, then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Re-run only the timed-out candidates in eval_code_log.jsonl files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("root", nargs="?", default="relay_cpr400",
                   help="directory holding <method>/<seed>/eval_code_log.jsonl, "
                        "or one such file (default: relay_cpr400)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="seconds allowed per candidate (default: 60)")
    p.add_argument("--jobs", "-j", type=int, default=4,
                   help="candidates evaluated in parallel (default: 4)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many candidates in total -- for a smoke test")
    p.add_argument("--dry-run", action="store_true",
                   help="evaluate and report, but do not touch the files")
    p.add_argument("--no-backup", dest="backup", action="store_false",
                   help="do not keep the original as <file>.bak")
    p.add_argument("--flush-every", type=int, default=25,
                   help="rewrite the log after this many candidates, so an interrupted "
                        "run keeps its progress (0 disables; default: 25)")
    p.add_argument("--no-mark", action="store_true",
                   help="do not add the 'rerun' provenance field to rewritten records")
    p.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR,
                   help=f"evaluator to run (default: {DEFAULT_EVALUATOR})")
    p.add_argument("--python", default=None,
                   help="interpreter used to run each candidate; it needs numpy and "
                        "scipy (default: the active environment, else a project venv)")
    p.add_argument("--only", action="append", default=None,
                   help="only logs whose path contains this substring; repeatable "
                        "(e.g. --only relayevolve --only seed1)")
    args = p.parse_args()

    args.python = args.python or default_python()
    args.evaluator = args.evaluator.resolve()
    if not args.evaluator.is_file():
        print(f"error: evaluator not found: {args.evaluator}", file=sys.stderr)
        return 2

    root = Path(args.root)
    if not root.exists():
        print(f"error: no such path: {root}", file=sys.stderr)
        return 2

    logs = find_logs(root)
    if args.only:
        logs = [f for f in logs if any(s in str(f) for s in args.only)]
    if not logs:
        print(f"error: no eval_code_log.jsonl found under {root}", file=sys.stderr)
        return 2

    if not can_evaluate(args.python):
        probe = subprocess.run(
            [args.python, "-c", "import numpy, scipy"], capture_output=True, text=True
        )
        print(f"error: {args.python} cannot import numpy/scipy -- every candidate "
              f"would fail identically.\n{probe.stderr.strip()}\n"
              f"Activate the right environment, or pass one with --python.", file=sys.stderr)
        return 2

    print(f"python    : {args.python}")
    print(f"evaluator : {args.evaluator}")
    print(f"logs      : {len(logs)}")
    print(f"timeout   : {args.timeout:.0f}s per candidate, {args.jobs} in parallel")
    if args.dry_run:
        print("mode      : DRY RUN (no files written)")
    print()

    budget = [0]
    started = time.time()
    all_stats = []
    for path in logs:
        if args.limit is not None and budget[0] >= args.limit:
            break
        all_stats.append(process_file(path, args, budget))

    print("\n=== summary ===")
    print(f"{'log':<28} {'timeouts':>9} {'-> ok':>7} {'-> error':>9} {'-> timeout':>11}")
    tot = {"timeouts": 0, "ok": 0, "error": 0, "timeout": 0}
    for s in all_stats:
        if not s["timeouts"]:
            continue
        print(f"{s['label']:<28} {s['timeouts']:>9} {s['ok']:>7} {s['error']:>9} {s['timeout']:>11}")
        for k in tot:
            tot[k] += s[k]
    print(f"{'TOTAL':<28} {tot['timeouts']:>9} {tot['ok']:>7} {tot['error']:>9} {tot['timeout']:>11}")
    print(f"\nelapsed: {time.time() - started:.1f}s")
    if tot["timeouts"]:
        recovered = 100.0 * tot["ok"] / tot["timeouts"]
        print(f"recovered into a real score: {tot['ok']}/{tot['timeouts']} ({recovered:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
