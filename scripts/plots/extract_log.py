#!/usr/bin/env python3
"""
Extract new-best (score, cost) trajectories from the raw run logs of every
search method, for *any* benchmark task.

Score convention (BLADE-aligned)
---------------------------------
BLADE reports its objective directly as ``score: <metric>`` and that raw metric
is the canonical scale ("score của blade là chuẩn nhất").  The native-stack
methods (ada / oe / gepa / evox) report the same raw metric on their
``Metrics: …`` line *and* a ``combined_score`` where, per the task evaluator,
either::

    combined_score = metric / BENCHMARK      # higher-is-better tasks
    combined_score = BENCHMARK / metric      # lower-is-better tasks

``combined_score`` is therefore always "higher == better" and == 1.0 at SOTA
for math tasks, so it is used to decide what counts as a *new best* regardless
of task direction.  The plotted y-value is the **raw metric** (BLADE style)
when the log exposes a single objective field; for ADRS-style multi-metric
logs (EPLB, …) the plotted y-value is ``combined_score`` itself.

What is written next to each method's ``run.txt``
-------------------------------------------------
  * ``frontier.csv`` — new-best points: ``iter,cost,score,combined_score``
                       (only strictly-improving points; ``cost`` is the
                       cumulative $ at the moment that best was found).
  * ``summary.csv``  — one row: ``total_cost,final_score,final_combined,n_newbest``
                       where ``total_cost`` is the **full run cost** (all money
                       spent, not just up to the last improvement).

Generality
----------
Nothing here is hard-coded to a single task.  The script auto-discovers every
``result/<task>/<method>/run.txt`` and, for each task, auto-detects the raw
metric field name on the native ``Metrics:`` line (the first numeric field that
isn't ``combined_score``/``eval_time``).  BLADE's ``score:`` is the raw metric
already, and its ``combined_score`` is recomputed from the task BENCHMARK when
available (else left blank — the plotter only needs the raw score + cost).

Run:
  python scripts/plots/extract_log.py                 # all tasks under result/
  python scripts/plots/extract_log.py EPLB            # one (or more) task(s)
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "result"
BENCH_ROOT = ROOT / "benchmarks" / "math"

# ---------------------------------------------------------------------------
# Per-task BENCHMARK (read from the evaluator; only needed for BLADE's
# combined_score, which is informational).  Falls back to None gracefully.
# ---------------------------------------------------------------------------
_BENCHMARK_RE = re.compile(r"^BENCHMARK\s*=\s*(?P<val>[-\d.eE]+)", re.MULTILINE)


def task_benchmark(task: str) -> float | None:
    ev = BENCH_ROOT / task / "evaluator" / "evaluator.py"
    if not ev.exists():
        return None
    m = _BENCHMARK_RE.search(ev.read_text())
    return float(m["val"]) if m else None


# ---------------------------------------------------------------------------
# Log regexes
# ---------------------------------------------------------------------------
# BLADE:  ... [Eval #134] ... NEW BEST ★ | source: ... | score: 2.320193 | best: ... | $1.174
_BLADE_NEWBEST = re.compile(
    r"\[Eval #(?P<iter>\d+)\].*?NEW BEST.*?score:\s*(?P<score>[-\d.]+).*?\$(?P<cost>[\d.]+)\s*$"
)
_BLADE_TOTAL = re.compile(r"Total cost\s*:\s*\$(?P<cost>[\d.]+)")
_BLADE_STATUS_COST = re.compile(r"\[Status\] Cost:\s*\$(?P<cost>[\d.]+)")

# Native stack:
#   ... Metrics: radii_sum=2.2189, combined_score=0.9379, eval_time=0.0 ... [cost=$0.4050, ...]
_NATIVE_METRICS_LINE = re.compile(r"Metrics:\s*(?P<body>.*?)\s*\[cost=\$(?P<cost>[\d.]+)")
_NATIVE_NEWBEST = re.compile(
    r"New best solution found(?:\s+at\s+iteration\s+(?P<iter>\d+))?"
)
_NATIVE_ITER = re.compile(r"Iteration (?P<iter>\d+):")
_KV = re.compile(r"([A-Za-z_]\w*)=(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
_ANY_COST = re.compile(r"cost=\$(?P<cost>[\d.]+)")

# fields on the Metrics line that are never the objective metric
_NON_METRIC = {"combined_score", "eval_time", "llm_calls", "error", "validity"}


def _pick_metric(kv: dict[str, float]) -> float | None:
    """Pick the plotted objective metric from a parsed Metrics line.

    Math tasks expose a single raw metric (e.g. ``radii_sum``,
    ``min_area_normalized``) beside ``combined_score`` — that raw value is
    what BLADE reports as ``score:`` and is what we plot.

    ADRS-style evaluators (EPLB, TXN, …) emit many auxiliary fields
    (balancedness, times, speed, …) with ``combined_score`` as the actual
    search objective.  When more than one non-aux candidate is present we
    therefore fall back to ``combined_score``.
    """
    candidates = [k for k in kv if k not in _NON_METRIC]
    if len(candidates) == 1:
        return kv[candidates[0]]
    if "combined_score" in kv and len(candidates) != 1:
        return kv["combined_score"]
    return kv[candidates[0]] if candidates else None


# ---------------------------------------------------------------------------
# Parsers (return list of new-best dicts)
# ---------------------------------------------------------------------------
def parse_blade(lines: list[str], benchmark: float | None) -> list[dict]:
    out, best = [], -1.0
    for ln in lines:
        m = _BLADE_NEWBEST.search(ln)
        if not m:
            continue
        score = float(m["score"])
        if score <= best:                      # BLADE prints monotone bests
            continue
        best = score
        out.append(dict(
            iter=int(m["iter"]),
            cost=float(m["cost"]),
            score=score,
            combined_score=(score / benchmark) if benchmark else "",
        ))
    return out


def parse_native(lines: list[str], benchmark: float | None) -> list[dict]:
    """A new best = the `Metrics:` line immediately preceding a `New best …`.

    `combined_score` (always higher-is-better) decides improvement, so this is
    correct for both maximise- and minimise-objective tasks; the raw metric is
    recorded for plotting.
    """
    out, best_combined = [], -1.0
    last = None                                 # (metric, combined, cost)
    last_iter = None
    for ln in lines:
        mi = _NATIVE_ITER.search(ln)
        if mi:
            last_iter = int(mi["iter"])
        mm = _NATIVE_METRICS_LINE.search(ln)
        if mm:
            kv = {k: float(v) for k, v in _KV.findall(mm["body"])}
            metric = _pick_metric(kv)
            combined = kv.get("combined_score")
            if metric is not None and combined is not None:
                last = (metric, combined, float(mm["cost"]))
            continue
        mn = _NATIVE_NEWBEST.search(ln)
        if mn and last is not None:
            metric, combined, cost = last
            if combined <= best_combined:
                continue
            best_combined = combined
            it = (
                int(mn["iter"]) if mn["iter"] is not None
                else last_iter if last_iter is not None
                else len(out) + 1
            )
            out.append(dict(
                iter=it,
                cost=cost,
                score=metric,
                combined_score=combined,
            ))
    return out


def total_cost(method: str, lines: list[str]) -> float:
    """Full run cost = everything spent (not just up to the last new best)."""
    if method == "blade":
        for ln in reversed(lines):
            m = _BLADE_TOTAL.search(ln)
            if m:
                return float(m["cost"])
        costs = [float(m["cost"]) for ln in lines
                 for m in [_BLADE_STATUS_COST.search(ln)] if m]
        return max(costs) if costs else 0.0
    costs = [float(m["cost"]) for ln in lines
             for m in [_ANY_COST.search(ln)] if m]
    return max(costs) if costs else 0.0


# method dir name -> parser.  Folder names vary slightly per task
# (circle_packing_rect uses "ada", heilbronn_triangle uses "adaevo"); both
# native-stack aliases map to the same parser.
PARSERS = {
    "blade": parse_blade,
    "ada": parse_native,
    "adaevo": parse_native,
    "oe": parse_native,
    "gepa": parse_native,
    "evox": parse_native,
}


def main(tasks: list[str] | None = None) -> None:
    task_dirs = sorted(p for p in RESULT_ROOT.iterdir() if p.is_dir())
    if tasks:
        wanted = set(tasks)
        task_dirs = [p for p in task_dirs if p.name in wanted]
        missing = wanted - {p.name for p in task_dirs}
        if missing:
            raise SystemExit(f"Unknown task(s) under result/: {sorted(missing)}")

    for task_dir in task_dirs:
        task = task_dir.name
        benchmark = task_benchmark(task)
        bstr = f"{benchmark:.6g}" if benchmark else "n/a"
        print(f"\n=== task: {task} (BENCHMARK={bstr}) ===")
        for method, parser in PARSERS.items():
            run = task_dir / method / "run.txt"
            if not run.exists():
                continue
            text = run.read_text()
            if not text.strip():
                print(f"  {method:6s}: empty run.txt — skipped")
                continue
            lines = text.splitlines()
            rows = parser(lines, benchmark)
            tcost = total_cost(method, lines)

            with (run.parent / "frontier.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["iter", "cost", "score", "combined_score"])
                w.writeheader()
                w.writerows(rows)

            final = rows[-1] if rows else dict(score=0.0, combined_score="")
            with (run.parent / "summary.csv").open("w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=["total_cost", "final_score", "final_combined", "n_newbest"]
                )
                w.writeheader()
                fc = final["combined_score"]
                w.writerow(dict(
                    total_cost=round(tcost, 4),
                    final_score=round(final["score"], 6),
                    final_combined=round(fc, 6) if isinstance(fc, float) else "",
                    n_newbest=len(rows),
                ))

            if rows:
                last = rows[-1]
                pct = (f"{100*last['combined_score']:.1f}% SOTA"
                       if isinstance(last["combined_score"], float) and benchmark
                       else "—")
                print(
                    f"  {method:6s}: {len(rows):3d} new-best pts | "
                    f"final score={last['score']:.4f} ({pct}) | "
                    f"last-best @ ${last['cost']:.2f} | TOTAL run cost ${tcost:.2f}"
                )
            else:
                print(f"  {method:6s}: no new-best points parsed (!)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
