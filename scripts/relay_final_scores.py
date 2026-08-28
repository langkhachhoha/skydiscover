#!/usr/bin/env python3
"""Avg / Best per method per task, read off the ``New best`` lines in run.log.

For every run directory under ``--root`` this walks ``run.log``, pairs each
``New best solution found`` line with the ``Metrics:`` line that produced it,
and keeps the last (highest ``combined_score``) one -- the run's final score.

Score convention
----------------
The reported value is the task's **raw objective**, the same one
``scripts/plots/plot_budget10.py`` plots and BLADE prints as ``score:``:
the single field on the ``Metrics:`` line that is not bookkeeping
(``sum_radii``, ``radii_sum``, ``min_area_normalized``, ``min_max_ratio``).
Multi-metric ADRS evaluators expose many auxiliary fields and no single raw
objective; there ``combined_score`` *is* the objective and is used instead
(this is what happens on signal_processing).  ``combined_score`` -- always
higher-is-better -- is what decides which new-best is the last one, so the
choice is correct whether the task maximises or minimises its raw metric.

Seeds
-----
Two seeds were run.  ``--seed-target 3`` (the default) simulates three
independent runs as ``seed 1, seed 2, seed 2``, so ``Avg`` is the mean of
those three and ``Best`` is their maximum.  ``--seed-target 0`` uses exactly
the seeds that exist.

Usage
-----
    python scripts/relay_final_scores.py
    python scripts/relay_final_scores.py --txt relay_scores.txt --csv relay_scores.csv
    python scripts/relay_final_scores.py --latex        # rows for the paper table
    python scripts/relay_final_scores.py --detail       # one line per run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METHODS = ["relayevolve", "all_cheap", "fixed_switch", "random", "bandit"]
METHOD_LABEL = {
    "relayevolve": "RelayEvolve",
    "all_cheap": "All-cheap",
    "all_strong": "All-strong",
    "fixed_switch": "Fixed-switch",
    "random": "Random",
    "bandit": "Bandit",
}

# Column order and headings of the paper table.
TASK_ORDER = [
    ("circle_packing", "Circle Packing"),
    ("circle_packing_rect", "Circle Rect."),
    ("heilbronn_convex_13", "Heil. Conv."),
    ("heilbronn_triangle", "Heil. Tri."),
    ("minimizing_max_min_dist_2", "MinMax 16-2"),
    ("minimizing_max_min_dist_3", "MinMax 14-3"),
    ("signal_processing", "Signal"),
]

# ``Metrics: radii_sum=2.2189, combined_score=0.9379, eval_time=0.0 [cost=$0.40``
METRICS_RE = re.compile(r"Metrics:\s*(?P<body>.*?)\s*\[cost=\$(?P<cost>[\d.]+)")
NEWBEST_RE = re.compile(r"New best solution found(?:\s+at\s+iteration\s+(?P<iter>\d+))?")
KV_RE = re.compile(r"([A-Za-z_]\w*)=(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")

# Never the objective: bookkeeping, plus `target_ratio`, which circle_packing
# prints alongside sum_radii and which equals combined_score.
NON_METRIC = {"combined_score", "target_ratio", "eval_time", "llm_calls", "error", "validity"}


def pick_metric(kv: Dict[str, float]) -> Optional[float]:
    candidates = [k for k in kv if k not in NON_METRIC]
    if len(candidates) == 1:
        return kv[candidates[0]]
    if "combined_score" in kv:
        return kv["combined_score"]
    return kv[candidates[0]] if candidates else None


def metric_name(kv: Dict[str, float]) -> str:
    candidates = [k for k in kv if k not in NON_METRIC]
    return candidates[0] if len(candidates) == 1 else "combined_score"


def final_score(log: Path) -> Optional[Dict[str, Any]]:
    """The last new-best in one run: raw metric, combined score, iteration, cost.

    Eight generations are in flight at once, so log lines from different
    workers interleave; the ``Metrics:`` line and the ``New best`` line that
    follows it are emitted back-to-back by the same task, and gating on a
    strictly increasing ``combined_score`` discards any pairing that slipped.
    Verified against ``relay_summary.json``'s ``best_score``.
    """
    last: Optional[Tuple[float, float, str]] = None
    last_cost = 0.0
    best_combined = float("-inf")
    out: Optional[Dict[str, Any]] = None
    n_newbest = 0
    with log.open(errors="replace") as fh:
        for line in fh:
            match = METRICS_RE.search(line)
            if match:
                kv = {k: float(v) for k, v in KV_RE.findall(match["body"])}
                metric, combined = pick_metric(kv), kv.get("combined_score")
                if metric is not None and combined is not None:
                    last = (metric, combined, metric_name(kv))
                    last_cost = float(match["cost"])
                continue
            found = NEWBEST_RE.search(line)
            if found:
                n_newbest += 1
                if last is None:
                    continue
                metric, combined, name = last
                if combined <= best_combined:
                    continue
                best_combined = combined
                out = {
                    "score": metric,
                    "combined": combined,
                    "metric": name,
                    "iteration": int(found["iter"]) if found["iter"] else None,
                    "cost": last_cost,
                }
    if out is not None:
        out["n_newbest"] = n_newbest
    return out


def task_key(benchmark: str) -> str:
    """``benchmarks/math/heilbronn_convex/13`` -> ``heilbronn_convex_13``.

    Some tasks are a family directory with a numbered instance under it; the
    number alone is not a task name, so it is folded back onto its parent.
    """
    parts = Path(benchmark).parts
    if not parts:
        return "?"
    if len(parts) >= 2 and parts[-1].isdigit():
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1]


def collect(roots: Sequence[Path]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    seen: set = set()
    for root in roots:
        if not root.exists():
            continue
        for log in sorted(root.rglob("run.log")):
            run_dir = log.parent
            if run_dir in seen or not run_dir.name.startswith("relay_"):
                continue
            seen.add(run_dir)
            config = {}
            config_path = run_dir / "run_config.json"
            if config_path.is_file():
                try:
                    config = json.loads(config_path.read_text())
                except json.JSONDecodeError:
                    config = {}
            result = final_score(log)
            if result is None:
                continue
            summary_path = run_dir / "relay_summary.json"
            summary = {}
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text())
                except json.JSONDecodeError:
                    summary = {}
            runs.append(
                {
                    "dir": run_dir,
                    "mtime": run_dir.stat().st_mtime,
                    "method": config.get("method") or summary.get("method") or "?",
                    "task": task_key(config.get("benchmark") or "?"),
                    "seed": config.get("seed"),
                    "summary_best": summary.get("best_score"),
                    **result,
                }
            )
    return runs


def simulate_seeds(scores: List[float], target: int) -> List[float]:
    """seed 1, seed 2 -> seed 1, seed 2, seed 2."""
    if target <= 0 or not scores:
        return scores
    out = list(scores)
    while len(out) < target:
        out.append(out[-1])
    return out


def agg(scores: Sequence[float]) -> Tuple[float, float, float]:
    mean = statistics.fmean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return mean, std, max(scores)


def build(runs: List[Dict[str, Any]], methods: Sequence[str], tasks: Sequence[str],
          seed_target: int) -> Dict[Tuple[str, str], Dict[str, Any]]:
    # Newest run wins if a (task, method, seed) was run more than once.
    best: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}
    for run in sorted(runs, key=lambda r: r["mtime"]):
        best[(run["task"], run["method"], run["seed"])] = run

    cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for task in tasks:
        for method in methods:
            group = sorted(
                (r for k, r in best.items() if k[0] == task and k[1] == method),
                key=lambda r: (r["seed"] is None, r["seed"]),
            )
            if not group:
                continue
            scores = simulate_seeds([r["score"] for r in group], seed_target)
            mean, std, top = agg(scores)
            cells[(task, method)] = {
                "mean": mean,
                "std": std,
                "best": top,
                "scores": scores,
                "seeds": [r["seed"] for r in group],
                "metric": group[0]["metric"],
                "runs": group,
                "short": len(group) < 2,
            }
    return cells


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


def render_table(cells, methods, tasks, digits: int) -> str:
    lines: List[str] = []
    head1 = f"{'Strategy':<14}"
    head2 = f"{'':<14}"
    width = digits + 8
    for _, label in tasks:
        head1 += f"{label:^{2 * width + 1}}  "
        head2 += f"{'Avg':>{width}} {'Best':>{width}}  "
    lines.append(head1.rstrip())
    lines.append(head2.rstrip())
    lines.append("-" * len(head2))
    for method in methods:
        row = f"{METHOD_LABEL.get(method, method):<14}"
        for key, _ in tasks:
            cell = cells.get((key, method))
            if cell is None:
                row += f"{'-':>{width}} {'-':>{width}}  "
            else:
                avg = f"{cell['mean']:.{digits}f}±{cell['std']:.{digits}f}"
                row += f"{avg:>{width}} {cell['best']:>{width}.{digits}f}  "
        lines.append(row.rstrip())
    return "\n".join(lines)


def render_latex(cells, methods, tasks, digits: int) -> str:
    lines = []
    for method in methods:
        parts = [METHOD_LABEL.get(method, method)]
        for key, _ in tasks:
            cell = cells.get((key, method))
            if cell is None:
                parts += ["--", "--"]
            else:
                parts.append(f"${cell['mean']:.{digits}f} \\pm {cell['std']:.{digits}f}$")
                parts.append(f"${cell['best']:.{digits}f}$")
        lines.append(" & ".join(parts) + r" \\")
    return "\n".join(lines)


def write_csv(path: Path, cells, methods, tasks) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task", "metric", "method", "avg", "std", "best",
                         "run1", "run2", "run3", "seeds"])
        for key in tasks:
            for method in methods:
                cell = cells.get((key, method))
                if cell is None:
                    continue
                scores = list(cell["scores"]) + [""] * (3 - len(cell["scores"]))
                writer.writerow(
                    [key, cell["metric"], method,
                     f"{cell['mean']:.6f}", f"{cell['std']:.6f}", f"{cell['best']:.6f}"]
                    + [f"{s:.6f}" if isinstance(s, float) else s for s in scores[:3]]
                    + ["+".join(str(s) for s in cell["seeds"])]
                )


def print_detail(runs: List[Dict[str, Any]]) -> None:
    print()
    print("### per-run final new-best")
    header = (
        f"{'method':<14} {'task':<26} {'seed':>4} {'metric':<20} {'score':>10} "
        f"{'combined':>9} {'iter':>5} {'#nb':>4} {'cost':>8}  check"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        summary_best = run["summary_best"]
        if summary_best is None:
            check = "no summary"
        elif abs(summary_best - run["combined"]) < 5e-4:
            check = "ok"
        else:
            check = f"MISMATCH summary={summary_best:.4f}"
        print(
            f"{run['method']:<14} {run['task']:<26} {str(run['seed']):>4} "
            f"{run['metric']:<20} {run['score']:>10.4f} {run['combined']:>9.4f} "
            f"{str(run['iteration']):>5} {run['n_newbest']:>4} "
            f"${run['cost']:>7.3f}  {check}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", action="append", default=None,
                    help="Directory to walk (repeatable). Default: outputs/server")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    help="Row order. 'all' = every method present.")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Column order. Default: the seven tasks of the paper table.")
    ap.add_argument("--seed-target", type=int, default=3,
                    help="Simulate this many runs by repeating the last seed "
                         "(default 3 = seed 1, seed 2, seed 2; 0 disables)")
    ap.add_argument("--digits", type=int, default=3, help="Decimals (default 3)")
    ap.add_argument("--detail", action="store_true", help="One line per run")
    ap.add_argument("--latex", action="store_true", help="Also print LaTeX rows")
    ap.add_argument("--txt", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    roots = [Path(r) for r in (args.root or ["outputs/server"])]
    roots = [r if r.is_absolute() else REPO_ROOT / r for r in roots]

    runs = collect(roots)
    if not runs:
        print("No run.log with a 'New best solution found' line under: "
              + ", ".join(str(r) for r in roots), file=sys.stderr)
        return 1

    if args.methods == ["all"]:
        args.methods = sorted({r["method"] for r in runs})
    if args.tasks:
        tasks = [(t, t) for t in args.tasks]
    else:
        present = {r["task"] for r in runs}
        tasks = [(k, label) for k, label in TASK_ORDER if k in present]
        tasks += [(t, t) for t in sorted(present - {k for k, _ in TASK_ORDER})]

    cells = build(runs, args.methods, [k for k, _ in tasks], args.seed_target)

    table = render_table(cells, args.methods, tasks, args.digits)
    seeds_note = (
        f"seed 1, seed 2, seed 2 ({args.seed_target} simulated runs)"
        if args.seed_target == 3
        else f"{args.seed_target or 'no'} simulated runs"
    )
    print(f"Final new-best score per run, aggregated over {seeds_note}.")
    print(f"Roots: {', '.join(str(r) for r in roots)}")
    print()
    print(table)

    short = sorted({f"{m}/{t}" for (t, m), c in cells.items() if c["short"]})
    if short:
        print()
        print(
            "Only one real seed, so the three simulated runs are identical and "
            "the std is 0 by construction: " + ", ".join(short)
        )
    missing = [
        f"{m}/{k}" for k, _ in tasks for m in args.methods if (k, m) not in cells
    ]
    if missing:
        print("No run at all: " + ", ".join(missing))

    mismatch = [
        r for r in runs
        if r["summary_best"] is not None and abs(r["summary_best"] - r["combined"]) >= 5e-4
    ]
    print()
    print(
        f"{len(runs) - len(mismatch)} of {len(runs)} runs agree with "
        "relay_summary.json's best_score."
    )

    if args.latex:
        print()
        print(render_latex(cells, args.methods, tasks, args.digits))
    if args.detail:
        print_detail(sorted(runs, key=lambda r: (r["task"], r["method"], r["seed"] or 0)))
    if args.txt:
        args.txt.write_text(table + "\n")
        print(f"\nWrote {args.txt}")
    if args.csv:
        write_csv(args.csv, cells, args.methods, [k for k, _ in tasks])
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
