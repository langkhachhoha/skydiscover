#!/usr/bin/env python3
"""List and aggregate RelayEvolve / baseline runs — no session name needed.

Every run writes ``run_config.json`` (what was asked for), ``relay_summary.json``
(what happened) and ``best/best_program_info.json`` (the authoritative test
score) into its own directory, so the results can always be recovered by
walking ``outputs/`` even when the tmux session that produced them is long gone.

    python scripts/relay_summarize.py                       # every run, newest first
    python scripts/relay_summarize.py --benchmark circle_packing
    python scripts/relay_summarize.py --agg                 # mean +/- std over seeds
    python scripts/relay_summarize.py --csv relay.csv
    python scripts/relay_summarize.py --path outputs/server/<run-id>   # one run, in full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def collect(roots: List[Path]) -> List[Dict[str, Any]]:
    """Every directory under *roots* that holds a relay_summary.json."""
    runs: List[Dict[str, Any]] = []
    seen: set = set()
    for root in roots:
        if not root.exists():
            continue
        for summary_path in sorted(root.rglob("relay_summary.json")):
            run_dir = summary_path.parent
            if run_dir in seen:
                continue
            seen.add(run_dir)
            runs.append(_describe(run_dir))
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def _describe(run_dir: Path) -> Dict[str, Any]:
    summary = _load(run_dir / "relay_summary.json")
    config = _load(run_dir / "run_config.json")
    best = _load(run_dir / "best" / "best_program_info.json")
    metrics = best.get("metrics") or {}

    # The test-mode re-evaluation is the authoritative number; fall back to the
    # search-time best when a run was killed before it ran.
    score = metrics.get("test_combined_score")
    if score is None:
        score = metrics.get("combined_score", summary.get("best_score"))

    totals = summary.get("totals") or {}
    tiers = summary.get("llm_calls_by_tier") or {}
    benchmark = config.get("benchmark") or "?"

    return {
        "dir": run_dir,
        "mtime": run_dir.stat().st_mtime,
        "method": summary.get("method") or config.get("method") or "?",
        "benchmark": Path(benchmark).name,
        "seed": config.get("seed"),
        "score": score,
        "test_score": metrics.get("test_combined_score"),
        "generations": summary.get("iterations_used"),
        "cost": totals.get("total_cost_usd"),
        "budget": summary.get("budget_usd"),
        "cheap_calls": tiers.get("cheap"),
        "strong_calls": tiers.get("strong"),
        "handoff": summary.get("handoff_iteration"),
        "handoff_reason": summary.get("handoff_reason") or summary.get("switch_reason"),
        "budget_stop": summary.get("budget_stop_triggered"),
        "wall_s": summary.get("wall_clock_s"),
        "prompt_tokens": totals.get("total_prompt_tokens"),
        "completion_tokens": totals.get("total_completion_tokens"),
        "curation": config.get("curation"),
        "relay_control": config.get("relay_control"),
    }


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "-"
    if spec and isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def print_table(runs: List[Dict[str, Any]]) -> None:
    if not runs:
        print("No runs found. Point --root at the directory the runs were written to.")
        return
    header = (
        f"{'method':<14} {'benchmark':<22} {'seed':>4} {'score':>12} {'gens':>5} "
        f"{'cost':>8} {'cheap':>6} {'strong':>6} {'handoff':>8}  run dir"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        stop = " [$]" if r["budget_stop"] else ""
        print(
            f"{r['method']:<14} {r['benchmark']:<22} {_fmt(r['seed']):>4} "
            f"{_fmt(r['score'], '.6f'):>12} {_fmt(r['generations']):>5} "
            f"{'$' + _fmt(r['cost'], '.3f'):>8} {_fmt(r['cheap_calls']):>6} "
            f"{_fmt(r['strong_calls']):>6} {_fmt(r['handoff']):>8}{stop}  "
            f"{r['dir'].relative_to(REPO_ROOT) if _under(r['dir']) else r['dir']}"
        )
    if any(r["budget_stop"] for r in runs):
        print("\n[$] = the run stopped because it reached its dollar budget.")


def _under(path: Path) -> bool:
    try:
        path.relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def print_aggregate(runs: List[Dict[str, Any]]) -> None:
    """Mean +/- std over seeds — the shape of the paper's results table."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in runs:
        if r["score"] is None:
            continue
        groups.setdefault((r["benchmark"], r["method"]), []).append(r)

    if not groups:
        print("No scored runs to aggregate.")
        return

    for benchmark in sorted({k[0] for k in groups}):
        print(f"\n=== {benchmark} ===")
        print(f"{'method':<14} {'n':>2} {'mean':>12} {'std':>10} {'best':>12} {'mean $':>8}")
        print("-" * 62)
        rows = [(m, g) for (b, m), g in groups.items() if b == benchmark]
        rows.sort(key=lambda item: -statistics.fmean(r["score"] for r in item[1]))
        for method, group in rows:
            scores = [r["score"] for r in group]
            costs = [r["cost"] for r in group if r["cost"] is not None]
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(
                f"{method:<14} {len(scores):>2} {statistics.fmean(scores):>12.6f} "
                f"{std:>10.6f} {max(scores):>12.6f} "
                f"{('$' + format(statistics.fmean(costs), '.3f')) if costs else '-':>8}"
            )
            if len(scores) == 1:
                print(f"{'':<14} (single seed — std is not meaningful yet)")


def print_detail(run_dir: Path) -> None:
    summary = _load(run_dir / "relay_summary.json")
    config = _load(run_dir / "run_config.json")
    if not summary:
        print(f"No relay_summary.json in {run_dir}", file=sys.stderr)
        return
    info = _describe(run_dir)

    print(f"=== {run_dir} ===\n")
    print("what was asked for:")
    for key in (
        "method",
        "benchmark",
        "cheap_model",
        "strong_model",
        "iterations",
        "dollars",
        "workers",
        "eval_timeout",
        "retries",
        "seed",
        "curation",
        "relay_control",
    ):
        if config.get(key) is not None:
            print(f"  {key:<14}: {config[key]}")

    print("\nwhat happened:")
    print(
        f"  score         : {_fmt(info['score'], '.6f')}"
        f"{' (test-mode)' if info['test_score'] is not None else ' (search-time)'}"
    )
    print(f"  generations   : {_fmt(info['generations'])}")
    print(
        f"  llm calls     : cheap={_fmt(info['cheap_calls'])} strong={_fmt(info['strong_calls'])}"
    )
    print(
        f"  cost          : ${_fmt(info['cost'], '.4f')} of ${_fmt(info['budget'], '.2f')}"
        f"{'  <- stopped on budget' if info['budget_stop'] else ''}"
    )
    print(
        f"  tokens        : in={_fmt(info['prompt_tokens'])} out={_fmt(info['completion_tokens'])}"
    )
    print(f"  wall clock    : {_fmt(info['wall_s'], '.0f')}s")
    if info["handoff"] is not None:
        print(f"  handoff       : generation {info['handoff']} ({info['handoff_reason']})")
        print(
            f"  cheap/strong  : {summary.get('cheap_iterations')} / "
            f"{summary.get('strong_iterations')} generations"
        )
        print(f"  seeds handed  : {len(summary.get('seeds') or [])}")

    blocks = summary.get("blocks") or []
    if blocks:
        print("\nGrow/Deepen blocks:")
        for b in blocks:
            print(
                f"  #{b['block']:<3} {b['action']:<12} traj={b['trajectory']} "
                f"+{b['iterations']} gens  gain={b['relay_gain']:.4f} "
                f"rel={b['relative_relay_gain']:.4f}  bank F={b['bank_objective']:.4f}"
                f"{'  (warmup)' if b.get('warmup') else ''}"
            )

    print(f"\nfiles:\n  best program  : {run_dir / 'best' / 'best_program.py'}")
    print(f"  cost/score curve: {run_dir / 'relay_progress.jsonl'}")
    print(f"  full log      : {run_dir / 'run.log'}")


def write_csv(runs: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "method",
        "benchmark",
        "seed",
        "score",
        "generations",
        "cost",
        "budget",
        "cheap_calls",
        "strong_calls",
        "handoff",
        "handoff_reason",
        "budget_stop",
        "wall_s",
        "prompt_tokens",
        "completion_tokens",
        "curation",
        "relay_control",
        "dir",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            row = {k: r.get(k) for k in fields}
            row["dir"] = str(r["dir"])
            writer.writerow(row)
    print(f"Wrote {len(runs)} rows to {path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root",
        action="append",
        default=None,
        help="Directory to search (repeatable). " "Default: outputs/server and outputs/relay.",
    )
    p.add_argument("--path", help="Show one run in full instead of the table.")
    p.add_argument("--method", help="Only this method.")
    p.add_argument("--benchmark", help="Substring match on the benchmark name.")
    p.add_argument("--seed", type=int, help="Only this seed.")
    p.add_argument("--agg", action="store_true", help="Mean +/- std over seeds.")
    p.add_argument("--csv", help="Also write the rows to this CSV file.")
    args = p.parse_args()

    if args.path:
        print_detail(Path(args.path).resolve())
        return 0

    roots = (
        [Path(r).resolve() for r in args.root]
        if args.root
        else [REPO_ROOT / "outputs" / "server", REPO_ROOT / "outputs" / "relay"]
    )
    runs = collect(roots)

    if args.method:
        runs = [r for r in runs if r["method"] == args.method]
    if args.benchmark:
        runs = [r for r in runs if args.benchmark in r["benchmark"]]
    if args.seed is not None:
        runs = [r for r in runs if r["seed"] == args.seed]

    if args.agg:
        print_aggregate(runs)
    else:
        print_table(runs)
    if args.csv:
        write_csv(runs, Path(args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
