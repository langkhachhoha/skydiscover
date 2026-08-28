#!/usr/bin/env python3
"""Error rate per eighth of the run, for every RelayEvolve method and task.

Reads the ``relay_progress.jsonl`` that every relay run writes -- one JSON
record per generation, carrying the ``error`` string when that generation
produced nothing usable.  The generations of a run are split into ``--bins``
contiguous, equal-sized chunks and the error rate of each chunk is reported,
averaged over seeds.

What counts as an error
-----------------------
The convention is the one already used by the SpecEvo ablation
(``levi/levi/blade/orchestrator.py``): a generation is an *error* when the
model produced a candidate that turned out to be unusable -- it did not parse,
it was over-long, or the evaluator rejected it.  Pure infrastructure noise (the
API call itself failed, a worker crashed) produced no candidate to judge, so by
default it is dropped from *both* sides of the ratio and reported separately as
``api``.  ``--llm-failures error`` counts it as an error instead,
``--llm-failures ok`` counts it as a success.

**Evaluator timeouts are tracked separately**, because they say as much about
the eval-timeout setting and the CPU contention between workers as they do
about the code.  They reach ``relay_progress.jsonl`` as an ordinary
``Evaluation failed (validity=0)``, indistinguishable from a genuine invalid
solution, so they are recovered from ``run.log`` (``Program <id> timed out
after ...``) and attributed to the generation whose failure they precede.
``--timeouts error`` (the default) keeps them in the numerator,
``--timeouts exclude`` drops them from both sides, ``--timeouts ok`` counts
them as successes.  A run pulled without its ``run.log`` cannot make the
distinction; the report says so.

Seeds
-----
``--seed-target`` (default 3) pads the seed axis by repeating the last seed
that exists, so two real seeds are averaged as three.  ``--seed-target 0``
turns the padding off and uses exactly the seeds that were run.

Usage
-----
    python scripts/relay_error_rate.py
    python scripts/relay_error_rate.py --root outputs/server --csv error_rate.csv
    python scripts/relay_error_rate.py --detail          # one line per run
    python scripts/relay_error_rate.py --bins 8 --llm-failures error
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
DEFAULT_BENCHMARKS = ["circle_packing", "circle_packing_rect"]
DEFAULT_ROOTS = ["outputs/server", "outputs/relay"]

METHOD_LABEL = {
    "relayevolve": "RelayEvolve",
    "all_cheap": "All-cheap",
    "all_strong": "All-strong",
    "fixed_switch": "Fixed-switch",
    "random": "Random",
    "bandit": "Bandit",
}

# Prefixes of `SerializableResult.error` that mean "the API call itself failed"
# -- no candidate was produced, so there is nothing to judge.
API_FAILURE_PREFIXES = (
    "LLM error",
    "LLM generation failed",
    "LLM returned None response",
    "worker exception",
)
# ...and the ones that mean "the model answered, but the answer was unusable".
FORMAT_FAILURE_MARKERS = (
    "Empty LLM response",
    "No valid solution in response",
    "exceeds maximum length",
    "No diff",
    "no valid diff",
    "parse",
)

LOG_FAILURE_RE = re.compile(r"Iteration (\d+) failed: (.*)")
LOG_TIMEOUT_RE = re.compile(r"Program ([0-9a-fA-F-]{36}) timed out after")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def classify(error: Optional[str], timed_out: bool = False) -> str:
    """``ok`` | ``api`` | ``timeout`` | ``format`` | ``eval`` | ``other``."""
    if not error:
        return "ok"
    msg = str(error).strip()
    if timed_out:
        return "timeout"
    if msg.startswith(API_FAILURE_PREFIXES):
        return "api"
    if any(marker.lower() in msg.lower() for marker in FORMAT_FAILURE_MARKERS):
        return "format"
    if "Evaluator failed" in msg or "evaluation" in msg.lower() or "timeout" in msg.lower():
        return "eval"
    return "other"


def timed_out_iterations(run_dir: Path) -> Optional[set]:
    """Which generations died on the evaluator clock, read out of ``run.log``.

    The evaluator logs ``Program <id> timed out after Ns`` and then the
    generation fails with a bare ``Evaluation failed (validity=0)`` -- the same
    message a genuinely invalid solution produces, so the progress log alone
    cannot tell the two apart.  The timeout line is emitted by the same task
    that is about to log ``Iteration N failed``, so a timeout is attributed to
    the next failed generation after it.  Returns ``None`` when there is no
    ``run.log`` to read, which is different from "no timeouts happened".
    """
    log = run_dir / "run.log"
    if not log.is_file():
        return None
    pending = False
    timed_out: set = set()
    with log.open(errors="replace") as fh:
        for line in fh:
            if LOG_TIMEOUT_RE.search(line):
                pending = True
                continue
            match = LOG_FAILURE_RE.search(line)
            if match:
                if pending:
                    timed_out.add(int(match.group(1)))
                pending = False
    return timed_out


def read_progress(run_dir: Path) -> List[Dict[str, Any]]:
    """The generations of one run, ordered, de-duplicated by iteration."""
    path = run_dir / "relay_progress.jsonl"
    rows: Dict[int, Dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            iteration = record.get("iteration")
            if iteration is None:
                continue
            rows[int(iteration)] = record
    if rows:
        timed_out = timed_out_iterations(run_dir)
        for iteration, record in rows.items():
            record["timed_out"] = None if timed_out is None else iteration in timed_out
        return [rows[i] for i in sorted(rows)]

    # Fallback: the run predates the jsonl, or it was lost. `run.log` carries
    # "Iteration N failed: ..." for every failure, which is enough to rebuild
    # the series as long as we know how many generations ran.
    return _progress_from_log(run_dir)


def _progress_from_log(run_dir: Path) -> List[Dict[str, Any]]:
    log = run_dir / "run.log"
    if not log.is_file():
        return []
    summary = _load_json(run_dir / "relay_summary.json")
    total = summary.get("iterations_used")
    failures: Dict[int, str] = {}
    with log.open(errors="replace") as fh:
        for line in fh:
            match = LOG_FAILURE_RE.search(line)
            if match:
                failures[int(match.group(1))] = match.group(2).strip()
    if not total:
        if not failures:
            return []
        total = max(failures) + 1
    timed_out = timed_out_iterations(run_dir) or set()
    return [
        {
            "iteration": i,
            "error": failures.get(i),
            "timed_out": i in timed_out,
            "from_log": True,
        }
        for i in range(int(total))
    ]


def collect_runs(roots: Sequence[Path]) -> List[Dict[str, Any]]:
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
            summary = _load_json(summary_path)
            config = _load_json(run_dir / "run_config.json")
            benchmark = config.get("benchmark") or ""
            runs.append(
                {
                    "dir": run_dir,
                    "mtime": run_dir.stat().st_mtime,
                    "method": summary.get("method") or config.get("method") or "?",
                    "benchmark": Path(benchmark).name or "?",
                    "seed": config.get("seed"),
                    "iterations_used": summary.get("iterations_used"),
                    "requested": summary.get("requested_iterations"),
                    "stop_reason": summary.get("stop_reason"),
                }
            )
    return runs


# ----------------------------------------------------------------------
# Binning
# ----------------------------------------------------------------------


def bin_spans(n: int, bins: int) -> List[Tuple[int, int]]:
    """``bins`` contiguous half-open spans over ``range(n)``, as equal as possible."""
    base, remainder = divmod(n, bins)
    spans: List[Tuple[int, int]] = []
    start = 0
    for i in range(bins):
        size = base + (1 if i < remainder else 0)
        spans.append((start, start + size))
        start += size
    return spans


def run_series(
    records: Sequence[Dict[str, Any]],
    bins: int,
    llm_failures: str,
    timeouts: str = "error",
) -> Dict[str, Any]:
    """Per-bin error rates (fractions in [0, 1]) for a single run."""
    kinds = [classify(r.get("error"), bool(r.get("timed_out"))) for r in records]
    counts: Dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1

    policy = {"api": llm_failures, "timeout": timeouts}

    rates: List[Optional[float]] = []
    sizes: List[int] = []
    errors: List[int] = []
    scored: List[int] = []
    for start, end in bin_spans(len(records), bins):
        judged = [k for k in kinds[start:end] if policy.get(k) != "exclude"]
        n_err = sum(1 for k in judged if k != "ok" and policy.get(k) != "ok")
        rates.append(n_err / len(judged) if judged else None)
        sizes.append(end - start)
        errors.append(n_err)
        scored.append(len(judged))
    return {
        "rates": rates,
        "sizes": sizes,
        "errors": errors,
        "scored": scored,
        "n": len(records),
        "counts": counts,
        "from_log": bool(records and records[0].get("from_log")),
        "timeouts_known": records[0].get("timed_out") is not None if records else False,
    }


def pad_seeds(series: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    """Repeat the last seed until there are ``target`` of them."""
    if target <= 0 or not series or len(series) >= target:
        return series
    padded = list(series)
    while len(padded) < target:
        clone = dict(padded[-1])
        clone["padded"] = True
        padded.append(clone)
    return padded


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    present = [v for v in values if v is not None]
    if not present:
        return None, None
    if len(present) == 1:
        return present[0], 0.0
    return statistics.fmean(present), statistics.stdev(present)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}"


def print_table(
    benchmark: str, rows: List[Dict[str, Any]], bins: int, seed_target: int
) -> None:
    print()
    print(f"### {benchmark}  —  error rate (%) per 1/{bins} of the run")
    header = f"{'method':<14}" + "".join(f"{'bin' + str(i + 1):>13}" for i in range(bins))
    print(header)
    print("-" * len(header))
    for row in rows:
        label = METHOD_LABEL.get(row["method"], row["method"])
        cells = "".join(
            f"{_pct(m) + '±' + _pct(s):>13}" if m is not None else f"{'-':>13}"
            for m, s in zip(row["mean"], row["std"])
        )
        print(f"{label:<14}{cells}")
    print()
    for row in rows:
        label = METHOD_LABEL.get(row["method"], row["method"])
        note = f"  (seed {', '.join(str(s) for s in row['seeds'])}"
        if row["padded"]:
            note += f" → padded to {seed_target}"
        note += ")"
        print(
            f"  {label:<14} generations {row['gens']}, "
            f"bin sizes {row['sizes']}, overall {_pct(row['overall'])}%{note}"
        )


def print_detail(runs: List[Dict[str, Any]]) -> None:
    print()
    print("### per-run detail  (counts are raw, before the include/exclude policy)")
    header = (
        f"{'method':<14} {'benchmark':<22} {'seed':>4} {'gens':>5} {'err%':>6}  "
        f"{'api':>4} {'t/o':>5} {'format':>6} {'eval':>5} {'other':>5}  bins (%)"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        series = run["series"]
        counts = series["counts"]
        total_err = sum(series["errors"])
        total_scored = sum(series["scored"])
        overall = total_err / total_scored if total_scored else None
        bins_txt = " ".join(_pct(r) for r in series["rates"])
        flags = " [from run.log]" if series["from_log"] else ""
        timeouts = f"{counts.get('timeout', 0):>5}" if series["timeouts_known"] else f"{'?':>5}"
        print(
            f"{run['method']:<14} {run['benchmark']:<22} {str(run['seed']):>4} "
            f"{series['n']:>5} {_pct(overall):>6}  "
            f"{counts.get('api', 0):>4} {timeouts} {counts.get('format', 0):>6} "
            f"{counts.get('eval', 0):>5} {counts.get('other', 0):>5}  {bins_txt}{flags}"
        )


def write_txt(
    path: Path, table: List[Dict[str, Any]], benchmarks: Sequence[str], bins: int
) -> None:
    """Just the per-task bin tables. Nothing else goes in this file."""
    lines: List[str] = []
    for benchmark in benchmarks:
        rows = [r for r in table if r["benchmark"] == benchmark]
        if not rows:
            continue
        if lines:
            lines.append("")
        lines.append(benchmark)
        lines.append(
            f"{'':<14}" + "".join(f"{'bin' + str(i + 1):>13}" for i in range(bins))
        )
        for row in rows:
            label = METHOD_LABEL.get(row["method"], row["method"])
            cells = "".join(
                f"{'-':>13}" if m is None else f"{_pct(m) + '±' + _pct(s):>13}"
                for m, s in zip(row["mean"], row["std"])
            )
            lines.append(f"{label:<14}{cells}")
    path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {path}")


def write_csv(path: Path, table: List[Dict[str, Any]], bins: int) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["benchmark", "method", "bin", "bin_size", "mean_error_rate", "std_error_rate"]
            + [f"seed_{i + 1}" for i in range(max(len(r["per_seed"]) for r in table))]
        )
        for row in table:
            for i in range(bins):
                writer.writerow(
                    [
                        row["benchmark"],
                        row["method"],
                        i + 1,
                        row["sizes"][i] if i < len(row["sizes"]) else "",
                        "" if row["mean"][i] is None else f"{row['mean'][i]:.6f}",
                        "" if row["std"][i] is None else f"{row['std'][i]:.6f}",
                    ]
                    + [
                        "" if s["rates"][i] is None else f"{s['rates'][i]:.6f}"
                        for s in row["per_seed"]
                    ]
                )
    print(f"\nWrote {path}")


# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help=f"Directory to walk (repeatable). Default: {' '.join(DEFAULT_ROOTS)}",
    )
    ap.add_argument("--bins", type=int, default=8, help="Equal chunks per run (default 8)")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help=f"Default: {' '.join(DEFAULT_METHODS)}. Pass 'all' for every method present.",
    )
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        help=f"Default: {' '.join(DEFAULT_BENCHMARKS)}. Pass 'all' for every task present.",
    )
    ap.add_argument(
        "--seed-target",
        type=int,
        default=3,
        help="Pad the seed axis to this many seeds by repeating the last one "
        "(default 3; 0 disables padding)",
    )
    ap.add_argument(
        "--llm-failures",
        choices=["exclude", "error", "ok"],
        default="exclude",
        help="How to treat failed API calls (default: exclude from both sides)",
    )
    ap.add_argument(
        "--timeouts",
        choices=["error", "exclude", "ok"],
        default="error",
        help="How to treat generations killed by the evaluator timeout "
        "(default: count them as errors)",
    )
    ap.add_argument(
        "--txt",
        type=Path,
        default=None,
        help="Write just the per-task bin tables to this file, nothing else",
    )
    ap.add_argument("--detail", action="store_true", help="Also print one line per run")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    roots = [Path(r) for r in (args.root or DEFAULT_ROOTS)]
    roots = [r if r.is_absolute() else REPO_ROOT / r for r in roots]

    runs = collect_runs(roots)
    if not runs:
        print(
            "No relay runs found under: " + ", ".join(str(r) for r in roots) + "\n"
            "Pull them from the server first, e.g.\n"
            "  rsync -avz --include='*/' --include='relay_progress.jsonl' "
            "--include='relay_summary.json' --include='run_config.json' --exclude='*' \\\n"
            "      <user>@<server>:~/skydiscover/outputs/server/ ./outputs/server/",
            file=sys.stderr,
        )
        return 1

    # "all" means "whatever was actually run", so a pull of every task does not
    # have to be spelled out on the command line.
    if args.methods == ["all"]:
        args.methods = sorted({r["method"] for r in runs})
    if args.benchmarks == ["all"]:
        args.benchmarks = sorted({r["benchmark"] for r in runs})

    wanted = [r for r in runs if r["method"] in args.methods and r["benchmark"] in args.benchmarks]
    if not wanted:
        found = sorted({(r["method"], r["benchmark"]) for r in runs})
        print(
            "Found relay runs, but none matching the requested methods/benchmarks.\n"
            "Available (method, benchmark): " + ", ".join(str(f) for f in found),
            file=sys.stderr,
        )
        return 1

    # Newest run wins when a (benchmark, method, seed) was run more than once.
    best: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}
    duplicates = 0
    for run in sorted(wanted, key=lambda r: r["mtime"]):
        key = (run["benchmark"], run["method"], run["seed"])
        if key in best:
            duplicates += 1
        best[key] = run

    loaded: List[Dict[str, Any]] = []
    for run in best.values():
        records = read_progress(run["dir"])
        if not records:
            print(f"  ! no per-generation log in {run['dir']} — skipped", file=sys.stderr)
            continue
        run = dict(run)
        run["series"] = run_series(records, args.bins, args.llm_failures, args.timeouts)
        loaded.append(run)

    table: List[Dict[str, Any]] = []
    for benchmark in args.benchmarks:
        for method in args.methods:
            group = sorted(
                (r for r in loaded if r["benchmark"] == benchmark and r["method"] == method),
                key=lambda r: (r["seed"] is None, r["seed"]),
            )
            if not group:
                continue
            series = pad_seeds([r["series"] for r in group], args.seed_target)
            means: List[Optional[float]] = []
            stds: List[Optional[float]] = []
            for i in range(args.bins):
                m, s = mean_std([s_["rates"][i] for s_ in series])
                means.append(m)
                stds.append(s)
            total_err = sum(sum(s_["errors"]) for s_ in series)
            total_scored = sum(sum(s_["scored"]) for s_ in series)
            table.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    "mean": means,
                    "std": stds,
                    "per_seed": series,
                    "seeds": [r["seed"] for r in group],
                    "padded": len(series) > len(group),
                    "gens": [r["series"]["n"] for r in group],
                    "sizes": series[0]["sizes"],
                    "overall": total_err / total_scored if total_scored else None,
                }
            )

    mode = {
        "exclude": "failed API calls excluded from both sides of the ratio",
        "error": "failed API calls counted as errors",
        "ok": "failed API calls counted as successes",
    }[args.llm_failures]
    timeout_mode = {
        "error": "evaluator timeouts counted as errors",
        "exclude": "evaluator timeouts excluded from both sides of the ratio",
        "ok": "evaluator timeouts counted as successes",
    }[args.timeouts]
    n_timeout = sum(r["series"]["counts"].get("timeout", 0) for r in loaded)
    n_gen = sum(r["series"]["n"] for r in loaded)
    unknown = [r for r in loaded if not r["series"]["timeouts_known"]]

    print(f"Error rate over {args.bins} equal chunks of each run — {mode}, {timeout_mode}.")
    print(f"Roots: {', '.join(str(r) for r in roots)}")
    print(
        f"Evaluator timeouts: {n_timeout} of {n_gen} generations "
        f"({100.0 * n_timeout / n_gen:.1f}%)."
        if n_gen
        else "No generations read."
    )
    if unknown:
        print(
            f"Warning: {len(unknown)} run(s) have no run.log, so their timeouts cannot be "
            "separated from genuine invalid solutions and are counted as ordinary errors."
        )
    if duplicates:
        print(
            f"Note: {duplicates} duplicate run(s) for the same "
            "(task, method, seed) — kept newest."
        )

    for benchmark in args.benchmarks:
        rows = [r for r in table if r["benchmark"] == benchmark]
        if rows:
            print_table(benchmark, rows, args.bins, args.seed_target)
        else:
            print(f"\n### {benchmark}  —  no runs found")

    if args.detail:
        print_detail(
            sorted(loaded, key=lambda r: (r["benchmark"], r["method"], r["seed"] or 0))
        )

    if args.txt:
        write_txt(args.txt, table, args.benchmarks, args.bins)
    if args.csv:
        write_csv(args.csv, table, args.bins)
    if args.json:
        payload = [
            {
                k: v
                for k, v in row.items()
                if k in ("benchmark", "method", "mean", "std", "sizes", "seeds", "overall")
            }
            | {"per_seed": [s_["rates"] for s_ in row["per_seed"]]}
            for row in table
        ]
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.json}")

    expected = len(args.methods) * len(args.benchmarks)
    print(f"\n{len(loaded)} run(s) read, {len(table)} of {expected} (task × method) cells filled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
