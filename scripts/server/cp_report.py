#!/usr/bin/env python3
"""Status and comparison tables for scripts/server/run_cp_queue.sh.

Reads job lines ``<lane> <task> <arm> <kind> <dir>`` on stdin, where kind is
``blade`` (SpecEvo) or ``baseline`` (a skydiscover search), and prints

  status   one row per job: state, best score, spend, evaluations, wall time —
           read live from the logs for a run still in flight.
  compare  per-job token / latency / score table (input, output and hidden
           reasoning tokens, mean seconds per LLM call and per candidate),
           plus a per-model split. With --out DIR it also writes
           comparison.md / .csv / .json there.

Every number is read from files the runs write themselves:
  SpecEvo  : summary.json, llm_calls.jsonl (one line per LLM call), run.log
  baseline : run/best/best_program_info.json, run/checkpoints/*, cost_log.jsonl
             (one line per LLM call), run.log
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# Calls whose answer is a candidate program; the rest (advisor, analyzer,
# summarize) steer the search but produce no solution themselves.
SOLUTION_ROLES = {"mutate", "crossover", "init_seed", "init_variant", "navigator", "navigator_variant"}
CP26_TARGET = 2.635  # AlphaEvolve's n=26 sum of radii; the baselines' combined_score divides by it

STATUS_RE = re.compile(
    r"\[Status\] Cost: \$([\d.]+) \| Evals: (\d+) .*?Best: ([-\w.]+) \| Elapsed: (\d+)s"
)
ITER_RE = re.compile(r"Iteration (\d+): .*?completed in ([\d.]+)s \(llm: ([\d.]+)s, eval: ([\d.]+)s\)")
SUM_RADII_RE = re.compile(r"(?<![\w])sum_radii=([\d.]+)")
STARTED_RE = re.compile(r"^ started at\s*: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", re.M)
FINISHED_RE = re.compile(r"^ finished at : (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", re.M)
EXIT_RE = re.compile(r"^ exit status : (\d+)", re.M)


# ---------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------
def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a line being written right now
    except OSError:
        pass
    return out


def _text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _pid_alive(d: Path) -> bool:
    try:
        pid = int((d / ".run.pid").read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def _wall_seconds(log: str, running: bool) -> float | None:
    starts = STARTED_RE.findall(log)
    if not starts:
        return None
    t0 = _ts(starts[0])
    ends = FINISHED_RE.findall(log)
    if ends and not running:
        return _ts(ends[-1]) - t0
    return time.time() - t0


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _call_stats(calls: list[dict]) -> dict:
    lat = [c["latency_s"] for c in calls if _num(c.get("latency_s")) is not None]
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(int(c.get("prompt_tokens") or 0) for c in calls),
        "output_tokens": sum(int(c.get("completion_tokens") or 0) for c in calls),
        "reasoning_tokens": sum(int(c.get("reasoning_tokens") or 0) for c in calls),
        "llm_cost": sum(float(c.get("cost_usd") or 0.0) for c in calls),
        "mean_s_per_call": statistics.fmean(lat) if lat else None,
        "median_s_per_call": statistics.median(lat) if lat else None,
        "p90_s_per_call": (sorted(lat)[math.ceil(0.9 * len(lat)) - 1] if lat else None),  # nearest rank
        "llm_seconds_total": sum(lat) if lat else None,
    }


def _by_model(calls: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for c in calls:
        groups.setdefault(str(c.get("model") or "?").rsplit("/", 1)[-1], []).append(c)
    return {m: _call_stats(cs) for m, cs in sorted(groups.items())}


# ---------------------------------------------------------------------------
# per-kind collectors
# ---------------------------------------------------------------------------
def collect_blade(d: Path) -> dict:
    summary = _json(d / "summary.json")
    log = _text(d / "run.log")
    running = _pid_alive(d)
    calls = _jsonl(d / "llm_calls.jsonl")
    r: dict = {"kind": "blade"}

    if summary:
        r["state"] = "DONE"
        r["best"] = _num(summary.get("best_score"))
        r["cost"] = _num(summary.get("total_cost"))
        r["candidates"] = summary.get("total_evaluations")
        r["stopped_by"] = _blade_stop(summary)
    else:
        m = None
        for m in STATUS_RE.finditer(log):
            pass
        if m:
            r["cost"], r["candidates"] = float(m.group(1)), int(m.group(2))
            r["best"] = _num(m.group(3))
        r["state"] = _unfinished_state(d, running, log)
    r["wall_s"] = _wall_seconds(log, running)

    if calls:
        r.update(_call_stats(calls))
        sol = [c for c in calls if c.get("role") in SOLUTION_ROLES]
        sol_lat = [c["latency_s"] for c in sol if _num(c.get("latency_s")) is not None]
        r["mean_s_per_solution_call"] = statistics.fmean(sol_lat) if sol_lat else None
        r["by_model"] = _by_model(calls)
        roles: dict[str, list[dict]] = {}
        for c in calls:
            roles.setdefault(str(c.get("role")), []).append(c)
        r["by_role"] = {k: _call_stats(v) for k, v in sorted(roles.items())}
    elif summary:  # a run from before llm_calls.jsonl existed
        r["llm_calls"] = summary.get("total_llm_calls")
        r["input_tokens"] = summary.get("total_prompt_tokens")
        r["output_tokens"] = summary.get("total_completion_tokens")
        r["reasoning_tokens"] = summary.get("total_reasoning_tokens")
    _per_candidate(r)
    return r


def _blade_stop(summary: dict) -> str:
    b = summary.get("budget") or {}
    cost, secs = _num(summary.get("total_cost")) or 0.0, _num(summary.get("runtime_seconds")) or 0.0
    if b.get("dollars") and cost >= float(b["dollars"]):
        return f"${b['dollars']:g} budget"
    if b.get("seconds") and secs >= float(b["seconds"]) - 5:
        return f"{float(b['seconds']) / 3600:g}h budget"
    if b.get("evaluations") and (summary.get("total_evaluations") or 0) >= int(b["evaluations"]):
        return "eval cap"
    return "-"


def collect_baseline(d: Path) -> dict:
    log = _text(d / "run.log")
    running = _pid_alive(d)
    calls = _jsonl(d / "cost_log.jsonl")
    r: dict = {"kind": "baseline"}

    info = _json(d / "run" / "best" / "best_program_info.json")
    best = _baseline_score(info)
    if info:
        r["state"] = "DONE"
    else:
        r["state"] = _unfinished_state(d, running, log)
        ckpts = sorted(
            (d / "run" / "checkpoints").glob("checkpoint_*/best_program_info.json"),
            key=lambda p: int(p.parent.name.rsplit("_", 1)[-1]) if p.parent.name.rsplit("_", 1)[-1].isdigit() else -1,
        )
        if ckpts:
            best = _baseline_score(_json(ckpts[-1]))
        seen = [float(x) for x in SUM_RADII_RE.findall(log)]
        if seen and (best is None or max(seen) > best):
            best = max(seen)  # newer than the last checkpoint
    r["best"] = best

    iters = ITER_RE.findall(log)
    r["candidates"] = len(iters)
    if iters:
        r["mean_s_per_solution_call"] = statistics.fmean(float(i[2]) for i in iters)  # llm time per iteration
    r["wall_s"] = _wall_seconds(log, running)
    totals = _json(d / "cost_log.totals.json") or {}
    r["cost"] = _num(totals.get("total_cost_usd"))
    if r["state"] == "DONE":
        if "Time budget of" in log:
            r["stopped_by"] = "time budget"
        elif "Spend budget of" in log:
            r["stopped_by"] = "$ budget"
        else:
            r["stopped_by"] = "iteration cap"
    if calls:
        r.update(_call_stats(calls))
        r["by_model"] = _by_model(calls)
        if r.get("cost") is None:
            r["cost"] = r["llm_cost"]
    _per_candidate(r)
    return r


def _baseline_score(info) -> float | None:
    if not info:
        return None
    m = info.get("metrics") or {}
    v = _num(m.get("sum_radii"))
    if v is None and _num(m.get("combined_score")) is not None:
        v = float(m["combined_score"]) * CP26_TARGET
    return v


def _unfinished_state(d: Path, running: bool, log: str) -> str:
    if running:
        return "RUNNING"
    if not d.exists():
        return "PENDING"
    code = _text(d / ".exit_code").strip()
    if code == "124":
        return "TIMEOUT"
    m = EXIT_RE.findall(log)
    if code or m:
        return "FAILED"
    return "STOPPED"  # dir exists, nothing running, no exit recorded: the lane was killed


def _per_candidate(r: dict) -> None:
    n, w = r.get("candidates"), r.get("wall_s")
    r["wall_s_per_candidate"] = (w / n) if (n and w) else None
    inp, out = r.get("input_tokens"), r.get("output_tokens")
    r["tokens_per_candidate"] = ((inp or 0) + (out or 0)) / n if (n and (inp or out)) else None


def collect(kind: str, d: Path) -> dict:
    if not d.exists():
        return {"kind": kind, "state": "PENDING"}
    return collect_blade(d) if kind == "blade" else collect_baseline(d)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def _f(v, fmt="{:.4f}", none="-"):
    return none if v is None else fmt.format(v)


def _dur(s) -> str:
    if s is None:
        return "-"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _k(v) -> str:
    if v is None:
        return "-"
    v = float(v)
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.0f}"


def read_jobs() -> list[tuple[str, str, str, str, Path]]:
    jobs = []
    for line in sys.stdin:
        parts = line.split()
        if len(parts) == 5:
            lane, task, arm, kind, d = parts
            jobs.append((lane, task, arm, kind, Path(d)))
    return jobs


def cmd_status(jobs) -> None:
    hdr = f"{'lane':<4} {'task':<20} {'arm':<20} {'state':<8} {'best':>8} {'cost':>7} {'evals':>6} {'wall':>6}  stopped by"
    print(hdr)
    print("-" * len(hdr))
    last_lane = None
    for lane, task, arm, kind, d in jobs:
        if last_lane is not None and lane != last_lane:
            print()
        last_lane = lane
        r = collect(kind, d)
        live = r["state"] == "RUNNING"
        print(
            f"{lane:<4} {task:<20} {arm:<20} {r['state']:<8} {_f(r.get('best')):>8} "
            f"{_f(r.get('cost'), '${:.2f}'):>7} {_f(r.get('candidates'), '{}'):>6} {_dur(r.get('wall_s')):>6}  "
            f"{'(live)' if live else r.get('stopped_by', '')}"
        )


COLUMNS = [
    # key, header, formatter
    ("label", "method", str),
    ("state", "state", str),
    ("best", "best sum_radii", lambda v: _f(v)),
    ("ratio", "/2.635", lambda v: _f(v)),
    ("wall_s", "wall", _dur),
    ("candidates", "candidates", lambda v: _f(v, "{}")),
    ("llm_calls", "LLM calls", lambda v: _f(v, "{}")),
    ("input_tokens", "input tok", _k),
    ("output_tokens", "output tok", _k),
    ("reasoning_tokens", "reasoning tok", _k),
    ("reasoning_share", "reasoning %", lambda v: _f(v, "{:.0%}")),
    ("tokens_per_candidate", "tok/candidate", _k),
    ("cost", "cost", lambda v: _f(v, "${:.2f}")),
    ("mean_s_per_call", "s/LLM call", lambda v: _f(v, "{:.1f}")),
    ("median_s_per_call", "median s/call", lambda v: _f(v, "{:.1f}")),
    ("mean_s_per_solution_call", "LLM s/solution", lambda v: _f(v, "{:.1f}")),
    ("wall_s_per_candidate", "wall s/candidate", lambda v: _f(v, "{:.0f}")),
    ("stopped_by", "stopped by", lambda v: v or "-"),
]


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def cmd_compare(jobs, out_dir: Path | None) -> None:
    results = []
    for lane, task, arm, kind, d in jobs:
        r = collect(kind, d)
        r["label"] = arm if len({j[1] for j in jobs}) == 1 else f"{task}/{arm}"
        r["task"], r["arm"], r["dir"] = task, arm, str(d)
        if r.get("best") is not None and task == "circle_packing":
            r["ratio"] = r["best"] / CP26_TARGET
        if r.get("output_tokens"):
            r["reasoning_share"] = (r.get("reasoning_tokens") or 0) / r["output_tokens"]
        results.append(r)

    headers = [h for _, h, _ in COLUMNS]
    rows = [[fmt(r.get(k)) if k not in ("label", "state") else str(r.get(k, "-")) for k, _, fmt in COLUMNS] for r in results]
    main = _md_table(headers, rows)

    mrows = []
    for r in results:
        for model, s in (r.get("by_model") or {}).items():
            share = (s["reasoning_tokens"] / s["output_tokens"]) if s["output_tokens"] else None
            mrows.append([
                r["label"], model, str(s["llm_calls"]), _k(s["input_tokens"]), _k(s["output_tokens"]),
                _k(s["reasoning_tokens"]), _f(share, "{:.0%}"), _f(s["llm_cost"], "${:.2f}"),
                _f(s["mean_s_per_call"], "{:.1f}"), _f(s["median_s_per_call"], "{:.1f}"), _f(s["p90_s_per_call"], "{:.1f}"),
            ])
    per_model = _md_table(
        ["method", "model", "calls", "input tok", "output tok", "reasoning tok", "reasoning %", "cost",
         "s/call", "median s/call", "p90 s/call"], mrows,
    ) if mrows else ""

    rrows = []
    for r in results:
        for role, s in (r.get("by_role") or {}).items():
            rrows.append([
                r["label"], role, str(s["llm_calls"]), _k(s["input_tokens"]), _k(s["output_tokens"]),
                _k(s["reasoning_tokens"]), _f(s["mean_s_per_call"], "{:.1f}"), _f(s["llm_cost"], "${:.2f}"),
            ])
    per_role = _md_table(
        ["method", "SpecEvo role", "calls", "input tok", "output tok", "reasoning tok", "s/call", "cost"], rrows,
    ) if rrows else ""

    notes = (
        "Notes\n"
        "- reasoning tok is the hidden-reasoning part of output tok (already counted in it).\n"
        "- s/LLM call: mean wall time of one API call, from the per-call logs.\n"
        "- LLM s/solution: SpecEvo = mean latency of the calls that return a program "
        "(mutate, crossover, init, navigator); baselines = mean LLM time per iteration "
        "(all of that iteration's calls), from the 'Iteration N ... (llm: Xs ...)' lines.\n"
        "- wall s/candidate: wall time / candidates evaluated, evaluation time included.\n"
        "- best sum_radii: SpecEvo scores with levi/examples/circle_packing, baselines with "
        "benchmarks/math/circle_packing; both are the sum of the 26 radii of a valid packing."
    )
    text = f"{main}\n\nPer model\n\n{per_model}\n" if per_model else f"{main}\n"
    if per_role:
        text += f"\nSpecEvo calls by role\n\n{per_role}\n"
    text += f"\n{notes}\n"
    print(text)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        (out_dir / "comparison.md").write_text(f"# Comparison ({stamp})\n\n{text}")
        with (out_dir / "comparison.csv").open("w", newline="") as fh:
            keys = [k for k, _, _ in COLUMNS] + ["task", "arm", "kind", "dir"]
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in keys})
        (out_dir / "comparison.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"written: {out_dir}/comparison.md, comparison.csv, comparison.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("status", "compare"))
    ap.add_argument("--out", type=Path, default=None, help="compare: also write comparison.{md,csv,json} here")
    args = ap.parse_args()
    jobs = read_jobs()
    if args.command == "status":
        cmd_status(jobs)
    else:
        cmd_compare(jobs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
