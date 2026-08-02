#!/usr/bin/env python3
"""
Cost-vs-score comparison for the $10-budget runs stored under ``logs/``.

Layout mirrors the reference figure: best-so-far step curves, one per search
method, with markers on every new-best point.  The x-axis is always $0 -> $10
(ticks 0,2,4,6,8,10); runs that spent less than $10 are extended flat at their
final score so every curve is comparable at the full budget.  The panels carry
no legend — they share a single one added downstream.

Log dialects handled (they are NOT all the same syntax)
-------------------------------------------------------
1. blade / SpecEvo::

       HH:MM:SS [INFO] [Eval #1894] <model> NEW BEST ★ | source: ... |
                score: 0.731578 | best: 0.731578 | $8.972
       Total cost        : $9.0669

2. native stack, math tasks (``[search.<method>]`` prefix)::

       ... [search.adaevolve] Metrics: runs_successfully=1.0000, ...,
           combined_score=0.5278, ... [cost=$0.0561/$9.00, llm_calls=1]
       ... [search.adaevolve] New best solution found at iteration 1 [cost=$...]

3. native stack, ADRS tasks (``[default_discovery_controller]`` prefix, a
   ``hit_rates=[...]`` list field, a ``🌟`` before "New best")::

       ... Metrics: combined_score=0.6805, runs_successfully=1.0000,
           hit_rates=[...], total_runtime=4.2208 [cost=$0.1398/$9.00, ...]
       ... 🌟 New best solution found at iteration 2 [cost=$0.1398/$9.00, ...]

   OpenEvolve additionally prints ``New best program <id> replaces <id>
   (0.6753 -> 0.6842)`` *before* the Metrics line; only the
   "New best solution found" line is treated as the commit point.

Score convention: the plotted y value is ``combined_score`` for the native
stack and ``score:`` for blade — for both benchmarks here those are the same
quantity (``overall_score`` for signal_processing, the SQL objective for
llm_sql), so the curves are directly comparable.

Usage
-----
    .venv/bin/python scripts/plots/plot_budget10.py             # every task
    .venv/bin/python scripts/plots/plot_budget10.py signal      # one task
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "logs"

BUDGET = 10.0                      # x-axis always runs to $10
XTICKS = [0, 2, 4, 6, 8, 10]

# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------
# log folder -> (benchmark name as it appears in the codebase, bottom of the
# y-axis, top of the y-axis).  ``None`` on either limit means "derive it from
# the data" — see _ylim.  circle_packing is pinned because GEPA's second point
# (2.16) sits far below where the run actually competes.
TASKS = {
    "signal":              ("signal processing",   None, 0.80),
    "sql":                 ("LLM-SQL",             None, 0.85),
    "eplb":                ("EPLB",                None, None),
    "txn":                 ("TXN Scheduling",      None, None),
    "circle_packing":      ("Circle Packing",      2.42, 2.65),
    "circle_packing_rect": ("Circle Packing Rectangle", 2.25, 2.38),
}

# method folder -> (label, colour, marker, is_ours)
STYLE = {
    "spec": ("SpecEvo",    "#D6263A", "o", True),
    "ada":  ("AdaEvolve",  "#8E44AD", "^", False),
    "oe":   ("OpenEvolve", "#117A65", "s", False),
    "gepa": ("GEPA",       "#2E86C1", "D", False),
    "evox": ("EvoX",       "#E08A1E", "v", False),
}
METHOD_ORDER = ["ada", "oe", "gepa", "evox", "spec"]   # ours drawn last / on top

# ---------------------------------------------------------------------------
# Log regexes
# ---------------------------------------------------------------------------
# the `[Eval #N]` prefix is not always present — SpecEvo's last new best on
# circle_packing_rect is logged without it, and requiring it silently drops the
# run's actual final score.
_BLADE_NEWBEST = re.compile(
    r"NEW BEST.*?score:\s*(?P<score>-?[\d.]+).*?\$(?P<cost>[\d.]+)\s*$"
)
_BLADE_TOTAL = re.compile(r"Total cost\s*:\s*\$(?P<cost>[\d.]+)")
_BLADE_STATUS_COST = re.compile(r"\[Status\]\s*Cost:\s*\$(?P<cost>[\d.]+)")

_NATIVE_METRICS = re.compile(r"Metrics:\s*(?P<body>.*?)\s*\[cost=\$(?P<cost>[\d.]+)")
# the seeded initial program, scored before any LLM call: "Database has 1
# programs, best program score is combined_score=0.6754, runs_successfully=…"
_NATIVE_INIT = re.compile(
    r"Database has 1 programs?, best program score is\s*(?P<body>.*)"
)
_NATIVE_NEWBEST = re.compile(r"New best solution found(?:\s+at\s+iteration\s+(?P<iter>\d+))?")
_NATIVE_ITER = re.compile(r"Iteration (?P<iter>\d+):")
_ANY_COST = re.compile(r"cost=\$(?P<cost>[\d.]+)")
# `hit_rates=[...]` must not match: the value has to start with a digit/sign.
_KV = re.compile(r"([A-Za-z_]\w*)=(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")

# fields on a Metrics line that are never the objective: bookkeeping, plus
# `target_ratio`, which is just the raw metric divided by the task BENCHMARK
# (circle_packing prints target_ratio == combined_score alongside sum_radii).
_NON_METRIC = {"combined_score", "target_ratio", "eval_time", "llm_calls",
               "error", "validity"}


def _is_blade_log(text: str) -> bool:
    return "NEW BEST" in text or "[blade]" in text


def _pick_metric(kv: dict[str, float]) -> float | None:
    """Objective value on a native ``Metrics:`` line, on SpecEvo's scale.

    SpecEvo logs its objective raw (``score: 2.465604`` on circle_packing), so
    the native curves have to be read the same way or the two are not on one
    axis.  A single-objective math task exposes exactly one raw metric beside
    the bookkeeping fields (``sum_radii``) — that is the value to plot.
    Multi-metric ADRS evaluators expose many auxiliary fields and there
    ``combined_score`` *is* the search objective, matching what SpecEvo reports
    for those tasks.
    """
    candidates = [k for k in kv if k not in _NON_METRIC]
    if len(candidates) == 1:
        return kv[candidates[0]]
    if "combined_score" in kv:
        return kv["combined_score"]
    return kv[candidates[0]] if candidates else None


class Run(NamedTuple):
    """One method's parsed run.

    ``pts`` are the (cost, score) points the log actually reports, best-so-far
    and strictly increasing.  ``n_seeded`` is 1 when the first of them is the
    initial program rather than a search result, so callers can report the
    number of *new bests* honestly.
    """
    pts: list[tuple[float, float]]
    total: float
    n_seeded: int = 0


class Curve(NamedTuple):
    """A plottable step curve spanning the full $0 -> $10 axis."""
    costs: list[float]
    scores: list[float]
    total: float
    m0: int             # first index carrying a marker (logged point)
    m1: int             # one past the last such index
    n_newbest: int


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_blade(lines: list[str]) -> Run:
    pts: list[tuple[float, float]] = []
    best = float("-inf")
    for ln in lines:
        m = _BLADE_NEWBEST.search(ln.rstrip())
        if not m:
            continue
        score = float(m["score"])
        if score <= best:
            continue
        best = score
        pts.append((float(m["cost"]), score))

    total = 0.0
    for ln in reversed(lines):
        m = _BLADE_TOTAL.search(ln)
        if m:
            total = float(m["cost"])
            break
    else:
        costs = [float(m["cost"]) for ln in lines
                 if (m := _BLADE_STATUS_COST.search(ln))]
        total = max(costs, default=0.0)
    return Run(pts, max(total, pts[-1][0] if pts else 0.0))


def parse_native(lines: list[str]) -> Run:
    """A new best = the ``Metrics:`` line immediately preceding "New best …".

    The seeded initial program is picked up as the $0 point when the log
    reports it, because best-so-far at $0 *is* that program's score.  This also
    covers a run that never improves on it and therefore logs no new best at
    all — AdaEvolve on llm_sql ends exactly where it started, at 0.6754.
    """
    pts: list[tuple[float, float]] = []
    best = float("-inf")
    n_seeded = 0
    pending: tuple[float, float] | None = None      # (cost, score)
    for ln in lines:
        mi = _NATIVE_INIT.search(ln)
        if mi and not pts:
            init = _pick_metric({k: float(v) for k, v in _KV.findall(mi["body"])})
            if init is not None:
                best = init
                pts.append((0.0, init))
                n_seeded = 1
            continue
        mm = _NATIVE_METRICS.search(ln)
        if mm:
            kv = {k: float(v) for k, v in _KV.findall(mm["body"])}
            score = _pick_metric(kv)
            if score is not None:
                pending = (float(mm["cost"]), score)
            continue
        if _NATIVE_NEWBEST.search(ln) and pending is not None:
            cost, score = pending
            if score > best:
                best = score
                pts.append((cost, score))

    costs = [float(m["cost"]) for ln in lines if (m := _ANY_COST.search(ln))]
    total = max(costs, default=0.0)
    return Run(pts, max(total, pts[-1][0] if pts else 0.0), n_seeded)


def load_method(run_log: Path) -> Curve | None:
    """Turn one ``run.log`` into a step curve spanning the whole $10 axis.

    Only the logged points get markers — the $0 lead-in (when the log doesn't
    report a starting score) and the flat tail out to the budget are synthetic.
    Incomplete runs are flat-extended to $10 the same way as finished-early ones.
    """
    text = run_log.read_text(errors="replace")
    if not text.strip():
        return None
    lines = text.splitlines()
    run = (parse_blade if _is_blade_log(text) else parse_native)(lines)
    if not run.pts:
        return None

    costs = [c for c, _ in run.pts]
    scores = [s for _, s in run.pts]
    lead = 1 if costs[0] > 0 else 0
    if lead:                                      # flat lead-in from $0
        costs.insert(0, 0.0)
        scores.insert(0, scores[0])
    if run.total > costs[-1]:                     # money spent after the last best
        costs.append(run.total)
        scores.append(scores[-1])
    if costs[-1] < BUDGET:                        # extend flat out to the full budget
        costs.append(BUDGET)
        scores.append(scores[-1])
    return Curve(costs, scores, run.total, lead, lead + len(run.pts),
                 len(run.pts) - run.n_seeded)


# ---------------------------------------------------------------------------
# Y range: ignore the very first eval of each run
# ---------------------------------------------------------------------------
def _step_value(costs: list[float], scores: list[float], x: float) -> float:
    """Best-so-far value at cost ``x`` (post-step lookup)."""
    y = scores[0]
    for c, s in zip(costs, scores):
        if c > x:
            break
        y = s
    return y


def _ylim(data: dict, ymin: float | None, ymax: float | None) -> tuple[float, float]:
    """Axis limits, taking any pinned value from TASKS and deriving the rest.

    The derived bottom crops so one bad first evaluation can't flatten every
    curve.

    A run's opening evaluation is often a wild outlier — SpecEvo's first
    llm_sql program scores 0.115, its first circle_packing one 1.60 against a
    2.6 finish — and leaving it in squeezes the whole comparison into a sliver
    at the top of the axes.  Each run therefore contributes its *second* logged
    point as its floor (its only point, if that is all it has).  The curves are
    best-so-far, hence monotone, so that floor really is the lowest value that
    run shows anywhere on screen.
    """
    lo = min(cv.scores[min(cv.m0 + 1, cv.m1 - 1)] for cv in data.values())
    hi = max(max(cv.scores) for cv in data.values())
    span = (hi - lo) or 1.0
    return (ymin if ymin is not None else lo - 0.10 * span,
            ymax if ymax is not None else hi + 0.08 * span)


def _covered_curves(data: dict[str, Curve], yspan: float) -> set[str]:
    """Methods an all-but-identical later curve would draw straight over.

    AdaEvolve never improves on the llm_sql seed (0.6754) and EvoX spends most
    of the run at 0.6749 — a gap of 0.0005, invisible at this scale, so EvoX
    (drawn second) erases AdaEvolve entirely.  Every line stays solid, so the
    fix is to widen the buried one until it shows as a rim around the line on
    top of it.
    """
    # only near-identical curves qualify: a wider tolerance starts thickening
    # lines that are merely close (SpecEvo vs GEPA mid-run on signal), and the
    # extra weight then reads as emphasis that isn't there.
    tol = 0.006 * yspan
    grid = [BUDGET * i / 200 for i in range(201)]
    ys = {m: [_step_value(cv.costs, cv.scores, x) for x in grid]
          for m, cv in data.items()}
    order = list(data)                             # data is in draw order
    return {
        early for i, early in enumerate(order)
        for late in order[i + 1:]
        if sum(abs(a - b) < tol for a, b in zip(ys[early], ys[late]))
        > 0.25 * len(grid)
    }


def style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, color="#DCDCDC", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=12)


def draw(ax, task_dir: Path, title: str, ymin: float | None = None,
         ymax: float | None = None, verbose: bool = True) -> dict[str, Curve]:
    data: dict[str, Curve] = {}
    for method in METHOD_ORDER:
        run_log = task_dir / method / "run.log"
        if not run_log.exists():
            continue
        loaded = load_method(run_log)
        if loaded is None:
            if verbose:
                print(f"  {method:5s}: no usable data — skipped")
            continue
        data[method] = loaded

    if not data:
        return {}

    ylo, yhi = _ylim(data, ymin, ymax)
    covered = _covered_curves(data, yhi - ylo)

    style_axes(ax)
    for method, cv in data.items():
        _, color, marker, ours = STYLE[method]
        lw = 3.4 if ours else 2.4
        ax.step(cv.costs, cv.scores, where="post", color=color,
                lw=lw + 1.8 if method in covered else lw,
                alpha=1.0 if ours else 0.9,
                zorder=6 if ours else 4, solid_capstyle="round")
        # markers only on logged points (skip the synthetic lead-in/tail)
        mx, my = cv.costs[cv.m0:cv.m1], cv.scores[cv.m0:cv.m1]
        ax.plot(mx, my, linestyle="none", marker=marker,
                ms=5.5 if ours else 4.5, color=color,
                mec="white", mew=0.6, zorder=7 if ours else 5)

    ax.set_xlim(0, BUDGET)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(XTICKS)
    ax.set_xlabel("Cost", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)

    # No legend on purpose: these panels share one legend added downstream.
    if verbose:
        for method, cv in data.items():
            print(f"  {STYLE[method][0]:11s}: {cv.n_newbest:3d} new bests | "
                  f"final score {cv.scores[-1]:.4f} | run stopped at ${cv.total:.2f}")
    return data


def save_legend(out: Path) -> None:
    """Standalone horizontal legend for the five methods (shared by panels)."""
    # SpecEvo first (ours), then the four baselines.
    order = ["spec", "evox", "ada", "gepa", "oe"]
    handles = []
    for method in order:
        label, color, marker, ours = STYLE[method]
        handles.append(
            plt.Line2D(
                [0], [0],
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=2.4 if ours else 1.8,
                markersize=9 if ours else 8,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=label,
            )
        )

    fig = plt.figure(figsize=(9.5, 0.55))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=14,
        handlelength=2.2,
        handletextpad=0.4,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08, dpi=220)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved -> {out}")


def main(tasks: list[str] | None = None) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.0,
                         "figure.dpi": 130})

    # Always emit the shared horizontal legend (for LaTeX panels).
    save_legend(LOG_ROOT / "cost_vs_score_budget10_legend.png")

    if tasks == ["legend"]:
        return

    candidates = tasks or [t for t in TASKS if (LOG_ROOT / t).is_dir()]
    drawn = []
    for task in candidates:
        task_dir = LOG_ROOT / task
        if not task_dir.is_dir():
            raise SystemExit(f"No such log folder: {task_dir}")
        title, ymin, ymax = TASKS.get(task, (task, None, None))
        print(f"\n=== {task}  ->  {title} ===")
        fig, ax = plt.subplots(figsize=(6.6, 4.3))
        if not draw(ax, task_dir, title, ymin, ymax):
            print("  (no data — nothing plotted)")
            plt.close(fig)
            continue
        out = task_dir / "cost_vs_score_budget10.png"
        fig.savefig(out, bbox_inches="tight", dpi=220)
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {out}")
        drawn.append((task, title, ymin, ymax))

    if len(drawn) > 1:                    # side-by-side panel of everything
        fig, axes = plt.subplots(1, len(drawn), figsize=(6.6 * len(drawn), 4.3))
        for ax, (task, title, ymin, ymax) in zip(axes, drawn):
            draw(ax, LOG_ROOT / task, title, ymin, ymax, verbose=False)
        fig.tight_layout()
        out = LOG_ROOT / "cost_vs_score_budget10_all.png"
        fig.savefig(out, bbox_inches="tight", dpi=220)
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
