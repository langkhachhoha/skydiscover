#!/usr/bin/env python3
"""
Cost-vs-quality frontier plot — works for any benchmark task under result/.

Design (deliberate, to be clearer / nicer than the iteration-based reference):
  * x-axis is **cumulative cost ($)**, not iteration  -> shows efficiency.
  * best-so-far is drawn as a true **step** curve (post-step), the correct
    semantics of a "best found so far" frontier.
  * the legend lives **inside** the axes.
  * the y-axis is cropped to the near-SOTA band (auto-computed from the data)
    so the gap between methods is legible.
  * each curve ends at a dot marking the run's *total* cost (all money spent),
    so a method is rewarded for reaching a high score *and* stopping cheaply.
  * our method (SpecEvo / blade) is emphasised: thicker line, higher z-order.

Usage:
  .venv/bin/python scripts/plots/plot_frontier.py [task]
where [task] is a folder under result/ (default: circle_packing_rect).
Run extract_log.py first to produce result/<task>/<method>/frontier.csv.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "result"
BENCH_ROOT = ROOT / "benchmarks" / "math"

# shared palette / display names; folder aliases (ada vs adaevo) both map here
STYLE = {
    "blade":  ("SpecEvo", "#D6263A", True),    # our method (data from LiteEvo/blade)
    "ada":    ("AdaEvolve", "#8E44AD", False),
    "adaevo": ("AdaEvolve", "#8E44AD", False),
    "gepa":   ("GEPA", "#2E86C1", False),
    "oe":     ("OpenEvolve", "#117A65", False),
    "evox":   ("EvoX", "#B9770E", False),
}

# per-task display config; y-limits/ticks are auto-derived if omitted
TASKS = {
    "circle_packing_rect": dict(title="Circle Packing (n=21)", ylabel="Sum of Radii"),
    "heilbronn_triangle":  dict(title="Heilbronn Triangle (n=11)", ylabel="Min Area (normalized)"),
    "EPLB":                dict(title="EPLB", ylabel="Combined Score"),
}


def task_benchmark(task: str) -> float | None:
    """Read the SOTA reference (BENCHMARK constant) from the task evaluator."""
    ev = BENCH_ROOT / task / "evaluator" / "evaluator.py"
    if not ev.exists():
        return None
    m = re.search(r"^BENCHMARK\s*=\s*([-\d.eE]+)", ev.read_text(), re.MULTILINE)
    return float(m.group(1)) if m else None


def discover_methods(task: str) -> list[str]:
    """Method folders present for this task, ordered baselines-first, ours-last.

    Skips methods whose frontier is empty *and* whose run spent $0 (e.g. an
    empty placeholder ``run.txt``) so they don't draw a degenerate (0,0) line.
    """
    tdir = RESULT_ROOT / task
    found = []
    for p in tdir.iterdir():
        if not p.is_dir() or not (p / "frontier.csv").exists():
            continue
        with (p / "frontier.csv").open() as f:
            n_rows = sum(1 for _ in csv.DictReader(f))
        total = 0.0
        summary = p / "summary.csv"
        if summary.exists():
            with summary.open() as f:
                row = next(csv.DictReader(f), None)
                if row:
                    total = float(row["total_cost"])
        if n_rows == 0 and total <= 0:
            continue
        found.append(p.name)
    return sorted(found, key=lambda m: STYLE.get(m, ("", "", False))[2])


def load(task: str, method: str) -> tuple[list[float], list[float], float]:
    """Return (cost, score, total_cost).

    The frontier is prepended with the origin (0,0) and extended with a final
    flat segment out to the run's *total* cost, so the curve ends at the real
    amount of money spent (rewarding methods that stop cheaply).
    """
    base = RESULT_ROOT / task / method
    costs, scores = [0.0], [0.0]
    with (base / "frontier.csv").open() as f:
        for row in csv.DictReader(f):
            costs.append(float(row["cost"]))
            scores.append(float(row["score"]))
    with (base / "summary.csv").open() as f:
        total = float(next(csv.DictReader(f))["total_cost"])
    if total > costs[-1]:
        costs.append(total)
        scores.append(scores[-1])
    return costs, scores, total


def style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, which="major", color="#DCDCDC", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)


def main(task: str = "circle_packing_rect") -> None:
    cfg = TASKS.get(task, dict(title=task, ylabel="Score"))
    benchmark = task_benchmark(task)
    methods = discover_methods(task)
    if not methods:
        raise SystemExit(f"No frontier.csv under result/{task}/ — run extract_log.py first.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 1.0,
        "figure.dpi": 130,
    })

    data = {m: load(task, m) for m in methods}
    xmax = max(total for _, _, total in data.values()) * 1.05

    # crop the y-axis to the near-SOTA band so the curves spread out: from a
    # little below the lowest final score up to a little above SOTA.
    finals = [score[-1] for _, score, _ in data.values()]
    lo, hi = min(finals), max(benchmark or max(finals), max(finals))
    span = hi - lo
    ymin = lo - 0.18 * span
    ymax = hi + 0.10 * span

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    style_axes(ax)

    for m in methods:
        _, color, ours = STYLE.get(m, (m, "#888888", False))
        cost, score, total = data[m]
        ax.step(
            cost, score, where="post",
            color=color,
            lw=3.0 if ours else 1.8,
            alpha=1.0 if ours else 0.85,
            zorder=6 if ours else 4,
            solid_capstyle="round",
        )
        # single endpoint dot = where the run stopped (total cost spent)
        ax.plot(
            [total], [score[-1]], "o",
            color=color, ms=9 if ours else 6.5,
            mec="white", mew=1.2,
            zorder=8 if ours else 5,
        )

    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Cumulative Cost (USD)", fontsize=13, fontweight="bold")
    ax.set_ylabel(cfg["ylabel"], fontsize=13, fontweight="bold")
    ax.set_title(cfg["title"], fontsize=15, fontweight="bold", pad=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # --- compact legend inside the plot --------------------------------------
    handles = [
        plt.Line2D([0], [0], color=STYLE.get(m, (m, "#888888", False))[1],
                   lw=3.0 if STYLE.get(m, (m, "", False))[2] else 1.8,
                   label=STYLE.get(m, (m, "", False))[0])
        for m in methods
    ]
    leg = ax.legend(
        handles=handles, loc="lower right", frameon=True, fontsize=11,
        framealpha=0.45, edgecolor="#CCCCCC", borderpad=0.7,
        labelspacing=0.4, handlelength=1.8,
    )
    leg.get_frame().set_linewidth(1.0)
    leg.set_zorder(10)  # keep labels readable above the curves
    for txt in leg.get_texts():
        if txt.get_text() == "SpecEvo":
            txt.set_fontweight("bold")
            txt.set_color(STYLE["blade"][1])

    out = RESULT_ROOT / task / "frontier_cost_vs_score.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"saved -> {out}")
    print(f"saved -> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "circle_packing_rect")
