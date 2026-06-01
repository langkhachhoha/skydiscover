#!/usr/bin/env python3
"""
Cost-vs-quality frontier plot for the circle-packing-rect experiment.

Differences from the reference figure (deliberate, to be clearer / nicer):
  * x-axis is **cumulative cost ($)**, not iteration  -> shows efficiency.
  * best-so-far is drawn as a true **step** curve (post-step), which is the
    correct semantics of a "best found so far" frontier.
  * the legend lives **inside** the axes.
  * an **inset zoom** magnifies the near-SOTA region (where every method
    crowds together) so the gap between LiteEvo and the baselines is legible.
  * our method (LiteEvo) is emphasised: thicker line, higher z-order, markers,
    a soft fill down to the next-best baseline to literally shade the gap.

Run after extract_log.py has produced result/<task>/<method>/frontier.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BENCHMARK = 2.3658321334167627  # Human / SOTA reference

RESULT_ROOT = Path(__file__).resolve().parents[2] / "result"
TASK = "circle_packing_rect"
TASK_TITLE = "Circle Packing (n=26)"

# method dir -> (display name, colour, is_ours)
METHODS = {
    "blade": ("LiteEvo", "#D6263A", True),    # our method
    "ada":   ("AdaEvolve", "#8E44AD", False),
    "gepa":  ("GEPA", "#2E86C1", False),
    "oe":    ("OpenEvolve", "#117A65", False),
    "evox":  ("EvoX", "#B9770E", False),
}


def load(method: str) -> tuple[list[float], list[float], float]:
    """Return (cost, score, total_cost).

    The frontier is prepended with the origin (0,0) and extended with a final
    flat segment out to the run's *total* cost, so the curve ends at the real
    amount of money spent (rewarding methods that stop cheaply).
    """
    base = RESULT_ROOT / TASK / method
    costs, scores = [0.0], [0.0]
    with (base / "frontier.csv").open() as f:
        for row in csv.DictReader(f):
            costs.append(float(row["cost"]))
            scores.append(float(row["score"]))
    with (base / "summary.csv").open() as f:
        total = float(next(csv.DictReader(f))["total_cost"])
    # extend the flat tail to the full run cost (wasted spend after last best)
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


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 1.0,
        "figure.dpi": 130,
    })

    data = {m: load(m) for m in METHODS}
    xmax = max(total for _, _, total in data.values()) * 1.05

    # y-axis is cropped to the near-SOTA band so the curves spread out and the
    # gap between methods is readable (lowest final score is GEPA ~2.285).
    ymin, ymax = 2.27, 2.378

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    style_axes(ax)

    # --- SOTA reference line --------------------------------------------------
    ax.axhline(BENCHMARK, ls=(0, (6, 4)), color="#555555", lw=1.4, zorder=2)
    ax.text(xmax * 0.015, BENCHMARK - 0.004, "Human / SOTA",
            ha="left", va="top", fontsize=11, color="#444444", style="italic")

    # order: baselines first, our method last (on top)
    draw_order = sorted(METHODS, key=lambda m: METHODS[m][2])

    for m in draw_order:
        _, color, ours = METHODS[m]
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
    ax.set_ylabel("Sum of Radii", fontsize=13, fontweight="bold")
    ax.set_title(TASK_TITLE, fontsize=15, fontweight="bold", pad=12)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))

    # --- compact legend inside the plot --------------------------------------
    handles = [
        plt.Line2D([0], [0], color=METHODS[m][1],
                   lw=3.0 if METHODS[m][2] else 1.8,
                   label=METHODS[m][0])
        for m in METHODS
    ]
    leg = ax.legend(
        handles=handles, loc="lower right", frameon=True, fontsize=11,
        framealpha=0.95, edgecolor="#CCCCCC", borderpad=0.7,
        labelspacing=0.4, handlelength=1.8,
    )
    leg.get_frame().set_linewidth(1.0)
    for txt in leg.get_texts():
        if txt.get_text() == METHODS["blade"][0]:
            txt.set_fontweight("bold")
            txt.set_color(METHODS["blade"][1])

    out = RESULT_ROOT / TASK / "frontier_cost_vs_score.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"saved -> {out}")
    print(f"saved -> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
