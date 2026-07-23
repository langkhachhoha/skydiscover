#!/usr/bin/env python3
"""Error-rate vs iteration for SpecEvo with / without advisor.

Two tasks (Circle Packing, Circle Packing Rect), each with solid (no advisor)
and dashed (with advisor) curves. Color encodes task; line style encodes
advisor setting.

Usage:
  .venv/bin/python scripts/plots/plot_advisor_error_rate.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "result" / "advisor_error_rate"

ITERATIONS = [50, 100, 150, 200, 250, 300, 350, 400]

# Error rates (%) at each iteration checkpoint
DATA = {
    "Circle Packing": {
        "color": "#D6263A",
        "no_advisor": [30.0, 36.0, 24.0, 22.0, 34.7, 20.4, 30.0, 38.0],
        "with_advisor": [30.0, 20.0, 18.0, 16.0, 8.0, 14.3, 6.0, 14.0],
    },
    "Circle Packing Rect": {
        "color": "#2E86C1",
        "no_advisor": [14.0, 26.0, 8.0, 12.2, 18.0, 8.2, 16.0, 12.0],
        "with_advisor": [18.0, 14.6, 12.0, 4.0, 2.0, 10.0, 6.0, 6.0],
    },
}


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.9,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.5))

    for task, cfg in DATA.items():
        c = cfg["color"]
        ax.plot(
            ITERATIONS,
            cfg["no_advisor"],
            color=c,
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            markerfacecolor=c,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.plot(
            ITERATIONS,
            cfg["with_advisor"],
            color=c,
            linestyle="--",
            linewidth=1.8,
            marker="s",
            markersize=4.2,
            markerfacecolor=c,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )

    ax.set_xlabel("Iteration", fontsize=10, labelpad=4)
    ax.set_ylabel("Error rate (%)", fontsize=10, labelpad=4)
    ax.set_xticks(ITERATIONS)
    ax.set_xlim(40, 410)
    ax.set_ylim(0, 48)
    ax.set_yticks([0, 10, 20, 30, 40])

    ax.tick_params(axis="both", labelsize=8, length=3.5, pad=2)
    ax.grid(True, which="major", color="#E6E6E6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Compact 2x2 legend inside axes, parked in the clear high band
    # (y≈38–43) over iter ~110–300 where every series stays ≤ 34.7%.
    # Column-fill → row1: tasks, row2: settings.
    task_names = list(DATA.keys())
    handles = [
        Patch(facecolor=DATA[task_names[0]]["color"], edgecolor="none", label=task_names[0]),
        Line2D([0], [0], color="#555555", lw=1.8, linestyle="-", label="No advisor"),
        Patch(facecolor=DATA[task_names[1]]["color"], edgecolor="none", label=task_names[1]),
        Line2D([0], [0], color="#555555", lw=1.8, linestyle="--", label="With advisor"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.45, 0.98),
        ncol=2,
        fontsize=7,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.95,
        handlelength=1.5,
        handleheight=0.8,
        borderpad=0.4,
        labelspacing=0.35,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"advisor_error_rate.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(path)


if __name__ == "__main__":
    main()
