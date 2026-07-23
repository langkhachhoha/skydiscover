#!/usr/bin/env python3
"""
Horizontal bar charts of per-method averages across CO-Bench categories.

Data = Average rows from the Packing / Cutting / Facility results table.
Style mirrors the "Base LLMs" horizontal-bar reference: bold section header,
value labels to the right of each bar, light vertical grid, x-axis cropped
to the near-data band so gaps between methods stay legible.

Usage:
  .venv/bin/python scripts/plots/plot_category_averages.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "result" / "category_averages"

# Fixed top→bottom order (do not sort by score)
METHODS = ("SpecEvo", "EvoX", "GEPA", "AdaEvolve", "OpenEvolve")

# Same palette as plot_frontier / plot_benchmark_cost_score
COLORS = {
    "OpenEvolve": "#117A65",
    "GEPA": "#2E86C1",
    "AdaEvolve": "#8E44AD",
    "EvoX": "#B9770E",
    "SpecEvo": "#D6263A",
}

# Average rows extracted from the results table
CATEGORIES = {
    "Packing Problems": {
        "OpenEvolve": 0.8346,
        "GEPA": 0.8432,
        "AdaEvolve": 0.8839,
        "EvoX": 0.8575,
        "SpecEvo": 0.8937,
    },
    "Cutting Problems": {
        "OpenEvolve": 0.7969,
        "GEPA": 0.8552,
        "AdaEvolve": 0.8518,
        "EvoX": 0.8559,
        "SpecEvo": 0.9657,
    },
    "Facility Location Problems": {
        "OpenEvolve": 0.9657,
        "GEPA": 0.9700,
        "AdaEvolve": 0.9672,
        "EvoX": 0.9258,
        "SpecEvo": 0.9454,
    },
    "Scheduling Problems": {
        "OpenEvolve": 0.8174,
        "GEPA": 0.8583,
        "AdaEvolve": 0.8346,
        "EvoX": 0.8047,
        "SpecEvo": 0.8700,
    },
}


def _xlim(vals: list[float]) -> tuple[float, float]:
    """Crop x to the data band with a little padding so gaps stand out."""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = max(0.012, 0.22 * span)
    return lo - pad, hi + pad * 1.35  # extra right room for value labels


def plot_category(title: str, scores: dict[str, float], out: Path) -> None:
    methods = list(METHODS)
    vals = [scores[m] for m in methods]
    n = len(methods)

    fig_h = 0.70 + 0.48 * n
    fig, ax = plt.subplots(figsize=(6.2, fig_h))
    y = np.arange(n)

    bars = ax.barh(
        y,
        vals,
        height=0.72,
        color=[COLORS[m] for m in methods],
        edgecolor="none",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=11)
    ax.invert_yaxis()

    # value frame = data band; exactly 5 evenly spaced ticks inside it.
    # xlim extends a bit past the last tick so value labels have room.
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-6)
    pad = max(0.012, 0.18 * span)
    tick_lo, tick_hi = lo - pad, hi + pad
    ticks = np.linspace(tick_lo, tick_hi, 5)
    ax.set_xlim(tick_lo, tick_hi + 0.28 * (tick_hi - tick_lo))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2f}" for t in ticks])

    ax.xaxis.grid(True, linestyle="-", color="#D8D8D8", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.xaxis.set_ticks_position("none")
    ax.set_xlabel("")

    # bold section header (top-left), like "Base LLMs"
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)

    xmin, xmax = ax.get_xlim()
    dx = 0.012 * (xmax - xmin)
    for bar, val in zip(bars, vals):
        cy = bar.get_y() + bar.get_height() / 2
        ax.text(
            val + dx,
            cy,
            f"{val:.4f}",
            ha="left",
            va="center",
            fontsize=10,
            color="#222222",
        )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
    print(f"saved -> {out.with_suffix('.pdf')}")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 1.0,
        "figure.dpi": 130,
    })
    slug = {
        "Packing Problems": "packing",
        "Cutting Problems": "cutting",
        "Facility Location Problems": "facility",
        "Scheduling Problems": "scheduling",
    }
    for title, scores in CATEGORIES.items():
        plot_category(title, scores, OUT_DIR / f"avg_{slug[title]}.png")


if __name__ == "__main__":
    main()
