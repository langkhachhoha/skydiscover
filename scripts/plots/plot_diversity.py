#!/usr/bin/env python3
"""SWDI / CDI vs iteration for SpecEvo multi-prompt vs single-prompt.

Two figures (one per diversity index), each with two benchmarks (Circle Packing,
Circle Packing Rect). Color encodes benchmark; line style encodes prompt setting
-- the same visual grammar as scripts/plots/plot_advisor_error_rate.py.

Reads the JSON emitted by scripts/diversity/compute_diversity.py.
Checkpoints 1..8 map to iterations 50..400 (step 50).

Usage:
  .venv/bin/python scripts/plots/plot_diversity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "result" / "diversity"

ITERATIONS = [50, 100, 150, 200, 250, 300, 350, 400]

BENCHMARKS = {
    "Circle Packing": {"dir": "circle_packing", "color": "#D6263A"},
    "Circle Packing Rect": {"dir": "circle_packing_rect", "color": "#2E86C1"},
}

# metric key -> (axis label, y limits, y ticks)
METRICS = {
    "swdi": ("SWDI", (2.1, 3.35), [2.2, 2.4, 2.6, 2.8, 3.0, 3.2]),
    "cdi": ("CDI", (3.55, 3.90), [3.55, 3.60, 3.65, 3.70, 3.75, 3.80, 3.85]),
}


def load() -> dict:
    data = {}
    for name, cfg in BENCHMARKS.items():
        path = ROOT / "diversity" / cfg["dir"] / "diversity_results.json"
        data[name] = json.loads(path.read_text())
    return data


def plot_metric(data: dict, metric: str) -> None:
    label, ylim, yticks = METRICS[metric]

    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    for name, cfg in BENCHMARKS.items():
        c = cfg["color"]
        rows = data[name]
        ax.plot(
            ITERATIONS,
            [r[metric] for r in rows["multi_prompt"]],
            color=c,
            linestyle="--",
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
            [r[metric] for r in rows["single_prompt"]],
            color=c,
            linestyle="-",
            linewidth=1.8,
            marker="s",
            markersize=4.2,
            markerfacecolor=c,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )

    ax.set_xlabel("Iteration", fontsize=14, labelpad=5)
    ax.set_ylabel(f"{label} ($\\uparrow$)", fontsize=14, labelpad=5)
    ax.set_xticks(ITERATIONS)
    ax.set_xlim(40, 410)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)

    ax.tick_params(axis="both", labelsize=12, length=3.5, pad=2)
    ax.grid(True, which="major", color="#E6E6E6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    names = list(BENCHMARKS)
    handles = [
        Patch(facecolor=BENCHMARKS[names[0]]["color"], edgecolor="none", label=names[0]),
        Line2D([0], [0], color="#555555", lw=1.8, linestyle="--", label="Multi-prompt"),
        Patch(facecolor=BENCHMARKS[names[1]]["color"], edgecolor="none", label=names[1]),
        Line2D([0], [0], color="#555555", lw=1.8, linestyle="-", label="Single-prompt"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.95,
        handlelength=1.8,
        handleheight=0.9,
        borderpad=0.5,
        labelspacing=0.4,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"diversity_{metric}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(path)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.linewidth": 0.9,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    data = load()
    for metric in METRICS:
        plot_metric(data, metric)


if __name__ == "__main__":
    main()
