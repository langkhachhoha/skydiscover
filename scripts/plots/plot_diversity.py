#!/usr/bin/env python3
"""SWDI / CDI vs checkpoint for SpecEvo multi-prompt vs single-prompt.

Two figures (one per diversity index), each with two benchmarks (Circle Packing,
Circle Packing Rect). Color encodes benchmark; line style encodes prompt setting
-- the same visual grammar as scripts/plots/plot_advisor_error_rate.py.

Reads the JSON emitted by scripts/diversity/compute_diversity.py.

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

    fig, ax = plt.subplots(figsize=(5.4, 3.5))

    for name, cfg in BENCHMARKS.items():
        c = cfg["color"]
        rows = data[name]
        x = [r["checkpoint"] for r in rows["multi_prompt"]]
        ax.plot(
            x,
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
            [r["checkpoint"] for r in rows["single_prompt"]],
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

    ax.set_xlabel("Checkpoint", fontsize=10, labelpad=4)
    ax.set_ylabel(f"{label} ($\\uparrow$)", fontsize=10, labelpad=4)
    ax.set_xticks(range(1, 9))
    ax.set_xlim(0.7, 8.3)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)

    ax.tick_params(axis="both", labelsize=8, length=3.5, pad=2)
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
        path = OUT_DIR / f"diversity_{metric}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(path)
    plt.close(fig)


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
    data = load()
    for metric in METRICS:
        plot_metric(data, metric)


if __name__ == "__main__":
    main()
