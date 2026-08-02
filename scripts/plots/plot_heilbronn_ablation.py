#!/usr/bin/env python3
"""Grouped bar chart: Heilbronn Triangle + EPLB hyperparameter ablation.

Three zones along the x-axis ("k-means clusters", "navigator interval",
"advisor interval"). Within each zone: 6 slim bars — first 3 Heilbronn
Triangle, last 3 EPLB — for parameter values 20/50/100. Matching values
share a color; EPLB bars add horizontal hatching. Y-axis is % of the best
score within each (benchmark, zone) group (best = 100%).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Higher is better.
HEILBRONN = {
    "k-means clusters": {"20": 0.0351, "50": 0.0361, "100": 0.0353},
    "navigator interval": {"20": 0.0361, "50": 0.0357, "100": 0.0350},
    "advisor interval": {"20": 0.0351, "50": 0.0359, "100": 0.0355},
}
EPLB = {
    "k-means clusters": {"20": 0.144386, "50": 0.1485, "100": 0.143624},
    "navigator interval": {"20": 0.1485, "50": 0.144382, "100": 0.144186},
    "advisor interval": {"20": 0.146079, "50": 0.1485, "100": 0.144789},
}

OUT_DIR = Path(__file__).resolve().parents[2] / "result" / "heilbronn_triangle" / "ablation"

ZONE_LABELS = ["k-means clusters", "navigator interval", "advisor interval"]
ZONE_COLORS = {
    "k-means clusters": ["#A9C2DD", "#6B97C4", "#3B6EA5"],       # blue
    "navigator interval": ["#EBBDA9", "#DD8E6E", "#D1603D"],    # terracotta
    "advisor interval": ["#B8DDB8", "#7FC87F", "#4CAF50"],      # green
}
SETTINGS = ["20", "50", "100"]
BENCHMARKS = [
    {"name": "Heilbronn Triangle", "data": HEILBRONN, "hatch": None},
    {"name": "EPLB", "data": EPLB, "hatch": "---"},
]


def to_pct(scores: dict[str, float]) -> list[float]:
    """Normalize a zone's scores so the best is 100%."""
    best = max(scores[s] for s in SETTINGS)
    return [100.0 * scores[s] / best for s in SETTINGS]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
            "hatch.linewidth": 0.7,
        }
    )

    all_pcts: list[float] = []
    for zone in ZONE_LABELS:
        for bench in BENCHMARKS:
            all_pcts.extend(to_pct(bench["data"][zone]))
    lo, hi = min(all_pcts), max(all_pcts)
    span = hi - lo
    y_bottom = lo - span * 0.45
    y_top = hi + span * 0.55

    fig, ax = plt.subplots(figsize=(3.8, 1.9))

    # Narrower bars so 6 columns fit in roughly the same figure size as before.
    bar_w = 0.085
    in_gap = 0.008                 # gap between bars within a benchmark trio
    group_gap = 0.04               # gap between Heilbronn and EPLB within a zone
    n = len(SETTINGS)
    n_bench = len(BENCHMARKS)
    trio_span = n * bar_w + (n - 1) * in_gap
    zone_span = n_bench * trio_span + (n_bench - 1) * group_gap
    zone_pitch = zone_span + 0.18  # gutter between zones

    zone_centres = []
    for zi, zone in enumerate(ZONE_LABELS):
        centre = zi * zone_pitch
        zone_centres.append(centre)
        colors = ZONE_COLORS[zone]

        # Two trios packed around the zone centre: Heilbronn then EPLB.
        trio_offsets = (np.arange(n_bench) - (n_bench - 1) / 2) * (trio_span + group_gap)
        for bi, bench in enumerate(BENCHMARKS):
            pcts = to_pct(bench["data"][zone])
            bar_offsets = (np.arange(n) - (n - 1) / 2) * (bar_w + in_gap)
            xs = centre + trio_offsets[bi] + bar_offsets
            edge = "#333333" if bench["hatch"] else "white"
            ax.bar(
                xs,
                pcts,
                width=bar_w,
                color=colors,
                edgecolor=edge,
                linewidth=0.4,
                hatch=bench["hatch"],
                zorder=3,
            )

    ax.set_xticks(zone_centres)
    ax.set_xticklabels(ZONE_LABELS, fontsize=6)
    ax.set_ylabel("Relative Score (%)", fontsize=6.5, labelpad=3)
    ax.axhline(100.0, color="#444444", linestyle="--", linewidth=0.7, zorder=2)
    ax.text(
        1.0,
        100.0,
        "100",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=5.5,
        color="#444444",
        clip_on=False,
    )
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", labelsize=6, pad=2, length=3)

    ax.set_ylim(y_bottom, y_top)
    half_zone = zone_span / 2
    ax.set_xlim(zone_centres[0] - half_zone - 0.06, zone_centres[-1] + half_zone + 0.06)

    yticks = np.linspace(lo - span * 0.2, hi + span * 0.2, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.1f}" for v in yticks])

    # Axis-break marker: y-axis does not start at zero.
    d = 0.012
    kw = dict(transform=ax.transAxes, color="#444444", clip_on=False, lw=0.8)
    ax.plot((-d, +d), (0.02 - d, 0.02 + d), **kw)
    ax.plot((-d, +d), (0.05 - d, 0.05 + d), **kw)

    grey_ramp = ["#C8C8C8", "#8A8A8A", "#4D4D4D"]
    legend_elements = [Patch(facecolor=grey_ramp[i], label=SETTINGS[i]) for i in range(n)]
    legend_elements += [
        Patch(
            facecolor="#B0B0B0",
            edgecolor="white",
            linewidth=0.4,
            label="Heilbronn Triangle",
        ),
        Patch(
            facecolor="#B0B0B0",
            edgecolor="#333333",
            linewidth=0.4,
            hatch="---",
            label="EPLB",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(legend_elements),
        fontsize=5.0,
        frameon=False,
        handlelength=0.9,
        columnspacing=0.55,
        handletextpad=0.25,
        title="Parameter value",
        title_fontsize=5.5,
        borderaxespad=0.0,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="y", color="#E8E8E8", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"hyperparam_ablation.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
        print(path)


if __name__ == "__main__":
    main()
