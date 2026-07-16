#!/usr/bin/env python3
"""Compact grouped bar chart: Heilbronn Triangle hyperparameter ablation.

Three zones along the x-axis, one per ablated knob ("k-means clusters",
"navigator frequency", and "advisor frequency"). Within each zone the three 
parameter values (20/50/100) sit as slim bars snug against each other, each 
a distinct shade. Scores cluster tightly (~0.0350-0.0361), so the y-axis is 
zoomed into the data band with a break marker at the origin to stay honest 
about the truncation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Higher is better.
KMEANS = {"20": 0.0351, "50": 0.0361, "100": 0.0353}
NAVIGATOR_FREQ = {"20": 0.0361, "50": 0.0357, "100": 0.0350}
ADVISOR_FREQ = {"20": 0.0351, "50": 0.0359, "100": 0.0355}

OUT_DIR = Path(__file__).resolve().parents[2] / "result" / "heilbronn_triangle" / "ablation"

# Each zone gets a single hue ramped light->dark across the three param values.
ZONES = [
    {
        "label": "k-means clusters",
        "scores": KMEANS,
        "colors": ["#A9C2DD", "#6B97C4", "#3B6EA5"],  # blue ramp
    },
    {
        "label": "navigator frequency",
        "scores": NAVIGATOR_FREQ,
        "colors": ["#EBBDA9", "#DD8E6E", "#D1603D"],  # terracotta ramp
    },
    {
        "label": "advisor frequency",
        "scores": ADVISOR_FREQ,
        "colors": ["#B8DDB8", "#7FC87F", "#4CAF50"],  # green ramp
    },
]
SETTINGS = ["20", "50", "100"]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
        }
    )

    all_scores = [zone["scores"][s] for zone in ZONES for s in SETTINGS]
    lo, hi = min(all_scores), max(all_scores)
    span = hi - lo
    y_bottom = lo - span * 0.45
    y_top = hi + span * 0.55

    fig, ax = plt.subplots(figsize=(3.8, 1.9))

    bar_w = 0.16
    in_gap = 0.012                # gap between slim bars inside a zone
    n = len(SETTINGS)
    zone_span = n * bar_w + (n - 1) * in_gap
    zone_pitch = zone_span + 0.22  # small gutter between zones

    label_off = span * 0.06
    zone_centres = []
    for zi, zone in enumerate(ZONES):
        centre = zi * zone_pitch
        zone_centres.append(centre)
        scores = [zone["scores"][s] for s in SETTINGS]
        best_score = max(scores)
        
        # Slim bars packed snug around the zone centre.
        offsets = (np.arange(n) - (n - 1) / 2) * (bar_w + in_gap)
        xs = centre + offsets
        ax.bar(xs, scores, width=bar_w, color=zone["colors"], edgecolor="white",
               linewidth=0.4, zorder=3)
        
        for x, score in zip(xs, scores):
            # Add percentage decrease text above the score number for non-best bars
            if score < best_score:
                decrease_pct = ((best_score - score) / best_score) * 100
                # Place red text above the score
                ax.text(x, score + label_off * 2.5, f'↓{decrease_pct:.1f}%', 
                       ha="center", va="bottom", fontsize=5, color="red", 
                       fontweight="bold")
            
            # Score number
            ax.text(x, score + label_off, f"{score:.4f}", ha="center", va="bottom",
                    fontsize=5, color="#333333")

    ax.set_xticks(zone_centres)
    ax.set_xticklabels([z["label"] for z in ZONES], fontsize=6)
    ax.set_ylabel("Score", fontsize=6.5, labelpad=3)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", labelsize=6, pad=2, length=3)

    ax.set_ylim(y_bottom, y_top)
    half_zone = zone_span / 2
    ax.set_xlim(zone_centres[0] - half_zone - 0.06, zone_centres[-1] + half_zone + 0.06)

    yticks = np.linspace(lo - span * 0.2, hi + span * 0.2, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.4f}" for v in yticks])

    # Axis-break marker on the y-axis: diagonal slashes signalling the axis
    # does not start at zero.
    d = 0.012
    kw = dict(transform=ax.transAxes, color="#444444", clip_on=False, lw=0.8)
    ax.plot((-d, +d), (0.02 - d, 0.02 + d), **kw)
    ax.plot((-d, +d), (0.05 - d, 0.05 + d), **kw)

    # Legend mapping each shade to its parameter value (shared meaning across
    # zones, shown with a neutral grey ramp).
    from matplotlib.patches import Patch
    grey_ramp = ["#C8C8C8", "#8A8A8A", "#4D4D4D"]
    legend_elements = [Patch(facecolor=grey_ramp[i], label=SETTINGS[i]) for i in range(n)]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.99),
              ncol=n, fontsize=5.5, frameon=False, handlelength=0.8, columnspacing=0.8,
              handletextpad=0.3, title="Parameter value", title_fontsize=5.5,
              borderaxespad=0.0)

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
