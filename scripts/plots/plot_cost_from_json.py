#!/usr/bin/env python3
"""
Horizontal total-cost bars from seed-run JSON files.

Same layout as plot_benchmark_cost_score.py: one horizontal bar per method,
showing the mean total cost (averaged over the 3 seed runs) with a std error
bar. SpecEvo is annotated with × cheaper vs the average of the baselines.

For each method we compute the per-seed total cost (sum over tasks), then take
the mean and std across the 3 seeds.

Usage:
  python scripts/plots/plot_cost_from_json.py
  python scripts/plots/plot_cost_from_json.py math_gpt.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "OpenEvolve": "#117A65",
    "GEPA": "#2E86C1",
    "AdaEvolve": "#8E44AD",
    "EvoX": "#B9770E",
    "SpecEvo": "#c9a227",
}

HIGHLIGHT = "SpecEvo"

# (json file in repo root, output png under image/)
DATASETS = [
    ("math_gpt.json", "image/math_gpt_cost.png"),
    ("math_kimi.json", "image/math_kimi_cost.png"),
    ("system_gpt.json", "image/system_gpt_cost.png"),
    ("system_kimi.json", "image/system_kimi_cost.png"),
]


def per_seed_totals(data: dict) -> dict[str, list[float]]:
    """For each method, return the list of total costs (one per seed)."""
    methods = data["methods"]
    totals: dict[str, list[float]] = {m: [] for m in methods}
    for seed in data["seeds"]:
        seed_total = {m: 0.0 for m in methods}
        for res in seed["results"]:
            for m in methods:
                seed_total[m] += res[m]["cost"]
        for m in methods:
            totals[m].append(seed_total[m])
    return totals


def plot(json_path: Path, out: Path) -> None:
    data = json.loads(json_path.read_text())
    methods = data["methods"]
    totals = per_seed_totals(data)

    means = {m: float(np.mean(totals[m])) for m in methods}
    stds = {m: float(np.std(totals[m])) for m in methods}

    baselines = [m for m in methods if m != HIGHLIGHT]
    baseline_avg = float(np.mean([means[m] for m in baselines]))
    cheaper_x = baseline_avg / means[HIGHLIGHT]

    mean_vals = [means[m] for m in methods]
    std_vals = [stds[m] for m in methods]

    n = len(methods)
    fig_h = 1.25 + 0.65 * n
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    y = np.arange(n)
    bar_h = 0.82
    bars = ax.barh(
        y,
        mean_vals,
        height=bar_h,
        color=[COLORS.get(m, "#888888") for m in methods],
        edgecolor="none",
        xerr=std_vals,
        error_kw=dict(ecolor="#333333", elinewidth=1.4, capsize=5, capthick=1.4),
    )

    ax.set_xlabel("Total cost ($)", fontsize=17)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=16)
    ax.tick_params(axis="x", labelsize=15)
    xmax = max(mean_vals[i] + std_vals[i] for i in range(n))
    cost_dx = max(2.0, xmax * 0.045)
    # Large fonts → cost text spans many data units; keep × clear of it.
    mult_x = means[HIGHLIGHT] + stds[HIGHLIGHT] + cost_dx + max(28.0, xmax * 0.48)
    ax.set_xlim(0, max(xmax * 1.65, mult_x + max(10.0, xmax * 0.20)))
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(y=0.02)

    hi_idx = methods.index(HIGHLIGHT)
    for i, (bar, mean, std) in enumerate(zip(bars, mean_vals, std_vals)):
        cy = bar.get_y() + bar.get_height() / 2
        ax.text(
            mean + std + cost_dx,
            cy,
            f"${mean:.1f} ± {std:.1f}",
            ha="left",
            va="center",
            fontsize=19,
        )
        if i == hi_idx:
            ax.text(
                mult_x,
                cy,
                f"{cheaper_x:.1f}×",
                ha="left",
                va="center",
                fontsize=24,
                fontweight="bold",
                color=COLORS[HIGHLIGHT],
            )

    title = f"{data.get('dataset', json_path.stem)} — {data.get('model', '')}".strip(" —")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[{json_path.name}] mean total cost: "
          + ", ".join(f"{m}=${means[m]:.1f}±{stds[m]:.1f}" for m in methods))
    print(f"[{json_path.name}] {HIGHLIGHT} vs avg. baseline: {cheaper_x:.2f}×")
    print(f"Wrote {out}")


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        targets = [
            (Path(a).name, f"image/{Path(a).stem}_cost.png") for a in sys.argv[1:]
        ]
    else:
        targets = DATASETS
    for json_name, png_name in targets:
        plot(ROOT / json_name, ROOT / png_name)


if __name__ == "__main__":
    main()
