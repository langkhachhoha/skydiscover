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

# (json file in repo root, output png in repo root)
DATASETS = [
    ("math_gpt.json", "math_gpt_cost.png"),
    ("math_kimi.json", "math_kimi_cost.png"),
    ("system_gpt.json", "system_gpt_cost.png"),
    ("system_kimi.json", "system_kimi_cost.png"),
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
    fig_h = 1.0 + 0.55 * n
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    y = np.arange(n)
    bar_h = 0.82
    bars = ax.barh(
        y,
        mean_vals,
        height=bar_h,
        color=[COLORS.get(m, "#888888") for m in methods],
        edgecolor="none",
        xerr=std_vals,
        error_kw=dict(ecolor="#333333", elinewidth=1.2, capsize=4, capthick=1.2),
    )

    ax.set_xlabel("Total cost ($)", fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    xmax = max(mean_vals[i] + std_vals[i] for i in range(n))
    ax.set_xlim(0, xmax * 1.5)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(y=0.02)

    hi_idx = methods.index(HIGHLIGHT)
    cost_dx = max(0.3, xmax * 0.015)
    mult_dx = max(2.6, xmax * 0.34)
    for i, (bar, mean, std) in enumerate(zip(bars, mean_vals, std_vals)):
        cy = bar.get_y() + bar.get_height() / 2
        ax.text(
            mean + std + cost_dx,
            cy,
            f"${mean:.1f} ± {std:.1f}",
            ha="left",
            va="center",
            fontsize=13,
        )
        if i == hi_idx:
            ax.text(
                mean + std + mult_dx,
                cy,
                f"{cheaper_x:.1f}×",
                ha="left",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=COLORS[HIGHLIGHT],
            )

    title = f"{data.get('dataset', json_path.stem)} — {data.get('model', '')}".strip(" —")
    ax.set_title(title, fontsize=14)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{json_path.name}] mean total cost: "
          + ", ".join(f"{m}=${means[m]:.1f}±{stds[m]:.1f}" for m in methods))
    print(f"[{json_path.name}] {HIGHLIGHT} vs avg. baseline: {cheaper_x:.2f}×")
    print(f"Wrote {out}")


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        targets = [(Path(a).name, Path(a).stem + "_cost.png") for a in sys.argv[1:]]
    else:
        targets = DATASETS
    for json_name, png_name in targets:
        plot(ROOT / json_name, ROOT / png_name)


if __name__ == "__main__":
    main()
