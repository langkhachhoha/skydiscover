#!/usr/bin/env python3
"""
Total-cost bars from seed-run JSON files.

Primary output: one compact 1×4 row of *vertical* bar charts (4 suites)
with a shared colour legend at the bottom (no per-panel method names).

Also emits the legacy per-suite horizontal charts.

Usage:
  .venv/bin/python scripts/plots/plot_cost_from_json.py
  .venv/bin/python scripts/plots/plot_cost_from_json.py math_gpt.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "image"

# Left→right bar / legend order. SpecEvo first (ours).
METHODS = ("SpecEvo", "EvoX", "AdaEvolve", "GEPA", "OpenEvolve")

COLORS = {
    "OpenEvolve": "#117A65",
    "GEPA": "#2E86C1",
    "AdaEvolve": "#8E44AD",
    "EvoX": "#B9770E",
    "SpecEvo": "#c9a227",
}

HIGHLIGHT = "SpecEvo"

# (json in repo root, short panel title, legacy individual png)
DATASETS = [
    ("math_gpt.json", "Math — GPT-5", "math_gpt_cost.png"),
    ("math_kimi.json", "Math — KIMI K2", "math_kimi_cost.png"),
    ("system_gpt.json", "System — GPT-5", "system_gpt_cost.png"),
    ("system_kimi.json", "System — KIMI K2", "system_kimi_cost.png"),
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


def _stats(json_path: Path) -> tuple[list[str], dict[str, float], dict[str, float], float]:
    data = json.loads(json_path.read_text())
    # Keep METHODS order; drop any missing.
    methods = [m for m in METHODS if m in data["methods"]]
    totals = per_seed_totals(data)
    means = {m: float(np.mean(totals[m])) for m in methods}
    stds = {m: float(np.std(totals[m])) for m in methods}
    baselines = [m for m in methods if m != HIGHLIGHT]
    cheaper_x = float(np.mean([means[m] for m in baselines])) / means[HIGHLIGHT]
    return methods, means, stds, cheaper_x


def _draw_cost_v(
    ax,
    title: str,
    methods: list[str],
    means: dict[str, float],
    stds: dict[str, float],
    cheaper_x: float,
) -> None:
    mean_vals = [means[m] for m in methods]
    std_vals = [stds[m] for m in methods]
    n = len(methods)
    x = np.arange(n)

    bars = ax.bar(
        x,
        mean_vals,
        width=0.70,
        color=[COLORS.get(m, "#888888") for m in methods],
        edgecolor="none",
        yerr=std_vals,
        error_kw=dict(ecolor="#333333", elinewidth=1.3, capsize=4.5, capthick=1.3),
        zorder=3,
    )

    top = max(mean_vals[i] + std_vals[i] for i in range(n))
    hi_idx = methods.index(HIGHLIGHT) if HIGHLIGHT in methods else -1
    # Headroom: cost label, then × above SpecEvo, then title strip.
    ax.set_ylim(0, top * 1.58)
    ax.set_xlim(-0.6, n - 0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("")

    for i, (bar, mean, std) in enumerate(zip(bars, mean_vals, std_vals)):
        cx = bar.get_x() + bar.get_width() / 2
        y = mean + std + top * 0.028
        # Single large cost label; std is already shown by the error bar.
        ax.text(
            cx, y,
            f"${mean:.1f}",
            ha="center", va="bottom",
            fontsize=20, color="#1a1a1a", zorder=4,
        )
        if i == hi_idx:
            ax.text(
                cx, y + top * 0.13,
                f"{cheaper_x:.1f}×",
                ha="center", va="bottom",
                fontsize=22, fontweight="bold",
                color=COLORS[HIGHLIGHT], zorder=5,
            )

    ax.text(
        0.0, 0.985, title,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=20, fontweight="bold",
        color="#111111", zorder=6,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
            "pad": 1.2,
        },
    )


def plot_cost_grid(out: Path) -> None:
    """1×4 vertical-bar cost row + shared method colour legend."""
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.4))
    for i, (ax, (json_name, title, _)) in enumerate(zip(axes, DATASETS)):
        methods, means, stds, cheaper_x = _stats(ROOT / json_name)
        _draw_cost_v(ax, title, methods, means, stds, cheaper_x)
        if i == 0:
            ax.set_ylabel("Total cost ($)", fontsize=18, fontweight="bold", labelpad=6)
        else:
            ax.set_ylabel("")

    handles = [Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(METHODS),
        frameon=False,
        fontsize=17,
        handlelength=1.25,
        handleheight=1.05,
        handletextpad=0.45,
        columnspacing=1.55,
        borderaxespad=0.0,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(
        left=0.04, right=0.995, top=0.93, bottom=0.13,
        wspace=0.14,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"saved -> {out}")
    print(f"saved -> {out.with_suffix('.pdf')}")


def plot_horizontal(json_path: Path, out: Path) -> None:
    """Legacy single-suite horizontal bar chart."""
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
    ax.barh(
        y,
        mean_vals,
        height=0.82,
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
    mult_x = means[HIGHLIGHT] + stds[HIGHLIGHT] + cost_dx + max(28.0, xmax * 0.48)
    ax.set_xlim(0, max(xmax * 1.65, mult_x + max(10.0, xmax * 0.20)))
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(y=0.02)

    hi_idx = methods.index(HIGHLIGHT)
    for i, (mean, std) in enumerate(zip(mean_vals, std_vals)):
        cy = y[i]
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
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 1.0,
        "figure.dpi": 130,
    })

    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            p = Path(a)
            plot_horizontal(ROOT / p.name, OUT_DIR / f"{p.stem}_cost.png")
        return

    plot_cost_grid(OUT_DIR / "cost_grid.png")
    for json_name, _, png_name in DATASETS:
        plot_horizontal(ROOT / json_name, OUT_DIR / png_name)


if __name__ == "__main__":
    main()
