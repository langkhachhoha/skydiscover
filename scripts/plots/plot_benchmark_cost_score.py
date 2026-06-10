#!/usr/bin/env python3
"""
Horizontal total-cost bars (tight, 3–5 methods).

LiteEvo is annotated with × cheaper vs the average of the baseline methods.

Usage:
  .venv/bin/python scripts/plots/plot_benchmark_cost_score.py
  .venv/bin/python scripts/plots/plot_benchmark_cost_score.py --suite kimi-k2
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    "LiteEvo": "#c9a227",
}


@dataclass(frozen=True)
class Suite:
    name: str
    benchmarks: tuple[str, ...]
    methods: tuple[str, ...]
    data: dict[str, dict[str, tuple[float, float]]]
    default_out: Path

    @property
    def baselines(self) -> tuple[str, ...]:
        return tuple(m for m in self.methods if m != "LiteEvo")


GPT5 = Suite(
    name="gpt-5",
    benchmarks=(
        "Circle Packing",
        "Circle Packing Rect",
        "heilbronn_convex",
        "heilbronn_triangle",
        "MinMax Distance (n=16, d=2)",
        "MinMax Distance (n=14, d=3)",
        "signal processing",
    ),
    methods=("OpenEvolve", "GEPA", "AdaEvolve", "EvoX", "LiteEvo"),
    data={
        "Circle Packing": {
            "OpenEvolve": (2.5414, 8.1094),
            "GEPA": (2.5240, 8.5237),
            "AdaEvolve": (2.6061, 8.4721),
            "EvoX": (2.4808, 8.2327),
            "LiteEvo": (2.6272, 1.5073),
        },
        "Circle Packing Rect": {
            "OpenEvolve": (2.3163, 9.8134),
            "GEPA": (2.2852, 6.5190),
            "AdaEvolve": (2.3447, 8.1753),
            "EvoX": (2.3596, 8.1220),
            "LiteEvo": (2.3626, 4.0994),
        },
        "heilbronn_convex": {
            "OpenEvolve": (0.0223, 11.6245),
            "GEPA": (0.0288, 9.5738),
            "AdaEvolve": (0.0234, 8.6866),
            "EvoX": (0.0274, 9.4314),
            "LiteEvo": (0.0298, 2.0219),
        },
        "heilbronn_triangle": {
            "OpenEvolve": (0.0361, 8.5446),
            "GEPA": (0.0317, 11.3371),
            "AdaEvolve": (0.0295, 9.9287),
            "EvoX": (0.0337, 7.2926),
            "LiteEvo": (0.0361, 1.1077),
        },
        "MinMax Distance (n=16, d=2)": {
            "OpenEvolve": (0.0769, 8.6347),
            "GEPA": (0.0769, 6.1959),
            "AdaEvolve": (0.0772, 8.2558),
            "EvoX": (0.0776, 8.7980),
            "LiteEvo": (0.0776, 1.5806),
        },
        "MinMax Distance (n=14, d=3)": {
            "OpenEvolve": (0.2381, 8.1190),
            "GEPA": (0.2240, 7.4516),
            "AdaEvolve": (0.2324, 8.5647),
            "EvoX": (0.2390, 7.9468),
            "LiteEvo": (0.2398, 1.3329),
        },
        "signal processing": {
            "OpenEvolve": (0.5897, 7.7906),
            "GEPA": (0.6363, 7.7180),
            "AdaEvolve": (0.7316, 6.8654),
            "EvoX": (0.7031, 6.3520),
            "LiteEvo": (0.6970, 3.8694),
        },
    },
    default_out=ROOT / "paper" / "figures" / "benchmark_cost_score.pdf",
)

KIMI_K2 = Suite(
    name="kimi-k2",
    benchmarks=(
        "Circle Packing",
        "Circle Packing Rect",
        "heilbronn_convex",
        "heilbronn_triangle",
        "MinMax Distance (n=16, d=2)",
        "MinMax Distance (n=14, d=3)",
        "signal processing",
    ),
    methods=("OpenEvolve", "GEPA", "AdaEvolve", "EvoX", "LiteEvo"),
    data={
        "Circle Packing": {
            "OpenEvolve": (2.3506, 1.7510),
            "GEPA": (2.4443, 1.8587),
            "AdaEvolve": (2.6002, 1.4844),
            "EvoX": (2.6009, 1.8493),
            "LiteEvo": (2.5654, 1.0023),
        },
        "Circle Packing Rect": {
            "OpenEvolve": (1.9865, 1.4461),
            "GEPA": (2.0003, 1.7479),
            "AdaEvolve": (2.2508, 1.6333),
            "EvoX": (2.2508, 1.8427),
            "LiteEvo": (2.3613, 0.8674),
        },
        "heilbronn_convex": {
            "OpenEvolve": (0.0176, 1.6041),
            "GEPA": (0.0176, 1.6270),
            "AdaEvolve": (0.0232, 1.6427),
            "EvoX": (0.0265, 1.4388),
            "LiteEvo": (0.0262, 0.6612),
        },
        "heilbronn_triangle": {
            "OpenEvolve": (0.0260, 2.0356),
            "GEPA": (0.0064, 1.3209),
            "AdaEvolve": (0.0313, 1.7596),
            "EvoX": (0.0267, 1.8894),
            "LiteEvo": (0.0334, 0.4174),
        },
        "MinMax Distance (n=16, d=2)": {
            "OpenEvolve": (0.0746, 1.8115),
            "GEPA": (0.0556, 1.6374),
            "AdaEvolve": (0.0745, 1.4600),
            "EvoX": (0.0705, 1.4570),
            "LiteEvo": (0.0776, 1.3401),
        },
        "MinMax Distance (n=14, d=3)": {
            "OpenEvolve": (0.2103, 1.6733),
            "GEPA": (0.2113, 1.5141),
            "AdaEvolve": (0.2214, 1.4140),
            "EvoX": (0.2113, 2.1089),
            "LiteEvo": (0.2358, 0.7214),
        },
        "signal processing": {
            "OpenEvolve": (0.6965, 2.5747),
            "GEPA": (0.6514, 2.2495),
            "AdaEvolve": (0.6333, 2.1871),
            "EvoX": (0.6192, 2.3201),
            "LiteEvo": (0.7030, 1.2502),
        },
    },
    default_out=ROOT / "paper" / "figures" / "benchmark_cost_score_kimi_k2.pdf",
)

SUITES = {s.name: s for s in (GPT5, KIMI_K2)}


def total_costs(suite: Suite) -> dict[str, float]:
    out = {m: 0.0 for m in suite.methods}
    for bench in suite.benchmarks:
        for m in suite.methods:
            out[m] += suite.data[bench][m][1]
    return out


def plot(suite: Suite, out: Path) -> None:
    costs = total_costs(suite)
    totals = [costs[m] for m in suite.methods]
    baseline_avg = float(np.mean([costs[m] for m in suite.baselines]))
    cheaper_x = baseline_avg / costs["LiteEvo"]

    n = len(suite.methods)
    fig_h = 1.0 + 0.55 * n
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    y = np.arange(n)
    bar_h = 0.82
    bars = ax.barh(
        y,
        totals,
        height=bar_h,
        color=[COLORS[m] for m in suite.methods],
        edgecolor="none",
    )

    ax.set_xlabel("Total cost ($)", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(suite.methods, fontsize=10)
    xmax = max(totals)
    ax.set_xlim(0, xmax * 1.22)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(y=0.02)

    le_idx = suite.methods.index("LiteEvo")
    cost_dx = 0.15
    mult_dx = max(5.5, xmax * 0.09)
    for i, (bar, val) in enumerate(zip(bars, totals)):
        cy = bar.get_y() + bar.get_height() / 2
        ax.text(val + cost_dx, cy, f"${val:.1f}", ha="left", va="center", fontsize=9)
        if i == le_idx:
            ax.text(
                val + mult_dx,
                cy,
                f"{cheaper_x:.1f}×",
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=COLORS["LiteEvo"],
            )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    if out.suffix.lower() == ".pdf":
        fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{suite.name}] total costs: {costs}")
    print(f"[{suite.name}] LiteEvo vs avg. baseline: {cheaper_x:.2f}×")
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="gpt-5",
        help="gpt-5: 5 methods × 7 benchmarks; kimi-k2: 5 methods × 7 benchmarks",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    suite = SUITES[args.suite]
    out = args.out or suite.default_out
    plot(suite, out)


if __name__ == "__main__":
    main()
