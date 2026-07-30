#!/usr/bin/env python3
"""
Horizontal bar charts of per-method averages across CO-Bench categories.

Data = Average rows from the results table (8 categories + overall).
Style mirrors the "Base LLMs" horizontal-bar reference: bold section header,
value labels to the right of each bar, light vertical grid, x-axis cropped
to the near-data band so gaps between methods stay legible.

The overall (36-problem) chart also annotates each method with best/second
counts across problems (e.g. 20/1), coloured red/blue with a legend.

Usage:
  .venv/bin/python scripts/plots/plot_category_averages.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "result" / "category_averages"

# Fixed top→bottom order (do not sort by score).
# Table columns are OpenEvolve → GEPA → AdaEvolve → EvoX → SpecEvo;
# we plot the reverse so SpecEvo leads.
METHODS = ("SpecEvo", "EvoX", "AdaEvolve", "GEPA", "OpenEvolve")

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
    # Table columns: OpenEvolve, GEPA, AdaEvolve, EvoX, SpecEvo
    "Routing Problems": {
        "OpenEvolve": 0.7693,
        "GEPA": 0.7707,
        "AdaEvolve": 0.6841,
        "EvoX": 0.7706,
        "SpecEvo": 0.7709,
    },
    "Assignment Problems": {
        "OpenEvolve": 1.0000,
        "GEPA": 0.9980,
        "AdaEvolve": 0.9997,
        "EvoX": 0.9999,
        "SpecEvo": 0.9971,
    },
    "Tree Problems": {
        "OpenEvolve": 0.8482,
        "GEPA": 0.7047,
        "AdaEvolve": 0.7392,
        "EvoX": 0.7617,
        "SpecEvo": 0.9474,
    },
    "Graph and Set Problems": {
        "OpenEvolve": 0.9585,
        "GEPA": 0.9701,
        "AdaEvolve": 0.9711,
        "EvoX": 0.9822,
        "SpecEvo": 0.9803,
    },
}

# Overall average across all 36 CO-Bench problems (+ best/second counts)
# Table columns: OpenEvolve, GEPA, AdaEvolve, EvoX, SpecEvo
OVERALL = {
    "title": "Average across 36 problems",
    "scores": {
        "OpenEvolve": 0.8633,
        "GEPA": 0.8740,
        "AdaEvolve": 0.8739,
        "EvoX": 0.8673,
        "SpecEvo": 0.9133,
    },
    # (#best, #second) across the 36 problems
    "best_second": {
        "SpecEvo": (20, 1),
        "EvoX": (8, 10),
        "AdaEvolve": (9, 13),
        "GEPA": (10, 6),
        "OpenEvolve": (8, 9),
    },
}

BEST_COLOR = "#C0392B"
SECOND_COLOR = "#1A5276"


def _style_axes(ax, methods: list[str], vals: list[float], title: str = "", right_pad: float = 0.28) -> None:
    n = len(methods)
    y = np.arange(n)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=14)
    ax.invert_yaxis()

    # value frame = data band; exactly 5 evenly spaced ticks inside it.
    # xlim extends a bit past the last tick so value labels have room.
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-6)
    pad = max(0.012, 0.18 * span)
    # For near-ceiling scores (e.g. assignment ≈ 1.0), keep the frame tight
    # so .2f tick labels stay distinct.
    if span < 0.02:
        pad = max(0.0008, 0.35 * span)
    tick_lo, tick_hi = lo - pad, hi + pad
    ticks = np.linspace(tick_lo, tick_hi, 5)
    ax.set_xlim(tick_lo, tick_hi + right_pad * (tick_hi - tick_lo))
    ax.set_xticks(ticks)
    # More decimals when the band is narrow so ticks don't collide after rounding
    tick_fmt = "{:.4f}" if (ticks[-1] - ticks[0]) < 0.05 else "{:.2f}"
    ax.set_xticklabels([tick_fmt.format(t) for t in ticks])

    ax.xaxis.grid(True, linestyle="-", color="#D8D8D8", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.xaxis.set_ticks_position("none")
    ax.set_xlabel("")
    if title:
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=10)
    else:
        ax.set_title("")


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
    _style_axes(ax, methods, vals, title)

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
            fontsize=12,
            color="#222222",
        )

    _save(fig, out)


def _text_right_x(fig, ax, text_artist) -> float:
    """Return the data-x coordinate at the right edge of a text artist."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = text_artist.get_window_extent(renderer=renderer)
    return ax.transData.inverted().transform((bb.x1, bb.y0))[0]


def plot_overall(
    scores: dict[str, float],
    best_second: dict[str, tuple[int, int]],
    out: Path,
) -> None:
    """Overall average chart with best/second count annotations (e.g. 20/1)."""
    methods = list(METHODS)
    vals = [scores[m] for m in methods]
    n = len(methods)

    # No title — keep figure short for tight LaTeX inclusion.
    fig_h = 0.22 + 0.48 * n
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    y = np.arange(n)

    bars = ax.barh(
        y,
        vals,
        height=0.72,
        color=[COLORS[m] for m in methods],
        edgecolor="none",
        zorder=3,
    )
    # Modest pad first; xlim is tightened to content after labels/legend.
    _style_axes(ax, methods, vals, title="", right_pad=0.35)
    # Crop vertical margins (no title) so PDF bbox stays tight.
    ax.set_ylim(n - 0.45, -0.45)

    xmin, xmax = ax.get_xlim()
    dx = 0.012 * (xmax - xmin)
    # Explicit gap so score and best/second never collide (trailing
    # spaces are ignored by matplotlib text extents).
    gap = 0.010 * (xmax - xmin)
    for bar, method, val in zip(bars, methods, vals):
        cy = bar.get_y() + bar.get_height() / 2
        n_best, n_second = best_second[method]

        t_score = ax.text(
            val + dx,
            cy,
            f"{val:.4f}",
            ha="left",
            va="center",
            fontsize=12,
            color="#222222",
        )
        x = _text_right_x(fig, ax, t_score) + gap

        t_best = ax.text(
            x,
            cy,
            f"{n_best}",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=BEST_COLOR,
        )
        x = _text_right_x(fig, ax, t_best)

        t_slash = ax.text(
            x,
            cy,
            "/",
            ha="left",
            va="center",
            fontsize=12,
            color="#444444",
        )
        x = _text_right_x(fig, ax, t_slash)

        ax.text(
            x,
            cy,
            f"{n_second}",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=SECOND_COLOR,
        )

    legend_handles = [
        Line2D(
            [0], [0], color=BEST_COLOR, lw=0, marker="s", markersize=8,
            label="Best",
        ),
        Line2D(
            [0], [0], color=SECOND_COLOR, lw=0, marker="s", markersize=8,
            label="Second",
        ),
    ]
    # Inside the plot, in the empty pocket right of the short bars.
    ax.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.905, 2.55),
        bbox_transform=ax.transData,
        fontsize=11,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        framealpha=0.92,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
    )

    # Crop right edge to the rightmost label/legend — kill empty white.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    right = max(vals)
    for t in ax.texts:
        bb = t.get_window_extent(renderer=renderer)
        right = max(right, inv.transform((bb.x1, bb.y0))[0])
    leg = ax.get_legend()
    if leg is not None:
        bb = leg.get_window_extent(renderer=renderer)
        right = max(right, inv.transform((bb.x1, bb.y0))[0])
    left, _ = ax.get_xlim()
    ax.set_xlim(left, right + 0.003)

    # Keep ticks only over the value band (not the annotation gutter),
    # so the rightmost tick label cannot overhang and create white margin.
    v_lo, v_hi = min(vals), max(vals)
    v_span = max(v_hi - v_lo, 1e-6)
    v_pad = max(0.012, 0.18 * v_span)
    ticks = np.linspace(v_lo - v_pad, v_hi + v_pad, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2f}" for t in ticks])

    _save(fig, out)


def _trim_whitespace_png(path: Path, *, threshold: int = 250, pad: int = 1) -> tuple[int, int, int, int]:
    """Crop near-white margins from a PNG. Returns (left, top, right, bottom) cut px."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    h, w = arr.shape[:2]
    mask = (arr < threshold).any(axis=2)
    if not mask.any():
        return (0, 0, 0, 0)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(h, r1 + pad)
    c1 = min(w, c1 + pad)
    im.crop((c0, r0, c1, r1)).save(path)
    return (c0, r0, w - c1, h - r1)


def _crop_pdf_margins(pdf_path: Path, cuts_px: tuple[int, int, int, int], png_size: tuple[int, int]) -> None:
    """Shrink PDF MediaBox using the same relative margins trimmed from the PNG."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter  # type: ignore
        except ImportError:
            return

    left, top, right, bottom = cuts_px
    if left == right == top == bottom == 0:
        return
    w_px, h_px = png_size
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    page = reader.pages[0]
    box = page.mediabox
    pw, ph = float(box.width), float(box.height)
    # PNG was saved at dpi with the same aspect as this PDF page.
    page.mediabox.lower_left = (
        float(box.left) + pw * left / w_px,
        float(box.bottom) + ph * bottom / h_px,
    )
    page.mediabox.upper_right = (
        float(box.right) - pw * right / w_px,
        float(box.top) - ph * top / h_px,
    )
    writer.add_page(page)
    with open(pdf_path, "wb") as f:
        writer.write(f)


def _save(fig, out: Path) -> None:
    # Tight crop for LaTeX \includegraphics (no wasted whitespace).
    fig.tight_layout(pad=0.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    png_path = out if out.suffix.lower() == ".png" else out.with_suffix(".png")
    pdf_path = out.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    from PIL import Image

    w0, h0 = Image.open(png_path).size
    cuts = _trim_whitespace_png(png_path)
    _crop_pdf_margins(pdf_path, cuts, (w0, h0))
    print(f"saved -> {png_path}")
    print(f"saved -> {pdf_path}")


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
        "Routing Problems": "routing",
        "Assignment Problems": "assignment",
        "Tree Problems": "tree",
        "Graph and Set Problems": "graph_set",
    }
    for title, scores in CATEGORIES.items():
        plot_category(title, scores, OUT_DIR / f"avg_{slug[title]}.png")

    plot_overall(
        OVERALL["scores"],
        OVERALL["best_second"],
        OUT_DIR / "avg_overall_36.png",
    )


if __name__ == "__main__":
    main()
