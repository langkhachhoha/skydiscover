#!/usr/bin/env python3
"""
Render the best circle-packing solution found by each method, side by side.

Each method ships a ``program.py`` exposing ``circle_packing21() -> ndarray``
of shape (N, 3) rows ``(x, y, r)``.  We execute it, then draw the circles in
their **minimum circumscribing rectangle** (the domain the evaluator scores on,
a rectangle of perimeter 4), one compact panel per method.

Design notes
------------
* multi-colour circles (a soft qualitative colormap) instead of one flat blue;
* thin white separation + subtle edge so neighbouring circles stay distinct;
* index label centred in each circle;
* per-panel title ``<Method> (sum of radii)``;
* tight layout, equal aspect — sized for a single paper column / figure row.

Run:  .venv/bin/python scripts/plots/plot_packings.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "result" / "circle_packing_rect"

# (folder, display name) in left-to-right order; our method last to stand out
PANELS = [
    ("oe",    "OpenEvolve"),
    ("ada",   "AdaEvolve"),
    ("evox",  "EvoX"),
    ("blade", "LiteEvo"),
]


def run_program(folder: str) -> np.ndarray:
    """Import <folder>/program.py in isolation and call circle_packing21()."""
    path = TASK_DIR / folder / "program.py"
    spec = importlib.util.spec_from_file_location(f"pack_{folder}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    circles = np.asarray(mod.circle_packing21(), dtype=float)
    return circles


def draw_panel(ax, circles: np.ndarray, name: str, ours: bool) -> None:
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # minimum circumscribing rectangle = the scored domain
    min_x, max_x = (x - r).min(), (x + r).max()
    min_y, max_y = (y - r).min(), (y + r).max()
    W, H = max_x - min_x, max_y - min_y
    pad = 0.02 * max(W, H)

    # colour circles by radius (sequential) — pretty *and* meaningful: bigger
    # circles read warmer, so the eye sees how area is distributed.
    cmap = plt.get_cmap("viridis")
    rmin, rmax = r.min(), r.max()
    norm = (r - rmin) / (rmax - rmin + 1e-12)
    colors = [cmap(0.12 + 0.78 * v) for v in norm]

    # background grid (drawn first, behind everything)
    gx = np.linspace(min_x, max_x, 6)
    gy = np.linspace(min_y, max_y, 6)
    for xv in gx:
        ax.axvline(xv, color="#D8D8D8", lw=0.6, zorder=0)
    for yv in gy:
        ax.axhline(yv, color="#D8D8D8", lw=0.6, zorder=0)

    # domain rectangle (the perimeter-4 bound)
    ax.add_patch(Rectangle((min_x, min_y), W, H, fill=False,
                           edgecolor="#888888", lw=1.2, zorder=1))

    for i, (cx, cy, cr) in enumerate(circles):
        ax.add_patch(Circle((cx, cy), cr, facecolor=colors[i],
                            edgecolor="#1A1A1A", lw=1.4, zorder=2))
        ax.text(cx, cy, str(i), ha="center", va="center",
                fontsize=max(5.5, min(8.0, 90 * cr / max(W, H))),
                color="white", fontweight="bold", zorder=3)

    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    sum_r = r.sum()
    title = f"{name} ({sum_r:.3f})"
    ax.set_title(title, fontsize=11,
                 fontweight="bold" if ours else "normal",
                 color="#C0392B" if ours else "#333333", pad=6)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    n = len(PANELS)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.7))

    for ax, (folder, name) in zip(axes, PANELS):
        circles = run_program(folder)
        draw_panel(ax, circles, name, ours=(folder == "blade"))
        print(f"  {name:11s}: N={len(circles)}  sum_r={circles[:,2].sum():.4f}")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.08)
    out = TASK_DIR / "best_packings.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print(f"\nsaved -> {out}")
    print(f"saved -> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
