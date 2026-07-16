#!/usr/bin/env python3
"""
Prompt-type efficiency analysis for BLADE runs (single-column pie figure).

Question
--------
Among the *evolutionary* prompts, what fraction of new-best discoveries does
each prompt type produce?  The seeding prompt (``init``) is **excluded** — we
only compare the mutate / crossover / paradigm prompts that drive search.

Prompt taxonomy (7 types, init excluded)
----------------------------------------
  mutate_general / focused_fix / mechanism_swap   — 3 mutate templates
  mutate_targeted                                 — analysis-guided mutate
  crossover_structural / component_swap           — 2 crossover templates
  paradigm_shift                                  — paradigm + paradigm_variant

Method
------
Per run: count new bests per type, convert to a within-run percentage (so each
run weighs equally), then average those percentages across runs and renormalise
to 100% over the 7 evolutionary types.

Run:  .venv/bin/python scripts/plots/prompt_efficiency.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "prompt_result"

_NEWBEST_SRC = re.compile(r"NEW BEST.*?source:\s*([a-z_]+)")


def _polar(angle_deg: float, r: float) -> tuple[float, float]:
    import math
    a = math.radians(angle_deg)
    return r * math.cos(a), r * math.sin(a)


def canon(src: str) -> str:
    return "paradigm_shift" if src.startswith("paradigm") else src


EXCLUDE = {"init"}  # seeding prompt is not an evolutionary operator

# (key, short label, colour) — distinct hues, grouped by family
TYPES = [
    ("mutate_general",           "Mutate\ngeneral",      "#4C72B0"),
    ("mutate_focused_fix",       "Mutate\nfocused fix",  "#DD8452"),
    ("mutate_mechanism_swap",    "Mutate\nmechanism",    "#55A868"),
    ("mutate_targeted",          "Mutate\ntargeted",     "#C44E52"),
    ("crossover_structural",     "Crossover\nstructural", "#8172B3"),
    ("crossover_component_swap", "Crossover\ncomponent",  "#CCB974"),
    # ("paradigm_shift",           "Paradigm\nshift",       "#64B5CD"),  # Removed
]
ORDER = [t[0] for t in TYPES]
LABELS = {t[0]: t[1] for t in TYPES}
COLORS = {t[0]: t[2] for t in TYPES}


def mean_shares() -> tuple[dict[str, float], int]:
    acc: dict[str, float] = defaultdict(float)
    runs = sorted(PROMPT_DIR.glob("*.txt"))
    used = 0
    for run in runs:
        counts: dict[str, int] = defaultdict(int)
        for ln in run.read_text().splitlines():
            m = _NEWBEST_SRC.search(ln)
            if not m:
                continue
            t = canon(m.group(1))
            if t in EXCLUDE:
                continue
            counts[t] += 1
        total = sum(counts.values())
        if total == 0:
            continue
        used += 1
        for t in ORDER:
            acc[t] += 100.0 * counts.get(t, 0) / total
    mean = {t: acc[t] / used for t in ORDER}
    s = sum(mean.values())
    mean = {t: v * 100.0 / s for t, v in mean.items()}   # renormalise to 100%
    return mean, used


def main() -> None:
    mean, n_runs = mean_shares()
    print(f"Averaged over {n_runs} runs (init excluded):")
    for t in sorted(ORDER, key=lambda x: -mean[x]):
        print(f"  {LABELS[t].replace(chr(10),' '):26s} {mean[t]:5.1f}%")

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    # compact, single-column friendly — labels live INSIDE the wedges so the
    # pie fills the whole frame (no wasted margin for outside text).
    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    sizes = [mean[t] for t in ORDER]
    colors = [COLORS[t] for t in ORDER]
    top = max(range(len(ORDER)), key=lambda i: sizes[i])
    explode = [0.045 if i == top else 0.0 for i in range(len(ORDER))]

    # "petal" look: thin white gaps between wedges, closed pie.
    wedges, _ = ax.pie(
        sizes, colors=colors, explode=explode,
        startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=2.5, antialiased=True),
    )

    # name (black) inside each wedge; percentage (wedge colour) just outside
    for w, t in zip(wedges, ORDER):
        ang = (w.theta2 + w.theta1) / 2.0
        cx, cy = w.center                 # accounts for the explode offset
        # name — inside
        ix, iy = _polar(ang, 0.62)
        ax.text(
            cx + ix, cy + iy, LABELS[t],
            ha="center", va="center", color="black",
            fontsize=8.0 if t != ORDER[top] else 8.8,
            fontweight="bold", linespacing=1.05,
        )
        # percentage — outside the rim
        ox, oy = _polar(ang, 1.12)
        ax.text(
            cx + ox, cy + oy, f"{mean[t]:.0f}%",
            ha="center", va="center", color=COLORS[t],
            fontsize=10 if t != ORDER[top] else 11.5,
            fontweight="bold", clip_on=False,
        )

    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out = PROMPT_DIR / "prompt_efficiency.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print(f"\nsaved -> {out}")
    print(f"saved -> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
