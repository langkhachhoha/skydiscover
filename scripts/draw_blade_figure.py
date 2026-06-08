"""
Generate a professional NeurIPS-style figure for the BLADE/AdaEvolve framework.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "text.usetex": False,
    "figure.dpi": 300,
})

C = {
    "bg": "#FAFBFC",
    "archive_bg": "#E8F5E9", "archive_bd": "#43A047",
    "ctrl_bg": "#F3E5F5",    "ctrl_bd": "#8E24AA",
    "guide_bg": "#FFF3E0",   "guide_bd": "#EF6C00",
    "loop_bg": "#ECEFF1",    "loop_bd": "#546E7A",
    "worker_bg": "#E3F2FD",  "worker_bd": "#1E88E5",
    "island": ["#66BB6A", "#42A5F5", "#FFA726", "#AB47BC"],
    "box_w": "#FFFFFF",
    "sample": "#A5D6A7", "prompt": "#CE93D8", "llm": "#90CAF9",
    "eval": "#FFE082", "update": "#A5D6A7",
    "input_bg": "#F5F5F5", "input_bd": "#BDBDBD",
    "best_bg": "#FFF9C4", "best_bd": "#F9A825",
    "err_bg": "#FFCDD2", "err_bd": "#E53935",
    "arrow": "#37474F", "arrow_l": "#90A4AE",
    "t_dark": "#212121", "t_mid": "#616161", "t_light": "#9E9E9E",
    "red": "#E53935", "green": "#2E7D32",
}


def rbox(ax, x, y, w, h, fc, ec=None, lw=1.2, alpha=0.85, r=0.012, z=1):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec or fc, lw=lw, alpha=alpha, zorder=z,
        transform=ax.transAxes)
    ax.add_patch(p)


def arr(ax, s, e, c=None, lw=1.2, cs="arc3,rad=0.0", z=5, ms=11,
        sty="-|>"):
    a = FancyArrowPatch(s, e, arrowstyle=sty, color=c or C["arrow"],
        lw=lw, connectionstyle=cs, mutation_scale=ms, zorder=z,
        transform=ax.transAxes)
    ax.add_patch(a)


def txt(ax, x, y, s, fs=8, c=None, ha="center", va="center",
        fw="normal", fi="normal", z=10):
    ax.text(x, y, s, fontsize=fs, color=c or C["t_dark"], ha=ha, va=va,
            fontweight=fw, fontstyle=fi, transform=ax.transAxes, zorder=z)


def sbox(ax, x, y, w, h, text, fc, tc=None, ec=None, fs=7.5, lw=0.8,
         z=4, fw="normal"):
    rbox(ax, x, y, w, h, fc, ec=ec or fc, lw=lw, z=z, r=0.008)
    txt(ax, x+w/2, y+h/2, text, fs=fs, c=tc or C["t_dark"], fw=fw, z=z+1)


def draw_island(ax, cx, cy, r, color, label, idx):
    circ = plt.Circle((cx, cy), r, fc=color, ec="white", lw=1.8,
                       alpha=0.72, zorder=3, transform=ax.transAxes)
    ax.add_patch(circ)
    np.random.seed(42 + idx)
    for _ in range(10):
        dx = np.random.uniform(-r*0.55, r*0.55)
        dy = np.random.uniform(-r*0.55, r*0.55)
        if dx**2 + dy**2 < (r*0.55)**2:
            d = plt.Circle((cx+dx, cy+dy), 0.004, fc="white", ec="none",
                           alpha=0.85, zorder=4, transform=ax.transAxes)
            ax.add_patch(d)
    txt(ax, cx, cy-r-0.018, label, fs=6.5, c=C["t_mid"], va="top")


def main():
    fig, ax = plt.subplots(1, 1, figsize=(17, 10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("auto"); ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    # ═══════════ TITLE ═══════════
    txt(ax, 0.50, 0.975,
        "BLADE: Budget-aware LLM-driven Adaptive Discovery and Evolution",
        fs=15, fw="bold", c=C["t_dark"])

    # ═══════════ INPUTS (Left) ═══════════
    ix, iy, iw, ih = 0.012, 0.55, 0.10, 0.37
    rbox(ax, ix, iy, iw, ih, C["input_bg"], ec=C["input_bd"], lw=1.0, z=2)
    txt(ax, ix+iw/2, iy+ih-0.022, "Inputs", fs=10, fw="bold", c=C["t_mid"])

    for label, ry in [("Problem\nDefinition", 0.76),
                      ("Evaluator\nFramework", 0.50),
                      ("Configuration\n(Budget B)", 0.24)]:
        sbox(ax, ix+0.008, iy+ih*ry-0.035, iw-0.016, 0.06, label,
             C["box_w"], ec="#E0E0E0", fs=7.2, lw=0.6)

    arr(ax, (ix+iw+0.005, iy+ih/2), (0.13, iy+ih/2), lw=1.8)

    # ═══════════ (A) MULTI-ISLAND QD ARCHIVE (Top, Green) ═══════════
    Ax, Ay, Aw, Ah = 0.135, 0.59, 0.52, 0.34
    rbox(ax, Ax, Ay, Aw, Ah, C["archive_bg"], ec=C["archive_bd"],
         lw=2.0, z=1, alpha=0.42)
    txt(ax, Ax+0.015, Ay+Ah-0.025,
        "(A) Multi-Island Quality-Diversity Archive",
        fs=11, fw="bold", c=C["archive_bd"], ha="left")

    # Elite Score formula
    sbox(ax, Ax+0.10, Ay+Ah-0.085, 0.32, 0.038,
         r"$\mathbf{Elite\ Score} = w_f \cdot rank_{fitness} + w_n \cdot rank_{novelty}$",
         "#C8E6C9", tc="#1B5E20", ec="#81C784", fs=8, lw=0.8)

    # Islands
    ipos = [(Ax+0.065, Ay+0.16), (Ax+0.18, Ay+0.16),
            (Ax+0.30, Ay+0.16), (Ax+0.42, Ay+0.16)]
    for j, ((px, py), lb) in enumerate(zip(ipos, [
        "Island 1\n(Balanced)", "Island 2\n(Quality)",
        "Island 3\n(Diversity)", "Island 4\n(Pareto)"])):
        draw_island(ax, px, py, 0.044, C["island"][j], lb, j)

    # Migration ring arrows
    for j in range(3):
        arr(ax, (ipos[j][0]+0.046, ipos[j][1]+0.012),
            (ipos[j+1][0]-0.046, ipos[j+1][1]+0.012),
            c=C["arrow_l"], lw=0.9)
    arr(ax, (ipos[3][0], ipos[3][1]+0.047),
        (ipos[0][0], ipos[0][1]+0.047),
        c=C["arrow_l"], lw=0.9, cs="arc3,rad=-0.2")
    txt(ax, (ipos[1][0]+ipos[2][0])/2, ipos[0][1]+0.07,
        "Ring Migration", fs=7, c=C["arrow_l"], fi="italic")

    # UCB label
    txt(ax, Ax+Aw-0.055, Ay+0.05, "UCB Island\nSelection",
        fs=8, c="#2E7D32", fw="bold")

    # ═══════════ (B) ADAPTIVE CONTROLLER (Left Middle, Purple) ═══════════
    Bx, By, Bw, Bh = 0.135, 0.305, 0.225, 0.26
    rbox(ax, Bx, By, Bw, Bh, C["ctrl_bg"], ec=C["ctrl_bd"],
         lw=2.0, z=1, alpha=0.42)
    txt(ax, Bx+0.015, By+Bh-0.025,
        "(B) Adaptive Controller",
        fs=11, fw="bold", c=C["ctrl_bd"], ha="left")

    # Accumulated signal G
    sbox(ax, Bx+0.015, By+Bh-0.095, Bw-0.03, 0.045,
         r"$G_t = \rho \cdot G_{t-1} + (1 - \rho) \cdot \delta^2$",
         "#E1BEE7", tc="#4A148C", ec="#CE93D8", fs=9, lw=0.8)

    # Search intensity I
    sbox(ax, Bx+0.015, By+Bh-0.155, Bw-0.03, 0.045,
         r"$I = I_{min} + \frac{I_{max} - I_{min}}{1 + \sqrt{G}}$",
         "#E1BEE7", tc="#4A148C", ec="#CE93D8", fs=9, lw=0.8)

    txt(ax, Bx+Bw/2, By+0.065,
        r"High $G$ $\Rightarrow$ Exploit (low $I$)" + "\n" +
        r"Low  $G$ $\Rightarrow$ Explore (high $I$)",
        fs=8, c="#6A1B9A", fi="italic")
    txt(ax, Bx+Bw/2, By+0.025,
        r"$\delta$ = normalized improvement signal per island",
        fs=6.5, c=C["t_light"])

    # Arrow: B -> A (selects island/mode)
    arr(ax, (Bx+Bw/2, By+Bh+0.005), (Bx+Bw/2, Ay-0.005),
        c=C["ctrl_bd"], lw=1.3)
    txt(ax, Bx+Bw/2+0.055, (By+Bh+Ay)/2,
        "Selects\nIsland &\nMode", fs=7, c=C["ctrl_bd"], fw="bold")

    # ═══════════ (C) GUIDE MODEL (Left Bottom, Orange) ═══════════
    Gx, Gy, Gw, Gh = 0.135, 0.045, 0.225, 0.235
    rbox(ax, Gx, Gy, Gw, Gh, C["guide_bg"], ec=C["guide_bd"],
         lw=2.0, z=1, alpha=0.42)
    txt(ax, Gx+0.015, Gy+Gh-0.025,
        "(C) Guide Model (Frontier Expert)",
        fs=11, fw="bold", c=C["guide_bd"], ha="left")

    sbox(ax, Gx+0.015, Gy+Gh-0.08, Gw-0.03, 0.035,
         "Paradigm Breakthrough", "#FFE0B2", tc="#BF360C",
         ec="#FFB74D", fs=8.5, lw=0.8, fw="bold")

    for gt, gy_ in [("6-Step Analysis Framework", Gy+Gh-0.125),
                     ("Stagnation -> Generate New Ideas", Gy+Gh-0.155),
                     ("Variation Operator Generation", Gy+Gh-0.185)]:
        txt(ax, Gx+Gw/2, gy_, gt, fs=7.5, c="#BF360C")

    txt(ax, Gx+Gw/2, Gy+0.02,
        "Frontier LLM (e.g., GPT-4o, Claude-3.5)",
        fs=6.5, c=C["t_light"], fi="italic")

    # Arrow: C -> Evolution loop (paradigm injection)
    arr(ax, (Gx+Gw+0.005, Gy+Gh*0.55),
        (0.52, 0.20),
        c=C["guide_bd"], lw=1.3, cs="arc3,rad=0.12")
    txt(ax, Gx+Gw+0.075, 0.215,
        "Inject\nParadigm", fs=7, c=C["guide_bd"], fw="bold")

    # Stagnation: B -> C
    arr(ax, (Bx+Bw/2, By-0.005),
        (Gx+Gw/2, Gy+Gh+0.005),
        c=C["red"], lw=1.1)
    txt(ax, Bx+Bw/2+0.068, (By+Gy+Gh)/2,
        "Stagnation\nDetected", fs=7, c=C["red"], fw="bold")

    # ═══════════ (D) EVOLUTION LOOP (Right, Main Flow) ═══════════
    Lx, Ly, Lw, Lh = 0.385, 0.045, 0.60, 0.52
    rbox(ax, Lx, Ly, Lw, Lh, C["loop_bg"], ec=C["loop_bd"],
         lw=2.0, z=1, alpha=0.30)
    txt(ax, Lx+0.015, Ly+Lh-0.025,
        "(D) Adaptive Evolution Loop",
        fs=11, fw="bold", c="#37474F", ha="left")

    # Step layout
    sy = Ly + Lh*0.55
    sh = 0.072
    gap = 0.022

    # Step boxes
    s1x, s1w = Lx+0.025, 0.09
    s2x, s2w = s1x+s1w+gap, 0.09
    s3x, s3w = s2x+s2w+gap, 0.115
    s4x, s4w = s3x+s3w+gap, 0.085
    s5x, s5w = s4x+s4w+gap, 0.085

    sbox(ax, s1x, sy-sh/2, s1w, sh,
         "Sample\nParent", C["sample"], tc=C["green"],
         ec="#66BB6A", fs=9, lw=1.2, fw="bold")

    # Sampling modes
    my = sy - sh/2 - 0.055
    mw = 0.029
    for mi, (mt, mc) in enumerate([("Explore", "#81D4FA"),
                                    ("Exploit", "#FFCC80"),
                                    ("Balance", "#CE93D8")]):
        sbox(ax, s1x+mi*(mw+0.002), my, mw, 0.025, mt, mc,
             fs=6, ec=mc, lw=0.5, tc="#424242")

    sbox(ax, s2x, sy-sh/2, s2w, sh,
         "Build\nPrompt", C["prompt"], tc="#6A1B9A",
         ec="#AB47BC", fs=9, lw=1.2, fw="bold")

    for ci, ct in enumerate(["+ Context Programs",
                             "+ Paradigm (if active)",
                             "+ Error History"]):
        txt(ax, s2x+s2w/2, my+0.01-ci*0.016, ct, fs=6, c="#7B1FA2")

    # Worker LLM highlight
    rbox(ax, s3x-0.007, sy-sh/2-0.01, s3w+0.014, sh+0.02,
         C["worker_bg"], ec=C["worker_bd"], lw=1.6, z=2, alpha=0.4)
    sbox(ax, s3x, sy-sh/2, s3w, sh,
         "Worker LLM\nGeneration", C["llm"], tc="#1565C0",
         ec="#42A5F5", fs=9, lw=1.2, fw="bold")
    txt(ax, s3x+s3w/2, my+0.005,
        "Smaller Model\n(e.g., GPT-4o-mini, Gemini Flash)",
        fs=6, c=C["t_light"], fi="italic")

    sbox(ax, s4x, sy-sh/2, s4w, sh,
         "Evaluate", C["eval"], tc="#F57F17",
         ec="#FFA726", fs=9, lw=1.2, fw="bold")

    sbox(ax, s5x, sy-sh/2, s5w, sh,
         "Update\nArchive", C["update"], tc=C["green"],
         ec="#66BB6A", fs=9, lw=1.2, fw="bold")

    # Step arrows
    for (x1, w1), (x2, _) in zip(
        [(s1x,s1w),(s2x,s2w),(s3x,s3w),(s4x,s4w)],
        [(s2x,s2w),(s3x,s3w),(s4x,s4w),(s5x,s5w)]):
        arr(ax, (x1+w1+0.003, sy), (x2-0.003, sy), lw=1.6)

    # Error retry loop
    ey_ = my - 0.04
    sbox(ax, s4x-0.015, ey_, 0.115, 0.028,
         "Error -> Retry with context", C["err_bg"],
         tc="#B71C1C", ec=C["err_bd"], fs=6.5, lw=0.7)
    arr(ax, (s4x-0.015, ey_+0.014),
        (s2x+s2w/2, my-0.03),
        c=C["red"], lw=0.9, cs="arc3,rad=0.22")

    # Update -> Archive (loop back to top)
    arr(ax, (s5x+s5w/2, sy+sh/2+0.01),
        (Ax+Aw-0.06, Ay-0.005),
        c=C["archive_bd"], lw=2.0, cs="arc3,rad=-0.15")
    txt(ax, s5x+s5w/2+0.025, sy+sh/2+0.075,
        "Valid: Add\nto Island", fs=7.5, c=C["green"], fw="bold")

    # Archive -> Sample (flow down)
    arr(ax, (Ax+0.065, Ay-0.005),
        (s1x+s1w/2, sy+sh/2+0.01),
        c=C["archive_bd"], lw=1.6, cs="arc3,rad=0.15")

    # Controller -> Sample
    arr(ax, (Bx+Bw+0.005, By+Bh*0.5),
        (s1x-0.008, sy),
        c=C["ctrl_bd"], lw=1.2, cs="arc3,rad=-0.1")

    # Evaluate -> Controller (feedback to update G)
    arr(ax, (s4x+s4w/2, sy-sh/2-0.01),
        (Bx+Bw-0.02, By+0.005),
        c=C["ctrl_bd"], lw=1.1, cs="arc3,rad=0.25")
    txt(ax, Bx+Bw+0.03, By+0.065,
        "Fitness\nfeedback", fs=6.5, c=C["ctrl_bd"], fw="bold")

    # ═══════════ BEST SOLUTION (Bottom Right) ═══════════
    bsx, bsy, bsw, bsh = 0.845, 0.055, 0.125, 0.085
    rbox(ax, bsx, bsy, bsw, bsh, C["best_bg"], ec=C["best_bd"],
         lw=2.5, z=3, alpha=0.95)
    txt(ax, bsx+bsw/2, bsy+bsh/2+0.012,
        "Best Solution(s)", fs=10, fw="bold", c="#E65100")
    txt(ax, bsx+bsw/2, bsy+bsh/2-0.015,
        "Under Budget", fs=7.5, c="#BF360C")

    arr(ax, (Lx+Lw-0.03, Ly+0.04),
        (bsx+bsw/2, bsy+bsh+0.005),
        c="#E65100", lw=2.0, cs="arc3,rad=-0.08")
    txt(ax, Lx+Lw-0.095, Ly+0.012,
        "Max Budget Reached", fs=7, c="#E65100", fw="bold")

    # ═══════════ LEGEND ═══════════
    legs = [("(A) QD Archive", C["archive_bg"], C["archive_bd"]),
            ("(B) Adaptive Controller", C["ctrl_bg"], C["ctrl_bd"]),
            ("(C) Guide Model", C["guide_bg"], C["guide_bd"]),
            ("(D) Evolution Loop", C["loop_bg"], C["loop_bd"])]
    for j, (lb, fc, ec) in enumerate(legs):
        lxx = 0.30 + j*0.175
        rbox(ax, lxx, 0.007, 0.014, 0.014, fc, ec=ec, lw=1.0, z=10, r=0.003)
        txt(ax, lxx+0.021, 0.014, lb, fs=7.5, ha="left", c=C["t_mid"])

    # ═══════════ SAVE ═══════════
    plt.tight_layout(pad=0.3)
    out = "/Users/apple/Desktop/All/NUS_INTERNSHIP/skydiscover/result/blade_framework_figure.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    fig.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    print(f"Saved: {out}")
    print(f"Saved: {out.replace('.png', '.pdf')}")
    plt.close()


if __name__ == "__main__":
    main()
