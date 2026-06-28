import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# ── Publication-quality global settings ───────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 9,
    'axes.linewidth': 0.5,
    'pdf.fonttype': 42,   # embed fonts (required by most venues: ACL, NeurIPS, ICML)
    'ps.fonttype': 42,
})


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_total_costs(data):
    methods = data['methods']
    tasks = data['tasks']
    seeds = data['seeds']
    total_costs = {method: 0 for method in methods}
    for task in tasks:
        for method in methods:
            costs = []
            for seed in seeds:
                for result in seed['results']:
                    if result['task'] == task:
                        costs.append(result[method]['cost'])
                        break
            total_costs[method] += np.mean(costs)
    return total_costs


def academic_cmap():
    """Muted green→cream→red colormap; low cost = green (good), high = red (bad)."""
    colors = ['#4CA864', '#A8D4A8', '#F4F4E0', '#F0B5A5', '#C83232']
    return LinearSegmentedColormap.from_list('cost', colors, N=256)


def cell_text_color(val, vmin, vmax, cmap):
    """Return dark or white text color based on perceived background luminance."""
    norm_val = np.clip((val - vmin) / (vmax - vmin), 0, 1)
    r, g, b, _ = cmap(norm_val)
    lum = 0.299 * r + 0.587 * g + 0.114 * b   # standard perceived luminance
    return '#111111' if lum > 0.50 else 'white'


def plot_cost_heatmap():
    FILES = {
        'Math (GPT-5)':     'math_gpt.json',
        'Math (KIMI K2)':   'math_kimi.json',
        'System (GPT-5)':   'system_gpt.json',
        'System (KIMI K2)': 'system_kimi.json',
    }
    METHODS = ['OpenEvolve', 'GEPA', 'AdaEvolve', 'EvoX', 'SpecEvo']
    YLABELS = ['OpenEvolve', 'GEPA', 'AdaEvolve', 'EvoX', 'SpecEvo (Ours)']
    DATASETS = list(FILES.keys())

    # Build cost matrix
    cost_matrix = np.zeros((len(METHODS), len(DATASETS)))
    for col_idx, filepath in enumerate(FILES.values()):
        data = load_json(filepath)
        costs = calculate_total_costs(data)
        for row_idx, method in enumerate(METHODS):
            cost_matrix[row_idx, col_idx] = costs[method]

    # SpecEvo cost-reduction ratios (vs. mean of all other methods per column)
    si = METHODS.index('SpecEvo')
    reductions = []
    for col in range(len(DATASETS)):
        others = np.mean([cost_matrix[i, col] for i in range(len(METHODS)) if i != si])
        reductions.append(others / cost_matrix[si, col])
    avg_reduction = np.mean(reductions)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.2, 3.8))

    cmap = academic_cmap()
    vmin, vmax = cost_matrix.min(), cost_matrix.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    ax.imshow(cost_matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_xticklabels(DATASETS, fontsize=10)
    ax.set_yticklabels(YLABELS, fontsize=10)

    # Bold SpecEvo y-label
    for tick in ax.get_yticklabels():
        if 'SpecEvo' in tick.get_text():
            tick.set_fontweight('bold')

    ax.tick_params(which='both', length=0, pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # White cell-separator grid (minor ticks)
    ax.set_xticks(np.arange(len(DATASETS)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(METHODS)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.2)
    ax.tick_params(which='minor', length=0)

    # Thicker separator between Math and System column groups
    ax.axvline(x=1.5, color='white', linewidth=3.0, zorder=3)

    # ── Cell annotations ──────────────────────────────────────────────────────
    for i, method in enumerate(METHODS):
        for j in range(len(DATASETS)):
            val = cost_matrix[i, j]
            txt_color = cell_text_color(val, vmin, vmax, cmap)
            cost_str = f'${val:.2f}' if val < 10 else f'${val:.1f}'

            if i == si:   # SpecEvo row: cost + reduction ratio
                ax.text(j, i - 0.15, cost_str,
                        ha='center', va='center',
                        color=txt_color, fontsize=12.5, fontweight='bold')
                ax.text(j, i + 0.24, f'({reductions[j]:.1f}×↓)',  # ×↓
                        ha='center', va='center',
                        color='#1a5c1a', fontsize=8.5)
            else:
                ax.text(j, i, cost_str,
                        ha='center', va='center',
                        color=txt_color, fontsize=12.5)

    # Thin dark border around SpecEvo row to mark the proposed method
    ax.add_patch(Rectangle((-0.5, si - 0.5), len(DATASETS), 1,
                            linewidth=1.4, edgecolor='#222222', facecolor='none',
                            linestyle='-', zorder=4))

    # Column-group labels above the tick labels
    ax.text(0.5, 1.155, 'Math Benchmarks',
            ha='center', va='bottom', fontsize=8.5, color='#555555',
            transform=ax.transAxes)
    ax.text(0.875, 1.155, 'System Benchmarks',
            ha='center', va='bottom', fontsize=8.5, color='#555555',
            transform=ax.transAxes)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, pad=0.015, fraction=0.033, aspect=22)
    cbar.set_label('Avg. Total Cost (USD)', rotation=270, labelpad=14, fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2, width=0.5)
    cbar.outline.set_linewidth(0.5)

    # ── Caption-style footnote (italic, subtle) ───────────────────────────────
    fig.text(0.50, -0.02,
             f'† SpecEvo (Ours) achieves {avg_reduction:.1f}× '
             f'average cost reduction relative to baselines.',
             ha='center', fontsize=8, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig('cost_heatmap.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('cost_heatmap.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print(f'Saved cost_heatmap.png / .pdf')

    # Summary stats
    print('\nCost reduction per dataset:')
    for j, ds in enumerate(DATASETS):
        print(f'  {ds:<22} {reductions[j]:.2f}×')
    print(f'  {"Average":<22} {avg_reduction:.2f}×')


if __name__ == '__main__':
    plot_cost_heatmap()
