"""fig2A: measured vs predicted activity of EpiCast-AlphaGenome in the held-out cell types.

Two stacked scatter panels titled with their cell type, HCT116 above A549, restricted to
the test chromosomes and to the sequences that were actually measured in that cell type.
Both panels share the axis
range and a 1:1 aspect so the two can be compared directly; the diagonal is y = x. The
PCC and the sample size are recomputed from the table rather than hard-coded.

Only one model is drawn, so a point's colour is free to carry the other thing that varies
here: `cell_colors` gives each panel its cell type's colour, the same green and blue those
two cells have in fig1C and fig3F.

Font sizes come from seaborn's "talk" context and are never set here: every figure in this
bundle ends up as one panel among several and gets scaled down, so the text has to start
out large. To make a figure look finer, grow its figsize; the text then reads smaller
relative to the whole. Margins are the same everywhere and do not need tuning because the
figure is saved with a tight bounding box.

Reads results/predictions, written by analysis/14.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from epicast import metrics
from epicast.utils.plot_utils import set_mpl_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cell_colors, figures_dir, predictions_dir, test_cell_types
from utils import safe_metric, valid_pairs

# measured activity and predictions side by side, written by analysis/14
pred_table = predictions_dir / "gosai_epicast_ag_vef.tsv"
fig_name = "fig2a_epicast_ag_scatter.pdf"

axis_limits = (-4, 6)
axis_ticks = [-4, -2, 0, 2, 4, 6]


def plot_scatter(ax, true, pred, cell_type):
    valid = valid_pairs(true, pred)
    x = true.loc[valid]
    y = pred.loc[valid]
    r = safe_metric(metrics.pearson, x, y)

    ax.scatter(
        x, y, s=2, alpha=0.25, color=cell_colors[cell_type], edgecolors="none", rasterized=True
    )
    ax.plot(axis_limits, axis_limits, color="gray", lw=1.0, linestyle="--", zorder=2)
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(cell_type)
    ax.set_xlabel("Measured activity")
    ax.set_ylabel("Predicted activity")
    ax.text(
        0.97,
        0.03,
        f"PCC = {r:.3f}\nn = {len(x):,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    ax.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    table = pd.read_csv(pred_table, sep="\t")
    print(f"[load] {pred_table} {table.shape}")
    test_df = table[table["split"] == "test"]

    set_mpl_params()
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(test_cell_types), 1, figsize=(6, 12), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, hspace=0.35)
    for ax, cell_type in zip(axes, test_cell_types):
        plot_scatter(ax, test_df[cell_type], test_df[f"{cell_type}_pred"], cell_type)

    out_path = figures_dir / fig_name
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.resolve()}")


if __name__ == "__main__":
    main()
