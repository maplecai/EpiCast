"""fig1C: how strongly each VEF tracks measured CRE activity, per pretrained model.

Four panels side by side sharing a y axis, one per pretrained model, titled with the model
name. The figure is twice as wide as it is tall, so each panel is roughly 1:2. Inside a
panel the x positions are the four assays; at each position sit the five cell-type-level
correlations as coloured points. The spread shown is therefore across cell types, not
across CREs.

The summary is a wide bar at the mean and a thin vertical line one sample SD (ddof=1) in
each direction, capped by two half-width bars. Nothing is filled, so the five points stay
the most visible thing in the figure. Unlike the quartiles of a box, a mean and an SD are
defined the same way at n=5 as at n=500; what they assume instead is that the five cell
types scatter symmetrically, which is not something this figure can show.

Font sizes come from seaborn's "talk" context and are never set here: every figure in this
bundle ends up as one panel among several and gets scaled down, so the text has to start
out large. To make a figure look finer, grow its figsize; the text then reads smaller
relative to the whole. Margins are the same everywhere and do not need tuning because the
figure is saved with a tight bounding box.

The points are not spread out horizontally: all five sit on the centre line, so a point
that hides another is genuinely a near-tie. The white outline is what keeps overlapping
points readable.

A point is PCC(VEF of that cell and assay, measured activity of that cell), over every
CRE where both are available. Enformer and Borzoi have no H3K27ac track for K562, HepG2
and A549, so those points are simply absent and the summary there rests on two cell types;
nothing is imputed.

Reads the MPRA table and the VEF matrices directly: this figure needs the per-sequence
VEF values, which no metric table carries.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast.utils.plot_utils import set_mpl_params
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import assays, cell_colors, cell_types, figures_dir, mpra_path, vef_paths

fig_name = "fig1c_vef_activity_correlation.pdf"
legend_name = "fig1c_legend.pdf"

# (VEF matrix key, panel title) left to right; alphagenome_prefix is the pre-CTCF-fix
# matrix and stays out
panel_models = [
    ("enformer", "Enformer"),
    ("borzoi", "Borzoi"),
    ("sei", "Sei"),
    ("alphagenome", "AlphaGenome"),
]

summary_width = 0.5
summary_linewidth = 1.5
ylim = (0.0, 0.7)
yticks = [0.0, 0.2, 0.4, 0.6]
colors = [cell_colors[ct] for ct in cell_types]


def draw_summary(ax, x, values, width=summary_width):
    """A bar at the mean and a capped line one sample SD in each direction."""
    mean, sd = np.nanmean(values), np.nanstd(values, ddof=1)
    ax.vlines(x, mean - sd, mean + sd, color="black", lw=1.0, zorder=3)
    ax.hlines(mean, x - width / 2, x + width / 2, color="black", lw=summary_linewidth, zorder=3)
    for cap in (mean - sd, mean + sd):
        ax.hlines(cap, x - width / 4, x + width / 4, color="black", lw=1.0, zorder=3)


def correlations(vef_df, mpra_df):
    """assay -> cell type -> PCC, over the CREs where both values exist."""
    return {
        assay: {
            cell_type: mpra_df[cell_type].corr(vef_df[f"{cell_type}_{assay}"])
            for cell_type in cell_types
        }
        for assay in assays
    }


def plot_panels(panels, save_path):
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 6), dpi=100, sharey=True)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, wspace=0.15)

    for ax, (model_label, corr) in zip(axes, panels):
        for x, assay in enumerate(assays):
            values = np.array([corr[assay][ct] for ct in cell_types], dtype=float)
            draw_summary(ax, x, values)
            ax.scatter(
                np.full(len(values), x), values,
                s=34, color=colors, edgecolor="white", linewidth=0.4, zorder=4,
            )

        ax.set_title(model_label)
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.set_xlim(-0.7, len(assays) - 0.3)
        ax.set_xticks(range(len(assays)))
        ax.set_xticklabels(assays, rotation=90)
        ax.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("PCC")
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_legend(save_path):
    fig, ax = plt.subplots(figsize=(2, 3), dpi=100)
    ax.set_axis_off()
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=ct)
        for ct, color in zip(cell_types, colors)
    ]
    ax.legend(handles=handles, loc="center", frameon=False, labelspacing=0.5)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    set_mpl_params()
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] {mpra_path} {mpra_df.shape}")

    panels = []
    for model_name, model_label in panel_models:
        vef_df = pd.read_csv(vef_paths[model_name], sep="\t")
        print(f"[load] {vef_paths[model_name]} {vef_df.shape}")
        panels.append((model_label, correlations(vef_df, mpra_df)))

    fig_path = figures_dir / fig_name
    plot_panels(panels, fig_path)
    legend_path = figures_dir / legend_name
    plot_legend(legend_path)

    for p in [fig_path, legend_path]:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
