"""fig1E: whether a virtual DNase residual tracks the activity residual of its own cell.

One row of ten panels sharing a y axis: two groups of five, Sei first and then AlphaGenome,
with a gap between them. Each group is about 4:3 and the whole figure 8:3. A panel is
titled with its target cell type; the two group headers are added by hand.

Font sizes come from seaborn's "talk" context and are never set here: every figure in this
bundle ends up as one panel among several and gets scaled down, so the text has to start
out large. To make a figure look finer, grow its figsize; the text then reads smaller
relative to the whole. Margins are the same everywhere and do not need tuning because the
figure is saved with a tight bounding box.

A panel fixes the target cell t whose activity residual is being explained and
draws one line per VEF cell context s, running from the whole CRE set to the union CTS
set. The matched line, s = t, is emphasized; the four unmatched ones are faded.

Residuals are taken against the same reference panel throughout: the mean of K562, HepG2
and SK-N-SH, for the activity and for the VEF alike.

The point of the figure is the change along each line. On the whole library a DNase
residual from any cell context correlates about equally with the target, because most of
the signal is shared; restricted to the CTS set the matched context pulls away, which is
the evidence that a VEF carries cell-context information and not just sequence strength.

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
from config import (
    cell_types,
    figures_dir,
    mpra_path,
    test_cell_types,
    train_cell_types,
    vef_paths,
)
from utils import build_masks

fig_name = "fig1e_dnase_residual_specificity.pdf"
legend_name = "fig1e_legend.pdf"

assay = "DNase"
# (VEF matrix key, group header)
vef_sources = [("sei", "Sei"), ("alphagenome", "AlphaGenome")]
# (mask key, x tick label)
subsets = [("total", "Whole"), ("all_cts_1_99", "CTS")]

matched_color = "black"
unmatched_color = "#BFBFBF"
matched_linewidth = 2.0
unmatched_linewidth = 1.0
matched_markersize = 8.0
unmatched_markersize = 6.0


def residual_correlations(mpra_df, vef_df, masks):
    """(target cell, VEF context) -> [PCC on the whole set, PCC on the CTS set]."""
    activity_residual = mpra_df[cell_types].sub(mpra_df[train_cell_types].mean(axis=1), axis=0)
    dnase = vef_df[[f"{ct}_{assay}" for ct in cell_types]]
    dnase.columns = cell_types
    vef_residual = dnase.sub(dnase[train_cell_types].mean(axis=1), axis=0)

    out = {}
    for target in cell_types:
        for context in cell_types:
            out[(target, context)] = [
                activity_residual.loc[masks[key], target].corr(vef_residual.loc[masks[key], context])
                for key, _ in subsets
            ]
    return out


def plot_panels(groups, save_path):
    fig = plt.figure(figsize=(24, 6), dpi=100)
    outer = fig.add_gridspec(
        1, len(groups), left=0.15, bottom=0.15, right=0.9, top=0.9, wspace=0.15
    )

    x = np.arange(len(subsets), dtype=float)
    shared = None
    for group, correlations in enumerate(groups):
        inner = outer[0, group].subgridspec(1, len(cell_types), wspace=0.35)
        for col, target in enumerate(cell_types):
            ax = fig.add_subplot(inner[0, col], sharey=shared)
            shared = shared or ax
            for context in cell_types:
                matched = context == target
                ax.plot(
                    x,
                    correlations[(target, context)],
                    color=matched_color if matched else unmatched_color,
                    lw=matched_linewidth if matched else unmatched_linewidth,
                    marker="o",
                    markersize=matched_markersize if matched else unmatched_markersize,
                    zorder=3 if matched else 2,
                )

            ax.axhline(0, color="gray", lw=1.0, linestyle="--")
            ax.set_title(target)
            ax.set_xlim(-0.35, len(subsets) - 0.65)
            ax.set_xticks(x)
            ax.set_xticklabels([label for _, label in subsets])
            # the two x labels nearly touch, so the tick marks carry which is which
            ax.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5)
            ax.spines[["top", "right"]].set_visible(False)
            # all ten panels share one y axis, so it is only labelled once, on the far left
            if group == 0 and col == 0:
                ax.set_ylabel("PCC")
            else:
                ax.tick_params(axis="y", labelleft=False)

    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_legend(save_path):
    fig, ax = plt.subplots(figsize=(3, 1), dpi=100)
    ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=matched_color, lw=matched_linewidth, marker="o",
               markersize=matched_markersize, label="Matched cell type"),
        Line2D([0], [0], color=unmatched_color, lw=unmatched_linewidth, marker="o",
               markersize=unmatched_markersize, label="Unmatched cell type"),
    ]
    ax.legend(handles=handles, loc="center", frameon=False, handlelength=2.0)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    set_mpl_params()
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] {mpra_path} {mpra_df.shape}")
    masks = build_masks(mpra_df, cell_types, train_cell_types, test_cell_types, verbose=False)
    masks["total"] = np.ones(len(mpra_df), dtype=bool)

    groups = []
    for source, source_label in vef_sources:
        vef_df = pd.read_csv(vef_paths[source], sep="\t")
        print(f"[load] {vef_paths[source]} {vef_df.shape}")
        correlations = residual_correlations(mpra_df, vef_df, masks)
        for target in cell_types:
            whole, cts = correlations[(target, target)]
            print(f"  [matched] {source_label} {target}: whole={whole:.3f} cts={cts:.3f}")
        groups.append(correlations)

    fig_path = figures_dir / fig_name
    plot_panels(groups, fig_path)
    legend_path = figures_dir / legend_name
    plot_legend(legend_path)

    for p in [fig_path, legend_path]:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
