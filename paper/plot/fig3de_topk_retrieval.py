"""fig3D / fig3E: top-k retrieval of CTS-high and CTS-low CREs by the two EpiCast models.

Each panel is a 2 (metric) x 2 (held-out cell type) grid of curves against the screened
top fraction on a log axis, from the top 0.01% to the top 10%, the top row titled with the
cell type of its column. Only EpiCast-Sei and
EpiCast-AlphaGenome are drawn; they keep the dark end of their Sei / AlphaGenome colour
family so the curves match the bars of figures 2 and 3.

The random expectation is 1 for the enrichment fold and 1 / prevalence for the number
needed to screen, so it is read off the evaluated subset rather than assumed to be 100:
the CTS tails are defined on the full measurable set and then cut to the test
chromosomes, which leaves the prevalence near but not exactly 1%.

Curves are the raw per-depth values, deliberately unsmoothed: the jitter at small k is
the discreteness of the selection and should stay visible.

Font sizes come from seaborn's "talk" context and are never set here: every figure in this
bundle ends up as one panel among several and gets scaled down, so the text has to start
out large. To make a figure look finer, grow its figsize; the text then reads smaller
relative to the whole. Margins are the same everywhere and do not need tuning because the
figure is saved with a tight bounding box.

Reads results/figure_metrics, written by analysis/15.
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
from config import figure_metrics_dir, figure_model_blocks, figures_dir, test_cell_types

# (retrieval table, output figure) in manuscript panel order
panels = [
    ("retrieval_cts_high.tsv", "fig3d_retrieval_cts_high.pdf"),
    ("retrieval_cts_low.tsv", "fig3e_retrieval_cts_low.pdf"),
]
legend_name = "fig3de_legend.pdf"

metric_rows = [("ef", "Enrichment fold"), ("nns", "NNS")]

min_frac = 1e-4
max_frac = 1e-1
x_ticks = [1e-4, 1e-3, 1e-2, 1e-1]
# plain text, not mathtext: matplotlib renders $\log_{10}$ in its own font, not Arial
x_label = "log10(top-ranked fraction)"

# darkest colour of the Sei and AlphaGenome blocks, i.e. the EpiCast entry of each
curve_models = [
    (models[-1][0], plt.get_cmap(colormap)(0.92), f"EpiCast-{block}")
    for block, colormap, models in figure_model_blocks[:2]
]


def plot_panel(table, save_path):
    fig, axes = plt.subplots(
        len(metric_rows), len(test_cell_types), figsize=(8, 6), dpi=100, sharey="row"
    )
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, hspace=0.3, wspace=0.25)

    for row, (metric, ylabel) in enumerate(metric_rows):
        ymax = 0.0
        for col, cell_type in enumerate(test_cell_types):
            ax = axes[row, col]
            window = table[
                (table["cell_type"] == cell_type)
                & (table["k_frac"] >= min_frac)
                & (table["k_frac"] <= max_frac)
            ]
            for model, color, _ in curve_models:
                curve = window[window["model"] == model]
                ax.plot(curve["k_frac"], curve[metric], color=color, lw=1.4)
                ymax = max(ymax, float(np.nanmax(curve[metric].to_numpy(dtype=float))))

            # the random expectation is 1 for EF and 1 / prevalence for NNS
            random_y = 1.0 if metric == "ef" else 1.0 / float(window["prevalence"].iloc[0])
            ax.axhline(random_y, color="gray", lw=1.0, linestyle="--")
            ymax = max(ymax, random_y)

            # a log axis labelled by its exponents rather than by 10^-4 and friends
            ax.set_xscale("log")
            ax.set_xlim(min_frac, max_frac)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([round(np.log10(tick)) for tick in x_ticks])
            ax.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(cell_type)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(metric_rows) - 1:
                ax.set_xlabel(x_label)

        axes[row, 0].set_ylim(0, ymax * 1.1)

    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_legend(save_path):
    fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
    ax.set_axis_off()
    handles = [Line2D([0], [0], color=color, lw=1.8, label=label) for _, color, label in curve_models]
    handles.append(Line2D([0], [0], color="gray", lw=1.0, linestyle="--", label="Random"))
    ax.legend(handles=handles, loc="center", frameon=False, handlelength=1.6, labelspacing=0.5)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    set_mpl_params()
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for table_name, fig_name in panels:
        path = figure_metrics_dir / table_name
        table = pd.read_csv(path, sep="\t")
        print(f"[load] {path} {table.shape}")

        out_path = figures_dir / fig_name
        plot_panel(table, out_path)
        saved.append(out_path)

    legend_path = figures_dir / legend_name
    plot_legend(legend_path)
    saved.append(legend_path)

    for p in saved:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
