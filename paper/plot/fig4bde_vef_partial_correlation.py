"""fig4B / fig4D / fig4E: what each VEF still explains once the other three are held fixed.

fig4B and fig4D are one row of eight panels, four assays under Sei and four under
AlphaGenome, with a wide gap between the two sources; every panel has two x positions,
the marginal VEF-activity correlation and the partial one given the other three VEFs.
fig4B is the absolute setting, fig4D the residual one. fig4E is a 2 x 2 grid, Sei and
AlphaGenome across, absolute and residual down, showing the standardized coefficients of
the four-VEF fit that produced those partial correlations.

Every point is one cell type and the box summarizes them, so colour means cell type here,
as in fig1C. The box is the conventional one, whiskers and caps included, which is what
C.Z. asked for on 2026-08-25; at n=5 read it as decoration rather than as statistics, since
the quartiles land on the second and fourth point and the 1.5 IQR rule calls a cell type an
outlier in 22 of these 48 positions. Fliers are off, so such a point is still drawn, just
as one of the five coloured dots. The two rejected alternatives are
`_fig4bde_vef_partial_correlation_bar.py` (median bar only) and
`_fig4bde_vef_partial_correlation_box.py` (box without whiskers).

Titles are two levels and are drawn here rather than added by hand: the source spans its
four assays in fig4B and fig4D, and in fig4E the setting sits above the source.

The three panels are the collinearity argument: absolute VEFs all correlate with activity
because they all track "this is a strong regulatory element", and the partial correlations
show how little of that is assay-specific. The residual setting asks the same question of
the cell-type-specific component, which is what EpiCast is actually built on.

Font sizes come from seaborn's "talk" context and are never set here: every figure in this
bundle ends up as one panel among several and gets scaled down, so the text has to start
out large. To make a figure look finer, grow its figsize; the text then reads smaller
relative to the whole. Margins are the same everywhere and do not need tuning because the
figure is saved with a tight bounding box.

Reads results/vef_partial_correlation, written by analysis/11_vef_partial_correlation.py.
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
from config import assays, cell_colors, cell_types, figures_dir, results_dir

metrics_path = results_dir / "vef_partial_correlation" / "vef_partial_correlation.csv"

# left group, right group
sources = [("sei", "Sei"), ("alphagenome", "AlphaGenome")]

# (setting in the metrics table, output figure) for the correlation panels
correlation_panels = [
    ("absolute", "fig4b_vef_partial_correlation.pdf"),
    ("residual", "fig4d_vef_residual_partial_correlation.pdf"),
]
# (metric column, x tick label) left to right inside a correlation panel
correlation_columns = [("marginal_r", "Marginal"), ("partial_r", "3 VEFs conditioned")]
# (setting, row label) top to bottom in fig4E; the label is what the fit regresses on
# what, so it doubles as the first line of the panel title
beta_rows = [
    ("absolute", "Activity-VEF"),
    ("residual", "Activity residual-VEF residual"),
]
beta_name = "fig4e_vef_regression_beta.pdf"
legend_name = "fig4bde_legend.pdf"

# a conventional box plot: IQR box, median line, 1.5 IQR whiskers with caps. Fliers are off
# because every value is already on the figure as a coloured point. White fill rather than
# unfilled, so the box is a real object to recolour when laying out; the median is black
# rather than the theme's orange, which is a cell-type colour here
box_style = {
    "widths": 0.5,
    "patch_artist": True,
    "showfliers": False,
    "boxprops": {"facecolor": "white"},
    "medianprops": {"color": "black"},
}
point_size = 34
colors = [cell_colors[ct] for ct in cell_types]


def draw_groups(ax, values, labels):
    """The five cell-type values at each x position, boxed."""
    for x, column in enumerate(values):
        ax.boxplot(column[np.isfinite(column)], positions=[x], **box_style)
        ax.scatter(
            np.full(len(column), x), column,
            s=point_size, color=colors, edgecolor="white", linewidth=0.4, zorder=4,
        )

    ax.axhline(0, color="gray", lw=1.0, linestyle="--")
    ax.set_xlim(-0.7, len(values) - 0.3)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=90)
    ax.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_correlation_panel(table, setting, save_path):
    fig = plt.figure(figsize=(32, 6), dpi=100)
    # the two sources are separated by a gap much wider than the one between assays
    outer = fig.add_gridspec(
        1, len(sources), left=0.15, bottom=0.15, right=0.9, top=0.9, wspace=0.25
    )

    shared = None
    for group, (source, source_label) in enumerate(sources):
        # one header centred over the group instead of the source repeated four times
        header = fig.add_subplot(outer[0, group])
        header.set_axis_off()
        header.set_title(source_label, pad=40)

        inner = outer[0, group].subgridspec(1, len(assays), wspace=0.4)
        for col, assay in enumerate(assays):
            ax = fig.add_subplot(inner[0, col], sharey=shared)
            shared = shared or ax

            rows = table[
                (table["vef_source"] == source)
                & (table["setting"] == setting)
                & (table["assay"] == assay)
            ].set_index("cell_type")
            values = [rows.loc[cell_types, column].to_numpy() for column, _ in correlation_columns]
            draw_groups(ax, values, [label for _, label in correlation_columns])

            ax.set_title(assay)
            # all eight panels share one y axis, so it is only labelled once, on the far left
            if group == 0 and col == 0:
                ax.set_ylabel("PCC")
            else:
                ax.tick_params(axis="y", labelleft=False)

    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_beta_panel(table, save_path):
    # a row at a time, not all four: the residual betas are far smaller than the absolute
    # ones, and one shared range would flatten the bottom row to a line
    fig, axes = plt.subplots(
        len(beta_rows), len(sources), figsize=(12, 12), dpi=100, sharey="row"
    )
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, hspace=0.5, wspace=0.15)

    for row, (setting, setting_label) in enumerate(beta_rows):
        for col, (source, source_label) in enumerate(sources):
            ax = axes[row, col]
            rows = table[
                (table["vef_source"] == source) & (table["setting"] == setting)
            ].pivot(index="cell_type", columns="assay", values="beta")
            values = [rows.loc[cell_types, assay].to_numpy() for assay in assays]
            draw_groups(ax, values, assays)

            # two levels: what is regressed on what, then which VEF source
            ax.set_title(f"{setting_label}\n{source_label}")
            if col == 0:
                ax.set_ylabel("Standardized \u03b2")

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

    table = pd.read_csv(metrics_path)
    print(f"[load] {metrics_path} {table.shape}")

    saved = []
    for setting, fig_name in correlation_panels:
        out_path = figures_dir / fig_name
        plot_correlation_panel(table, setting, out_path)
        saved.append(out_path)

    beta_path = figures_dir / beta_name
    plot_beta_panel(table, beta_path)
    saved.append(beta_path)

    legend_path = figures_dir / legend_name
    plot_legend(legend_path)
    saved.append(legend_path)

    for p in saved:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
