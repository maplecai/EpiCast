"""Why these four assays: availability and predictive performance on one pair of axes.

Every assay that can be scored at all is placed by how many biosamples AlphaGenome covers
it in (availability) against how well its VEF predicts measured activity (performance).

A VEF exists for a cell type only if the assay has a track there, so "comparable across
cell types" is a hard filter rather than a convenience. The three marker tiers are that
filter, and they are nested, so the legend counts are cumulative rather than the size of
each colour group: 25 assays have a track in the three reference cell types, 10 of those
have one in all five, and the paper uses 4 of those 10. Reading the legend top to bottom
therefore gives the availability funnel, which is why it needs no separate panel.

Which points are named is set by hand in `label_assays`; the rest form one dense
low-coverage cluster whose individual identities the figure does not argue about.

The performance axis is the correlation with *absolute* activity, averaged over the cell
types an assay covers. That choice is deliberate: this ranking reproduces across the two
held-out cell types (Spearman rho = 0.89), whereas the ranking on residual activity does
not (rho = -0.43, n.s.), so residual performance cannot support a claim about which
assays to pick and is left to the tables.

Reads results/vef_assay_selection/, written by analysis/18_vef_assay_selection.py.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epicast.utils.plot_utils import set_mpl_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cell_types, figures_dir, results_dir, train_cell_types

metrics_dir = results_dir / "vef_assay_selection"
tradeoff_name = "figs1a_assay_coverage_vs_performance.pdf"

# an assay needs a track in every Gosai cell type before it can enter a model
n_cell_types = len(cell_types)
marker_size = 70

# Which points carry a name, listed by hand. These are the 11 best covered assays, i.e.
# the high-coverage end where the choice was actually made; the low-coverage candidates
# stay as an unlabelled cluster. Edit this list to relabel the figure.
label_assays = [
    "DNase",
    "H3K4me3",
    "H3K4me1",
    "H3K27ac",
    "H3K36me3",
    "CTCF",
    "H3K27me3",
    "H3K9ac",
    "H3K4me2",
    "H3K79me2",
    "POLR2AphosphoS5",
]
chosen_color = "#08599C"
usable_color = "#57B8D0"
other_color = "#B0B0B0"



def load():
    pool = pd.read_csv(metrics_dir / "candidate_pool.csv")
    univariate = pd.read_csv(metrics_dir / "assay_univariate.csv")
    performance = univariate.groupby("assay")["absolute_r"].mean()
    return pool.set_index("assay").assign(absolute_r=performance).reset_index()


def place_labels(ax, table):
    """Annotate every point, nudging labels that would sit on top of each other.

    adjustText is not installed, so this is a deterministic greedy pass in axes
    coordinates: try above, then below, then progressively further out, and keep the
    first offset whose box misses every label already placed.
    """
    x_span = (np.log10(table["coverage"].max()) - np.log10(table["coverage"].min())) or 1.0
    y_span = (table["absolute_r"].max() - table["absolute_r"].min()) or 1.0
    # small boxes keep a label next to its own marker, which is what disambiguates it
    label_w, label_h = 0.068, 0.027
    # sideways before below: dropping a label under its marker tends to land it on top
    # of whatever point sits beneath, which is exactly the confusion to avoid
    offsets = [
        (0, 1), (1, 0), (-1, 0), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (0, 2), (2, 0), (-2, 0), (0, -2),
    ]

    def to_axes(row):
        return (
            (np.log10(row["coverage"]) - np.log10(table["coverage"].min())) / x_span,
            (row["absolute_r"] - table["absolute_r"].min()) / y_span,
        )

    # markers are obstacles too, otherwise a label lands on a neighbouring point
    occupied = [(*to_axes(row), label_w * 0.35, label_h * 0.5) for _, row in table.iterrows()]
    named = table[table["assay"].isin(label_assays)]
    # the complete assays go first, so they take the slot straight above their marker
    priority = named.assign(
        rank=named["chosen"].astype(int) + (named["n_gosai_cells"] == n_cell_types).astype(int)
    )
    for _, row in priority.sort_values(["rank", "coverage"], ascending=False).iterrows():
        ax_x, ax_y = to_axes(row)
        for dx, dy in offsets:
            box_x, box_y = ax_x + dx * label_w, ax_y + dy * label_h
            if all(
                abs(box_x - px) > (label_w * 0.5 + pw) or abs(box_y - py) > (label_h * 0.5 + ph)
                for px, py, pw, ph in occupied
            ):
                occupied.append((box_x, box_y, label_w * 0.5, label_h * 0.5))
                break
        ax.annotate(
            row["assay"],
            (row["coverage"], row["absolute_r"]),
            xytext=(dx * 26, dy * 11 + (4 if dy >= 0 else -4)),
            textcoords="offset points",
            ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
            va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
            fontsize=10,
            color="black",
            zorder=4,
        )


def plot_tradeoff(table):
    fig, ax = plt.subplots(figsize=(10.5, 7.5))

    complete = table["n_gosai_cells"] == n_cell_types
    # The counts are cumulative, not the size of each colour group: the three tiers are
    # nested, so 25 -> 10 -> 4 is the availability funnel and carries it in the legend.
    groups = [
        (
            ~complete,
            other_color,
            f"available in the {len(train_cell_types)} reference cell types (n = {len(table)})",
        ),
        (
            complete & ~table["chosen"],
            usable_color,
            f"available in all {n_cell_types} cell types (n = {int(complete.sum())})",
        ),
        (
            table["chosen"],
            chosen_color,
            f"selected for EpiCast (n = {int(table['chosen'].sum())})",
        ),
    ]
    # one size for all three tiers: colour alone carries the tier, so a reader never has
    # to compare areas to tell them apart
    for mask, color, label in groups:
        subset = table[mask]
        ax.scatter(
            subset["coverage"], subset["absolute_r"], s=marker_size, color=color,
            edgecolors="black", linewidths=0.8, zorder=3, label=label,
        )

    ax.set_xscale("log")
    place_labels(ax, table)
    ax.margins(x=0.13, y=0.10)
    ax.set_xlabel("Biosamples with this assay in AlphaGenome")
    ax.set_ylabel("Pearson r with measured activity")
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    ax.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5)
    ax.tick_params(axis="x", which="minor", bottom=True, length=2.0)
    ax.spines[["top", "right"]].set_visible(False)

    out_path = figures_dir / tradeoff_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")



def main():
    set_mpl_params()
    figures_dir.mkdir(parents=True, exist_ok=True)
    table = load()
    print(f"[load] {len(table)} candidate assays")
    plot_tradeoff(table)


if __name__ == "__main__":
    main()
