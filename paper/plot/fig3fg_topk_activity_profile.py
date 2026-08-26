"""fig3F / fig3G: measured activity profile of the CREs the EpiCast models rank first.

One figure per ranking task, fig3F for CTS-high and fig3G for CTS-low, each a row of four
panels: EpiCast-Sei on HCT116 and on A549, then EpiCast-AlphaGenome on the same two. The
panel title names both the model and the target cell, so the pair a panel belongs to is
readable without the hand-added headers.

For every screening depth the CREs are ranked by predicted residual activity (negated
for CTS-low), the top fraction is taken, and the mean measured activity of that one
selected set is read out with its SEM. A model that had merely learnt to find generally
active elements would move all the curves together; the target cell curve separating
from the training cells is the cell-type-specific signal.

Each panel draws the three training cell types plus its own target, four curves, and the
legend collects all five colours. The other held-out cell type is left out because the
two are assayed on almost disjoint parts of the library: among the 36 CREs of the top
0.1% for HCT116 only 2 carry an A549 measurement, so its mean would be an artefact of
two sequences drawn on the same axis as means over hundreds.

The ranking universe is the test chromosomes restricted to the CREs actually measured
in the target cell type, the same pool that analysis/09 scores, so the depths here
line up with fig 3D and 3E. Both tasks share one y axis range so 3F and 3G can be read
against each other.

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
import numpy as np
import pandas as pd
import seaborn as sns
from epicast.utils.plot_utils import set_mpl_params
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    cell_colors,
    cell_types,
    figures_dir,
    predictions_dir,
    test_cell_types,
    train_cell_types,
)

legend_name = "fig3fg_legend.pdf"
eval_split = "test"

# (prediction table, panel label) in panel order within a figure
models = [
    ("gosai_epicast_sei_vef.tsv", "EpiCast-Sei"),
    ("gosai_epicast_ag_vef.tsv", "EpiCast-AlphaGenome"),
]
# (task, sign applied to the predicted residual before ranking, output figure)
tasks = [
    ("CTS-high", 1.0, "fig3f_topk_activity_cts_high.pdf"),
    ("CTS-low", -1.0, "fig3g_topk_activity_cts_low.pdf"),
]

min_frac = 1e-4
max_frac = 1e-1
x_ticks = [1e-4, 1e-3, 1e-2, 1e-1]
# plain text, not mathtext: matplotlib renders $\log_{10}$ in its own font, not Arial
x_label = "log10(top-ranked fraction)"
n_depths = 40
target_linewidth = 2.0
train_linewidth = 1.0


def activity_profile(table, mask, score, panel_cell_types):
    """Mean and SEM of measured activity in each cell type, over a grid of depths."""
    order = np.argsort(-score[mask].to_numpy())
    rows = table.index[mask][order]
    n_eval = len(rows)
    fracs = np.logspace(np.log10(min_frac), np.log10(max_frac), n_depths)

    means = {ct: [] for ct in panel_cell_types}
    sems = {ct: [] for ct in panel_cell_types}
    for frac in fracs:
        selected = table.loc[rows[: max(1, round(frac * n_eval))], panel_cell_types]
        for ct in panel_cell_types:
            values = selected[ct].dropna()
            means[ct].append(values.mean())
            sems[ct].append(values.std(ddof=1) / np.sqrt(len(values)))
    return fracs, means, sems


def plot_panels(profiles, ylim, save_path):
    fig, axes = plt.subplots(1, len(profiles), figsize=(24, 6), dpi=100, sharey=True)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, wspace=0.3)

    for ax, (model_label, target, fracs, means, sems) in zip(axes, profiles):
        for ct in train_cell_types + [target]:
            mean = np.asarray(means[ct], dtype=float)
            sem = np.asarray(sems[ct], dtype=float)
            lw = target_linewidth if ct == target else train_linewidth
            ax.plot(fracs, mean, color=cell_colors[ct], lw=lw)
            ax.fill_between(fracs, mean - sem, mean + sem, color=cell_colors[ct], alpha=0.2, lw=0)

        ax.axhline(0, color="gray", lw=1.0, linestyle="--")
        ax.set_title(f"{model_label}\n{target}")
        ax.set_ylim(ylim)
        # a log axis labelled by its exponents rather than by 10^-4 and friends
        ax.set_xscale("log")
        ax.set_xlim(min_frac, max_frac)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([round(np.log10(tick)) for tick in x_ticks])
        ax.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Measured activity")
    fig.supxlabel(x_label)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_legend(save_path):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.set_axis_off()
    handles = [Line2D([0], [0], color=cell_colors[ct], lw=2.0, label=ct) for ct in cell_types]
    ax.legend(handles=handles, loc="center", frameon=False, handlelength=1.6, labelspacing=0.5)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    set_mpl_params()
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    for table_name, _ in models:
        tables[table_name] = pd.read_csv(predictions_dir / table_name, sep="\t")
        print(f"[load] {table_name} {tables[table_name].shape}")

    task_profiles = []
    for task, sign, _ in tasks:
        profiles = []
        for table_name, model_label in models:
            table = tables[table_name]
            train_mean_pred = table[[f"{ct}_pred" for ct in train_cell_types]].mean(axis=1)
            eval_mask = table["split"] == eval_split

            for target in test_cell_types:
                mask = eval_mask & table[target].notna()
                score = sign * (table[f"{target}_pred"] - train_mean_pred)
                panel_cell_types = train_cell_types + [target]
                fracs, means, sems = activity_profile(table, mask, score, panel_cell_types)
                print(f"  [profile] {task} {model_label} {target} n_eval={int(mask.sum()):,}")
                profiles.append((model_label, target, fracs, means, sems))
        task_profiles.append(profiles)

    # the two tasks are separate files but keep one y range, so 3F and 3G stay comparable
    bounds = [
        np.asarray(means[ct]) + sign * np.asarray(sems[ct])
        for profiles in task_profiles
        for _, _, _, means, sems in profiles
        for ct in means
        for sign in (-1.0, 1.0)
    ]
    span = np.concatenate(bounds)
    ylim = (span.min() - 0.1, span.max() + 0.1)

    saved = []
    for profiles, (_, _, fig_name) in zip(task_profiles, tasks):
        fig_path = figures_dir / fig_name
        plot_panels(profiles, ylim, fig_path)
        saved.append(fig_path)

    legend_path = figures_dir / legend_name
    plot_legend(legend_path)
    saved.append(legend_path)

    for p in saved:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
