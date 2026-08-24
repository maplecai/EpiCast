"""fig3g: measured activity of the variants EpiCast (AlphaGenome) ranks at the
extremes of predicted residual, i.e. the variants it calls CTS_high/CTS_low.

Two selection rules are plotted side by side: the 1/99 percentile cut-off used by
analysis/08 and analysis/09 (`pct1` files), and a fixed top/bottom 100 (`n100` files).

Residual is pred_c - mean(pred over train cell types), the same quantity as fig3b.
HCT116 and A549 are assayed in only part of the library, so every quantity is computed
inside that measured subset, the universe the true CTS labels already live in:
percentile cut-offs over all measured variants (analysis/08), top-k ranking inside the
evaluated pool of test & measured variants (analysis/09). Each panel therefore shows
the train cell types plus the selection cell type, not the other test cell type.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast.utils.plot_utils import set_mpl_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    cell_types,
    figures_dir,
    predictions_dir,
    test_cell_types,
    train_cell_types,
)
from utils import build_cts_labels

# measured activity and predictions side by side, written by analysis/14
pred_table = predictions_dir / "gosai_epicast_ag_vef.tsv"
eval_split = "test"
tasks = ["CTS_high", "CTS_low"]
cts_low_pct = 1
cts_high_pct = 99
top_n = 100

cell_palette = dict(zip(cell_types, sns.color_palette("tab10", len(cell_types), desat=0.45)))
box_step = 0.55
box_width = 0.36


def select_by_percentile(gap_measured: pd.Series, gap_eval: pd.Series) -> dict[str, pd.Index]:
    """Percentile cut-offs over the measured subset, as in analysis/08."""
    _, _, q_hi, q_lo = build_cts_labels(
        gap_measured, low_pct=cts_low_pct, high_pct=cts_high_pct
    )
    print(f"  [pct] q{cts_high_pct}={q_hi:.4f} q{cts_low_pct}={q_lo:.4f}")
    return {
        "CTS_high": gap_eval.index[gap_eval > q_hi],
        "CTS_low": gap_eval.index[gap_eval < q_lo],
    }


def select_by_count(gap_measured: pd.Series, gap_eval: pd.Series) -> dict[str, pd.Index]:
    """Top/bottom k inside the evaluated pool, as in analysis/09's p@k."""
    return {
        "CTS_high": gap_eval.nlargest(top_n).index,
        "CTS_low": gap_eval.nsmallest(top_n).index,
    }


selectors = [
    (f"pct{cts_low_pct}", select_by_percentile),
    (f"n{top_n}", select_by_count),
]


def plot_box(
    activity_df: pd.DataFrame, plot_cell_types: list[str], save_path: Path, title: str
) -> None:
    groups = [activity_df[ct].dropna().to_numpy() for ct in plot_cell_types]
    positions = np.arange(len(plot_cell_types), dtype=float) * box_step

    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=100)
    fig.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.9)
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2},
    )
    for patch, cell_type in zip(bp["boxes"], plot_cell_types):
        patch.set_facecolor(cell_palette[cell_type])
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
        patch.set_alpha(0.65)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=0)
    ax.set_xlim(positions[0] - box_step * 0.7, positions[-1] + box_step * 0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_cell_types, fontsize=12, rotation=30, ha="right")
    ax.set_ylabel("Measured CRE activity (z-score)")
    ax.set_title(title, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def main() -> None:
    table = pd.read_csv(pred_table, sep="\t")
    print(f"[load] {pred_table} {table.shape}")

    eval_mask = (table["split"] == eval_split).to_numpy()
    train_mean_pred = table[[f"{ct}_pred" for ct in train_cell_types]].mean(axis=1)

    set_mpl_params()
    sns.set_theme(style="whitegrid", context="talk")
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for cell_type in test_cell_types:
        measured = table[cell_type].notna().to_numpy()
        cell_eval = eval_mask & measured
        gap_pred = table[f"{cell_type}_pred"] - train_mean_pred
        plot_cell_types = train_cell_types + [cell_type]
        print(f"[select] {cell_type} n_eval={int(cell_eval.sum())}")

        for suffix, selector in selectors:
            selected = selector(gap_pred[measured], gap_pred[cell_eval])
            for task in tasks:
                index = selected[task]
                title = f"{cell_type} ({task}, {eval_split}, n = {len(index):,})"
                out_path = (
                    figures_dir
                    / f"fig3g_{cell_type}_{task.lower()}_{suffix}_activity_boxplot.pdf"
                )
                plot_box(table.loc[index, plot_cell_types], plot_cell_types, out_path, title)
                saved_paths.append(out_path)

    for p in saved_paths:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
