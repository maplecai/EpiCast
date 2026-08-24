import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from epicast import metrics
from epicast.utils.plot_utils import set_mpl_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cell_types, figures_dir, model_styles, predictions_dir
from utils import safe_metric

epicast_color = model_styles["epicast_ag_vef"][1]
# measured activity and predictions side by side, written by analysis/14
pred_table = predictions_dir / "gosai_epicast_ag_vef.tsv"
fig_name_template = "fig2b_epicast_ag_vef_scatter_{cell_type}.pdf"


def plot_scatter(ax, true: pd.Series, pred: pd.Series, cell_type: str) -> None:
    valid = ~(true.isna() | pred.isna())
    x = true.loc[valid]
    y = pred.loc[valid]
    r = safe_metric(metrics.pearson, x, y)

    ax.scatter(x, y, s=4, alpha=0.45, color=epicast_color, edgecolors="none", rasterized=True)
    lims = (-4, 6)
    ax.plot(lims, lims, color="black", lw=1, linestyle="--", zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Measured CRE activity ({cell_type})", fontsize=12)
    ax.set_ylabel(f"EpiCast prediction ({cell_type})", fontsize=12)
    ax.text(
        0.03,
        0.97,
        f"r = {r:.3f}\nn = {len(x):,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    table = pd.read_csv(pred_table, sep="\t")
    print(f"[load] {pred_table} {table.shape}")
    test_df = table[table["split"] == "test"]

    set_mpl_params()
    sns.set_theme(style="white", context="talk")
    figures_dir.mkdir(parents=True, exist_ok=True)

    for cell_type in cell_types:
        true = test_df[cell_type]
        pred = test_df[f"{cell_type}_pred"]

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
        plot_scatter(ax, true, pred, cell_type)

        out_path = figures_dir / fig_name_template.format(cell_type=cell_type)
        fig.savefig(out_path, dpi=400)
        plt.close(fig)
        print(f"[save] {out_path.resolve()}")


if __name__ == "__main__":
    main()
