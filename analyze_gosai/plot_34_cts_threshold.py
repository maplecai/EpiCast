import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epicast import metrics

from gosai_eval import safe_metric
from gosai_io import build_leave_one_out_merged_pred_df, build_true_df
from gosai_masks import build_masks
from gosai_plot import set_plot_theme


cell_types = ["K562", "HepG2", "SK-N-SH", "HCT116", "A549"]
mpra_path = "data/gosai_mpra/gosai_mpra_760679_zscore.tsv"
percentiles = [90, 95, 96, 97, 98, 99]
model_order = ["linear", "seq_only", "epicast"]
model_labels = {"linear": "linear", "seq_only": "seq only", "epicast": "epicast"}

loo_pred_paths = {
    "linear": {
        cell_type: f"analyze_gosai/results/vef_only/linear_leave_out_{cell_type}_pred.npy"
        for cell_type in cell_types
    },
    "seq_only": {
        cell_type: f"analyze_gosai/results/seq_only/leave_one_out_pred_{cell_type}.npy"
        for cell_type in cell_types
    },
    "epicast": {
        "K562": "saved/0418_gosai_ag_vef_final_1/0418_074130/preds.npy",
        "HepG2": "saved/0418_gosai_ag_vef_final_2/0418_074100/preds.npy",
        "SK-N-SH": "saved/0418_gosai_ag_vef_final_3/0418_074041/preds.npy",
        "HCT116": "saved/0418_gosai_ag_vef_final_4/0418_073825/preds.npy",
        "A549": "saved/0418_gosai_ag_vef_final_5/0418_073743/preds.npy",
    },
}


def load_model_pred_tables(mpra_df):
    return {
        model_name: build_leave_one_out_merged_pred_df(mpra_df, loo_pred_paths[model_name], cell_types)
        for model_name in model_order
    }


def pearson_on_mask(true_series: pd.Series, pred_series: pd.Series, mask: np.ndarray) -> float:
    x = true_series.loc[mask]
    y = pred_series.loc[mask]
    return float(safe_metric(metrics.pearson, x, y))


def compute_mean_pearson_by_threshold(mpra_df, true_df, masks, model_pred_dfs):
    rows = []
    test_mask = masks["test"]

    for q in percentiles:
        cell_masks = []
        for cell_type in cell_types:
            other_cell_types = [ct for ct in cell_types if ct != cell_type]
            delta = (mpra_df[cell_type] - mpra_df[other_cell_types].mean(axis=1)).abs()
            delta_test = delta.loc[test_mask].dropna()
            thr = float(np.percentile(delta_test, q))
            cell_masks.append(test_mask & (delta >= thr).fillna(False).to_numpy())

        for model_name, pred_df in model_pred_dfs.items():
            per_ct = [
                pearson_on_mask(true_df[f"{ct}_true"], pred_df[f"{ct}_pred"], cell_masks[i])
                for i, ct in enumerate(cell_types)
            ]
            rows.append({"cts_percentile": q, "model": model_name, "pearson_mean": float(np.nanmean(per_ct))})

    return pd.DataFrame(rows)


def plot_mean_cts_curve(result_df: pd.DataFrame, save_path: str) -> None:
    set_plot_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.88)

    for model_name in model_order:
        cur = result_df[result_df["model"] == model_name].sort_values("cts_percentile")
        ax.plot(cur["cts_percentile"], cur["pearson_mean"], marker="o", label=model_labels[model_name])

    ax.set_xlim(88, 101)
    ax.set_xlabel("CTS percentile", fontsize=12)
    ax.set_ylabel("Mean Pearson (across cell types)", fontsize=12)
    ax.set_title("Performance vs. cell-type specificity threshold (test)", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    masks = build_masks(mpra_df, cell_types)
    true_df = build_true_df(mpra_df, cell_types)
    model_pred_dfs = load_model_pred_tables(mpra_df)
    result_df = compute_mean_pearson_by_threshold(mpra_df, true_df, masks, model_pred_dfs)
    print(result_df)

    out_path = "analyze_gosai/figures/fig3.4_unseen_ct_performance_vs_cts_threshold.pdf"
    plot_mean_cts_curve(result_df, out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
