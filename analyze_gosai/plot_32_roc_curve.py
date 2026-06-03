from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from gosai_plot import set_plot_theme


model_names = [
    "sei_dnase",
    "enformer_dnase",
    "borzoi_dnase",
    "alphagenome_dnase",
    "linear",
    "mlp",
    "xgb",
    "lgbm",
    "seq_only",
    "epicast",
]

model_plot_names = [
    "Sei DNase",
    "Enformer DNase",
    "Borzoi DNase",
    "AlphaGenome DNase",
    "Linear",
    "MLP",
    "XGBoost",
    "LightGBM",
    "Seq only",
    "EpiCast",
]

colors = [
    "#B7D9D3",
    "#8FC2C8",
    "#6FA8C6",
    "#4F7FA8",
    "#E7D8A6",
    "#DDBE8A",
    "#D39E7A",
    "#C97F6D",
    "#B6A9CC",
    "#8E7FAF",
]

model_label_map = dict(zip(model_names, model_plot_names))
model_color_map = dict(zip(model_names, colors))

cell_types = ["K562", "HepG2", "SK-N-SH", "HCT116", "A549"]
figures_dir = Path("analyze_gosai/figures")
fig_prefix = "fig3.2_"
mpra_path = Path("data/gosai_mpra/gosai_mpra_760679_zscore.tsv")
loo_pred_dir = Path("analyze_gosai/results/loo_pred")
positive_threshold = 1.96

dnase_pred_paths = {
    "sei_dnase": "data/gosai_mpra/gosai_mpra_760679_sei_vef_logit.tsv",
    "enformer_dnase": "data/gosai_mpra/gosai_mpra_760679_enformer_vef_log1p.tsv",
    "borzoi_dnase": "data/gosai_mpra/gosai_mpra_760679_borzoi_vef_log1p.tsv",
    "alphagenome_dnase": "data/gosai_mpra/gosai_mpra_760679_ag_vef_x10_log1p.tsv",
}
loo_model_names = ["linear", "mlp", "xgb", "lgbm", "seq_only", "epicast"]


def load_model_preds(test_mask):
    preds = {}

    for model_name, pred_path in dnase_pred_paths.items():
        pred_df = pd.read_csv(pred_path, sep="\t")
        preds[model_name] = pred_df.loc[test_mask, [f"{ct}_DNase" for ct in cell_types]].to_numpy()

    for model_name in loo_model_names:
        preds[model_name] = np.load(loo_pred_dir / f"{model_name}_loo_pred.npy")[test_mask]

    return preds


def plot_roc_one_setting(cell_idx: int, preds: dict, true_values: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    y_true = (true_values[:, cell_idx] > positive_threshold).astype(int)

    for model_name in model_names:
        y_score = preds[model_name][:, cell_idx]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = roc_auc_score(y_true, y_score)
        label = f"{model_label_map[model_name]} (AUROC={auroc:.4f})"
        ax.plot(fpr, tpr, color=model_color_map[model_name], linewidth=1.8, label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="_nolegend_")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC comparison  test  ·  {cell_types[cell_idx]}")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="lower right")

    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def main() -> None:
    set_plot_theme(style="whitegrid", context="notebook")
    figures_dir.mkdir(parents=True, exist_ok=True)

    mpra_df = pd.read_csv(mpra_path, sep="\t")
    test_mask = mpra_df["chr"].isin(["chr7", "chr13"]).to_numpy()
    true_values = mpra_df.loc[test_mask, cell_types].to_numpy(dtype=float)
    preds = load_model_preds(test_mask)

    for cell_idx, cell_type in enumerate(cell_types):
        save_path = figures_dir / f"{fig_prefix}test_{cell_type}_roc_multi_model.pdf"
        plot_roc_one_setting(cell_idx, preds, true_values, save_path)


if __name__ == "__main__":
    main()
