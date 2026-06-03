import os
import warnings

import pandas as pd

from gosai_eval import evaluate_full_model_binary, evaluate_loo_model_binary
from gosai_io import build_true_df, load_dnase_pred_df
from gosai_masks import build_masks

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
warnings.filterwarnings("ignore")


mpra_path = "data/gosai_mpra/gosai_mpra_760679_zscore.tsv"
positive_threshold = 1.96
cell_types = ["K562", "HepG2", "SK-N-SH", "HCT116", "A549"]
vef_only_models = ["linear", "mlp", "xgb", "lgbm"]
loo_model_names = vef_only_models + ["seq_only", "epicast"]

loo_pred_paths = {
    "linear": {
        cell_type: f"analyze_gosai/results/vef_only/linear_leave_out_{cell_type}_pred.npy"
        for cell_type in cell_types
    },
    "mlp": {
        cell_type: f"analyze_gosai/results/vef_only/mlp_leave_out_{cell_type}_pred.npy"
        for cell_type in cell_types
    },
    "xgb": {
        cell_type: f"analyze_gosai/results/vef_only/xgb_leave_out_{cell_type}_pred.npy"
        for cell_type in cell_types
    },
    "lgbm": {
        cell_type: f"analyze_gosai/results/vef_only/lgbm_leave_out_{cell_type}_pred.npy"
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

dnase_pred_paths = {
    "sei_dnase": "data/gosai_mpra/gosai_mpra_760679_sei_vef_logit.tsv",
    "enformer_dnase": "data/gosai_mpra/gosai_mpra_760679_enformer_vef_log1p.tsv",
    "borzoi_dnase": "data/gosai_mpra/gosai_mpra_760679_borzoi_vef_log1p.tsv",
    "alphagenome_dnase": "data/gosai_mpra/gosai_mpra_760679_ag_vef_x10_log1p.tsv",
}
model_order = list(dnase_pred_paths) + loo_model_names


def main():
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(mpra_df.shape)

    output_dir = "analyze_gosai/results/loo_binary_curves"
    os.makedirs(output_dir, exist_ok=True)

    masks = build_masks(mpra_df, cell_types)
    true_df = build_true_df(mpra_df, cell_types)
    splits = ["test"]
    all_model_results = []

    for split in splits:
        for model_name, pred_path in dnase_pred_paths.items():
            print("loading", model_name, pred_path)
            pred_df = load_dnase_pred_df(pred_path, cell_types)
            all_model_results.append(
                evaluate_full_model_binary(model_name, pred_df, true_df, cell_types, masks, split, positive_threshold)
            )

        for model_name, pred_paths in loo_pred_paths.items():
            all_model_results.append(
                evaluate_loo_model_binary(model_name, pred_paths, true_df, cell_types, masks, split, positive_threshold)
            )

    combined_df = pd.concat(all_model_results, axis=0, ignore_index=True)
    combined_path = os.path.join(output_dir, "all_models_loo_binary_metrics.tsv")
    combined_df.to_csv(combined_path, sep="\t", index=False)
    print("saved:", combined_path)

    summary_df = combined_df[combined_df["cell_type"] == "loo_mean"].copy()
    summary_path = os.path.join(output_dir, "summary_loo_mean_binary_metrics.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print("saved:", summary_path)
    print(summary_df)

    for split in splits:
        for metric_name in ["auroc", "auprc"]:
            sub = combined_df[
                (combined_df["split"] == split)
                & (combined_df["cell_type"] != "loo_mean")
            ].copy()
            wide_df = sub.pivot(index="model", columns="cell_type", values=metric_name)
            wide_df = wide_df.reindex(index=model_order)
            wide_df = wide_df[cell_types]
            wide_df["mean"] = wide_df.mean(axis=1, skipna=True)

            safe_split = split.replace("&", "_and_").replace("|", "_or_")
            wide_path = os.path.join(output_dir, f"leave_one_out_{safe_split}_{metric_name}.tsv")
            wide_df.to_csv(wide_path, sep="\t")
            print("saved:", wide_path)
            print(wide_df)


if __name__ == "__main__":
    main()
