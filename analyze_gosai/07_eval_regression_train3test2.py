import os
import warnings
from pathlib import Path

import pandas as pd
from epicast import metrics
from scipy.stats import ConstantInputWarning

from gosai_eval import evaluate_full_model_on_cells, evaluate_single_pred_model
from gosai_io import build_true_df, load_dnase_pred_df
from gosai_masks import build_masks

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

mpra_path = "data/gosai_mpra/gosai_mpra_760679_zscore.tsv"
train_cell_types = ["K562", "HepG2", "SK-N-SH"]
test_cell_types = ["HCT116", "A549"]
cell_types = train_cell_types + test_cell_types

vef_only_models = ["linear", "mlp", "xgb", "lgbm"]
vef_pred_dir = "analyze_gosai/results/vef_only_train3test2"

# train3test2 EpiCast: 0428 configs use train K562/HepG2/SK-N-SH, valid HCT116/A549
epicast_exp_root = "saved/0428_gosai_ag_vef_final_3_ablation_VEF_1"
# epicast_pred_path = "saved/.../preds.npy"

dnase_pred_paths = {
    "sei_dnase": "data/gosai_mpra/gosai_mpra_760679_sei_vef_logit.tsv",
    "enformer_dnase": "data/gosai_mpra/gosai_mpra_760679_enformer_vef_log1p.tsv",
    "borzoi_dnase": "data/gosai_mpra/gosai_mpra_760679_borzoi_vef_log1p.tsv",
    "alphagenome_dnase": "data/gosai_mpra/gosai_mpra_760679_ag_vef_x10_log1p.tsv",
}
model_order = list(dnase_pred_paths) + vef_only_models + ["epicast"]


def resolve_latest_preds_npy(exp_root):
    root = Path(exp_root)
    candidates = [(p.stat().st_mtime, p) for p in root.glob("*/preds.npy")]
    if not candidates:
        raise FileNotFoundError(f"No preds.npy under {root}")
    return max(candidates, key=lambda t: t[0])[1]


def main():
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(mpra_df.shape)

    masks = build_masks(mpra_df, cell_types)
    true_df = build_true_df(mpra_df, cell_types)
    splits = ["test", "test&specific", "test&all_specific"]
    metric_fn_map = {"pearson": metrics.pearson, "spearman": metrics.spearman}
    all_model_results = []

    epicast_pred_path = resolve_latest_preds_npy(epicast_exp_root)
    print("epicast ->", epicast_pred_path)

    for split in splits:
        for model_name, pred_path in dnase_pred_paths.items():
            print("loading", model_name, pred_path)
            pred_df = load_dnase_pred_df(pred_path, cell_types)
            all_model_results.append(
                evaluate_full_model_on_cells(
                    model_name,
                    pred_df,
                    true_df,
                    test_cell_types,
                    masks,
                    split,
                    metric_fn_map,
                )
            )

        for model_name in vef_only_models:
            pred_path = os.path.join(vef_pred_dir, f"{model_name}_pred.npy")
            all_model_results.append(
                evaluate_single_pred_model(
                    model_name,
                    pred_path,
                    true_df,
                    cell_types,
                    test_cell_types,
                    masks,
                    split,
                    metric_fn_map,
                )
            )

        all_model_results.append(
            evaluate_single_pred_model(
                "epicast",
                str(epicast_pred_path),
                true_df,
                cell_types,
                test_cell_types,
                masks,
                split,
                metric_fn_map,
            )
        )

    output_dir = "analyze_gosai/results/train3test2_correlation"
    os.makedirs(output_dir, exist_ok=True)
    combined_df = pd.concat(all_model_results, axis=0, ignore_index=True)

    combined_path = os.path.join(output_dir, "all_models_train3test2_correlation.tsv")
    combined_df.to_csv(combined_path, sep="\t", index=False)
    print("saved:", combined_path)

    summary_df = combined_df[combined_df["cell_type"] == "test_mean"].copy()
    summary_path = os.path.join(output_dir, "summary_test_mean_correlation.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print("saved:", summary_path)
    print(summary_df)

    for split in splits:
        for metric_name in metric_fn_map:
            sub = combined_df[
                (combined_df["split"] == split)
                & (combined_df["metric"] == metric_name)
                & (combined_df["cell_type"] != "test_mean")
            ].copy()
            wide_df = sub.pivot(index="model", columns="cell_type", values="value")
            wide_df = wide_df.reindex(index=model_order)
            wide_df = wide_df[test_cell_types]
            wide_df["mean"] = wide_df.mean(axis=1, skipna=True)

            wide_path = os.path.join(output_dir, f"train3test2_{split}_{metric_name}.tsv")
            wide_df.to_csv(wide_path, sep="\t")
            print("saved:", wide_path)
            print(wide_df)


if __name__ == "__main__":
    main()
