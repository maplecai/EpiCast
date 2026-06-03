import os
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import ConstantInputWarning

from gosai_eval import evaluate_retrieval_single_pred
from gosai_io import load_pred_df
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
epicast_exp_root = "saved/0428_gosai_ag_vef_final_3_ablation_VEF_1"


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
    k_list = [100, 1000, 10000]
    output_dir = "analyze_gosai/results/train3test2_specific_retrieval"
    os.makedirs(output_dir, exist_ok=True)

    epicast_pred_path = resolve_latest_preds_npy(epicast_exp_root)
    print("epicast ->", epicast_pred_path)

    model_pred_paths = {
        model_name: os.path.join(vef_pred_dir, f"{model_name}_pred.npy")
        for model_name in vef_only_models
    }
    model_pred_paths["epicast"] = str(epicast_pred_path)

    all_model_results = []
    for split in ["test"]:
        for model_name, pred_path in model_pred_paths.items():
            print("loading", model_name, pred_path)
            pred_df = load_pred_df(pred_path, cell_types)
            result_df = evaluate_retrieval_single_pred(
                model_name=model_name,
                pred_df=pred_df,
                all_cell_types=cell_types,
                test_cell_types=test_cell_types,
                masks=masks,
                split=split,
                k_list=k_list,
                specific_score_mode="max",
            )
            out_path = os.path.join(output_dir, f"{model_name}_train3test2_specific_retrieval.tsv")
            result_df.to_csv(out_path, sep="\t", index=False)
            print("saved:", out_path)
            all_model_results.append(result_df)

        combined_df = pd.concat(all_model_results, axis=0, ignore_index=True)
        combined_path = os.path.join(output_dir, "all_models_train3test2_specific_retrieval.tsv")
        combined_df.to_csv(combined_path, sep="\t", index=False)
        print("saved:", combined_path)

        summary_df = combined_df[combined_df["cell_type"] == "test_mean"].copy()
        summary_path = os.path.join(output_dir, "summary_test_mean_specific_retrieval.tsv")
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print("saved:", summary_path)
        print(summary_df)


if __name__ == "__main__":
    main()
