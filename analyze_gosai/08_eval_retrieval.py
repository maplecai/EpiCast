import os
import warnings

import pandas as pd
from scipy.stats import ConstantInputWarning

from gosai_eval import evaluate_loo_model_retrieval
from gosai_masks import build_masks

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
warnings.filterwarnings("ignore", category=ConstantInputWarning)


mpra_path = "data/gosai_mpra/gosai_mpra_760679_zscore.tsv"
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


def main():
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(mpra_df.shape)

    masks = build_masks(mpra_df, cell_types)
    k_list = [100, 1000, 10000]
    output_dir = "analyze_gosai/results/loo_specific_retrieval"
    os.makedirs(output_dir, exist_ok=True)

    all_model_results = []
    for split in ["test"]:
        for model_name in vef_only_models + ["epicast"]:
            result_df = evaluate_loo_model_retrieval(
                model_name=model_name,
                pred_paths=loo_pred_paths[model_name],
                cell_types=cell_types,
                masks=masks,
                split=split,
                k_list=k_list,
                specific_score_mode="max",
            )
            out_path = os.path.join(output_dir, f"{model_name}_loo_specific_retrieval.tsv")
            result_df.to_csv(out_path, sep="\t", index=False)
            print("saved:", out_path)
            all_model_results.append(result_df)

        combined_df = pd.concat(all_model_results, axis=0, ignore_index=True)
        combined_path = os.path.join(output_dir, "all_models_loo_specific_retrieval.tsv")
        combined_df.to_csv(combined_path, sep="\t", index=False)
        print("saved:", combined_path)

        summary_df = combined_df[combined_df["cell_type"] == "loo_mean"].copy()
        summary_path = os.path.join(output_dir, "summary_loo_mean_specific_retrieval.tsv")
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print("saved:", summary_path)
        print(summary_df)


if __name__ == "__main__":
    main()
