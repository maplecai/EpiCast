from pathlib import Path

import numpy as np

from gosai_io import merge_loo_pred


cell_types = ["K562", "HepG2", "SK-N-SH", "HCT116", "A549"]
model_names = ["linear", "mlp", "xgb", "lgbm", "seq_only", "epicast"]
output_dir = Path("analyze_gosai/results/loo_pred")

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


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        merged_pred = merge_loo_pred(loo_pred_paths[model_name], cell_types)
        output_path = output_dir / f"{model_name}_loo_pred.npy"
        np.save(output_path, merged_pred)
        print("saved:", output_path, merged_pred.shape)


if __name__ == "__main__":
    main()
