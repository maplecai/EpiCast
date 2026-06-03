from pathlib import Path
import os
import warnings

import pandas as pd
from epicast import metrics
from scipy.stats import ConstantInputWarning

from gosai_eval import safe_metric
from gosai_io import build_true_df, load_pred_df
from gosai_masks import build_masks, get_mask

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

cell_types = ["K562", "HepG2", "SK-N-SH", "HCT116", "A549"]
mpra_path = "data/gosai_mpra/gosai_mpra_760679_zscore.tsv"
output_dir = "analyze_gosai/results/ablation_sk_n_sh"

ablation_experiments = {
    "VEF_1_no_DNase": ("saved/0427_gosai_ag_vef_final_3_ablation_VEF_1", "H3K4me3+H3K27ac+CTCF"),
    "VEF_2_no_H3K4me3": ("saved/0427_gosai_ag_vef_final_3_ablation_VEF_2", "DNase+H3K27ac+CTCF"),
    "VEF_3_no_H3K27ac": ("saved/0427_gosai_ag_vef_final_3_ablation_VEF_3", "DNase+H3K4me3+CTCF"),
    "VEF_4_no_CTCF": ("saved/0427_gosai_ag_vef_final_3_ablation_VEF_4", "DNase+H3K4me3+H3K27ac"),
}

baseline_models = {
    "epicast_full_baseline": ("saved/0418_gosai_ag_vef_final_3/0418_074041/pred_8566.npy", "EpiCast full assays (0418 LOO SK-N-SH)"),
}


def resolve_latest_preds_npy(exp_root):
    root = Path(exp_root)
    candidates = [(p.stat().st_mtime, p) for p in root.glob("*/preds.npy")]
    return max(candidates, key=lambda t: t[0])[1]


def evaluate_target_cell(model_key, pred_df, true_df, masks, splits, metric_fn_map, target_cell):
    rows = []
    for split in splits:
        eval_mask = get_mask(split, masks, cell_type=target_cell)
        x = true_df.loc[eval_mask, f"{target_cell}_true"]
        y = pred_df.loc[eval_mask, f"{target_cell}_pred"]

        for metric_name, metric_fn in metric_fn_map.items():
            rows.append(
                {
                    "model": model_key,
                    "split": split,
                    "target_cell": target_cell,
                    "metric": metric_name,
                    "residual": False,
                    "n_eval": int(eval_mask.sum()),
                    "value": safe_metric(metric_fn, x, y),
                }
            )

    return pd.DataFrame(rows)


def split_to_filename(split_name):
    return split_name.replace("&", "_").replace("|", "_").replace("/", "-").replace(" ", "")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print("mpra_df:", mpra_df.shape)

    masks = build_masks(mpra_df, cell_types)
    true_df = build_true_df(mpra_df, cell_types)
    metric_fn_map = {"pearson": metrics.pearson, "spearman": metrics.spearman}
    splits = ["test", "test&specific", "test&all_specific"]
    all_long = []

    for model_key, (exp_root, assay_label) in ablation_experiments.items():
        pred_path = resolve_latest_preds_npy(exp_root)
        print(model_key, assay_label, "->", pred_path)
        pred_df = load_pred_df(pred_path, cell_types)

        for target_cell in cell_types:
            all_long.append(
                evaluate_target_cell(model_key, pred_df, true_df, masks, splits, metric_fn_map, target_cell).assign(
                    assays_note=assay_label,
                    preds_path=str(pred_path),
                )
            )

    for model_key, (pred_rel, assay_label) in baseline_models.items():
        pred_path = Path(pred_rel)
        print(model_key, assay_label, "->", pred_path)
        pred_df = load_pred_df(pred_path, cell_types)

        for target_cell in cell_types:
            all_long.append(
                evaluate_target_cell(model_key, pred_df, true_df, masks, splits, metric_fn_map, target_cell).assign(
                    assays_note=assay_label,
                    preds_path=str(pred_path),
                )
            )

    long_df = pd.concat(all_long, axis=0, ignore_index=True)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path = out_dir / "ablation_all_cells_long.tsv"
    long_df.to_csv(long_path, sep="\t", index=False)
    print("saved:", long_path)
    long_df.to_csv(out_dir / "ablation_sk_n_sh_long.tsv", sep="\t", index=False)
    print("saved:", out_dir / "ablation_sk_n_sh_long.tsv")

    sub = long_df.loc[~long_df["residual"]].copy()
    for split_name in splits:
        for metric_name in metric_fn_map:
            msub = sub[(sub["split"] == split_name) & (sub["metric"] == metric_name)]
            wide = msub.pivot_table(index="model", columns="target_cell", values="value", aggfunc="first")
            wide = wide.reindex(columns=cell_types)
            wide_path = out_dir / f"ablation_{split_to_filename(split_name)}_{metric_name}_raw.tsv"
            wide.to_csv(wide_path, sep="\t")
            print(f"\n{'=' * 60}\nsplit={split_name!r} | metric={metric_name}\n{'=' * 60}")
            print("saved:", wide_path)
            print(wide)


if __name__ == "__main__":
    main()
