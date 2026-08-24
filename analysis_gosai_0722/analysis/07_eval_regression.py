import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from epicast import metrics
from scipy.stats import ConstantInputWarning

from config import (
    build_models,
    cell_types,
    eval_model_names,
    mpra_path,
    results_dir,
    test_cell_types,
    train_cell_types,
)
from utils import (
    build_masks,
    get_mask,
    load_pred_dfs,
    load_residual_eval_dfs,
    load_true_df,
    safe_metric,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

splits = ["test", "test&cts_1_99", "test&all_cts_1_99", "test&cts_5_95", "test&all_cts_5_95"]
metric_fn_map = {
    "pearson": metrics.pearson,
    "spearman": metrics.spearman,
    "mae": metrics.mae,
    "rmse": metrics.rmse,
}
activity_output_dir = results_dir / "correlation"
residual_output_dir = results_dir / "correlation_residual"


models = build_models(eval_model_names)


def evaluate_model(model_name, pred_df, true_df, masks, eval_cell_types, splits, metric_fn_map):
    rows = []
    for split in splits:
        for metric_name, metric_fn in metric_fn_map.items():
            split_values = []
            for cell_type in eval_cell_types:
                eval_mask = get_mask(split, masks, cell_type=cell_type)
                x = true_df.loc[eval_mask, f"{cell_type}_true"]
                y = pred_df.loc[eval_mask, f"{cell_type}_pred"]
                value = safe_metric(metric_fn, x, y)
                rows.append(
                    {
                        "model": model_name,
                        "split": split,
                        "cell_type": cell_type,
                        "metric": metric_name,
                        "n_eval": int(eval_mask.sum()),
                        "value": value,
                    }
                )
                split_values.append(value)
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "cell_type": "test_mean",
                    "metric": metric_name,
                    "n_eval": np.nan,
                    "value": np.nanmean(split_values),
                }
            )
    return pd.DataFrame(rows)


def save_results(combined_df, pred_dfs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "all_models_correlation.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"[save] {combined_path}")

    model_order = list(pred_dfs.keys())
    model_type_map = {name: model_type for name, _, model_type in models}
    for split in splits:
        for metric_name in metric_fn_map:
            sub = combined_df[
                (combined_df["split"] == split)
                & (combined_df["metric"] == metric_name)
                & (combined_df["cell_type"].isin(cell_types))
            ].copy()
            wide_df = sub.pivot(index="model", columns="cell_type", values="value")
            wide_df = wide_df.reindex(index=model_order, columns=cell_types)
            wide_df.columns.name = None
            wide_df.insert(0, "model_type", wide_df.index.map(model_type_map))
            wide_df = wide_df.reset_index()
            wide_path = output_dir / f"{split}_{metric_name}.csv"
            wide_df.to_csv(wide_path, index=False)
            print(f"[save] {wide_path}")
            print(wide_df)


def main():
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] {mpra_path} {mpra_df.shape}")

    masks = build_masks(
        mpra_df,
        cell_types,
        train_cell_types=train_cell_types,
        test_cell_types=test_cell_types,
    )
    true_df = load_true_df(mpra_df, cell_types)
    pred_dfs = load_pred_dfs(
        models, cell_types, train_cell_types, test_cell_types, n_variants=len(mpra_df)
    )

    print("\n=== activity ===")
    activity_results = []
    for model_name, pred_df in pred_dfs.items():
        activity_results.append(
            evaluate_model(model_name, pred_df, true_df, masks, cell_types, splits, metric_fn_map)
        )
    activity_df = pd.concat(activity_results, axis=0, ignore_index=True)
    save_results(activity_df, pred_dfs, activity_output_dir)

    resid_true_df, resid_pred_dfs = load_residual_eval_dfs(
        mpra_df, pred_dfs, cell_types, train_cell_types
    )

    print("\n=== residual ===")
    residual_results = []
    for model_name, pred_df in resid_pred_dfs.items():
        residual_results.append(
            evaluate_model(
                model_name, pred_df, resid_true_df, masks, cell_types, splits, metric_fn_map
            )
        )
    residual_df = pd.concat(residual_results, axis=0, ignore_index=True)
    save_results(residual_df, resid_pred_dfs, residual_output_dir)


if __name__ == "__main__":
    main()
