import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    build_models,
    cell_types,
    eval_model_names,
    mpra_path,
    results_dir,
    test_cell_types,
    train_cell_types,
)
from utils import build_cts_labels, build_masks, load_pred_dfs

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

output_dir = results_dir / "classification"
curve_dir = output_dir / "curves"
eval_split = "test"
tasks = ["CTS_high", "CTS_low"]
# saved ROC/PR curves are thinned to this many points; far above plot resolution
curve_points = 2000

models = build_models(eval_model_names)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict:
    n_eval = len(y_true)
    n_pos = int(y_true.sum())
    out = {
        "n_eval": n_eval,
        "n_pos": n_pos,
        "prevalence": n_pos / n_eval if n_eval else np.nan,
    }
    if n_pos == 0 or n_pos == n_eval:
        out.update(
            {
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "auroc": np.nan,
                "auprc": np.nan,
            }
        )
        return out

    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["auroc"] = float(roc_auc_score(y_true, score))
    out["auprc"] = float(average_precision_score(y_true, score))
    return out


def thin(n: int, k: int) -> np.ndarray:
    """Indices spanning 0..n-1 at about k points, always keeping both endpoints."""
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).astype(int))


def roc_pr_curves(y_true: np.ndarray, score: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thinned ROC and PR curves; empty frames when the task has no positives."""
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return pd.DataFrame(), pd.DataFrame()

    fpr, tpr, _ = roc_curve(y_true, score)
    idx = thin(len(fpr), curve_points)
    roc_df = pd.DataFrame({"fpr": fpr[idx], "tpr": tpr[idx]})
    roc_df["auroc"] = float(roc_auc_score(y_true, score))

    precision, recall, _ = precision_recall_curve(y_true, score)
    idx = thin(len(recall), curve_points)
    pr_df = pd.DataFrame({"recall": recall[idx], "precision": precision[idx]})
    pr_df["auprc"] = float(average_precision_score(y_true, score))
    pr_df["prevalence"] = n_pos / len(y_true)
    return roc_df, pr_df


def evaluate_model_on_cell(
    model_name: str,
    cell_type: str,
    eval_mask: np.ndarray,
    true_high: pd.Series,
    true_low: pd.Series,
    pred_high: pd.Series,
    pred_low: pd.Series,
    score_high: np.ndarray,
    score_low: np.ndarray,
) -> tuple[list[dict], dict]:
    rows = []
    curves = {}
    y_true_high = true_high.to_numpy()[eval_mask].astype(bool)
    y_pred_high = pred_high.to_numpy()[eval_mask].astype(bool)
    y_true_low = true_low.to_numpy()[eval_mask].astype(bool)
    y_pred_low = pred_low.to_numpy()[eval_mask].astype(bool)
    s_high = score_high[eval_mask]
    s_low = score_low[eval_mask]

    for task, y_true, y_pred, score in [
        ("CTS_high", y_true_high, y_pred_high, s_high),
        ("CTS_low", y_true_low, y_pred_low, s_low),
    ]:
        row = classification_metrics(y_true, y_pred, score)
        row.update({"model": model_name, "cell_type": cell_type, "task": task})
        rows.append(row)
        curves[task] = roc_pr_curves(y_true, score)
    return rows, curves


def save_by_cell_type(result_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> None:
    model_type_map = {name: model_type for name, _, model_type in models}
    metric_cols = ["precision", "recall", "f1", "auroc", "auprc"]

    for cell_type in test_cell_types:
        for task in tasks:
            sub = result_df[(result_df["cell_type"] == cell_type) & (result_df["task"] == task)]
            rows = []
            for model in model_order:
                hit = sub[sub["model"] == model]
                row = {"model": model, "model_type": model_type_map[model]}
                for m in metric_cols:
                    row[m] = hit.iloc[0][m] if not hit.empty else np.nan
                rows.append(row)
            wide_df = pd.DataFrame(rows)[["model", "model_type"] + metric_cols]
            wide_path = output_dir / f"{eval_split}_{cell_type}_{task}_by_model.csv"
            wide_df.to_csv(wide_path, index=False)
            print(f"[save] {wide_path}")
            print(wide_df)


def save_by_metric(result_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> None:
    """One wide (model x cell type) csv per (task, metric), same layout as analysis/07."""
    model_type_map = {name: model_type for name, _, model_type in models}
    metric_cols = ["precision", "recall", "f1", "auroc", "auprc"]

    for task in tasks:
        sub = result_df[result_df["task"] == task]
        for metric in metric_cols:
            wide_df = sub.pivot(index="model", columns="cell_type", values=metric)
            wide_df = wide_df.loc[model_order, cell_types]
            wide_df.insert(0, "model_type", [model_type_map[m] for m in model_order])
            path = output_dir / f"{eval_split}_{task}_{metric}.csv"
            wide_df.to_csv(path)
            print(f"[save] {path}")


def save_curves(curve_rows: dict, curve_dir: Path) -> None:
    """One csv per (cell type, task, curve kind), models stacked in a `model` column."""
    curve_dir.mkdir(parents=True, exist_ok=True)
    for (cell_type, task, kind), frames in curve_rows.items():
        curve_df = pd.concat(frames, ignore_index=True)
        path = curve_dir / f"{eval_split}_{cell_type}_{task}_{kind}.csv"
        curve_df.to_csv(path, index=False)
        print(f"[save] {path} {curve_df.shape}")


def main() -> None:
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] {mpra_path} {mpra_df.shape}")

    masks = build_masks(
        mpra_df,
        cell_types,
        train_cell_types=train_cell_types,
        test_cell_types=test_cell_types,
        verbose=False,
    )
    eval_mask = masks[eval_split]
    train_mean_true = mpra_df[train_cell_types].mean(axis=1)

    true_high_by_cell = {}
    true_low_by_cell = {}
    for cell_type in cell_types:
        gap = mpra_df[cell_type] - train_mean_true
        true_high, true_low, q99, q01 = build_cts_labels(gap)
        true_high_by_cell[cell_type] = true_high
        true_low_by_cell[cell_type] = true_low
        print(
            f"[label] {cell_type} true CTS_high={int(true_high.sum())} "
            f"CTS_low={int(true_low.sum())} q99={q99:.4f} q01={q01:.4f}"
        )

    pred_dfs = load_pred_dfs(
        models, cell_types, train_cell_types, test_cell_types, n_variants=len(mpra_df)
    )
    train_pred_cols = [f"{ct}_pred" for ct in train_cell_types]

    all_rows = []
    curve_rows = {}
    for model_name, pred_df in pred_dfs.items():
        train_mean_pred = pred_df[train_pred_cols].mean(axis=1)
        for cell_type in cell_types:
            gap_pred = pred_df[f"{cell_type}_pred"] - train_mean_pred
            # Threshold inside the measured subset so the predicted positive rate
            # matches the true one: the true gap is NaN wherever the cell type was
            # not assayed, so its percentiles already live in that subset.
            measured = mpra_df[cell_type].notna().to_numpy()
            _, _, q_hi, q_lo = build_cts_labels(gap_pred[measured])
            pred_high, pred_low = gap_pred > q_hi, gap_pred < q_lo
            score_high = gap_pred.to_numpy(dtype=float)
            score_low = (-gap_pred).to_numpy(dtype=float)
            cell_eval = eval_mask & measured
            rows, curves = evaluate_model_on_cell(
                model_name,
                cell_type,
                cell_eval,
                true_high_by_cell[cell_type],
                true_low_by_cell[cell_type],
                pred_high,
                pred_low,
                score_high,
                score_low,
            )
            all_rows.extend(rows)

            if cell_type not in test_cell_types:
                continue
            for task, (roc_df, pr_df) in curves.items():
                for kind, curve_df in [("roc", roc_df), ("pr", pr_df)]:
                    if curve_df.empty:
                        continue
                    curve_df.insert(0, "model", model_name)
                    curve_rows.setdefault((cell_type, task, kind), []).append(curve_df)

    result_df = pd.DataFrame(all_rows)
    meta_cols = ["n_eval", "n_pos", "prevalence"]
    metric_cols = ["precision", "recall", "f1", "auroc", "auprc"]
    result_df = result_df[["model", "cell_type", "task"] + metric_cols + meta_cols]

    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "all_models_classification.csv"
    result_df.to_csv(long_path, index=False)
    print(f"[save] {long_path}")

    model_order = list(pred_dfs.keys())
    save_by_cell_type(result_df, model_order, output_dir)
    save_by_metric(result_df, model_order, output_dir)
    save_curves(curve_rows, curve_dir)


if __name__ == "__main__":
    main()
