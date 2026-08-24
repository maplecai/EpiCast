import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
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
from utils import build_cts_labels, build_masks, load_pred_dfs

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

output_dir = results_dir / "retrieval"
curve_dir = output_dir / "curves"
eval_split = "test"
tasks = ["CTS_high", "CTS_low"]
k_list = [100, 1000, 10000]
# ties are broken at random (a seq-only model scores every test variant equally)
tiebreak_seed = 0
# saved curves are thinned to this many log-spaced k values
curve_points = 3000

models = build_models(eval_model_names)


def retrieval_curve(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Metrics at every cut-off k, ranking by descending score with random tiebreak."""
    n_eval = len(scores)
    n_pos = int(labels.sum())
    rng = np.random.default_rng(tiebreak_seed)
    order = np.lexsort((rng.random(n_eval), -np.asarray(scores, dtype=float)))
    cum_pos = np.cumsum(labels[order].astype(int))
    k = np.arange(1, n_eval + 1)
    precision = cum_pos / k
    prevalence = n_pos / n_eval
    nns = np.full(n_eval, np.nan)
    np.divide(k, cum_pos, out=nns, where=cum_pos > 0)
    return pd.DataFrame(
        {
            "k": k,
            "k_frac": k / n_eval,
            "k_pct": k / n_eval * 100.0,
            "precision": precision,
            "recall": cum_pos / n_pos if n_pos > 0 else np.full(n_eval, np.nan),
            "ef": precision / prevalence if prevalence > 0 else np.full(n_eval, np.nan),
            "nns": nns,
            "prevalence": prevalence,
            "n_pos": n_pos,
            "n_eval": n_eval,
        }
    )


def thin_log(curve_df: pd.DataFrame, n_points: int) -> pd.DataFrame:
    """Keep ~n_points rows on a log-spaced k grid, so small k stay densely sampled."""
    n = len(curve_df)
    if n <= n_points:
        return curve_df
    idx = np.unique(np.geomspace(1, n, n_points).astype(int) - 1)
    return curve_df.iloc[idx].reset_index(drop=True)


def retrieval_metrics(curve_df: pd.DataFrame, k_list: list[int]) -> dict:
    """p@k and ef@k read off the full curve, so summary and curve always agree."""
    n_eval = int(curve_df["n_eval"].iloc[0])
    n_pos = int(curve_df["n_pos"].iloc[0])
    out = {
        "n_eval": n_eval,
        "n_pos": n_pos,
        "prevalence": n_pos / n_eval if n_eval else np.nan,
    }
    for k in k_list:
        row = curve_df.iloc[min(k, n_eval) - 1]
        out[f"p@{k}"] = row["precision"] if n_pos else np.nan
        out[f"ef@{k}"] = row["ef"] if n_pos else np.nan
    return out


def save_by_cell_type(result_df: pd.DataFrame, model_order: list[str], output_dir: Path) -> None:
    model_type_map = {name: model_type for name, _, model_type in models}
    metric_cols = [f"p@{k}" for k in k_list] + [f"ef@{k}" for k in k_list]

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


def save_curves(curve_rows: dict, curve_dir: Path) -> None:
    """One csv per (cell type, task), models stacked in a `model` column."""
    curve_dir.mkdir(parents=True, exist_ok=True)
    for (cell_type, task), frames in curve_rows.items():
        curve_df = pd.concat(frames, ignore_index=True)
        path = curve_dir / f"{eval_split}_{cell_type}_{task}_curve.csv"
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
        true_high, true_low, _, _ = build_cts_labels(gap)
        true_high_by_cell[cell_type] = true_high
        true_low_by_cell[cell_type] = true_low
        print(
            f"[label] {cell_type} CTS_high={int(true_high.sum())} CTS_low={int(true_low.sum())}"
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
            score_high = gap_pred.to_numpy(dtype=float)
            score_low = -score_high
            cell_eval = eval_mask & mpra_df[cell_type].notna().to_numpy()

            for task, labels, scores in [
                ("CTS_high", true_high_by_cell[cell_type], score_high),
                ("CTS_low", true_low_by_cell[cell_type], score_low),
            ]:
                y = labels.to_numpy()[cell_eval].astype(bool)
                curve_df = retrieval_curve(scores[cell_eval], y)

                row = retrieval_metrics(curve_df, k_list)
                row.update({"model": model_name, "cell_type": cell_type, "task": task})
                all_rows.append(row)

                if cell_type not in test_cell_types:
                    continue
                thinned = thin_log(curve_df, curve_points)
                thinned.insert(0, "model", model_name)
                curve_rows.setdefault((cell_type, task), []).append(thinned)

    metric_cols = [f"p@{k}" for k in k_list] + [f"ef@{k}" for k in k_list]
    col_order = ["model", "cell_type", "task", "n_eval", "n_pos", "prevalence"] + metric_cols
    result_df = pd.DataFrame(all_rows)[col_order]

    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "all_models_retrieval.csv"
    result_df.to_csv(long_path, index=False)
    print(f"[save] {long_path}")

    model_order = list(pred_dfs.keys())
    save_by_cell_type(result_df, model_order, output_dir)
    save_curves(curve_rows, curve_dir)


if __name__ == "__main__":
    main()
