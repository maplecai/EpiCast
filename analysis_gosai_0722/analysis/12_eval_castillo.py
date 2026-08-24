"""Zero-shot evaluation on the independent Castillo MPRA (fig5).

Adapted from the final analysis written by C.Z.; the metric definitions are his
and are kept unchanged. Only the plotting was split out into plot/fig5, and the
paths and model registry now come from config.

Analysis choices, all deliberate and different from the Gosai evaluation:
* All 8,152 genomic and synthetic CREs are scored together, no train/test split:
  no model saw this dataset, so every sequence is held out.
* Measured and predicted activities are used raw. The Castillo activities were
  never z-scored, so MAE and RMSE are reported on the original scale and are only
  compared between models, never against the Gosai numbers.
* CTS-high: target - max(other six cell types) >= gap.
  CTS-low:  min(other six cell types) - target >= gap.
  An absolute margin, not a percentile tail. A positive CTS-high label therefore
  guarantees the element really is the most active of the seven cell types, which
  is what "selective" has to mean for design.
* Residual: target activity - mean over all seven cell types. This is the
  reference panel for the residual regression only; the CTS labels use the
  max/min gap above. The two quantities are not the same thing.
* Each model is ranked by the gap between its own predictions, so the score is
  oriented the same way as the label it is scored against.

Writes tidy tables that plot/fig5_castillo_metrics.py consumes directly: they are
already one row per (model, cell type, setting), so routing them through
analysis/15 would only reshape them twice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (
    castillo_cell_types,
    castillo_cts_gap,
    castillo_model_names,
    castillo_model_styles,
    castillo_screen_pcts,
    predictions_dir,
    results_dir,
)

output_dir = results_dir / "castillo"
regression_settings = ["All activity", "CTS-union activity", "CTS-union residual"]


def load_tables():
    """Measured activity plus one prediction frame per model, columns = cell types.

    The measured activity is read from the EpiCast table; every model table repeats
    the same measured columns in the same row order, which is asserted here rather
    than trusted.
    """
    frames = {}
    for name in castillo_model_names:
        path = predictions_dir / f"castillo_{name}.tsv"
        frames[name] = pd.read_csv(path, sep="\t")
        print(f"[load] {path.name} {frames[name].shape}")

    truth = frames["epicast_ag_vef"][castillo_cell_types].copy()
    pred_cols = [f"{cell}_pred" for cell in castillo_cell_types]
    predictions = {}
    for name, frame in frames.items():
        missing = [column for column in pred_cols if column not in frame]
        assert not missing, f"{name}: missing prediction columns {missing}"
        assert len(frame) == len(truth), f"{name}: {len(frame)} rows, expected {len(truth)}"
        predictions[name] = frame[pred_cols].set_axis(castillo_cell_types, axis=1)
    return truth, predictions


def gap_scores(frame, cell):
    """CTS-low and CTS-high gaps for one cell type, both oriented positive-is-more."""
    others = [other for other in castillo_cell_types if other != cell]
    low = frame[others].min(axis=1).to_numpy() - frame[cell].to_numpy()
    high = frame[cell].to_numpy() - frame[others].max(axis=1).to_numpy()
    return low, high


def regression_metrics(observed, predicted):
    error = predicted - observed
    observed_ranks = pd.Series(observed).rank(method="average").to_numpy()
    predicted_ranks = pd.Series(predicted).rank(method="average").to_numpy()
    return {
        "pcc": float(np.corrcoef(observed, predicted)[0, 1]),
        "scc": float(np.corrcoef(observed_ranks, predicted_ranks)[0, 1]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def auroc(labels, scores):
    """Rank-based AUROC, so no sklearn dependency and ties handled by mid-rank."""
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def auprc(labels, scores):
    ordered = labels[np.argsort(-scores, kind="stable")].astype(int)
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / ordered.sum())


def define_cts(truth):
    """Per-cell CTS-high/low labels and the union mask across all cell types."""
    labels = {}
    union = np.zeros(len(truth), dtype=bool)
    for cell in castillo_cell_types:
        low_score, high_score = gap_scores(truth, cell)
        labels[cell] = {
            "CTS-low": low_score >= castillo_cts_gap,
            "CTS-high": high_score >= castillo_cts_gap,
        }
        union |= labels[cell]["CTS-low"] | labels[cell]["CTS-high"]
    return labels, union


def eval_regression(truth, predictions, union):
    true_residual = truth.sub(truth.mean(axis=1), axis=0)
    rows = []
    for name, prediction in predictions.items():
        pred_residual = prediction.sub(prediction.mean(axis=1), axis=0)
        for cell in castillo_cell_types:
            settings = {
                "All activity": (truth[cell], prediction[cell], np.ones(len(truth), dtype=bool)),
                "CTS-union activity": (truth[cell], prediction[cell], union),
                "CTS-union residual": (true_residual[cell], pred_residual[cell], union),
            }
            for setting, (observed, predicted, subset) in settings.items():
                rows.append(
                    {
                        "model": name,
                        "model_label": castillo_model_styles[name][0],
                        "cell_type": cell,
                        "setting": setting,
                        "n": int(subset.sum()),
                        **regression_metrics(
                            observed.to_numpy()[subset], predicted.to_numpy()[subset]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def eval_classification(predictions, labels):
    rows = []
    for name, prediction in predictions.items():
        for cell in castillo_cell_types:
            pred_low, pred_high = gap_scores(prediction, cell)
            scores_by_task = {"CTS-high": pred_high, "CTS-low": pred_low}
            for task, scores in scores_by_task.items():
                task_labels = labels[cell][task]
                n_pos = int(task_labels.sum())
                prevalence = float(task_labels.mean())
                scorable = 0 < n_pos < len(task_labels)
                raw_auprc = auprc(task_labels, scores) if scorable else np.nan
                for screen_pct in castillo_screen_pcts:
                    k = max(1, int(np.ceil(len(scores) * screen_pct / 100)))
                    selected = np.argsort(-scores, kind="stable")[:k]
                    hits = int(task_labels[selected].sum())
                    rows.append(
                        {
                            "model": name,
                            "model_label": castillo_model_styles[name][0],
                            "cell_type": cell,
                            "task": task,
                            "screen_pct": screen_pct,
                            "n": len(scores),
                            "n_pos": n_pos,
                            "k": k,
                            "hits": hits,
                            "prevalence": prevalence,
                            "auroc": auroc(task_labels, scores) if scorable else np.nan,
                            "auprc": raw_auprc,
                            # AUPRC baseline is the prevalence, which differs a lot
                            # between cell types here, so rescale to 0 = random,
                            # 1 = perfect before comparing cells in one boxplot
                            "normalized_auprc": (raw_auprc - prevalence) / (1 - prevalence),
                            "ef": (hits / k) / prevalence if prevalence > 0 else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def count_cts(truth, labels, union):
    return pd.DataFrame(
        [
            {
                "cell_type": cell,
                "cts_high_n": int(labels[cell]["CTS-high"].sum()),
                "cts_low_n": int(labels[cell]["CTS-low"].sum()),
                "cts_union_n": int(union.sum()),
                "total_n": len(truth),
            }
            for cell in castillo_cell_types
        ]
    )


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    truth, predictions = load_tables()
    labels, union = define_cts(truth)
    print(f"[cts] gap >= {castillo_cts_gap:g}: union {union.sum()} / {len(truth)} sequences")

    regression = eval_regression(truth, predictions, union)
    classification = eval_classification(predictions, labels)
    counts = count_cts(truth, labels, union)

    for df, name in [
        (regression, "castillo_regression_metrics.csv"),
        (classification, "castillo_classification_metrics.csv"),
        (counts, "castillo_cts_counts.csv"),
    ]:
        path = output_dir / name
        df.to_csv(path, index=False)
        print(f"[save] {path} {df.shape}")


if __name__ == "__main__":
    main()
