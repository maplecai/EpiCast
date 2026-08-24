"""Score every model under both cell-type-specificity definitions.

Two definitions of a cell-type-specific (CTS) element are compared:

- "mean"   the residual against the mean of the reference panel, which is the
           definition used in the manuscript;
- "mingap" the gap to the strongest (or weakest) single reference cell type, the
           definition of Gosai et al., so that a positive score guarantees the
           element really is the most active of the cell types compared.

Both the labels and the ranking score change together: a model is always scored
with the same quantity that defines the labels, otherwise the comparison would
penalise every model for a mismatch rather than for its predictions.

Note that the two definitions differ in how they treat the sequence-only model.
Its prediction in a held-out cell type is the mean over the training cell types,
so its mean-residual is identically zero and its ranking is degenerate, whereas
its min-gap score varies between elements and is therefore a usable baseline.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from epicast import metrics
from scipy.stats import ConstantInputWarning
from sklearn.metrics import average_precision_score, roc_auc_score

from config import (
    build_models,
    cell_types,
    eval_model_names,
    mpra_path,
    results_dir,
    test_cell_types,
    train_cell_types,
)
from utils import build_cts_labels, build_masks, build_mingap_labels, load_pred_dfs, safe_metric

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

output_dir = results_dir / "mingap"
k_frac = 0.01  # retrieval depth for EF@k, as a fraction of the evaluated pool


def specificity_scores(frame, cell_type, columns):
    """(high, low) specificity scores under both definitions for one cell type."""
    reference = [ct for ct in train_cell_types if ct != cell_type]
    target = frame[columns[cell_type]]
    mean_residual = target - frame[[columns[ct] for ct in train_cell_types]].mean(axis=1)
    reference_cols = [columns[ct] for ct in reference]
    return {
        "mean": (mean_residual, mean_residual),
        "mingap": (
            target - frame[reference_cols].max(axis=1),
            target - frame[reference_cols].min(axis=1),
        ),
    }


def enrichment_at_k(scores, labels, fraction):
    n = len(scores)
    k = max(1, int(round(n * fraction)))
    n_pos = int(labels.sum())
    if n_pos == 0:
        return np.nan
    rng = np.random.default_rng(0)
    order = np.lexsort((rng.random(n), -np.asarray(scores, dtype=float)))
    hits = int(labels[order][:k].sum())
    return (hits / k) / (n_pos / n)


def main():
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] {mpra_path} {mpra_df.shape}")
    masks = build_masks(
        mpra_df, cell_types, train_cell_types, test_cell_types, verbose=False
    )
    models = build_models(eval_model_names)
    pred_dfs = load_pred_dfs(
        models, cell_types, train_cell_types, test_cell_types, n_variants=len(mpra_df)
    )

    true_columns = {ct: ct for ct in cell_types}
    pred_columns = {ct: f"{ct}_pred" for ct in cell_types}

    rows = []
    for cell_type in test_cell_types:
        measured = mpra_df[cell_type].notna().to_numpy()
        evaluated = masks["test"] & measured

        truth = specificity_scores(mpra_df, cell_type, true_columns)
        labels = {
            "mean": build_cts_labels(truth["mean"][0]),
            "mingap": build_mingap_labels(mpra_df, cell_type, train_cell_types),
        }

        for definition in ("mean", "mingap"):
            high, low, _, _ = labels[definition]
            subset = evaluated & (high | low).to_numpy()
            for model_name, pred_df in pred_dfs.items():
                scores = specificity_scores(pred_df, cell_type, pred_columns)
                row = {
                    "definition": definition,
                    "cell_type": cell_type,
                    "model": model_name,
                    "n_subset": int(subset.sum()),
                    # absolute activity, restricted to the CTS subset of that definition
                    "pcc_activity": safe_metric(
                        metrics.pearson,
                        mpra_df.loc[subset, cell_type],
                        pred_df.loc[subset, f"{cell_type}_pred"],
                    ),
                    # the specificity score itself
                    "pcc_specificity": safe_metric(
                        metrics.pearson,
                        truth[definition][0][subset],
                        scores[definition][0][subset],
                    ),
                }
                for task, label, sign in (("high", high, 1.0), ("low", low, -1.0)):
                    y = label.to_numpy()[evaluated].astype(bool)
                    s = sign * scores[definition][0 if task == "high" else 1].to_numpy()[evaluated]
                    if y.sum() == 0 or y.all() or not np.isfinite(s).any() or np.std(s) == 0:
                        row[f"auroc_{task}"] = np.nan
                        row[f"auprc_{task}"] = np.nan
                        row[f"ef_{task}"] = np.nan
                    else:
                        row[f"auroc_{task}"] = roc_auc_score(y, s)
                        row[f"auprc_{task}"] = average_precision_score(y, s)
                        row[f"ef_{task}"] = enrichment_at_k(s, y, k_frac)
                rows.append(row)

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "mingap_vs_mean_all_models.csv"
    result.to_csv(long_path, index=False)
    print(f"[save] {long_path}")

    metric_labels = {
        "pcc_activity": "absolute activity PCC (within CTS subset)",
        "pcc_specificity": "specificity score PCC",
        "auroc_high": "CTS-high AUROC",
        "auprc_high": "CTS-high AUPRC",
        "ef_high": f"CTS-high EF@{k_frac:.0%}",
        "auroc_low": "CTS-low AUROC",
        "auprc_low": "CTS-low AUPRC",
        "ef_low": f"CTS-low EF@{k_frac:.0%}",
    }
    for metric, label in metric_labels.items():
        wide = result.pivot_table(
            index="model", columns=["cell_type", "definition"], values=metric
        ).reindex(eval_model_names)
        wide = wide.reindex(
            columns=pd.MultiIndex.from_product([test_cell_types, ["mean", "mingap"]])
        )
        path = output_dir / f"{metric}.csv"
        wide.to_csv(path)
        print(f"\n=== {label} ===")
        print(wide.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\n=== subset sizes ===")
    print(
        result.pivot_table(index="cell_type", columns="definition", values="n_subset")
        .astype(int)
        .to_string()
    )


if __name__ == "__main__":
    main()
