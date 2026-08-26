"""Two ranking scores against one set of Castillo CTS labels (side analysis, not fig5).

A Castillo CTS label is an absolute margin between measured activities: CTS-high means the
target cell beats every other evaluated cell by at least `castillo_cts_gap`. The score a
model is screened by should be the same quantity, and analysis/12 already does that, it
ranks each model by its own predicted gap. This script keeps that arm and adds the one the
manuscript Methods describes instead, the predicted residual against the mean of all seven
cell types, and scores both against identical labels, so the only thing that differs is the
ranking score:

- mingap    CTS-high  yhat_c - max_{j!=c} yhat_j     CTS-low  min_{j!=c} yhat_j - yhat_c
- residual  CTS-high  yhat_c - mean_j yhat_j         CTS-low  the negative of that

Both are oriented positive-is-more-specific, so one descending sort serves every metric.
Labels, metric definitions and screening depths are imported from analysis/12 rather than
reimplemented, which is what makes the mingap arm reproduce the fig5 numbers exactly.

The summary averages over cell types with more than five ground-truth positives, the same
filter fig5 uses, because AUROC/AUPRC/EF on two or three positives say nothing about the
score and would dominate a mean.

Reads results/predictions/castillo_*.tsv, writes results/castillo_ranking_score/. fig5 and
its inputs are untouched.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (
    castillo_cell_types,
    castillo_min_positives,
    castillo_model_names,
    castillo_screen_pcts,
    results_dir,
)

output_dir = results_dir / "castillo_ranking_score"
tasks = ["CTS-high", "CTS-low"]


def load_eval_module():
    """analysis/12 starts with a digit, so it cannot be imported by name."""
    path = Path(__file__).resolve().parent / "12_eval_castillo.py"
    spec = importlib.util.spec_from_file_location("castillo_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def residual_scores(frame, cell):
    """CTS-low and CTS-high residual scores, ordered and oriented like gap_scores."""
    residual = frame[cell].to_numpy() - frame[castillo_cell_types].mean(axis=1).to_numpy()
    return -residual, residual


def evaluate(ev, predictions, labels):
    rows = []
    for name, prediction in predictions.items():
        for cell in castillo_cell_types:
            gap_low, gap_high = ev.gap_scores(prediction, cell)
            residual_low, residual_high = residual_scores(prediction, cell)
            scores = {
                "mingap": {"CTS-high": gap_high, "CTS-low": gap_low},
                "residual": {"CTS-high": residual_high, "CTS-low": residual_low},
            }
            for score_name, by_task in scores.items():
                for task in tasks:
                    score = by_task[task]
                    task_labels = labels[cell][task]
                    n_pos = int(task_labels.sum())
                    prevalence = float(task_labels.mean())
                    scorable = 0 < n_pos < len(task_labels)
                    raw_auprc = ev.auprc(task_labels, score) if scorable else np.nan
                    for screen_pct in castillo_screen_pcts:
                        k = max(1, int(np.ceil(len(score) * screen_pct / 100)))
                        selected = np.argsort(-score, kind="stable")[:k]
                        hits = int(task_labels[selected].sum())
                        rows.append(
                            {
                                "model": name,
                                "score": score_name,
                                "cell_type": cell,
                                "task": task,
                                "screen_pct": screen_pct,
                                "n": len(score),
                                "n_pos": n_pos,
                                "k": k,
                                "hits": hits,
                                "prevalence": prevalence,
                                "auroc": ev.auroc(task_labels, score) if scorable else np.nan,
                                "auprc": raw_auprc,
                                "normalized_auprc": (raw_auprc - prevalence) / (1 - prevalence),
                                "ef": (hits / k) / prevalence if prevalence > 0 else np.nan,
                            }
                        )
    return pd.DataFrame(rows)


def summarize(table):
    """One row per (task, metric, model): both scores averaged over the kept cell types."""
    kept = table[table["n_pos"] >= castillo_min_positives]
    ranked = kept.drop_duplicates(["model", "score", "cell_type", "task"])

    blocks = [ranked.assign(metric=metric, value=ranked[metric]) for metric in ("auroc", "normalized_auprc")]
    for screen_pct in castillo_screen_pcts:
        depth = kept[kept["screen_pct"] == screen_pct]
        blocks.append(depth.assign(metric=f"ef@{screen_pct:g}%", value=depth["ef"]))
    long = pd.concat(blocks)

    summary = long.pivot_table(index=["task", "metric", "model"], columns="score", values="value")
    summary["delta"] = summary["mingap"] - summary["residual"]
    summary["n_cells"] = long[long["score"] == "mingap"].groupby(["task", "metric", "model"]).size()
    return summary.reset_index()


def report(table, summary):
    pd.set_option("display.width", 200)

    counts = table.drop_duplicates(["cell_type", "task"]).pivot(
        index="cell_type", columns="task", values="n_pos"
    )
    print("\n=== ground-truth positives per cell type ===")
    print(counts.loc[castillo_cell_types, tasks].to_string())
    print(f"(cell types with >= {castillo_min_positives} positives enter the summary)")

    for task in tasks:
        for metric in summary["metric"].unique():
            block = summary[(summary["task"] == task) & (summary["metric"] == metric)]
            print(f"\n=== {task}: {metric}, mean over {block['n_cells'].iloc[0]} cell types ===")
            print(
                block.set_index("model")[["mingap", "residual", "delta"]]
                .reindex(castillo_model_names)
                .round(4)
                .to_string()
            )


def main():
    ev = load_eval_module()
    output_dir.mkdir(parents=True, exist_ok=True)

    truth, predictions = ev.load_tables()
    labels, _ = ev.define_cts(truth)

    table = evaluate(ev, predictions, labels)
    summary = summarize(table)

    for df, name in [
        (table, "castillo_ranking_score_metrics.csv"),
        (summary, "castillo_ranking_score_summary.csv"),
    ]:
        path = output_dir / name
        df.to_csv(path, index=False)
        print(f"[save] {path} {df.shape}")

    report(table, summary)


if __name__ == "__main__":
    main()
