#!/usr/bin/env python3
"""Reproduce the final Castillo CTS analysis and publication-style PDF.

Final analysis choices
----------------------
* All 8,152 genomic and synthetic CREs are analyzed together.
* Raw measured and predicted activities are used without normalization.
* CTS-high: target - max(other six cell types) >= 1.
* CTS-low:  min(other six cell types) - target >= 1.
* CTS union: sequences labeled CTS-high or CTS-low in any cell type.
* Residual: target activity - mean(activity across all seven cell types).
* CTS-low cell types with <= 1 positive sequence are omitted from plots.
* Classification panels show AUROC, normalized AUPRC, 2% EF, and 5% EF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CELLS = ["K562", "HepG2", "SK-N-SH", "GM12878", "WERI-Rb-1", "MCF-7", "HeLa-S3"]
MODELS = ["DHS64", "DNase-AG", "AG-VEF-only", "EpiCast-AG"]
MODEL_FILES = {
    "DHS64": "castillo_dhs64.tsv",
    "DNase-AG": "castillo_ag_dnase.tsv",
    "AG-VEF-only": "castillo_vef_only.tsv",
    "EpiCast-AG": "castillo_epicast_ag_vef.tsv",
}
MODEL_COLORS = {
    "DHS64": "#DEDEDE",
    "DNase-AG": "#84A87C",
    "AG-VEF-only": "#6EC893",
    "EpiCast-AG": "#4A57CF",
}
CELL_COLORS = {
    "K562": "#D73027",
    "HepG2": "#F28E2B",
    "SK-N-SH": "#E6AB02",
    "GM12878": "#E83E8C",
    "WERI-Rb-1": "#8E44AD",
    "MCF-7": "#00A6D6",
    "HeLa-S3": "#6B3E26",
}
REGRESSION_SETTINGS = ["All activity", "CTS-union activity", "CTS-union residual"]
GAP = 1.0
SCREEN_PERCENTAGES = (2.0, 5.0)
BOX_EDGE = "#666666"
BOX_LINEWIDTH = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/cubicstone/Downloads/share/data"),
        help="Directory containing the four Castillo TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for the PDF and metric tables.",
    )
    return parser.parse_args()


def validate_and_load(data_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames = {model: pd.read_csv(data_dir / filename, sep="\t") for model, filename in MODEL_FILES.items()}
    truth = frames["EpiCast-AG"][CELLS].copy()
    n = len(truth)
    for model, frame in frames.items():
        required = [f"{cell}_pred" for cell in CELLS]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{model} is missing prediction columns: {missing}")
        if len(frame) != n:
            raise ValueError(f"{model} has {len(frame):,} rows; expected {n:,}")
    predictions = {
        model: frame[[f"{cell}_pred" for cell in CELLS]].set_axis(CELLS, axis=1)
        for model, frame in frames.items()
    }
    return truth, predictions


def extreme_scores(frame: pd.DataFrame, cell: str) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-oriented CTS-low and CTS-high scores, respectively."""
    others = [other for other in CELLS if other != cell]
    low = frame[others].min(axis=1).to_numpy() - frame[cell].to_numpy()
    high = frame[cell].to_numpy() - frame[others].max(axis=1).to_numpy()
    return low, high


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    observed_ranks = pd.Series(observed).rank(method="average").to_numpy()
    predicted_ranks = pd.Series(predicted).rank(method="average").to_numpy()
    return {
        "PCC": float(np.corrcoef(observed, predicted)[0, 1]),
        "SCC": float(np.corrcoef(observed_ranks, predicted_ranks)[0, 1]),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
    }


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    ordered = labels[np.argsort(-scores, kind="stable")].astype(int)
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / ordered.sum())


def define_cts(truth: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    labels: dict[str, dict[str, np.ndarray]] = {}
    union = np.zeros(len(truth), dtype=bool)
    for cell in CELLS:
        low_score, high_score = extreme_scores(truth, cell)
        labels[cell] = {"CTS-low": low_score >= GAP, "CTS-high": high_score >= GAP}
        union |= labels[cell]["CTS-low"] | labels[cell]["CTS-high"]
    return labels, union


def calculate_metrics(
    truth: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels, union = define_cts(truth)
    true_residual = truth.sub(truth.mean(axis=1), axis=0)
    regression_rows: list[dict] = []
    classification_rows: list[dict] = []

    for model, prediction in predictions.items():
        predicted_residual = prediction.sub(prediction.mean(axis=1), axis=0)
        for cell in CELLS:
            regression_inputs = {
                "All activity": (truth[cell], prediction[cell], np.ones(len(truth), dtype=bool)),
                "CTS-union activity": (truth[cell], prediction[cell], union),
                "CTS-union residual": (true_residual[cell], predicted_residual[cell], union),
            }
            for setting, (observed, predicted, subset) in regression_inputs.items():
                regression_rows.append(
                    {
                        "model": model,
                        "cell_type": cell,
                        "setting": setting,
                        "n": int(subset.sum()),
                        **regression_metrics(observed.to_numpy()[subset], predicted.to_numpy()[subset]),
                    }
                )

            predicted_low, predicted_high = extreme_scores(prediction, cell)
            score_by_task = {"CTS-high": predicted_high, "CTS-low": predicted_low}
            for task in ("CTS-high", "CTS-low"):
                task_labels = labels[cell][task]
                scores = score_by_task[task]
                prevalence = float(task_labels.mean())
                for screen_pct in SCREEN_PERCENTAGES:
                    k = max(1, int(np.ceil(len(scores) * screen_pct / 100)))
                    selected = np.argsort(-scores, kind="stable")[:k]
                    hits = int(task_labels[selected].sum())
                    raw_auprc = auprc(task_labels, scores) if 0 < task_labels.sum() < len(task_labels) else np.nan
                    classification_rows.append(
                        {
                            "model": model,
                            "cell_type": cell,
                            "task": task,
                            "screen_pct": screen_pct,
                            "n": len(scores),
                            "n_positive": int(task_labels.sum()),
                            "k": k,
                            "hits": hits,
                            "prevalence": prevalence,
                            "AUROC": auroc(task_labels, scores) if 0 < task_labels.sum() < len(task_labels) else np.nan,
                            "AUPRC": raw_auprc,
                            "normalized_AUPRC": (raw_auprc - prevalence) / (1 - prevalence),
                            "EF": (hits / k) / prevalence if prevalence > 0 else np.nan,
                        }
                    )

    counts = pd.DataFrame(
        [
            {
                "cell_type": cell,
                "CTS_high_n": int(labels[cell]["CTS-high"].sum()),
                "CTS_low_n": int(labels[cell]["CTS-low"].sum()),
                "CTS_union_n": int(union.sum()),
                "total_n": len(truth),
            }
            for cell in CELLS
        ]
    )
    return pd.DataFrame(regression_rows), pd.DataFrame(classification_rows), counts


def padded_limits(values: pd.Series, *, zero_floor: bool = False, ceiling: float | None = None) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 0.1)
    lower = 0.0 if zero_floor else min(0.0, low - 0.08 * span)
    upper = high + 0.10 * span
    if ceiling is not None:
        upper = min(ceiling, upper)
    return lower, upper


def figure_scales(regression: pd.DataFrame, classification: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        "PCC": padded_limits(regression["PCC"], ceiling=1.0),
        "SCC": padded_limits(regression["SCC"], ceiling=1.0),
        "MAE": padded_limits(regression["MAE"], zero_floor=True),
        "RMSE": padded_limits(regression["RMSE"], zero_floor=True),
        "AUROC": (0.0, 1.0),
        "normalized_AUPRC": padded_limits(classification["normalized_AUPRC"], ceiling=1.0),
        "EF2": padded_limits(classification.loc[classification["screen_pct"].eq(2), "EF"], zero_floor=True),
        "EF5": padded_limits(classification.loc[classification["screen_pct"].eq(5), "EF"], zero_floor=True),
    }


def draw_boxplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylim: tuple[float, float],
    *,
    units: list[str],
    baseline: float | None = None,
) -> None:
    arrays = [
        data[data["model"].eq(model)].set_index("cell_type").reindex(units)[metric].to_numpy(float)
        for model in MODELS
    ]
    positions = np.arange(len(MODELS))
    boxplot = ax.boxplot(arrays, positions=positions, widths=0.56, patch_artist=True, showfliers=False)
    for index, model in enumerate(MODELS):
        box = boxplot["boxes"][index]
        box.set(facecolor=MODEL_COLORS[model], edgecolor=BOX_EDGE, alpha=0.42, linewidth=BOX_LINEWIDTH)
        boxplot["medians"][index].set(color=BOX_EDGE, linewidth=BOX_LINEWIDTH)
        for component in ("whiskers", "caps"):
            for line in boxplot[component][2 * index : 2 * index + 2]:
                line.set(color=BOX_EDGE, linewidth=BOX_LINEWIDTH)
    for cell_index, cell in enumerate(units):
        values = np.array([array[cell_index] for array in arrays])
        valid = np.isfinite(values)
        ax.scatter(positions[valid], values[valid], s=21, color=CELL_COLORS[cell],
                   edgecolor="white", linewidth=0.3, zorder=4)
    if baseline is not None:
        ax.axhline(baseline, color="#777777", linestyle="--", linewidth=0.8)
    ax.set(ylim=ylim, xticks=positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="y", labelrotation=90, labelsize=7)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(
    regression: pd.DataFrame,
    classification: pd.DataFrame,
    counts: pd.DataFrame,
    output_path: Path,
) -> None:
    scales = figure_scales(regression, classification)
    fig, axes = plt.subplots(4, 5, figsize=(12, 12))
    regression_metrics_order = ["PCC", "SCC", "MAE", "RMSE"]
    classification_rows = [
        ("AUROC", "AUROC", 5.0),
        ("normalized_AUPRC", "Normalized AUPRC", 5.0),
        ("EF", "2% EF", 2.0),
        ("EF", "5% EF", 5.0),
    ]

    for row, metric in enumerate(regression_metrics_order):
        for column, setting in enumerate(REGRESSION_SETTINGS):
            subset = regression[regression["setting"].eq(setting)]
            draw_boxplot(axes[row, column], subset, metric, scales[metric], units=CELLS,
                         baseline=0 if metric in {"PCC", "SCC"} else None)
            axes[row, column].set_ylabel(metric)
            if row == 0:
                n = int(subset["n"].iloc[0])
                axes[row, column].set_title(f"{setting.replace('CTS-union', 'CTS union')}\n(n={n:,} sequences)", fontsize=9)

        class_metric, class_label, screen_pct = classification_rows[row]
        for offset, task in enumerate(("CTS-high", "CTS-low")):
            column = 3 + offset
            subset = classification[
                classification["task"].eq(task) & classification["screen_pct"].eq(screen_pct)
            ]
            positive_counts = subset[subset["model"].eq(MODELS[0])].set_index("cell_type")["n_positive"]
            units = CELLS if task == "CTS-high" else [cell for cell in CELLS if positive_counts[cell] > 1]
            baseline = 0.5 if class_metric == "AUROC" else (0.0 if class_metric == "normalized_AUPRC" else 1.0)
            scale_key = f"EF{int(screen_pct)}" if class_metric == "EF" else class_metric
            draw_boxplot(axes[row, column], subset, class_metric, scales[scale_key], units=units, baseline=baseline)
            axes[row, column].set_ylabel(class_label)
            if row == 0:
                axes[row, column].set_title(
                    f"{task}\n(n+={int(positive_counts[units].sum()):,}; {len(units)}/7 cells)", fontsize=9
                )

    for axis in axes[-1]:
        axis.set_xticklabels(MODELS, rotation=38, ha="right", fontsize=7.5)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color=CELL_COLORS[cell],
                   markeredgecolor="white", label=cell)
        for cell in CELLS
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.85, 0.5), frameon=False, title="Cell types")
    union_n = int(counts["CTS_union_n"].iloc[0])
    total_n = int(counts["total_n"].iloc[0])
    fig.suptitle(
        f"Castillo All sequences: cell-wise boxplots; CTS gap >= {GAP:g}; screen = 2% and 5%\n"
        f"CTS union = {union_n:,}/{total_n:,} sequences",
        x=0.43, y=0.985, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 0.84, 0.94), h_pad=1.9, w_pad=1.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    truth, predictions = validate_and_load(args.data_dir)
    regression, classification, counts = calculate_metrics(truth, predictions)

    regression.to_csv(args.output_dir / "castillo_final_regression_metrics.csv", index=False)
    classification.to_csv(args.output_dir / "castillo_final_classification_metrics.csv", index=False)
    counts.to_csv(args.output_dir / "castillo_final_cts_counts.csv", index=False)
    make_figure(
        regression,
        classification,
        counts,
        args.output_dir / "castillo_final_gap1_combined_metrics.pdf",
    )


if __name__ == "__main__":
    main()
