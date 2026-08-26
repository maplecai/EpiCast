"""Export the Gosai metric tables that plot/ reads: one TSV per evaluation.

Table names describe their content rather than a figure number, so renumbering a
manuscript panel never invalidates a file name. The mapping to the current
manuscript panels is:

    activity_test / activity_cts        -> fig 2B / 2C
    residual_cts                        -> fig 3A
    cts_high / cts_low                  -> fig 3B / 3C
    retrieval_cts_high / _cts_low       -> fig 3D / 3E
    residual_test, *_roc, *_pr          -> no manuscript panel, kept for reference

Castillo (fig5) is not exported here. analysis/12_eval_castillo.py writes its
metric tables directly, because its CTS definition and its metric set differ from
the Gosai ones and reshaping them through this script would only obscure that.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import (
    cell_types,
    figure_metrics_dir,
    figure_model_names,
    model_styles,
    results_dir,
    test_cell_types,
)

output_dir = figure_metrics_dir
float_format = "%.5f"
regression_metrics = ["pearson", "spearman", "mae", "rmse"]


def model_label(name: str) -> str:
    return model_styles[name][0]


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, float_format=float_format)
    print(f"[save] {path.name} {df.shape}")


def keep_figure_models(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model"].isin(figure_model_names)].copy()


def order_models(df: pd.DataFrame) -> pd.DataFrame:
    df["model"] = pd.Categorical(df["model"], figure_model_names, ordered=True)
    return df.sort_values(["model", "cell_type"]).reset_index(drop=True)


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df.insert(1, "model_label", df["model"].map(model_label))
    return df


def pivot_regression(long_df: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = keep_figure_models(long_df)
    sub = sub[
        (sub["split"] == split)
        & (sub["metric"].isin(regression_metrics))
        & (sub["cell_type"].isin(cell_types))
    ]
    n_eval = (
        sub[sub["metric"] == "pearson"][["model", "cell_type", "n_eval"]]
        .drop_duplicates()
        .assign(n_eval=lambda d: d["n_eval"].astype(int))
    )
    wide = sub.pivot(index=["model", "cell_type"], columns="metric", values="value")
    wide = wide.reset_index()
    wide.columns.name = None
    wide = wide.merge(n_eval, on=["model", "cell_type"])
    wide["cell_type"] = pd.Categorical(wide["cell_type"], cell_types, ordered=True)
    wide = add_label(order_models(wide))
    return wide[["model", "model_label", "cell_type", "n_eval", *regression_metrics]]


def export_classification(task: str, stem: str) -> None:
    src = pd.read_csv(results_dir / "classification/all_models_classification.csv")
    sub = keep_figure_models(src)
    sub = sub[(sub["task"] == task) & (sub["cell_type"].isin(cell_types))]
    sub["cell_type"] = pd.Categorical(sub["cell_type"], cell_types, ordered=True)
    sub = add_label(order_models(sub))
    out = sub[
        ["model", "model_label", "cell_type", "n_eval", "n_pos", "prevalence", "auroc", "auprc"]
    ]
    save(out, output_dir / f"{stem}.tsv")


def export_roc_pr(task: str, stem: str) -> None:
    for curve, xy in [("roc", ("fpr", "tpr")), ("pr", ("recall", "precision"))]:
        frames = []
        for cell_type in test_cell_types:
            path = results_dir / "classification/curves" / f"test_{cell_type}_{task}_{curve}.csv"
            df = keep_figure_models(pd.read_csv(path))
            df.insert(0, "cell_type", cell_type)
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        out["cell_type"] = pd.Categorical(out["cell_type"], test_cell_types, ordered=True)
        out = add_label(order_models(out))
        x, y = xy
        summary = ["auroc"] if curve == "roc" else ["auprc", "prevalence"]
        save(
            out[["cell_type", "model", "model_label", x, y, *summary]],
            output_dir / f"{stem}_{curve}.tsv",
        )


def export_retrieval(task: str, stem: str) -> None:
    frames = []
    for cell_type in test_cell_types:
        path = results_dir / "retrieval/curves" / f"test_{cell_type}_{task}_curve.csv"
        df = keep_figure_models(pd.read_csv(path))
        df.insert(0, "cell_type", cell_type)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["cell_type"] = pd.Categorical(out["cell_type"], test_cell_types, ordered=True)
    out = add_label(order_models(out))
    cols = [
        "cell_type",
        "model",
        "model_label",
        "k",
        "k_frac",
        "k_pct",
        "precision",
        "recall",
        "ef",
        "nns",
        "prevalence",
        "n_pos",
        "n_eval",
    ]
    save(out[cols], output_dir / f"{stem}.tsv")


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    activity = pd.read_csv(results_dir / "correlation/all_models_correlation.csv")
    residual = pd.read_csv(results_dir / "correlation_residual/all_models_correlation.csv")

    save(pivot_regression(activity, "test"), output_dir / "activity_test.tsv")
    save(pivot_regression(activity, "test&all_cts_1_99"), output_dir / "activity_cts.tsv")
    save(pivot_regression(residual, "test"), output_dir / "residual_test.tsv")
    save(pivot_regression(residual, "test&all_cts_1_99"), output_dir / "residual_cts.tsv")

    export_classification("CTS_high", "cts_high")
    export_classification("CTS_low", "cts_low")
    export_roc_pr("CTS_high", "cts_high")
    export_roc_pr("CTS_low", "cts_low")
    export_retrieval("CTS_high", "retrieval_cts_high")
    export_retrieval("CTS_low", "retrieval_cts_low")


if __name__ == "__main__":
    main()
