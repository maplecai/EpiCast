"""Zero-shot Castillo metrics for the Sei-based models, alongside the AlphaGenome ones.

analysis/12_eval_castillo.py is the pipeline's Castillo entry point and only scores the
four models that go into fig5, all of them AlphaGenome-based. Nothing had ever scored
EpiCast-Sei or the Sei VEF-only models here, because no Castillo dataset config carried
a Sei VEF matrix. analysis/02_extract_castillo_sei_vef.py now builds one, so this script
scores them with the metric definitions of analysis/12, imported rather than reimplemented.

The Sei models are scored under both z-score references produced by the extraction
script (self / assay, see its docstring). The AlphaGenome models and DHS64 are read from
results/predictions/ and reported unchanged as the reference column.

Reads only, writes results/castillo_sei/. Does not touch fig5.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from config import (
    assays,
    castillo_cell_types,
    castillo_dir,
    castillo_model_names,
    epicast_sei_run,
    predictions_dir,
    results_dir,
)
from utils import apply_dnase_linear

output_dir = results_dir / "castillo_sei"
# the column order inside every castillo prediction npy, set by the dataset config
npy_cell_types = ["GM12878", "SK-N-SH", "WERI-Rb-1", "HepG2", "K562", "MCF-7", "HeLa-S3"]
sei_vef_paths = {
    "self": castillo_dir / "castillo_mpra_sei_vef_logit_zscore_self.tsv",
    "assay": castillo_dir / "castillo_mpra_sei_vef_logit_zscore_assay.tsv",
}
sei_epicast_preds = {
    "self": epicast_sei_run.parent / "castillo_preds_pad_N_sei_self.npy",
    "assay": epicast_sei_run.parent / "castillo_preds_pad_N_sei_assay.npy",
}
sei_mlp_path = results_dir / "vef_only/sei_vef/mlp.joblib"
sei_dnase_linear_path = results_dir / "vef_only/sei_dnase/linear.joblib"


def load_eval_module():
    """analysis/12 starts with a digit, so it cannot be imported by name."""
    path = Path(__file__).resolve().parent / "12_eval_castillo.py"
    spec = importlib.util.spec_from_file_location("castillo_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference():
    """Measured activity plus the four AlphaGenome-side predictions already exported."""
    frames = {}
    for name in castillo_model_names:
        path = predictions_dir / f"castillo_{name}.tsv"
        frames[name] = pd.read_csv(path, sep="\t")
        print(f"[load] {path.name} {frames[name].shape}")

    truth = frames["epicast_ag_vef"][castillo_cell_types].copy()
    pred_cols = [f"{cell}_pred" for cell in castillo_cell_types]
    predictions = {
        name: frame[pred_cols].set_axis(castillo_cell_types, axis=1)
        for name, frame in frames.items()
    }
    return truth, predictions


def predict_vef_only(model, vef_df):
    columns = [f"{cell}_{assay}" for cell in castillo_cell_types for assay in assays]
    x_flat = vef_df[columns].to_numpy().reshape(len(vef_df) * len(castillo_cell_types), len(assays))
    return pd.DataFrame(
        model.predict(x_flat).reshape(len(vef_df), len(castillo_cell_types)),
        columns=castillo_cell_types,
    )


def build_sei_predictions():
    mlp = joblib.load(sei_mlp_path)
    dnase_linear = joblib.load(sei_dnase_linear_path)
    print(f"[load] {sei_mlp_path}")
    print(f"[load] {sei_dnase_linear_path}")

    predictions = {}
    for reference, vef_path in sei_vef_paths.items():
        vef_df = pd.read_csv(vef_path, sep="\t")
        print(f"[load] {vef_path.name} {vef_df.shape}")

        pred_path = sei_epicast_preds[reference]
        epicast = pd.DataFrame(np.load(pred_path), columns=npy_cell_types)
        print(f"[load] {pred_path.name} {epicast.shape}")

        predictions[f"epicast_sei_vef ({reference})"] = epicast[castillo_cell_types]
        predictions[f"mlp_sei_vef ({reference})"] = predict_vef_only(mlp, vef_df)
        predictions[f"linear_sei_dnase ({reference})"] = apply_dnase_linear(
            dnase_linear, vef_df, castillo_cell_types
        )
    return predictions


def eval_regression(ev, truth, predictions, union):
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
                        "cell_type": cell,
                        "setting": setting,
                        "n": int(subset.sum()),
                        **ev.regression_metrics(
                            observed.to_numpy()[subset], predicted.to_numpy()[subset]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def eval_classification(ev, predictions, labels):
    rows = []
    for name, prediction in predictions.items():
        for cell in castillo_cell_types:
            pred_low, pred_high = ev.gap_scores(prediction, cell)
            for task, scores in {"CTS-high": pred_high, "CTS-low": pred_low}.items():
                task_labels = labels[cell][task]
                n_pos = int(task_labels.sum())
                prevalence = float(task_labels.mean())
                scorable = 0 < n_pos < len(task_labels)
                raw_auprc = ev.auprc(task_labels, scores) if scorable else np.nan
                for screen_pct in ev.castillo_screen_pcts:
                    k = max(1, int(np.ceil(len(scores) * screen_pct / 100)))
                    selected = np.argsort(-scores, kind="stable")[:k]
                    hits = int(task_labels[selected].sum())
                    rows.append(
                        {
                            "model": name,
                            "cell_type": cell,
                            "task": task,
                            "screen_pct": screen_pct,
                            "n_pos": n_pos,
                            "auroc": ev.auroc(task_labels, scores) if scorable else np.nan,
                            "normalized_auprc": (raw_auprc - prevalence) / (1 - prevalence),
                            "ef": (hits / k) / prevalence if prevalence > 0 else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def report(regression, classification):
    pd.set_option("display.width", 200)
    for setting in ["All activity", "CTS-union activity", "CTS-union residual"]:
        block = regression[regression["setting"] == setting]
        print(f"\n=== {setting}: Pearson r ===")
        table = block.pivot(index="model", columns="cell_type", values="pcc")[castillo_cell_types]
        table["mean"] = table.mean(axis=1)
        print(table.round(4).to_string())

    for task in ["CTS-high", "CTS-low"]:
        block = classification[classification["task"] == task]
        print(f"\n=== {task}: AUROC ===")
        table = block.drop_duplicates(["model", "cell_type"]).pivot(
            index="model", columns="cell_type", values="auroc"
        )[castillo_cell_types]
        table["mean"] = table.mean(axis=1)
        print(table.round(4).to_string())

        print(f"\n=== {task}: enrichment factor @5% ===")
        block5 = block[block["screen_pct"] == 5.0]
        table = block5.pivot(index="model", columns="cell_type", values="ef")[castillo_cell_types]
        table["mean"] = table.mean(axis=1)
        print(table.round(3).to_string())


def main():
    ev = load_eval_module()
    output_dir.mkdir(parents=True, exist_ok=True)

    truth, predictions = load_reference()
    predictions.update(build_sei_predictions())

    labels, union = ev.define_cts(truth)
    print(f"\n[cts] gap >= {ev.castillo_cts_gap:g}: union {union.sum()} / {len(truth)} sequences")

    regression = eval_regression(ev, truth, predictions, union)
    classification = eval_classification(ev, predictions, labels)

    for df, name in [(regression, "regression_metrics.csv"), (classification, "classification_metrics.csv")]:
        path = output_dir / f"castillo_sei_{name}"
        df.to_csv(path, index=False)
        print(f"[save] {path} {df.shape}")

    report(regression, classification)


if __name__ == "__main__":
    main()
