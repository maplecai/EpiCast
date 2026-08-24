"""VEF-only metrics for the AlphaGenome preprocessing variant "B2".

Four preprocessing variants of the AlphaGenome VEF matrix are of interest. All of
them share the corrected track indexing except A, which read a neighbouring
transcription factor as CTCF:

    A   CTCF column wrong; DNase from the 1-bp head; log1p(10x) with the 128-bp
        heads first divided by the bin width
    B   as A but with the CTCF column fixed
    B2  as B but DNase also read from the 128-bp head, so that all four assays
        share one read-out and one transform: log1p(10x/128)
    C   all four assays from the 128-bp heads with no rescaling: log1p(x)

B2 isolates the effect of the DNase read-out alone, because its transform is
identical to B's for every assay. This script builds the B2 matrix, fits the
VEF-only regressors on it and scores them on the held-out cell lines, then prints
A, B2 and C side by side. B's numbers come from an earlier run whose prediction
files were overwritten by C and are quoted in the summary rather than recomputed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
from epicast import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import epicast

from config import (
    assays,
    cell_types,
    data_dir,
    mpra_path,
    project_root,
    results_dir,
    test_cell_types,
    train_cell_types,
)
from utils import (
    ag_dataset_keys,
    ag_head_starts,
    ag_padded_metadata,
    ag_track_columns,
    build_cts_labels,
    build_masks,
    get_mask,
)

pred_path = project_root / "alphagenome_vef/gosai_ag_pred_760k_pad_0.h5"
ag_metadata_path = project_root / "alphagenome_vef/metadata_padded.tsv"
out_vef_path = data_dir / "gosai_mpra_760679_ag_vef_x10_log1p_128bp.tsv"
output_dir = results_dir / "vef_variant_b2"
bin_width = 128.0


def build_b2_vef():
    """All four assays from the 128-bp heads, divided by the bin width, log1p(10x)."""
    if out_vef_path.exists():
        print(f"[load] {out_vef_path}")
        return pd.read_csv(out_vef_path, sep="\t")

    metadata = ag_padded_metadata(ag_metadata_path)
    starts = ag_head_starts(metadata, pred_path)
    datasets = ag_dataset_keys(pred_path)
    columns = ag_track_columns(metadata, starts, cell_types, assays)
    print(f"[datasets] {datasets}")

    raw = {}
    with h5py.File(pred_path, "r") as f:
        for dataset in sorted(set(datasets.values())):
            block = f[dataset][:, :]
            for cell_type in cell_types:
                for assay in assays:
                    if datasets[assay] != dataset:
                        continue
                    raw[f"{cell_type}_{assay}"] = np.asarray(
                        block[:, int(columns.at[cell_type, assay])], dtype=np.float64
                    )
            del block

    order = [f"{c}_{a}" for c in cell_types for a in assays]
    vef = pd.DataFrame(np.log1p(pd.DataFrame(raw)[order] / bin_width * 10.0))
    vef.to_csv(out_vef_path, sep="\t", index=False)
    print(f"[save] {out_vef_path} {vef.shape}")
    return vef


def pooled(frame_x, frame_y, mask, cells):
    x = np.concatenate(
        [frame_x.loc[mask, [f"{c}_{a}" for a in assays]].to_numpy() for c in cells]
    )
    y = np.concatenate([frame_y.loc[mask, c].to_numpy() for c in cells])
    return x, y


def main():
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    vef = build_b2_vef()
    masks = build_masks(mpra_df, cell_types, train_cell_types, test_cell_types, verbose=False)

    x_total, _ = pooled(vef, mpra_df, masks["total"], cell_types)
    x_train, y_train = pooled(vef, mpra_df, masks["train"], train_cell_types)
    x_train, y_train = epicast.utils.remove_nan(x_train, y_train)
    print(f"[split] train X {x_train.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    preds = {}
    for name, model in [
        ("linear", LinearRegression()),
        ("xgb", XGBRegressor(random_state=0)),
        ("mlp", MLPRegressor(random_state=0)),
    ]:
        print(f"[train] {name}")
        model.fit(x_train, y_train)
        p = model.predict(x_total).reshape(len(cell_types), len(mpra_df)).T
        np.save(output_dir / f"{name}_pred.npy", p)
        preds[name] = pd.DataFrame(p, columns=cell_types)

    variants = {
        "A 原始": "../analysis_gosai_0722_backup/results/vef_only/ag_vef",
        "C 现在": "results/vef_only/ag_vef",
    }
    train_mean_true = mpra_df[train_cell_types].mean(axis=1)

    rows = []
    for name in preds:
        sources = {"B2 新增": preds[name]}
        for label, folder in variants.items():
            sources[label] = pd.DataFrame(
                np.load(Path(folder) / f"{name}_pred.npy"), columns=cell_types
            )
        for label, pred in sources.items():
            train_mean_pred = pred[train_cell_types].mean(axis=1)
            for cell_type in test_cell_types:
                evaluated = masks["test"] & mpra_df[cell_type].notna().to_numpy()
                cts = get_mask("test&all_cts_1_99", masks, cell_type=cell_type)
                gap_true = mpra_df[cell_type] - train_mean_true
                high, low, _, _ = build_cts_labels(gap_true)
                score = (pred[cell_type] - train_mean_pred).to_numpy()
                rows.append(
                    {
                        "model": name,
                        "variant": label,
                        "cell_type": cell_type,
                        "pcc_activity": metrics.pearson(
                            pred.loc[evaluated, cell_type], mpra_df.loc[evaluated, cell_type]
                        ),
                        "pcc_residual": metrics.pearson(
                            score[cts], gap_true.to_numpy()[cts]
                        ),
                        "auroc_high": roc_auc_score(high.to_numpy()[evaluated], score[evaluated]),
                        "auroc_low": roc_auc_score(low.to_numpy()[evaluated], -score[evaluated]),
                    }
                )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "variant_comparison.csv", index=False)
    print(f"[save] {output_dir / 'variant_comparison.csv'}")

    order = ["A 原始", "B2 新增", "C 现在"]
    for metric, label in [
        ("pcc_activity", "绝对活性 PCC"),
        ("pcc_residual", "残差活性 PCC (test∩CTS)"),
        ("auroc_high", "CTS-high AUROC"),
        ("auroc_low", "CTS-low AUROC"),
    ]:
        wide = result.pivot_table(
            index=["model", "cell_type"], columns="variant", values=metric
        )[order]
        print(f"\n=== {label} ===")
        print(wide.to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
