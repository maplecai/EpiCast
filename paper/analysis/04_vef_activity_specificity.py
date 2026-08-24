"""Compute VEF–activity specificity correlations (same as fig1ff/fig1fff, no CTS mask, no plot)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from epicast import metrics

from config import assays, cell_types, mpra_path, train_cell_types, vef_paths

mpra_df = pd.read_csv(mpra_path, sep="\t")
print(f"[load] {mpra_path} {mpra_df.shape}")

model_vef_paths = {name: vef_paths[name] for name in ["sei", "alphagenome"]}

train_mean_activity = mpra_df[train_cell_types].mean(axis=1)


def residual_activity(cell_type):
    return (mpra_df[cell_type] - train_mean_activity).to_numpy()


def residual_vef(vef_df, cell_type, assay):
    train_mean = vef_df[[f"{ct}_{assay}" for ct in train_cell_types]].mean(axis=1)
    return (vef_df[f"{cell_type}_{assay}"] - train_mean).to_numpy()


def compute_vef_specificity_corr(vef_df, mpra_df, cell_types, assay):
    corr = pd.DataFrame(index=[f"{c}_{assay}" for c in cell_types], columns=cell_types, dtype=float)
    for c1 in cell_types:
        for c2 in cell_types:
            x = vef_df[f"{c1}_{assay}"].to_numpy()
            y = mpra_df[c2].to_numpy()
            corr.loc[f"{c1}_{assay}", c2] = metrics.pearson(x, y)
    return corr


def compute_residual_corr(vef_df, cell_types, assay):
    corr = pd.DataFrame(index=[f"{c}_{assay}" for c in cell_types], columns=cell_types, dtype=float)
    for c1 in cell_types:
        x = residual_vef(vef_df, c1, assay)
        for c2 in cell_types:
            y = residual_activity(c2)
            corr.loc[f"{c1}_{assay}", c2] = metrics.pearson(x, y)
    return corr


for model_name, vef_path in model_vef_paths.items():
    vef_df = pd.read_csv(vef_path, sep="\t")
    print(f"[load] {vef_path} {vef_df.shape}")

    for assay in assays:
        corr = compute_vef_specificity_corr(vef_df, mpra_df, cell_types, assay)
        print(f"[{model_name} {assay} absolute]")
        print(corr)
        print()

        corr_resid = compute_residual_corr(vef_df, cell_types, assay)
        print(f"[{model_name} {assay} residual]")
        print(corr_resid)
        print()
