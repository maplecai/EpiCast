"""Normalize raw VEF matrices and report their correlation with MPRA activity.

AlphaGenome and Sei VEF are already normalized by their extraction scripts
(x10_log1p and logit respectively), so only their baseline correlation is
reported here. Enformer and Borzoi raw matrices are log1p-transformed and saved.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from epicast import metrics

from config import assays, cell_types, data_dir, mpra_raw_path

vef_raw_paths = {
    "alphagenome": data_dir / "gosai_mpra_760679_ag_vef_raw.tsv",
    "enformer": data_dir / "gosai_mpra_760679_enformer_vef_raw.tsv",
    "borzoi": data_dir / "gosai_mpra_760679_borzoi_vef_raw.tsv",
}
log1p_models = ["enformer", "borzoi"]


def compute_metric(mpra_df, vef_df, metric_fn, cell_types, assays):
    corr_mat = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    for cell_type in cell_types:
        for assay in assays:
            corr_mat.loc[cell_type, assay] = metric_fn(
                vef_df[f"{cell_type}_{assay}"], mpra_df[cell_type]
            )
    return corr_mat


def main():
    mpra_df = pd.read_csv(mpra_raw_path, sep="\t")
    print(f"[load] {mpra_raw_path} {mpra_df.shape}")

    for model_name, vef_path in vef_raw_paths.items():
        vef_raw_df = pd.read_csv(vef_path, sep="\t")
        print(f"[load] {vef_path} {vef_raw_df.shape}")
        print(f"[{model_name} raw] pearson")
        print(compute_metric(mpra_df, vef_raw_df, metrics.pearson, cell_types, assays))

        if model_name not in log1p_models:
            continue

        vef_log1p_df = np.log1p(vef_raw_df)
        out_path = vef_path.with_name(vef_path.name.replace("_raw.tsv", "_log1p.tsv"))
        vef_log1p_df.to_csv(out_path, sep="\t", index=False)
        print(f"[save] {out_path} {vef_log1p_df.shape}")
        print(f"[{model_name} log1p] pearson")
        print(compute_metric(mpra_df, vef_log1p_df, metrics.pearson, cell_types, assays))


if __name__ == "__main__":
    main()
