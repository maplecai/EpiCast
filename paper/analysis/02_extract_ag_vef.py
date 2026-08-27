"""Extract the AlphaGenome VEF matrix for the Gosai MPRA from raw h5 predictions.

The preprocessing variant is selected with --variant and defaults to the published
one; see utils.ag_variants for what the variants are and why they are defined
there rather than here.

Track indexing is handled by the shared helpers in utils (see the comment there):
column numbers are derived from the padded metadata and checked against the h5
widths, because the earlier hard-coded offsets were two positions off for the TF
head and made the "CTCF" column read a neighbouring factor.

1. Build (cell_type, assay) -> h5 column from the padded metadata.
2. Read out all four assays from their 128-bp resolution heads.
3. Save the raw matrix and its log1p transform, used by the downstream scripts.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from epicast import metrics

from config import assays, cell_types, data_dir, mpra_path, project_root
from utils import (
    ag_extract_vef,
    ag_head_starts,
    ag_log1p,
    ag_padded_metadata,
    ag_track_columns,
    ag_default_variant,
    ag_variants,
)

pred_path = project_root / "data/AlphaGenome/gosai_ag_pred_760k_pad_0.h5"
ag_metadata_path = project_root / "data/AlphaGenome/metadata_padded.tsv"
track_columns_path = project_root / "data/AlphaGenome/gosai_ag_track_columns.csv"


def output_paths(variant):
    suffix = ag_variants[variant]["suffix"]
    return (
        data_dir / f"gosai_mpra_760679_ag_vef_raw_{suffix.split('_')[-1]}.tsv",
        data_dir / f"gosai_mpra_760679_ag_vef_{suffix}.tsv",
    )


def compute_pearson(mpra_df, vef_df):
    corr = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    for cell_type in cell_types:
        for assay in assays:
            corr.loc[cell_type, assay] = metrics.pearson(
                vef_df[f"{cell_type}_{assay}"], mpra_df[cell_type]
            )
    return corr


def main(variant):
    out_raw_path, out_log1p_path = output_paths(variant)
    print(f"[variant] {variant}: {ag_variants[variant]}")
    metadata = ag_padded_metadata(ag_metadata_path)
    print(f"[load] {ag_metadata_path} {metadata.shape}")

    starts = ag_head_starts(metadata, pred_path, variant)
    print(f"[head starts] {starts}")

    track_columns = ag_track_columns(metadata, starts, cell_types, assays)
    print("[h5 column per (cell type, assay)]")
    print(track_columns)
    track_columns.to_csv(track_columns_path)
    print(f"[save] {track_columns_path}")

    vef_raw = ag_extract_vef(pred_path, track_columns, cell_types, assays, variant)
    print(f"[extract] raw {vef_raw.shape}")

    vef_raw.to_csv(out_raw_path, sep="\t", index=False)
    print(f"[save] {out_raw_path}")

    vef_log1p = ag_log1p(vef_raw, variant)
    vef_log1p.to_csv(out_log1p_path, sep="\t", index=False)
    print(f"[save] {out_log1p_path}")

    if mpra_path.exists():
        mpra_df = pd.read_csv(mpra_path, sep="\t")
        print(f"[load] {mpra_path} {mpra_df.shape}")
        print("[pearson] log1p")
        print(compute_pearson(mpra_df, vef_log1p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(ag_variants), default=ag_default_variant)
    main(parser.parse_args().variant)
