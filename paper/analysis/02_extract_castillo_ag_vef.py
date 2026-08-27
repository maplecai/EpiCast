"""Extract the AlphaGenome VEF matrix for the Castillo-Hair MPRA.

Uses the same shared helpers as the Gosai extraction, which fixes two problems in
the earlier version of this step:

- the TF head offset was two positions off, so the "CTCF" column read a
  neighbouring transcription factor;
- DNase was read from the 128-bp head and divided by the bin width, whereas the
  Gosai VEFs the model is trained on use the 1-bp head. Evaluating a model on a
  differently defined feature is a train/inference mismatch, so DNase is now read
  from the 1-bp head here as well.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import assays, castillo_all_cell_types, castillo_cell_types, castillo_dir, project_root
from utils import (
    ag_extract_vef,
    ag_head_starts,
    ag_log1p,
    ag_padded_metadata,
    ag_track_columns,
    ag_default_variant,
    ag_variants,
)

pred_path = castillo_dir / "castillo_mpra_ag_pred.h5"
ag_metadata_path = project_root / "data/AlphaGenome/metadata_padded.tsv"
track_columns_path = project_root / "data/AlphaGenome/castillo_ag_track_columns.csv"


def output_paths(variant):
    suffix = ag_variants[variant]["suffix"]
    return (
        castillo_dir / f"castillo_mpra_ag_vef_raw_{suffix.split('_')[-1]}.tsv",
        castillo_dir / f"castillo_mpra_ag_vef_{suffix}.tsv",
    )


def available_cell_types(metadata):
    """Castillo cell lines that AlphaGenome covers for all four assays."""
    real = metadata[
        (metadata["name"] != "Padding") & (metadata["genetically_modified"] == False)
    ]
    keep = []
    for cell_type in castillo_all_cell_types:
        ok = True
        for assay in assays:
            title = {"DNase": "DNase-seq", "CTCF": "TF ChIP-seq"}.get(assay, "Histone ChIP-seq")
            rows = real[(real["Assay title"] == title) & (real["biosample_name"] == cell_type)]
            if title == "Histone ChIP-seq":
                rows = rows[rows["histone_mark"] == assay]
            elif title == "TF ChIP-seq":
                rows = rows[rows["transcription_factor"] == assay]
            ok &= len(rows) == 1
        if ok:
            keep.append(cell_type)
    return keep


def main(variant):
    out_raw_path, out_log1p_path = output_paths(variant)
    print(f"[variant] {variant}: {ag_variants[variant]}")
    metadata = ag_padded_metadata(ag_metadata_path)
    print(f"[load] {ag_metadata_path} {metadata.shape}")

    starts = ag_head_starts(metadata, pred_path, variant)
    print(f"[head starts] {starts}")

    cell_types = available_cell_types(metadata)
    print(f"[cell types] {len(cell_types)} of {len(castillo_all_cell_types)} covered: {cell_types}")
    missing = set(castillo_cell_types) - set(cell_types)
    assert not missing, f"evaluated cell lines without full VEF coverage: {sorted(missing)}"

    track_columns = ag_track_columns(metadata, starts, cell_types, assays)
    print("[h5 column per (cell type, assay)]")
    print(track_columns)
    track_columns.to_csv(track_columns_path)
    print(f"[save] {track_columns_path}")

    vef_raw = ag_extract_vef(pred_path, track_columns, cell_types, assays, variant)
    print(f"[extract] raw {vef_raw.shape}")
    print(vef_raw.describe().T[["mean", "std", "min", "max"]])

    vef_raw.to_csv(out_raw_path, sep="\t", index=False)
    print(f"[save] {out_raw_path}")

    vef_log1p = ag_log1p(vef_raw, variant)
    vef_log1p.to_csv(out_log1p_path, sep="\t", index=False)
    print(f"[save] {out_log1p_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(ag_variants), default=ag_default_variant)
    main(parser.parse_args().variant)
