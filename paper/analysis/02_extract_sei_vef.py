"""Extract Sei VEF matrix from raw Sei predictions.

Compared with predict_CRE_activity/1.1_Gosai_Sei_pred_VEF.ipynb:
- Same core transform: mean(logit(tracks)) over selected Sei tracks.
- Same cell-type name mapping and AUROC>0.95 track filter.
- SK-N-SH merges Neuroblastoma + RA Neuron tracks.
- Current downstream file gosai_mpra_760679_sei_vef_logit.tsv is the
  column-wise z-scored version of that logit VEF (notebook cell 22-23),
  not the raw logit values.

Writes to new filenames (logit_raw / logit_zscore), not the existing
gosai_mpra_760679_sei_vef_logit.tsv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
from epicast import metrics
from epicast.utils import logit

from config import assays, cell_types, data_dir, mpra_path, project_root

pred_path = project_root / "predict_CRE_activity/outputs/Gosai_MPRA_Sei_pred.h5"
tracks_info_path = project_root / "data/Sei/Sei_tracks_info.csv"
metadata_pivot_path = project_root / "data/Sei/metadata_pivot_vef.csv"
out_logit_path = data_dir / "gosai_mpra_760679_sei_vef_logit_raw.tsv"
out_zscore_path = data_dir / "gosai_mpra_760679_sei_vef_logit_zscore.tsv"
cell_standard_names = {
    "K562": "K562_Leukemia_Cell",
    "HepG2": "HepG2_Hepatocellular_Carcinoma",
    "SK-N-SH": "SK-N-SH_Neuroblastoma_cell_Brain",
    "HCT116": "HCT-116_Colorectal_cancer_cell_line",
    "A549": "A549",
}
auroc_threshold = 0.95


def build_metadata_pivot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["AUROC"] > auroc_threshold]
    pivot = df.pivot_table(
        values="index",
        index="cell_type",
        columns="assay",
        aggfunc=list,  # pyrefly: ignore
    )
    return pivot.map(lambda x: x if isinstance(x, list) else [])


def build_track_table(pivot: pd.DataFrame) -> pd.DataFrame:
    track = pd.DataFrame(index=cell_types, columns=assays)
    for cell_type in cell_types:
        track.loc[cell_type] = pivot.loc[cell_standard_names[cell_type]][assays]
    track.loc["SK-N-SH"] = (
        pivot.loc["SK-N-SH_Neuroblastoma_cell_Brain"][assays]
        + pivot.loc["SK-N-SH_RA_Neuron_Brain"][assays]
    )
    return track


def extract_vef_logit(pred_path: Path, track: pd.DataFrame) -> pd.DataFrame:
    vef = {}
    with h5py.File(pred_path, "r") as f:
        print(f"[load] {pred_path}")
        print(f"  data {f['data'].shape}")
        pred_array = f["data"][:]

    for cell_type in cell_types:
        for assay in assays:
            indices = track.loc[cell_type, assay]
            col = f"{cell_type}_{assay}"
            if isinstance(indices, list) and len(indices) > 0:
                vef[col] = logit(pred_array[:, indices]).mean(1)
            else:
                vef[col] = np.full(pred_array.shape[0], np.nan)
    return pd.DataFrame(vef)


def column_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=1)


def compute_pearson(mpra_df: pd.DataFrame, vef_df: pd.DataFrame) -> pd.DataFrame:
    corr = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    for cell_type in cell_types:
        for assay in assays:
            corr.loc[cell_type, assay] = metrics.pearson(
                vef_df[f"{cell_type}_{assay}"],
                mpra_df[cell_type],
            )
    return corr


def main() -> None:
    pivot = build_metadata_pivot(tracks_info_path)
    pivot.to_csv(metadata_pivot_path)
    print(f"[save] {metadata_pivot_path} {pivot.shape}")

    track = build_track_table(pivot)
    print("[track n]")
    print(track.map(len))

    vef_logit = extract_vef_logit(pred_path, track)
    print(f"[extract] logit {vef_logit.shape}")
    print(vef_logit.describe())

    vef_logit.to_csv(out_logit_path, sep="\t", index=False)
    print(f"[save] {out_logit_path}")

    vef_zscore = column_zscore(vef_logit)
    vef_zscore.to_csv(out_zscore_path, sep="\t", index=False)
    print(f"[save] {out_zscore_path}")

    if mpra_path.exists():
        mpra_df = pd.read_csv(mpra_path, sep="\t")
        print(f"[load] {mpra_path} {mpra_df.shape}")
        print("[pearson] logit")
        print(compute_pearson(mpra_df, vef_logit))
        print("[pearson] logit_zscore")
        print(compute_pearson(mpra_df, vef_zscore))


if __name__ == "__main__":
    main()
