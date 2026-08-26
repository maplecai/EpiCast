"""Extract the Sei VEF matrix for the Castillo MPRA.

Mirrors analysis/02_extract_sei_vef.py so that EpiCast-Sei sees the same kind of
input it was trained on: keep the Sei tracks with AUROC > 0.95, average the logit
of the selected tracks, then z-score each column. The Gosai matrix the model was
trained on (gosai_mpra_760679_sei_vef_logit.tsv) is the z-scored form, not the raw
logit, so skipping the z-score here would put the VEF on a different scale.

Cell-type names are Sei's, matched by hand against the seven Castillo cell types
that the AlphaGenome VEF also covers. SK-N-SH merges the neuroblastoma and the
RA-neuron entries, exactly as the Gosai extraction does.

Two normalisations are written because the z-score reference is a real choice and
the two are not interchangeable:

    self   each column standardised on Castillo itself. Same procedure as was
           applied to Gosai, uniform across all seven cell types, but it uses
           statistics of the evaluation set.
    assay  each column standardised with the mean and sd of that assay pooled
           over the five Gosai cell types. Carries the training scale across to
           cell types Gosai never had, at the cost of assuming one scale per
           assay.

Sei has no H3K27ac track for WERI-Rb-1 at any AUROC, so that column has no data.
It is written as NaN in the raw file and filled with 0 (the standardised column
mean) in both normalised files, which means every WERI-Rb-1 number downstream
rests on three assays instead of four.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from epicast import metrics
from epicast.utils import logit

from config import assays, castillo_cell_types, castillo_dir, castillo_mpra_path, data_dir, project_root

pred_path = castillo_dir / "sei_pred.npy"
tracks_info_path = project_root / "data/Sei/Sei_tracks_info.csv"
gosai_logit_raw_path = data_dir / "gosai_mpra_760679_sei_vef_logit_raw.tsv"
out_raw_path = castillo_dir / "castillo_mpra_sei_vef_logit_raw.tsv"
out_self_path = castillo_dir / "castillo_mpra_sei_vef_logit_zscore_self.tsv"
out_assay_path = castillo_dir / "castillo_mpra_sei_vef_logit_zscore_assay.tsv"
auroc_threshold = 0.95
logit_eps = 1e-6

# hand-matched Sei cell-type names; a list means the tracks of the entries are pooled
sei_names = {
    "K562": ["K562_Leukemia_Cell"],
    "HepG2": ["HepG2_Hepatocellular_Carcinoma"],
    "SK-N-SH": ["SK-N-SH_Neuroblastoma_cell_Brain", "SK-N-SH_RA_Neuron_Brain"],
    "GM12878": ["GM12878_B_Lymphocyte_Blood"],
    "WERI-Rb-1": ["WERI-Rb-1_Eye"],
    "MCF-7": ["MCF-7_Epithelium_Mammary_Gland"],
    "HeLa-S3": ["HeLa-S3_Epithelium_Cervix"],
}


def build_track_table():
    df = pd.read_csv(tracks_info_path)
    df = df[df["AUROC"] > auroc_threshold]
    pivot = df.pivot_table(values="index", index="cell_type", columns="assay", aggfunc=list)
    pivot = pivot.map(lambda x: x if isinstance(x, list) else [])

    track = pd.DataFrame(index=castillo_cell_types, columns=assays, dtype=object)
    for cell_type, names in sei_names.items():
        for assay in assays:
            track.loc[cell_type, assay] = sum((pivot.loc[name][assay] for name in names), [])
    return track


def extract_vef_logit(track):
    pred_array = np.load(pred_path)
    print(f"[load] {pred_path} {pred_array.shape}")

    # 1656 of the stored float32 probabilities are exactly 1.0, which sends logit to
    # inf and poisons the whole column mean. The Gosai h5 tops out at 0.99999106 and
    # never needed this; clipping affects at most 14 rows of any single track.
    n_saturated = int((pred_array >= 1.0).sum())
    pred_array = np.clip(pred_array, logit_eps, 1.0 - logit_eps)
    print(f"  clipped {n_saturated} saturated probabilities to 1 - {logit_eps:g}")

    vef = {}
    for cell_type in castillo_cell_types:
        for assay in assays:
            indices = track.loc[cell_type, assay]
            col = f"{cell_type}_{assay}"
            if indices:
                vef[col] = logit(pred_array[:, indices]).mean(1)
            else:
                vef[col] = np.full(pred_array.shape[0], np.nan)
    return pd.DataFrame(vef)


def zscore_self(vef):
    return ((vef - vef.mean()) / vef.std(ddof=1)).fillna(0.0)


def zscore_by_assay(vef):
    gosai = pd.read_csv(gosai_logit_raw_path, sep="\t")
    print(f"[load] {gosai_logit_raw_path} {gosai.shape}")
    out = {}
    for assay in assays:
        pooled = gosai[[c for c in gosai.columns if c.endswith(f"_{assay}")]].to_numpy().ravel()
        mean, std = pooled.mean(), pooled.std(ddof=1)
        print(f"  {assay:8s} gosai pooled mean {mean:8.4f}  sd {std:7.4f}")
        for cell_type in castillo_cell_types:
            col = f"{cell_type}_{assay}"
            out[col] = (vef[col] - mean) / std
    return pd.DataFrame(out)[vef.columns].fillna(0.0)


def compute_pearson(mpra_df, vef_df):
    corr = pd.DataFrame(index=castillo_cell_types, columns=assays, dtype=float)
    for cell_type in castillo_cell_types:
        for assay in assays:
            corr.loc[cell_type, assay] = metrics.pearson(
                vef_df[f"{cell_type}_{assay}"], mpra_df[cell_type]
            )
    return corr


def main():
    track = build_track_table()
    print("[track n]")
    print(track.map(len))

    vef_raw = extract_vef_logit(track)
    vef_raw.to_csv(out_raw_path, sep="\t", index=False, float_format="%.5f")
    print(f"[save] {out_raw_path} {vef_raw.shape}")

    for path, vef in [
        (out_self_path, zscore_self(vef_raw)),
        (out_assay_path, zscore_by_assay(vef_raw)),
    ]:
        vef.to_csv(path, sep="\t", index=False, float_format="%.5f")
        print(f"[save] {path} {vef.shape}")
        mpra_df = pd.read_csv(castillo_mpra_path, sep="\t")
        print(f"[pearson vs measured activity] {path.name}")
        print(compute_pearson(mpra_df, vef))


if __name__ == "__main__":
    main()
