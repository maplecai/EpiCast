"""Performance evidence for the choice of the four VEF assays.

The paper picks DNase, H3K4me3, H3K27ac and CTCF. Availability alone is a weak
justification, so this script adds the performance axis, using only data already on
disk: every AlphaGenome track is in the prediction h5, so any assay's VEF is a slice
away and no sequence-to-function model has to be re-run.

Three things are reported.

1. The availability funnel. A VEF exists for a cell type only if the assay has a
   track there, so "comparable across cell types" is a hard filter, not a
   convenience. The funnel counts how many of AlphaGenome's assays survive each
   requirement, which turns the availability argument into a number.

2. A univariate ranking of every assay that can be scored at all. The comparison pool
   is deliberately as wide as the data allows: an assay is included for a cell type
   whenever it has a track in that cell type and in the three training cell types,
   which is the minimum needed to form a residual. That admits 17 assays on HCT116 and
   12 on A549, rather than only the 10 that happen to cover all five cell types.

3. VEF-only subset comparison, to answer "would a different four do better". Subsets
   are fitted with the recipe of analysis/05 and scored on the two held-out cell types.
   Only assays covering all five cell types can enter a subset, since a model needs the
   same feature set everywhere.

Both the ranking and the subset scores are given on absolute and on residual activity.
Absolute activity is mostly set by how strong a sequence is in every cell type, so it
separates assays poorly; the residual is the quantity the paper's claim is about.

Note on reading the per-cell-type residuals: for a training cell type the residual
still contains 2/3 of its own activity, because that cell is inside the reference mean.
Only HCT116 and A549 give a residual that is a pure across-cell difference, so the
held-out columns are the ones to judge by.

Track indexing reuses utils.ag_head_starts, which derives each head's first column from
the padded metadata and checks it against the h5 widths. Head starts are a property of
the head, not of the assay, so the four verified starts also place every other histone
mark and transcription factor. Values use variant b, the published preprocessing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

import epicast
from epicast import metrics

from config import (
    assays,
    castillo_cell_types,
    cell_types,
    mpra_path,
    project_root,
    results_dir,
    test_cell_types,
    train_cell_types,
)
from utils import ag_bin_width, ag_head_starts, ag_padded_metadata, ag_variants, build_basic_masks, safe_metric

pred_path = project_root / "data/AlphaGenome/gosai_ag_pred_760k_pad_0.h5"
ag_metadata_path = project_root / "data/AlphaGenome/metadata_padded.tsv"
output_dir = results_dir / "vef_assay_selection"
variant = "b"
n_random_subsets = 20
random_seed = 0

head_dataset = {"dnase": "dnase_1", "histone": "chip_histone", "tf": "chip_tf"}
# ag_head_starts is keyed by assay; these three cover all three heads
head_probe_assay = {"dnase": "DNase", "histone": "H3K4me3", "tf": "CTCF"}


def named_tracks(metadata):
    """Real, unmodified tracks labelled with the assay and the head they live in."""
    real = metadata[
        (metadata["name"] != "Padding") & (metadata["genetically_modified"] == False)
    ].copy()

    def assay_of(row):
        title = row["Assay title"]
        if title == "DNase-seq":
            return "DNase", "dnase"
        if title == "Histone ChIP-seq":
            return row["histone_mark"], "histone"
        if title == "TF ChIP-seq":
            return row["transcription_factor"], "tf"
        return None, None

    named = real.apply(assay_of, axis=1, result_type="expand")
    real["assay"], real["head"] = named[0], named[1]
    return real[real["assay"].notna()]


def assays_covering(tracks, required):
    """Assays with a track in every one of `required` cell types."""
    subset = tracks[tracks["biosample_name"].isin(required)]
    counts = subset.groupby("assay")["biosample_name"].nunique()
    return set(counts[counts == len(required)].index)


def availability_funnel(tracks):
    stages = [
        # ATAC lives in its own head that no variant reads and is redundant with DNase,
        # so the pool starts from the three heads the VEF pipeline actually uses
        ("assay types in the DNase/histone/TF heads", set(tracks["assay"])),
        (
            "with a track in at least one Gosai cell type",
            set(tracks[tracks["biosample_name"].isin(cell_types)]["assay"]),
        ),
        ("in all 3 training cell types (residual definable)", assays_covering(tracks, train_cell_types)),
        ("in all 5 Gosai cell types (can enter a model)", assays_covering(tracks, cell_types)),
        (
            "and in all 7 Castillo cell types (usable in the paper)",
            assays_covering(tracks, cell_types) & assays_covering(tracks, castillo_cell_types),
        ),
    ]
    print("=== availability funnel ===")
    for label, members in stages:
        print(f"  {len(members):5d}  {label}")
    print(f"\n  usable on both datasets: {sorted(stages[-1][1])}")
    print(f"  the paper uses         : {sorted(assays)}")

    funnel = pd.DataFrame(
        [{"stage": i, "requirement": label, "n_assays": len(members)} for i, (label, members) in enumerate(stages)]
    )
    funnel.to_csv(output_dir / "availability_funnel.csv", index=False)
    return stages[-1][1]


def candidate_pool(tracks):
    """Assays scorable at all, with their coverage and the column of each cell type.

    Membership requires the three training cell types, the minimum for a residual.
    Columns are recorded for whichever of the five cell types the assay has, so an
    assay missing A549 is still ranked on HCT116 instead of being dropped outright.
    """
    keep = assays_covering(tracks, train_cell_types)
    rows = []
    for (assay, head), group in tracks.groupby(["assay", "head"]):
        if assay not in keep:
            continue
        present = group[group["biosample_name"].isin(cell_types)]
        assert present["biosample_name"].is_unique, f"{assay}: duplicate track in one cell type"
        rows.append(
            {
                "assay": assay,
                "head": head,
                "coverage": int(group["biosample_name"].nunique()),
                "cells": sorted(present["biosample_name"]),
                "columns": {
                    row["biosample_name"]: int(row["padded_index"]) for _, row in present.iterrows()
                },
            }
        )
    return pd.DataFrame(rows).sort_values("assay").reset_index(drop=True)


def extract_vef(pool, starts):
    """log1p(10x) VEF for every candidate assay and cell type it covers, variant b."""
    spec = ag_variants[variant]
    frames = {}
    for head, group in pool.groupby("head"):
        dataset = head_dataset[head]
        start = starts[head_probe_assay[head]]
        with h5py.File(pred_path, "r") as f:
            block = f[dataset][:, :]
        print(f"[read] {dataset} {block.shape}")
        for _, row in group.iterrows():
            for cell_type, padded_index in row["columns"].items():
                values = np.asarray(block[:, padded_index - start], dtype=np.float64)
                if spec["per_bp"] and dataset != "dnase_1":
                    values = values / ag_bin_width
                frames[f"{cell_type}_{row['assay']}"] = np.log1p(values * spec["multiplier"])
        del block
    return pd.DataFrame(frames)


def residual_of(frame, column_of, target_cells):
    """Value minus the mean over the three training cell types, the bundle's reference."""
    train_mean = pd.concat([frame[column_of(ct)] for ct in train_cell_types], axis=1).mean(axis=1)
    return pd.DataFrame({ct: frame[column_of(ct)] - train_mean for ct in target_cells})


def univariate(vef, mpra_df, masks, pool):
    activity_residual = residual_of(mpra_df, lambda ct: ct, cell_types)
    rows = []
    for _, row in pool.iterrows():
        assay, cells = row["assay"], row["cells"]
        vef_residual = residual_of(vef, lambda ct: f"{ct}_{assay}", cells)
        for cell_type in cells:
            evaluated = masks["test"] & mpra_df[cell_type].notna().to_numpy()
            rows.append(
                {
                    "assay": assay,
                    "coverage": row["coverage"],
                    "cell_type": cell_type,
                    "absolute_r": safe_metric(
                        metrics.pearson,
                        mpra_df.loc[evaluated, cell_type],
                        vef.loc[evaluated, f"{cell_type}_{assay}"],
                    ),
                    "residual_r": safe_metric(
                        metrics.pearson,
                        activity_residual.loc[evaluated, cell_type],
                        vef_residual.loc[evaluated, cell_type],
                    ),
                }
            )
    return pd.DataFrame(rows)


def get_x_y(vef, mpra_df, mask, fit_cell_types, feature_assays):
    """analysis/05's sample construction: one row per (CRE, cell type), no cell-type code."""
    x, y = [], []
    for cell_type in fit_cell_types:
        columns = [f"{cell_type}_{assay}" for assay in feature_assays]
        x.append(vef.loc[mask, columns].to_numpy())
        y.append(mpra_df.loc[mask, cell_type].to_numpy())
    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)


def score_subset(name, feature_assays, vef, mpra_df, masks):
    x_train, y_train = get_x_y(vef, mpra_df, masks["train"], train_cell_types, feature_assays)
    x_train, y_train = epicast.utils.remove_nan(x_train, y_train)
    x_total, _ = get_x_y(vef, mpra_df, masks["total"], cell_types, feature_assays)

    model = LinearRegression().fit(x_train, y_train)
    pred = pd.DataFrame(
        model.predict(x_total).reshape(len(cell_types), len(mpra_df)).T, columns=cell_types
    )
    pred_residual = pred.sub(pred[train_cell_types].mean(axis=1), axis=0)
    activity_residual = residual_of(mpra_df, lambda ct: ct, cell_types)

    rows = []
    for cell_type in test_cell_types:
        evaluated = masks["test"] & mpra_df[cell_type].notna().to_numpy()
        true_gap = activity_residual.loc[evaluated, cell_type]
        pred_gap = pred_residual.loc[evaluated, cell_type]
        # CTS tails inside the measured subset, the bundle's sub-universe rule
        high = (true_gap > np.percentile(true_gap, 99)).to_numpy()
        low = (true_gap < np.percentile(true_gap, 1)).to_numpy()
        rows.append(
            {
                "subset": name,
                "n_assays": len(feature_assays),
                "cell_type": cell_type,
                "absolute_r": safe_metric(
                    metrics.pearson, mpra_df.loc[evaluated, cell_type], pred[cell_type][evaluated]
                ),
                "residual_r": safe_metric(metrics.pearson, true_gap, pred_gap),
                "cts_high_auroc": roc_auc_score(high, pred_gap),
                "cts_low_auroc": roc_auc_score(low, -pred_gap),
            }
        )
    return rows


def build_subsets(pool, univariate_table, usable_both):
    """Competing feature sets. Only assays covering all five cell types can enter one."""
    model_ready = sorted(a for a, cells in zip(pool["assay"], pool["cells"]) if len(cells) == 5)
    ranked = (
        univariate_table[univariate_table["cell_type"].isin(test_cell_types)]
        .groupby("assay")["residual_r"]
        .mean()
        .reindex(model_ready)
        .sort_values(ascending=False)
    )
    subsets = {
        "chosen 4": list(assays),
        "chosen 4 + H3K4me1": list(assays) + ["H3K4me1"],
        f"usable on both datasets ({len(usable_both)})": sorted(usable_both),
        "top 4 by residual r": list(ranked.index[:4]),
        f"all model-ready ({len(model_ready)})": model_ready,
        "DNase only": ["DNase"],
    }
    rng = np.random.default_rng(random_seed)
    for i in range(n_random_subsets):
        subsets[f"random 4 #{i + 1}"] = sorted(rng.choice(model_ready, size=4, replace=False))
    return subsets


def report_univariate(univariate_table, pool):
    chosen = set(assays)
    print("\n=== univariate correlation, test chromosomes ===")
    for cell_type in cell_types:
        block = univariate_table[univariate_table["cell_type"] == cell_type]
        block = block.sort_values("residual_r", ascending=False)
        block = block.assign(chosen=[a in chosen for a in block["assay"]])
        tag = "held out" if cell_type in test_cell_types else "training"
        print(f"\n--- {cell_type} ({tag}), {len(block)} assays scorable ---")
        print(
            block[["assay", "coverage", "absolute_r", "residual_r", "chosen"]]
            .round(4)
            .to_string(index=False)
        )


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = ag_padded_metadata(ag_metadata_path)
    starts = ag_head_starts(metadata, pred_path, variant)
    tracks = named_tracks(metadata)
    usable_both = availability_funnel(tracks)

    pool = candidate_pool(tracks)
    flags = pool.assign(
        n_gosai_cells=pool["cells"].map(len),
        usable_both=[a in usable_both for a in pool["assay"]],
        chosen=[a in assays for a in pool["assay"]],
    )
    flags[["assay", "head", "coverage", "n_gosai_cells", "usable_both", "chosen"]].to_csv(
        output_dir / "candidate_pool.csv", index=False
    )

    mpra_df = pd.read_csv(mpra_path, sep="\t")
    masks = build_basic_masks(mpra_df)
    vef = extract_vef(pool, starts)
    print(f"[vef] {vef.shape}")

    univariate_table = univariate(vef, mpra_df, masks, pool)
    univariate_table.to_csv(output_dir / "assay_univariate.csv", index=False)
    report_univariate(univariate_table, pool)

    rows = []
    for name, feature_assays in build_subsets(pool, univariate_table, usable_both).items():
        rows += score_subset(name, feature_assays, vef, mpra_df, masks)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "subset_comparison.csv", index=False)

    print("\n=== subset comparison, mean over the two held-out cell types ===")
    wide = table.groupby("subset")[
        ["n_assays", "absolute_r", "residual_r", "cts_high_auroc", "cts_low_auroc"]
    ].mean()
    random_rows = wide.index.str.startswith("random 4")
    collapsed = pd.concat(
        [
            wide[~random_rows],
            wide[random_rows]
            .agg(["mean", "std", "max"])
            .rename(index={"mean": "random 4 (mean)", "std": "random 4 (sd)", "max": "random 4 (best)"}),
        ]
    )
    print(collapsed.round(4).to_string())


if __name__ == "__main__":
    main()
