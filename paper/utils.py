from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# --- AlphaGenome track indexing -------------------------------------------------
# Each AlphaGenome output head is a fixed-width tensor whose columns correspond to
# the rows of the *padded* metadata for that assay type, real tracks first and
# explicit "Padding" rows filling the tail. Column numbers must therefore be
# derived from the padded metadata:
#
#     h5 column = padded metadata index - start of that assay's head
#
# Deriving indices from the padding-removed metadata and subtracting hard-coded
# offsets silently produced the wrong column for the TF head, so the head starts
# are computed from the data and checked against the h5 widths here. Both the
# Gosai and the Castillo extraction scripts use these helpers so the two cannot
# drift apart.

# assay -> "Assay title" of the output head it lives in
ag_assay_title = {
    "DNase": "DNase-seq",
    "H3K4me3": "Histone ChIP-seq",
    "H3K27ac": "Histone ChIP-seq",
    "CTCF": "TF ChIP-seq",
}

ag_dnase_keys = ("dnase_128", "dnase")  # the 128-bp key name differs between files
ag_bin_width = 128.0

# Preprocessing variants of the AlphaGenome VEF matrix. They differ only in how
# the tracks are read out and rescaled before log1p; the track indexing is shared
# and always derived from the padded metadata.
#
#   b   DNase from the 1-bp head, the 128-bp heads divided by the bin width, and
#       log1p(10x). This is the published matrix: it keeps the read-out of the
#       original analysis and differs from it only in the corrected CTCF column.
#   b2  as b but DNase also from the 128-bp head, so a single read-out and a
#       single transform apply to all four assays. Used to isolate the effect of
#       the DNase read-out.
#   c   all four assays from the 128-bp heads with no rescaling, log1p(x), so the
#       preprocessing carries no scale constant at all.
#
# Recorded here rather than in the extraction scripts so that a variant cannot be
# silently redefined by editing one script, which is how the CTCF column came to
# be read from the wrong track in the first place.
ag_variants = {
    "b": {"dnase_1bp": True, "per_bp": True, "multiplier": 10.0,
          "suffix": "x10_log1p_dnase1"},
    "b2": {"dnase_1bp": False, "per_bp": True, "multiplier": 10.0,
           "suffix": "x10_log1p_128bp"},
    "c": {"dnase_1bp": False, "per_bp": False, "multiplier": 1.0,
          "suffix": "log1p_128bp"},
}
ag_default_variant = "b"


def ag_padded_metadata(path):
    metadata = pd.read_csv(path, sep="\t")
    metadata["padded_index"] = metadata.index
    return metadata


def ag_dataset_keys(pred_path, variant=ag_default_variant):
    """assay -> h5 dataset the variant reads it from."""
    with h5py.File(pred_path, "r") as f:
        keys = set(f.keys())
    if ag_variants[variant]["dnase_1bp"]:
        dnase = "dnase_1"
    else:
        dnase = next((key for key in ag_dnase_keys if key in keys), None)
    assert dnase in keys, f"{dnase!r} missing from {pred_path}"
    return {
        "DNase": dnase,
        "H3K4me3": "chip_histone",
        "H3K27ac": "chip_histone",
        "CTCF": "chip_tf",
    }


def ag_head_starts(metadata, pred_path, variant=ag_default_variant):
    """First column of each head, verified against the h5 dataset widths."""
    datasets = ag_dataset_keys(pred_path, variant)
    with h5py.File(pred_path, "r") as f:
        widths = {key: f[key].shape[1] for key in f.keys()}

    starts = {}
    for assay, title in ag_assay_title.items():
        dataset = datasets[assay]
        block = metadata.loc[metadata["Assay title"] == title, "padded_index"]
        start, width = int(block.min()), widths[dataset]
        head = metadata[
            (metadata["padded_index"] >= start) & (metadata["padded_index"] < start + width)
        ]
        n_real = int((head["Assay title"] == title).sum())
        n_pad = int((head["name"] == "Padding").sum())
        assert n_real + n_pad == width, f"{title}: {n_real}+{n_pad} != {width}"
        assert int(block.max()) == start + n_real - 1, f"{title}: real rows not contiguous"
        starts[assay] = start
    return starts


def ag_track_columns(metadata, starts, cell_types, assays):
    """(cell_type, assay) -> column index inside the corresponding h5 dataset."""
    real = metadata[
        (metadata["name"] != "Padding") & (metadata["genetically_modified"] == False)
    ]

    columns = {}
    for assay in assays:
        title = ag_assay_title[assay]
        rows = real[real["Assay title"] == title]
        if title == "Histone ChIP-seq":
            rows = rows[rows["histone_mark"] == assay]
        elif title == "TF ChIP-seq":
            rows = rows[rows["transcription_factor"] == assay]
        rows = rows[rows["biosample_name"].isin(cell_types)]

        counts = rows["biosample_name"].value_counts()
        missing = set(cell_types) - set(counts.index)
        assert not missing, f"{assay}: no track for {sorted(missing)}"
        assert (counts == 1).all(), f"{assay}: multiple tracks for {list(counts[counts > 1].index)}"
        columns[assay] = {
            row["biosample_name"]: int(row["padded_index"]) - starts[assay]
            for _, row in rows.iterrows()
        }

    table = pd.DataFrame(columns).loc[list(cell_types), list(assays)]
    assert (table >= 0).all().all()
    return table.astype(int)


def ag_extract_vef(pred_path, track_columns, cell_types, assays, variant=ag_default_variant):
    """Raw VEF matrix for a preprocessing variant, before the log1p transform.

    The 128-bp heads report bin sums, so a variant that asks for a per-base-pair
    scale divides them by the bin width. The 1-bp DNase head is already averaged
    over the insert during prediction and is never divided.
    """
    spec = ag_variants[variant]
    datasets = ag_dataset_keys(pred_path, variant)
    vef = {}
    with h5py.File(pred_path, "r") as f:
        # each chunk spans every column, so read a dataset once and then slice
        for dataset in sorted(set(datasets.values())):
            block = f[dataset][:, :]
            for cell_type in cell_types:
                for assay in assays:
                    if datasets[assay] != dataset:
                        continue
                    values = np.asarray(
                        block[:, int(track_columns.at[cell_type, assay])], dtype=np.float64
                    )
                    if spec["per_bp"] and dataset != "dnase_1":
                        values = values / ag_bin_width
                    vef[f"{cell_type}_{assay}"] = values
            del block
    columns = [f"{c}_{a}" for c in cell_types for a in assays]
    return pd.DataFrame(vef)[columns]


def ag_log1p(vef_raw, variant=ag_default_variant):
    """The transform applied to AlphaGenome VEFs: log1p of the scaled raw values."""
    multiplier = ag_variants[variant]["multiplier"]
    return pd.DataFrame(np.log1p(vef_raw * multiplier), columns=vef_raw.columns)


def load_pred_df(pred_path, cell_types):
    pred = np.load(pred_path)
    return pd.DataFrame(pred, columns=[f"{ct}_pred" for ct in cell_types])


def load_train_pred_df(pred_path, train_cell_types, test_cell_types):
    """Load preds that only cover train cell types; fill test with train mean.

    Consequence for CTS tasks: the residual gap of every test cell type is
    identically 0, so such a model cannot rank test-cell-type-specific variants.
    Its precision/recall come out as 0 and its AUROC as 0.5 by construction.
    """
    pred = np.load(pred_path)
    train_cols = [f"{ct}_pred" for ct in train_cell_types]
    pred_df = pd.DataFrame(pred, columns=train_cols)
    mean_pred = pred_df[train_cols].mean(axis=1)
    for ct in test_cell_types:
        pred_df[f"{ct}_pred"] = mean_pred
    return pred_df


def load_dnase_pred_df(pred_path, cell_types):
    vef_df = pd.read_csv(pred_path, sep="\t")
    pred_df = pd.DataFrame(index=vef_df.index)
    for ct in cell_types:
        pred_df[f"{ct}_pred"] = vef_df[f"{ct}_DNase"]
    return pred_df


def apply_dnase_linear(model, vef_df, cell_types):
    """DNase VEF mapped into activity units by the fitted single-feature linear model.

    Columns are plain cell type names, matching the other prediction helpers in
    analysis/14, not the "{ct}_pred" convention used by load_pred_dfs.
    """
    pred_df = pd.DataFrame(index=vef_df.index)
    for ct in cell_types:
        pred_df[ct] = model.predict(vef_df[[f"{ct}_DNase"]].to_numpy()).ravel()
    return pred_df


def load_pred_dfs(models, cell_types, train_cell_types, test_cell_types, n_variants=None):
    """Load model predictions. models: list of (name, path, kind).

    kind:
      - "dnase": VEF tsv with {ct}_DNase columns
      - "seq-only": npy covering only train cell types; fill test with train mean
      - "seq-only-all-train": npy covering all cell types from all-train seq model
      - "vef-only" / "seq-vef": npy covering all cell_types

    Predictions carry no variant key, so they are matched to the MPRA table by
    row order. Pass n_variants=len(mpra_df) to catch silent misalignment.
    """
    pred_dfs = OrderedDict()
    for model_name, pred_path, kind in models:
        if kind == "dnase":
            pred_df = load_dnase_pred_df(pred_path, cell_types)
        elif kind == "seq-only":
            pred_df = load_train_pred_df(pred_path, train_cell_types, test_cell_types)
        elif kind in {"vef-only", "seq-vef", "seq-only-all-train"}:
            pred_df = load_pred_df(pred_path, cell_types)
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        if n_variants is not None and len(pred_df) != n_variants:
            raise ValueError(
                f"{model_name}: {len(pred_df)} rows in {pred_path}, expected {n_variants}"
            )
        print(f"[load] {model_name} {pred_df.shape}")
        pred_dfs[model_name] = pred_df
    return pred_dfs


def load_true_df(mpra_df, cell_types):
    true_df = pd.DataFrame(index=mpra_df.index)
    for cell_type in cell_types:
        true_df[f"{cell_type}_true"] = mpra_df[cell_type]
    return true_df


def load_residual_eval_dfs(mpra_df, pred_dfs, cell_types, train_cell_types):
    """Per-variant residual w.r.t. train cell-type mean (same as fig3d)."""
    train_mean_true = mpra_df[train_cell_types].mean(axis=1)
    true_df = pd.DataFrame(index=mpra_df.index)
    for cell_type in cell_types:
        true_df[f"{cell_type}_true"] = mpra_df[cell_type] - train_mean_true

    train_pred_cols = [f"{ct}_pred" for ct in train_cell_types]
    resid_pred_dfs = OrderedDict()
    for model_name, pred_df in pred_dfs.items():
        train_mean_pred = pred_df[train_pred_cols].mean(axis=1)
        resid_pred_df = pd.DataFrame(index=pred_df.index)
        for cell_type in cell_types:
            resid_pred_df[f"{cell_type}_pred"] = (
                pred_df[f"{cell_type}_pred"] - train_mean_pred
            )
        resid_pred_dfs[model_name] = resid_pred_df
    return true_df, resid_pred_dfs


def safe_metric(metric_fn, x, y):
    valid = ~(pd.isna(x) | pd.isna(y))
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return metric_fn(x, y)


def cts_thresholds(gap: pd.Series, low_pct: float = 1, high_pct: float = 99) -> tuple[float, float]:
    vals = gap.dropna().to_numpy(dtype=float)
    return float(np.percentile(vals, high_pct)), float(np.percentile(vals, low_pct))


def build_cts_labels(
    gap: pd.Series,
    low_pct: float = 1,
    high_pct: float = 99,
) -> tuple[pd.Series, pd.Series, float, float]:
    q_hi, q_lo = cts_thresholds(gap, low_pct=low_pct, high_pct=high_pct)
    return gap > q_hi, gap < q_lo, q_hi, q_lo


def mingap_scores(mpra_df, cell_type, reference_cell_types):
    """Smallest gap between the target cell type and any reference cell type.

    This is the definition used by Gosai et al.: an element is cell-type-specific
    in the high direction when its activity exceeds *every* reference cell type,
    so the quantity to rank by is the gap to the strongest reference,

        high = A_c - max_{k in P, k != c} A_k

    and symmetrically for the low direction against the weakest reference. Unlike
    the residual against the panel mean, a positive high score guarantees that the
    element really is the most active of the cell types compared, which is what
    "selective" means for design. The reference panel is restricted to the fully
    measured cell lines, so the comparison set does not vary between elements.
    """
    references = [ct for ct in reference_cell_types if ct != cell_type]
    activity = mpra_df[cell_type]
    return (
        activity - mpra_df[references].max(axis=1),
        activity - mpra_df[references].min(axis=1),
    )


def build_mingap_labels(mpra_df, cell_type, reference_cell_types, low_pct=1, high_pct=99):
    """CTS-high / CTS-low labels from the min-gap score, with their thresholds."""
    high_score, low_score = mingap_scores(mpra_df, cell_type, reference_cell_types)
    q_hi = float(np.percentile(high_score.dropna().to_numpy(dtype=float), high_pct))
    q_lo = float(np.percentile(low_score.dropna().to_numpy(dtype=float), low_pct))
    return high_score > q_hi, low_score < q_lo, q_hi, q_lo


def build_mingap_masks(
    mpra_df,
    cell_types,
    reference_cell_types,
    low_pct=1,
    high_pct=99,
    key="mingap_1_99",
    verbose=True,
):
    masks = {}
    for cell_type in cell_types:
        high, low, q_hi, q_lo = build_mingap_labels(
            mpra_df, cell_type, reference_cell_types, low_pct=low_pct, high_pct=high_pct
        )
        masks[f"{cell_type}_{key}"] = (high | low).to_numpy()
        if verbose:
            print(
                f"{cell_type}_{key}: {masks[f'{cell_type}_{key}'].sum()} "
                f"(q{high_pct}={q_hi:.4f}, q{low_pct}={q_lo:.4f})"
            )

    all_key = f"all_{key}"
    all_mask = np.zeros(len(mpra_df), dtype=bool)
    for cell_type in cell_types:
        all_mask |= masks[f"{cell_type}_{key}"]
    masks[all_key] = all_mask
    if verbose:
        print(f"{all_key}:", masks[all_key].sum())
    return masks


def build_basic_masks(mpra_df):
    return {
        "total": np.ones(len(mpra_df), dtype=bool),
        "train": ~mpra_df["chr"].isin(["chr7", "chr13", "chr19", "chr21", "chrX"]),
        "val": mpra_df["chr"].isin(["chr19", "chr21", "chrX"]),
        "test": mpra_df["chr"].isin(["chr7", "chr13"]),
    }


def build_specific_masks(mpra_df, cell_types, verbose=True):
    masks = {}
    for cell_type in cell_types:
        other_cell_types = [ct for ct in cell_types if ct != cell_type]
        second_highest = mpra_df[other_cell_types].max(axis=1)
        gap_vs_second = mpra_df[cell_type] - second_highest
        q99 = np.percentile(gap_vs_second.dropna(), 99)
        masks[f"{cell_type}_specific"] = (gap_vs_second > q99).to_numpy()
        if verbose:
            print(f"{cell_type}_specific:", masks[f"{cell_type}_specific"].sum())

    all_specific = np.zeros(len(mpra_df), dtype=bool)
    for cell_type in cell_types:
        all_specific |= masks[f"{cell_type}_specific"]
    masks["all_specific"] = all_specific
    if verbose:
        print("all_specific:", masks["all_specific"].sum())
    return masks


def build_cts_tail_masks(
    mpra_df,
    cell_types,
    train_cell_types,
    low_pct,
    high_pct,
    key,
    verbose=True,
):
    masks = {}
    train_mean = mpra_df[train_cell_types].mean(axis=1)
    for cell_type in cell_types:
        gap = mpra_df[cell_type] - train_mean
        vals = gap.dropna()
        q_hi = np.percentile(vals, high_pct)
        q_lo = np.percentile(vals, low_pct)
        masks[f"{cell_type}_{key}"] = ((gap > q_hi) | (gap < q_lo)).to_numpy()
        if verbose:
            print(f"{cell_type}_{key}:", masks[f"{cell_type}_{key}"].sum())

    all_key = f"all_{key}"
    all_mask = np.zeros(len(mpra_df), dtype=bool)
    for cell_type in cell_types:
        all_mask |= masks[f"{cell_type}_{key}"]
    masks[all_key] = all_mask
    if verbose:
        print(f"{all_key}:", masks[all_key].sum())
    return masks


def build_masks(mpra_df, cell_types, train_cell_types=None, test_cell_types=None, verbose=True):
    masks = build_basic_masks(mpra_df)
    masks.update(build_specific_masks(mpra_df, cell_types, verbose=verbose))
    if train_cell_types is not None and test_cell_types is not None:
        masks.update(
            build_cts_tail_masks(
                mpra_df, cell_types, train_cell_types,
                low_pct=1, high_pct=99, key="cts_1_99", verbose=verbose,
            )
        )
        masks.update(
            build_cts_tail_masks(
                mpra_df, cell_types, train_cell_types,
                low_pct=5, high_pct=95, key="cts_5_95", verbose=verbose,
            )
        )
        masks.update(
            build_mingap_masks(
                mpra_df, cell_types, train_cell_types,
                low_pct=1, high_pct=99, key="mingap_1_99", verbose=verbose,
            )
        )
    return masks


def get_mask(split, masks, cell_type=None):
    split = split.strip()
    if split in ["specific", "cts_1_99", "cts_5_95", "mingap_1_99"]:
        if cell_type is None:
            raise ValueError(f"split={split!r} requires cell_type")
        return masks[f"{cell_type}_{split}"]
    if "|" in split:
        left, right = split.split("|", 1)
        return get_mask(left, masks, cell_type) | get_mask(right, masks, cell_type)
    if "&" in split:
        left, right = split.split("&", 1)
        return get_mask(left, masks, cell_type) & get_mask(right, masks, cell_type)
    if split not in masks:
        raise ValueError(f"Unknown split: {split}")
    return masks[split]
