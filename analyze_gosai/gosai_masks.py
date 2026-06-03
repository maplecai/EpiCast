import numpy as np


def build_basic_masks(mpra_df):
    return {
        "total": np.ones(len(mpra_df), dtype=bool),
        "train": ~mpra_df["chr"].isin(["chr7", "chr13", "chr19", "chr21", "chrX"]),
        "val": mpra_df["chr"].isin(["chr19", "chr21", "chrX"]),
        "test": mpra_df["chr"].isin(["chr7", "chr13"]),
    }


def build_specific_masks(mpra_df, cell_types):
    masks = {}

    for cell_type in cell_types:
        other_cell_types = [ct for ct in cell_types if ct != cell_type]
        second_highest = mpra_df[other_cell_types].max(axis=1)
        gap_vs_second = mpra_df[cell_type] - second_highest

        q99 = np.percentile(gap_vs_second.dropna(), 99)
        masks[f"{cell_type}_specific"] = (gap_vs_second > q99).to_numpy()
        print(f"{cell_type}_specific:", masks[f"{cell_type}_specific"].sum())

    all_specific = np.zeros(len(mpra_df), dtype=bool)
    for cell_type in cell_types:
        all_specific |= masks[f"{cell_type}_specific"]

    masks["all_specific"] = all_specific
    print("all_specific:", masks["all_specific"].sum())

    return masks


def build_masks(mpra_df, cell_types):
    masks = build_basic_masks(mpra_df)
    masks.update(build_specific_masks(mpra_df, cell_types))
    return masks


def get_mask(split, masks, cell_type=None):
    split = split.strip()

    if split == "specific":
        if cell_type is None:
            raise ValueError("split='specific' requires cell_type")
        return masks[f"{cell_type}_specific"]

    if "|" in split:
        left, right = split.split("|", 1)
        return get_mask(left, masks, cell_type) | get_mask(right, masks, cell_type)

    if "&" in split:
        left, right = split.split("&", 1)
        return get_mask(left, masks, cell_type) & get_mask(right, masks, cell_type)

    if split not in masks:
        raise ValueError(f"Unknown split: {split}")

    return masks[split]
