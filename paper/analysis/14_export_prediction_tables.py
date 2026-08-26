"""Export one self-describing table per model: measured activity next to predictions.

The prediction npy files carry no keys and are matched to the MPRA tables by row
order, which is easy to get wrong downstream. These tables put both sides in one
file so a reader never has to re-derive the alignment, which is why everything
downstream (plot/, analysis/12) reads results/predictions/ rather than saved/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from config import (
    epicast_ag_castillo_pred,
    assays,
    build_models,
    castillo_all_cell_types,
    castillo_cell_types,
    castillo_mpra_path,
    castillo_vef_path,
    cell_types,
    eval_model_names,
    mpra_path,
    predictions_dir,
    project_root,
    results_dir,
    test_cell_types,
    train_cell_types,
)
from utils import apply_dnase_linear, build_basic_masks, load_pred_dfs

output_dir = predictions_dir
float_format = "%.5f"

gosai_meta_cols = ["id", "chr", "pos"]
# every scored model, not just the ones that reach a figure: these per-sequence
# tables are the reusable data layer, figure membership is decided in plot/
gosai_export_models = eval_model_names

castillo_meta_cols = ["id", "category", "source", "target"]
# column order inside the EpiCast prediction npy, which differs from the plot order
npy_cell_types = ["GM12878", "SK-N-SH", "WERI-Rb-1", "HepG2", "K562", "MCF-7", "HeLa-S3"]
castillo_epicast_pred_paths = {
    # EpiCast-Sei is deliberately not exported: every configs/*castillo_dataset* file
    # carries an AlphaGenome VEF matrix, so the Sei checkpoint was run on a VEF it was
    # never trained on and its Castillo predictions are not interpretable. Restoring
    # this line requires an inference config pointing at data/castillo_mpra/sei_vef.tsv.
    # "epicast_sei_vef": epicast_sei_castillo_pred,
    "epicast_ag_vef": epicast_ag_castillo_pred,
}
vef_only_model_path = results_dir / "vef_only/ag_vef/mlp.joblib"
dnase_linear_model_path = results_dir / "vef_only/ag_dnase/linear.joblib"
dhs64_pred_path = project_root / "enhancer-design/castillo_dhs64_pred_merged.tsv"
dhs64_metadata_path = (
    project_root / "enhancer-design/data/dhs_index/dhs64_training/selected_biosample_metadata.xlsx"
)
castillo_to_dhs64 = {
    "GM12878": "GM12878",
    "SK-N-SH": "SKNSH",
    "WERI-Rb-1": "WERI_Rb1",
    "HepG2": "HepG2",
    "K562": "K562",
    "MCF-7": "MCF7",
    "HeLa-S3": "HeLaS3",
}


def load_dhs64_pred(cell_types):
    metadata = pd.read_excel(dhs64_metadata_path)
    celltype_to_idx = {name: i for i, name in enumerate(metadata["Biosample name"])}
    pred_df = pd.read_csv(dhs64_pred_path, sep="\t")
    print(f"[load] {dhs64_pred_path} {pred_df.shape}")
    return pd.DataFrame(
        {
            cell_type: pred_df[f"reg_{celltype_to_idx[castillo_to_dhs64[cell_type]]}"]
            for cell_type in cell_types
        }
    )


def load_ag_dnase_pred(vef_df, cell_types):
    model = joblib.load(dnase_linear_model_path)
    print(f"[load] {dnase_linear_model_path}")
    return apply_dnase_linear(model, vef_df, cell_types)


def predict_vef_only(vef_df, cell_types):
    model = joblib.load(vef_only_model_path)
    print(f"[load] {vef_only_model_path}")
    columns = [f"{ct}_{assay}" for ct in cell_types for assay in assays]
    x_flat = vef_df[columns].to_numpy().reshape(len(vef_df) * len(cell_types), len(assays))
    return pd.DataFrame(
        model.predict(x_flat).reshape(len(vef_df), len(cell_types)), columns=cell_types
    )


def load_epicast_pred(pred_path, cell_types):
    pred_df = pd.DataFrame(np.load(pred_path), columns=npy_cell_types)
    print(f"[load] {pred_path} {pred_df.shape}")
    return pred_df[cell_types]


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, float_format=float_format)
    size_mb = path.stat().st_size / 1048576
    print(f"[save] {path} {df.shape} {size_mb:.0f}M")


def split_labels(mpra_df: pd.DataFrame) -> pd.Series:
    masks = build_basic_masks(mpra_df)
    split = pd.Series("train", index=mpra_df.index)
    split[masks["val"]] = "val"
    split[masks["test"]] = "test"
    return split


def export_gosai() -> None:
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] {mpra_path} {mpra_df.shape}")

    base_df = mpra_df[gosai_meta_cols].copy()
    base_df["split"] = split_labels(mpra_df)
    for cell_type in cell_types:
        base_df[cell_type] = mpra_df[cell_type]

    pred_dfs = load_pred_dfs(
        build_models(gosai_export_models),
        cell_types,
        train_cell_types,
        test_cell_types,
        n_variants=len(mpra_df),
    )
    for model_name, pred_df in pred_dfs.items():
        out_df = pd.concat([base_df, pred_df], axis=1)
        save(out_df, output_dir / f"gosai_{model_name}.tsv")


def export_castillo() -> None:
    mpra_df = pd.read_csv(castillo_mpra_path, sep="\t")
    print(f"[load] {castillo_mpra_path} {mpra_df.shape}")
    vef_df = pd.read_csv(castillo_vef_path, sep="\t")
    print(f"[load] {castillo_vef_path} {vef_df.shape}")

    preds = {
        "dhs64": load_dhs64_pred(castillo_cell_types),
        "vef_only": predict_vef_only(vef_df, castillo_cell_types),
        "linear_ag_dnase": load_ag_dnase_pred(vef_df, castillo_cell_types),
    }
    for model_name, pred_path in castillo_epicast_pred_paths.items():
        preds[model_name] = load_epicast_pred(pred_path, castillo_cell_types)

    base_df = mpra_df[castillo_meta_cols + castillo_all_cell_types].copy()
    for model_name, pred_df in preds.items():
        out_df = base_df.copy()
        for cell_type in castillo_cell_types:
            out_df[f"{cell_type}_pred"] = pred_df[cell_type].to_numpy()
        save(out_df, output_dir / f"castillo_{model_name}.tsv")


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_gosai()
    export_castillo()


if __name__ == "__main__":
    main()
