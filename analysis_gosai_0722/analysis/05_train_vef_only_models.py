import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import epicast

from config import (
    assays,
    cell_types,
    mpra_path,
    results_dir,
    test_cell_types,
    train_cell_types,
    vef_paths,
)
from utils import build_basic_masks

dnase_sources = {
    "sei": results_dir / "vef_only/sei_dnase",
    "enformer": results_dir / "vef_only/enformer_dnase",
    "borzoi": results_dir / "vef_only/borzoi_dnase",
    "alphagenome": results_dir / "vef_only/ag_dnase",
}

# Models fitted on all four assays. Only AlphaGenome is retrained here: the Sei
# VEF matrix is unchanged by the track-indexing fix and these fits are
# deterministic given the seed, so its stored predictions still apply.
full_sources = {
    "alphagenome": results_dir / "vef_only/ag_vef",
}


def full_models():
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(random_state=0),
        "lasso": Lasso(random_state=0),
        "xgb": XGBRegressor(random_state=0),
        "mlp": MLPRegressor(random_state=0),
    }


def get_X_y(df_x, df_y, mask, cell_types, assays):
    X, y = [], []
    for cell_type in cell_types:
        x_cols = [f"{cell_type}_{assay}" for assay in assays]
        X.append(df_x.loc[mask, x_cols].to_numpy())
        y.append(df_y.loc[mask, cell_type].to_numpy())
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def params_to_save(model):
    params = model.get_params(deep=True)
    if hasattr(model, "coef_"):
        params["coef_"] = np.asarray(model.coef_).tolist()
    if hasattr(model, "intercept_"):
        params["intercept_"] = np.asarray(model.intercept_).tolist()
    return params


def fit_and_save(name, model, X_train, y_train, X_total, n_variants, output_dir):
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[train] model={name}  started={ts}")
    model.fit(X_train, y_train)
    y_total_pred = model.predict(X_total).reshape(len(cell_types), n_variants).T

    out_path = output_dir / f"{name}_pred.npy"
    np.save(out_path, y_total_pred)
    print(f"[save] predictions {y_total_pred.shape} (variants x cell types) -> {out_path}")

    model_path = output_dir / f"{name}.joblib"
    joblib.dump(model, model_path)
    print(f"[save] model -> {model_path}")

    params_path = output_dir / f"{name}_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_to_save(model), f, ensure_ascii=False, indent=2, default=str)
    print(f"[save] params -> {params_path}")


def train_source(source, output_dir, feature_assays, models, mpra_df, masks):
    vef_df = pd.read_csv(vef_paths[source], sep="\t")
    print(f"[load] {source} VEF: {vef_df.shape[0]} variants x {vef_df.shape[1]} columns")
    print(f"[feat] assays: {', '.join(feature_assays)}")

    X_total, _ = get_X_y(vef_df, mpra_df, masks["total"], cell_types, feature_assays)
    X_train, y_train = get_X_y(vef_df, mpra_df, masks["train"], train_cell_types, feature_assays)
    X_train, y_train = epicast.utils.remove_nan(X_train, y_train)
    print(f"[split] train set: X {X_train.shape}, y {y_train.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        fit_and_save(name, model, X_train, y_train, X_total, len(mpra_df), output_dir)


def main():
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] MPRA labels: {mpra_df.shape[0]} variants x {mpra_df.shape[1]} columns")
    masks = build_basic_masks(mpra_df)
    print(f"[split] train cell types ({len(train_cell_types)}): {', '.join(train_cell_types)}")
    print(f"[split] test cell types ({len(test_cell_types)}): {', '.join(test_cell_types)}")

    print("\n=== DNase-only linear ===")
    for source, output_dir in dnase_sources.items():
        train_source(source, output_dir, ["DNase"], {"linear": LinearRegression()}, mpra_df, masks)

    print("\n=== all four assays ===")
    for source, output_dir in full_sources.items():
        train_source(source, output_dir, assays, full_models(), mpra_df, masks)


if __name__ == "__main__":
    main()
