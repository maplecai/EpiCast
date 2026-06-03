import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import epicast

from gosai_masks import build_basic_masks


def get_X_y(df_x, df_y, mask, cell_types, assays):
    X, y = [], []
    for cell_type in cell_types:
        x_cols = [f"{cell_type}_{assay}" for assay in assays]
        X.append(df_x.loc[mask, x_cols].to_numpy())
        y.append(df_y.loc[mask, cell_type].to_numpy())
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def main():
    mpra_path = "data/gosai_mpra/gosai_mpra_760679_zscore.tsv"
    vef_path = "data/gosai_mpra/gosai_mpra_760679_ag_vef_x10_log1p.tsv"
    train_cell_types = ["K562", "HepG2", "SK-N-SH"]
    test_cell_types = ["HCT116", "A549"]
    cell_types = train_cell_types + test_cell_types
    assays = ["DNase", "H3K4me3", "H3K27ac", "CTCF"]
    output_dir = "analyze_gosai/results/vef_only_train3test2"

    mpra_df = pd.read_csv(mpra_path, sep="\t")
    vef_df = pd.read_csv(vef_path, sep="\t")
    print(f"[load] MPRA labels: {mpra_df.shape[0]} variants x {mpra_df.shape[1]} columns")
    print(f"[load] VEF features:  {vef_df.shape[0]} variants x {vef_df.shape[1]} columns")

    masks = build_basic_masks(mpra_df)
    X_total, _ = get_X_y(vef_df, mpra_df, masks["total"], cell_types, assays)
    X_train, y_train = get_X_y(vef_df, mpra_df, masks["train"], train_cell_types, assays)
    X_train, y_train = epicast.utils.remove_nan(X_train, y_train)

    print(f"[split] train cell types ({len(train_cell_types)}): {', '.join(train_cell_types)}")
    print(f"[split] test cell types ({len(test_cell_types)}): {', '.join(test_cell_types)}")
    print(f"[split] train set: X {X_train.shape}, y {y_train.shape}")

    os.makedirs(output_dir, exist_ok=True)

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(random_state=0),
        "lasso": Lasso(random_state=0),
        "sgd": SGDRegressor(random_state=0),
        "hgb": HistGradientBoostingRegressor(random_state=0),
        "lgbm": LGBMRegressor(random_state=0), # slow
        "xgb": XGBRegressor(random_state=0),
        "mlp": MLPRegressor(random_state=0),
    }

    for name, model in models.items():
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"[train] model={name}  started={ts}")
        model.fit(X_train, y_train)
        y_total_pred = model.predict(X_total)
        y_total_pred = y_total_pred.reshape(len(cell_types), len(mpra_df)).T
        out_path = os.path.join(output_dir, f"{name}_pred.npy")
        print(f"[save]  predictions {y_total_pred.shape} (variants x cell types) -> {out_path}")
        np.save(out_path, y_total_pred)


if __name__ == "__main__":
    main()
