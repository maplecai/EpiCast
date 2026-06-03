import numpy as np
import pandas as pd


def load_mpra(path):
    return pd.read_csv(path, sep="\t")


def build_true_df(mpra_df, cell_types):
    true_df = pd.DataFrame(index=mpra_df.index)

    for cell_type in cell_types:
        true_df[f"{cell_type}_true"] = mpra_df[cell_type]

    return true_df


def load_pred_df(pred_path, cell_types):
    pred_cols = [f"{ct}_pred" for ct in cell_types]
    pred = np.load(pred_path)
    return pd.DataFrame(pred, columns=pred_cols)


def load_dnase_pred_df(pred_path, cell_types):
    vef_df = pd.read_csv(pred_path, sep="\t")
    pred_df = pd.DataFrame(index=vef_df.index)

    for cell_type in cell_types:
        pred_df[f"{cell_type}_pred"] = vef_df[f"{cell_type}_DNase"]

    return pred_df


def merge_loo_pred(pred_paths, cell_types):
    first_pred = np.load(pred_paths[cell_types[0]])
    merged_pred = np.empty((first_pred.shape[0], len(cell_types)), dtype=float)

    for cell_idx, cell_type in enumerate(cell_types):
        pred = np.load(pred_paths[cell_type])
        merged_pred[:, cell_idx] = pred[:, cell_idx]

    return merged_pred


def build_leave_one_out_merged_pred_df(mpra_df, pred_paths, cell_types):
    pred_df = pd.DataFrame(index=mpra_df.index)

    for cell_type in cell_types:
        part = load_pred_df(pred_paths[cell_type], cell_types)
        pred_df[f"{cell_type}_pred"] = part[f"{cell_type}_pred"].to_numpy()

    return pred_df
