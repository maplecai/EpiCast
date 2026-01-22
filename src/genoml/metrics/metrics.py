import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from scipy import stats
from ..utils import *

import numpy as np
import numpy as np

import numpy as np

def metrics_each_channel(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-12,
    dtype=np.float32,
) -> dict:
    """
    x, y: (..., C)
    返回 dict: {"pearson": (C,), "mse": (C,), "r2": (C,)}
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim < 2:
        raise ValueError(f"expect at least 2D (..., C), got {x.ndim}D")

    t = x.reshape(-1, x.shape[-1]).astype(dtype, copy=False)  # (M, C)
    p = y.reshape(-1, y.shape[-1]).astype(dtype, copy=False)  # (M, C)

    # diff
    d = t - p
    mse = np.mean(d * d, axis=0)

    # r2: 1 - ss_res/ss_tot
    t_mean = np.mean(t, axis=0)
    tc = t - t_mean
    ss_res = np.sum(d * d, axis=0)
    ss_tot = np.sum(tc * tc, axis=0)

    r2 = np.full(t.shape[1], np.nan, dtype=np.float64)
    ok_r2 = ss_tot > eps
    r2[ok_r2] = 1.0 - (ss_res[ok_r2] / ss_tot[ok_r2])

    # pearson
    p_mean = np.mean(p, axis=0)
    pc = p - p_mean
    num = np.sum(tc * pc, axis=0)
    den = np.sqrt(np.sum(tc * tc, axis=0) * np.sum(pc * pc, axis=0))

    pearson = np.full(t.shape[1], np.nan, dtype=np.float64)
    ok_p = den > eps
    pearson[ok_p] = num[ok_p] / den[ok_p]

    return {"pearson": pearson, "mse": mse.astype(np.float64), "r2": r2}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = remove_nan(x, y)
    if len(x) >= 2:
        r, p = pearsonr(x, y)
    else:
        print('after remove nan, len(x) < 2, pearson = nan')
        r, p = np.nan, np.nan
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = remove_nan(x, y)
    if len(x) >= 2:
        r, p = spearmanr(x, y)
    else:
        print('after remove nan, len(x) < 2, spearman = nan')
        r, p = np.nan, np.nan
    return r, p


# def mse(x: np.ndarray, y: np.ndarray) -> float:
#     x, y = remove_nan(x, y)
#     if len(x) >= 2:
#         mse = mean_squared_error(x, y)
#     else:
#         print('after remove nan, len(x) < 2, mse = nan')
#         mse = np.nan
#     return mse
