import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from scipy.stats import pearsonr, spearmanr
import sklearn.metrics
import scipy.stats
from ..utils import *

import numpy as np
from numpy.typing import ArrayLike

def metrics_each_channel(
    x: ArrayLike,
    y: ArrayLike,
    eps: float = 1e-12
) -> dict:
    """
    x, y: (..., C)
    返回 dict: {"pearson": (C,), "mse": (C,), "r2": (C,)}
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim < 2:
        raise ValueError(f"expect at least 2D (..., C), got {x.ndim}D")

    t = x.reshape(-1, x.shape[-1])  # (M, C)
    p = y.reshape(-1, y.shape[-1])  # (M, C)

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


def pearson(x: ArrayLike, y: ArrayLike) -> float:
    x, y = remove_nan(x, y)
    if len(x) >= 2:
        r, p = scipy.stats.pearsonr(x, y)
    else:
        r, p = np.nan, np.nan
    return r


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    x, y = remove_nan(x, y)
    if len(x) >= 2:
        r, p = scipy.stats.spearmanr(x, y)
    else:
        r, p = np.nan, np.nan
    return r


def r2_score(x: ArrayLike, y: ArrayLike) -> float:
    x, y = remove_nan(x, y)
    if len(x) >= 2:
        r2 = sklearn.metrics.r2_score(x, y)
    else:
        r2 = np.nan
    return r2


def mse(x: ArrayLike, y: ArrayLike) -> float:
    x, y = remove_nan(x, y)
    if len(x) >= 1:
        mse = sklearn.metrics.mean_squared_error(x, y)
    else:
        mse = np.nan
    return mse

def rmse(x: ArrayLike, y: ArrayLike) -> float:
    x, y = remove_nan(x, y)
    if len(x) >= 1:
        mse = sklearn.metrics.mean_squared_error(x, y)
        rmse = np.sqrt(mse)
    else:
        rmse = np.nan
    return rmse

def mae(x: ArrayLike, y: ArrayLike) -> float:
    x, y = remove_nan(x, y)
    if len(x) >= 1:
        mae = sklearn.metrics.mean_absolute_error(x, y)
    else:
        mae = np.nan
    return mae
