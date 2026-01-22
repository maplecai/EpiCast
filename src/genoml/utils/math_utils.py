import os
import random
import numpy as np
import pandas as pd
import torch

from typing import List, Callable


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(x, eps=0):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x/(1-x))


def remove_nan(x, y):
    if len(x) != len(y):
        raise ValueError('len(x) must be equal to len(y)')
    x_mask = ~np.isnan(x)
    y_mask = ~np.isnan(y)
    mask = x_mask & y_mask
    x = x[mask]
    y = y[mask]
    return x, y
