import os
import random
import numpy as np
import pandas as pd
import torch

from typing import List, Callable


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(x, eps=0.0):
    x = np.asarray(x)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x/(1-x))

