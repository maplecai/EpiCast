import os
import yaml
import random
import numpy as np
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from typing import List, Callable


def set_seed(seed:int = 42) -> None:
    '''
    设置随机数种子
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_cell_type_specific_metrics(
        y_pred: np.ndarray, 
        y_true: np.ndarray, 
        metric_funcs: List[Callable], 
        cell_types: List[str]
    ):
        metric_names = [type(m).__name__ for m in metric_funcs]
        metric_df = pd.DataFrame(index=cell_types, columns=metric_names)
        for idx, cell_type in enumerate(cell_types):
            if len(y_true.shape) == 1:
                indice = (y_true['cell_type'] == cell_type)
                y_pred_0 = y_pred[indice]
                y_true_0 = y_true[indice]
            elif len(y_true.shape) == 2:
                y_pred_0 = y_pred[:, idx]
                y_true_0 = y_true[:, idx]
            else:
                raise ValueError(f'{y_pred.shape = }, {y_true.shape = }')
            for m in metric_funcs:
                metric_name = type(m).__name__
                score = m(y_pred_0, y_true_0)
                metric_df.loc[cell_type, metric_name] = score
        return metric_df


def split_dataset(index_list, train_valid_test_ratio):
    """
    Split the dataset into train, valid, and test sets.
    """
    total_size = len(index_list)
    train_ratio, valid_ratio, test_ratio = train_valid_test_ratio
    train_split = int(total_size * train_ratio)
    valid_split = int(total_size * (train_ratio+valid_ratio))
    
    # if split_mode is None:
    #     pass
    # elif split_mode =='order':
    #     pass
    # elif split_mode == 'random':
    #     np.random.shuffle(index_list)
    # else:
    #     raise ValueError

    train_indice = index_list[:train_split]
    valid_indice = index_list[train_split:valid_split]
    test_indice = index_list[valid_split:]

    return train_indice, valid_indice, test_indice


def filter_by_column(table, filter_column, filter_in_list=None, filter_not_in_list=None):
    if filter_column is not None:
        if filter_in_list is not None:
            filtered_index = table[filter_column].isin(filter_in_list)
            table = table[filtered_index]
        if filter_not_in_list is not None:
            filtered_index = ~table[filter_column].isin(filter_not_in_list)
            table = table[filtered_index]
    return table


def remove_nan(x, y, verbose=False):
    assert len(x) == len(y), 'len(x) must be equal to len(y)'
    if len(x.shape) == 2:
        x_mask = (~np.isnan(x)).all(axis=1)
    else:
        x_mask = ~np.isnan(x)
    y_mask = ~np.isnan(y)
    mask = x_mask & y_mask
    x = x[mask]
    y = y[mask]

    # if len(mask) == 0:
    #     print('len(x) = 0')
    if mask.sum() / len(mask) < 0.1 and verbose:
        print(f'{mask.sum()} of {len(mask)} values are non-nan.')
    return x, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(x):
    return np.log(x/(1-x))


def pearson(x, y, allow_nan=True):
    assert len(x) == len(y)
    x, y = remove_nan(x, y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    r, _ = pearsonr(x, y)
    return r


def spearman(x, y, allow_nan=True):
    assert len(x) == len(y)
    assert len(x) > 0
    x, y = remove_nan(x, y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    r, _ = spearmanr(x, y)
    return r