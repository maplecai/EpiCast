import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def reverse(seq: str) -> str:
    '''反向'''
    return seq[::-1]


def complement(seq: str) -> str:
    '''互补'''
    dic = {
        'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
        'a': 't', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n',
    }
    return ''.join(dic[c] for c in seq)


def rc_seq(seq: str) -> str:
    '''反向互补'''
    return reverse(complement(seq))


def seqs_rc(seqs: list[str]) -> list[str]:
    return [rc_seq(seq) for seq in seqs]


def rc_onehot(onehot: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    '''反向互补'''
    if isinstance(onehot, np.ndarray):
        onehot = np.flip(onehot, axis=(-1,-2))
    elif isinstance(onehot, torch.Tensor):
        onehot = torch.flip(onehot, dims=(-1,-2))
    else:
        raise ValueError('onehot must be a numpy array or a torch tensor.')
    return onehot


def rc_onehots(onehots: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return rc_onehot(onehots)


def str2onehot(seq: str, N_fill_value: int = 0.25) -> np.ndarray:
    '''
    str -> onehot
    N_fill_value: 碱基N对应的值 0.25(default)
    '''
    base2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4,
                'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4,}
    seq = np.array(list(seq), dtype='U1')
    indices = np.array([base2idx[b] for b in seq], dtype=int)
    onehot = np.zeros((len(seq), 4))
    # 对N的位置整行赋值N_fill_value
    N_pos = (indices == 4)
    onehot[N_pos, :] = N_fill_value
    # 非N的位置：直接用整型索引赋值为1
    not_N_pos = ~N_pos
    onehot[np.arange(len(seq))[not_N_pos], indices[not_N_pos]] = 1
    return onehot


def strs2onehots(seqs: list[str], N_fill_value: int = 0.25) -> np.ndarray:
    '''
    序列s->onehot矩阵s
    seqs: list of sequences (must be of the same length)
    N_fill_value: 碱基N对应的值 0.25(default)
    '''
    return np.array([str2onehot(seq, N_fill_value) for seq in seqs])


def onehot2str(onehot: np.ndarray | torch.Tensor) -> str:
    '''onehot矩阵->序列'''
    if isinstance(onehot, torch.Tensor):
        onehot = onehot.detach().cpu().numpy()
    mapping = np.array(['A', 'C', 'G', 'T'], dtype='U1')
    bases = np.full(len(onehot), 'N', dtype='U1')
    
    # 使用 np.where 找到值为 1 的位置
    row_indices, col_indices = np.where(onehot == 1)
    bases[row_indices] = mapping[col_indices]
    return ''.join(bases)


def onehots2strs(onehots: np.ndarray | torch.Tensor) -> list[str]:
    '''onehot矩阵s->序列s'''
    seqs = [onehot2str(onehot) for onehot in onehots]
    return seqs


def crop_seq(seq: str, length: int, crop_method: str = 'center') -> str:
    assert length <= len(seq), 'crop length must <= sequence length'
    if crop_method == 'center':
        start = (len(seq) - length) // 2
    elif crop_method == 'left':
        start = 0
    elif crop_method == 'right':
        start = len(seq) - length
    elif crop_method == 'random':
        start = np.random.randint(0, len(seq) - length)
    else:
        raise ValueError('crop_method must be "center", "left", "right" or "random"')
    return seq[start: start + length]


def crop_seqs(seqs: list[str], length: int, crop_method: str = 'center') -> list[str]:
    return [crop_seq(seq, length, crop_method) for seq in seqs]


# def crop_onehot(onehot: np.ndarray, length: int) -> np.ndarray:
#     assert length < onehot.shape[0], 'crop length must >= sequence length'
#     start = (onehot.shape[0] - length) // 2
#     return onehot[start: start + length]


# def crop_onehots(onehots: np.ndarray, length: int) -> np.ndarray:
#     assert length < onehots.shape[1], 'crop length must >= sequence length'
#     start = (onehots.shape[1] - length) // 2
#     return onehots[:, start: start + length]


def pad_seq(seq: str, padded_length: int, padding_method='N', padding_postition='both', upstream_seq: str=None, downstream_seq: str=None) -> str:
    assert padded_length >= len(seq), 'padded_length must >= sequence length'
    padding_len = padded_length - len(seq)
    if padding_postition == 'both':
        left_len = padding_len // 2
        right_len = padding_len - left_len
    elif padding_postition == 'left':
        left_len = padding_len
        right_len = 0
    elif padding_postition == 'right':
        left_len = 0
        right_len = padding_len
    else:
        raise ValueError('padding_postition must be "both", "left", or "right"')

    if padding_method == 'N':
        upstream_seq = 'N' * left_len
        downstream_seq = 'N' * right_len
    elif padding_method == 'random':
        upstream_seq = "".join(np.random.choice(['A', 'C', 'G', 'T'], left_len))
        downstream_seq = "".join(np.random.choice(['A', 'C', 'G', 'T'], right_len))
    elif padding_method == 'given':
        upstream_seq = upstream_seq[-left_len:]
        downstream_seq = downstream_seq[:right_len]
    else:
        raise ValueError('padding_method must be "N", "random", or "given"')
    padded_seq = upstream_seq + seq + downstream_seq
    return padded_seq


def pad_seqs(seqs: list[str], padded_length: int, padding_method='N', upstream_seq: str=None, downstream_seq: str=None) -> list[str]:
    return [pad_seq(seq, padded_length, padding_method, upstream_seq, downstream_seq) for seq in seqs]


# def pad_onehot_N(onehot, padded_length, N_fill_value=0.25):
#     """
#     Pad onehot with N to a padded_length
#     onehot.shape = (seq_len, 4)
#     """
#     assert onehot.shape[1] == 4, 'onehot shape must be (seq_len, 4)'
#     seq = onehot2str(onehot)
#     padded_seq = pad_seq(seq, padded_length)
#     padded_onehot = str2onehot(padded_seq, N_fill_value=N_fill_value)
#     return padded_onehot


# def pad_onehots_N(onehots : np.ndarray, target_length, N_fill_value=0.25):
#     """
#     Pad onehots with N's to a target length.
#     onehots.shape = (batch_size, seq_len, 4)
#     axis corresponds to the seq_len axis
#     """
#     padded_onehots = [pad_onehot_N(onehot, target_length, N_fill_value) for onehot in onehots]
#     padded_onehots = np.stack(padded_onehots, axis=0)
#     return padded_onehots


def random_seq(length: int) -> str:
    return ''.join(np.random.choice(['A', 'C', 'G', 'T'], length))


def random_seqs(length: int, num: int) -> list[str]:
    return [random_seq(length) for _ in range(num)]


def random_onehot(length: int) -> np.ndarray:
    return str2onehot(random_seq(length))


def random_onehots(length: int, num: int) -> np.ndarray:
    return np.array([random_onehot(length) for _ in range(num)])
