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

def seq_reverse_complement(seq: str):
    '''反向互补'''
    return reverse(complement(seq))

def onehots_reverse_complement(onehots):
    '''反向互补'''
    if isinstance(onehots, np.ndarray):
        onehots_rc = np.flip(onehots, axis=(-1,-2))
    elif isinstance(onehots, torch.Tensor):
        onehots_rc = torch.flip(onehots, dims=(-1,-2))
    else:
        raise ValueError('onehots must be a numpy array or a torch tensor.')
    return onehots_rc

# def str2onehot(seq: str, N_fill_value=0.25) -> np.ndarray:
#     '''
#     str -> onehot
#     N_fill_value: 碱基N对应的值 0.25(default)
#     '''
#     dic = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
#     onehot = np.zeros((len(seq), 4))
#     for i, c in enumerate(seq):
#         if c in ['A', 'C', 'G', 'T']:
#             onehot[i, dic[c]] = 1
#         elif c == 'N':
#             onehot[i, :] += N_fill_value
#     return onehot

# def strs2onehots(seqs: list[str], N_fill_value=0.25) -> np.ndarray:
#     '''序列s->onehot矩阵s'''
#     onehots = np.array([str2onehot(seq, N_fill_value) for seq in seqs])
#     return onehots

def str2onehot(seq: str, N_fill_value=0.25) -> np.ndarray:
    '''
    str -> onehot
    N_fill_value: 碱基N对应的值 0.25(default)
    '''
    # optimized version
    dic = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    onehot = np.zeros((len(seq), 4))
    seq_array = np.array(list(seq))
    for base, index in dic.items():
        onehot[seq_array == base, index] = 1
    onehot[seq_array == 'N', :] = N_fill_value
    return onehot

def strs2onehots(seqs: list[str], N_fill_value=0.25) -> np.ndarray:
    '''
    序列s->onehot矩阵s
    seqs: list of sequences (must be of the same length)
    N_fill_value: 碱基N对应的值 0.25(default)
    '''
    # 假设所有序列的长度相同
    seq_length = len(seqs[0])
    num_seqs = len(seqs)
    
    # 初始化一个onehot矩阵 [num_seqs, seq_length, 4]
    onehot = np.zeros((num_seqs, seq_length, 4))
    
    # 将序列转换为矩阵
    seqs_array = np.array([list(seq) for seq in seqs])
    
    # 创建字典，将 ACGT 映射到数值
    dic = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # 矢量化处理 ACGT
    for base, index in dic.items():
        onehot[seqs_array == base, :, index] = 1

    # 处理 N 的情况
    onehot[seqs_array == 'N', :] = N_fill_value
    
    return onehot



def onehot2str(onehot: np.ndarray) -> str:
    '''onehot矩阵->序列'''
    dic = {0:'A', 1:'C', 2:'G', 3:'T'}
    seq = ''
    onehot = np.array(onehot)
    num = np.argmax(onehot, axis=-1)
    for i in num:
        seq += dic[i]
    return seq

def onehots2strs(onehots: np.ndarray) -> list[str]:
    '''onehot矩阵s->序列s'''
    seqs = [onehot2str(onehot) for onehot in onehots]
    return seqs

def pad_seq(seq: str, padded_len: int, padding_mode='N', upstream_seq: str=None, downstream_seq: str=None) -> str:
    assert padded_len >= len(seq), 'Padded length must >= sequence length'
    padding_len = padded_len - len(seq)
    left_len = padding_len // 2
    right_len = padding_len - left_len

    if padding_mode == 'N':
        upstream_seq = 'N' * left_len
        downstream_seq = 'N' * right_len
    elif padding_mode == 'random':
        upstream_seq = "".join(np.random.choice(['A', 'C', 'G', 'T'], left_len))
        downstream_seq = "".join(np.random.choice(['A', 'C', 'G', 'T'], right_len))
    elif padding_mode == 'given':
        upstream_seq = upstream_seq[-left_len:]
        downstream_seq = downstream_seq[:right_len]
    else:
        raise ValueError('padding_mode must be "N", "random", or "given"')
    padded_seq = upstream_seq + seq + downstream_seq
    return padded_seq


def pad_onehot_N(onehot, padded_len, N_fill_value=0.25):
    """
    Pad onehot with N to a padded_len
    onehot.shape = (seq_len, 4)
    """
    assert onehot.shape[1] == 4, 'onehot shape must be (seq_len, 4)'
    seq = onehot2str(onehot)
    padded_seq = pad_seq(seq, padded_len)
    padded_onehot = str2onehot(padded_seq, N_fill_value=N_fill_value)
    return padded_onehot


    # if onehot.shape[0] == target_length:
    #     return onehot
    # elif onehot.shape[0] > target_length:
    #     mid_point = onehot.shape[0] // 2
    #     onehot = onehot[mid_point - target_length // 2: mid_point + target_length // 2, :]
    #     return onehot
    # else:
    #     total_pad_length = target_length - onehot.shape[0]
    #     left_pad_length = total_pad_length // 2
    #     right_pad_length = total_pad_length - left_pad_length
    #     if type(onehot) == np.ndarray:
    #         padded_onehot = np.zeros((target_length, 4))
    #     elif type(onehot) == torch.Tensor:
    #         padded_onehot = torch.zeros((target_length, 4))
    #     else:
    #         raise ValueError('onehot must be a numpy array or a torch tensor.')
    #     padded_onehot[:left_pad_length, :] = N_fill_value
    #     padded_onehot[left_pad_length: -right_pad_length, :] = onehot
    #     padded_onehot[-right_pad_length:, :] = N_fill_value
    #     return padded_onehot



def pad_onehots_N(onehots, target_length, N_fill_value=0.25):
    """
    Pad onehots with N's to a target length.
    onehots.shape = (batch_size, seq_len, 4)
    axis corresponds to the seq_len axis
    """
    padded_onehots = [pad_onehot_N(onehot, target_length, N_fill_value) for onehot in onehots]
    if isinstance(onehots, np.ndarray):
        padded_onehots = np.stack(padded_onehots, axis=0)
    elif isinstance(onehots, torch.Tensor):
        padded_onehots = torch.stack(padded_onehots, dim=0)
    else:
        raise ValueError('onehots must be a numpy array or a torch tensor.')
    return padded_onehots

