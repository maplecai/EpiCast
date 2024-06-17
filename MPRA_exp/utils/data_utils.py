import os
import yaml
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse_func

# MPRA_UPSTREAM = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
# MPRA_DOWNSTREAM = 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'

def load_txt(file_dir: str) -> list:
    with open(file_dir, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def save_txt(file_dir: str, data: list) -> None:
    with open(file_dir, 'w') as f:
        for d in data:
            f.write(f"{d}\n")
    return

def load_pickle(file_dir: str):
    with open(file_dir, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(file_dir: str, data):
    with open(file_dir, 'wb') as f:
        data = pickle.dump(data, f)
    return data

def reverse(seq: str):
    '''反向'''
    return seq[::-1]

def complement(seq: str):
    '''互补'''
    dic = {
        'A':'T', 'C':'G', 'G':'C', 'T':'A', 'N':'N',
        'a':'t', 'c':'g', 'g':'c', 't':'a', 'n':'n',
        }
    seq1 = ''
    for c in seq:
        seq1 += dic[c]
    return seq1

def seq_reverse_complement(seq: str):
    '''反向互补'''
    return reverse(complement(seq))

def onehots_reverse_complement(onehots: np.ndarray):
    '''反向互补'''
    if isinstance(onehots, np.ndarray):
        onehots_rc = onehots[..., ::-1, ::-1].copy()
    elif isinstance(onehots, torch.Tensor):
        onehots_rc = onehots.flip(-2, -1)
    else:
        raise ValueError('onehots must be a numpy array or a torch tensor.')
    return onehots_rc

def str2onehot(seq: str, N_fill_value=0.25, dot_fill_value=0) -> np.ndarray:
    '''序列->onehot矩阵
    N: 碱基N对应的值 0.25(default) 或 0
    '''
    dic = {'A':0, 'a':0, 'C':1, 'c':1, 'G':2, 'g':2, 'T':3, 't':3, 'N':4, 'n':4, '.': 5}
    onehot = np.zeros((len(seq), 6))
    for i, c in enumerate(seq):
        onehot[i, dic[c]] = 1
    onehot = onehot + onehot[i, 4] * N_fill_value + onehot[i, 5] * dot_fill_value
    onehot = onehot[:, :4]
    return onehot

def strs2onehots(seqs: list[str], N_fill_value=0.25, dot_fill_value=0) -> np.ndarray:
    '''序列s->onehot矩阵s'''
    onehots = np.array([str2onehot(seq, N_fill_value, dot_fill_value) for seq in seqs])
    return onehots

def onehot2str(onehot: np.ndarray) -> str:
    '''onehot矩阵->序列'''
    dic = {0:'A', 1:'C', 2:'G', 3:'T', 4:'N', 5:'.'}
    seq = ''
    num = np.argmax(onehot, axis=-1)
    for i in num:
        seq += dic[i]
    return seq

def onehots2strs(onehots: np.ndarray) -> str:
    '''onehot矩阵s->序列s'''
    seqs = [onehot2str(onehot) for onehot in onehots]
    return seqs


def pad_seq(seq, padded_len, upstream_seq=None, downstream_seq=None):
    assert padded_len >= len(seq), 'Padded length must >= sequence length'
    padding_len = padded_len - len(seq)
    left_len = padding_len // 2
    right_len = padding_len - left_len
    if upstream_seq == None and downstream_seq == None:
        padded_seq = 'N' * left_len + seq + 'N' * right_len
    else:
        padded_seq = upstream_seq[-left_len:] + seq + downstream_seq[:right_len]
    return padded_seq

# def pad_seqs(seq, target_length):
#     padded_seqs = [pad_seq_N(s, target_length) for s in seq]
#     return padded_seqs


# def pad_seq_given(seq, padded_len, upstream_seq, downstream_seq):
#     padding_len = padded_len - len(seq)
#     assert padding_len <= (len(upstream_seq) + len(downstream_seq)), 'Not enough padding available'
#     left_len = padding_len // 2
#     right_len = padding_len - left_len
#     up_pad_seq = upstream_seq[-left_len:]
#     down_pad_seq = downstream_seq[:right_len]
#     padded_seq = up_pad_seq + seq + down_pad_seq
#     return padded_seq

# def pad_seq_N(seq, padded_len):
#     if padded_len == len(seq):
#         padded_seq = seq
#     elif padded_len > len(seq):
#         padding_len = padded_len - len(seq)
#         left_len = padding_len // 2
#         right_len = padding_len - left_len
#         padded_seq = 'N' * left_len + seq + 'N' * right_len
#     # elif padded_len < len(seq):
#     #     mid_point = len(seq) // 2
#     #     left_len = padded_len // 2
#     #     right_len = padded_len - left_len
#     #     padded_seq = seq[mid_point - left_len: mid_point + right_len]
        
#     return padded_seq

# def pad_seqs_N(seq, target_length):
#     padded_seqs = [pad_seq_N(s, target_length) for s in seq]
#     return padded_seqs

def pad_onehot_N(onehot, target_length, N_fill_value=0.25):
    """
    Pad onehot with N's to a target length.
    onehot.shape = (seq_len, 4)
    """
    if onehot.shape[0] == 4 and onehot.shape[1] != 4:
        onehot = onehot.T

    if onehot.shape[0] == target_length:
        return onehot
    elif onehot.shape[0] > target_length:
        mid_point = onehot.shape[0] // 2
        onehot = onehot[mid_point - target_length // 2: mid_point + target_length // 2, :]
        return onehot
    else:
        total_pad_length = target_length - onehot.shape[0]
        left_pad_length = total_pad_length // 2
        right_pad_length = total_pad_length - left_pad_length
        if type(onehot) == np.ndarray:
            padded_onehot = np.zeros((target_length, 4))
        elif type(onehot) == torch.Tensor:
            padded_onehot = torch.zeros((target_length, 4))
        else:
            raise ValueError('onehot must be a numpy array or a torch tensor.')
        padded_onehot[:left_pad_length, :] = N_fill_value
        padded_onehot[left_pad_length: -right_pad_length, :] = onehot
        padded_onehot[-right_pad_length:, :] = N_fill_value
        return padded_onehot



def pad_onehots_N(onehots, target_length, N_fill_value=0.25):
    """
    Pad onehots with N's to a target length.
    onehots.shape = (batch_size, seq_len, 4)
    axis corresponds to the seq_len axis
    """
    padded_onehots = [pad_onehot_N(onehot, target_length, N_fill_value) for onehot in onehots]
    if type(onehots) == np.ndarray:
        padded_onehots = np.stack(padded_onehots, axis=0)
    elif type(onehots) == torch.Tensor:
        padded_onehots = torch.stack(padded_onehots, dim=0)
    else:
        raise ValueError('onehots must be a numpy array or a torch tensor.')
    return padded_onehots



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