import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyfaidx import Fasta

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


# def str2onehot(seq: str, N_fill_value: int = 0.25, dtype=torch.Tensor) -> np.ndarray | torch.Tensor:
#     '''
#     str -> onehot
#     N_fill_value: 碱基N对应的值 0.25(default)
#     '''
#     base2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4,
#                 'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4,}
#     seq = np.array(list(seq), dtype='U1')
#     indices = np.array([base2idx[b] for b in seq], dtype=int)
#     onehot = np.zeros((len(seq), 4))
#     # 对N的位置整行赋值N_fill_value
#     N_pos = (indices == 4)
#     onehot[N_pos, :] = N_fill_value
#     # 非N的位置：直接用整型索引赋值为1
#     not_N_pos = ~N_pos
#     onehot[np.arange(len(seq))[not_N_pos], indices[not_N_pos]] = 1
#     return onehot


_LUT = np.full(256, 5, dtype=np.uint8)
for a,i in zip(b"ACGTNacgtn", [0,1,2,3,4,0,1,2,3,4]):
    _LUT[a] = i

def str2onehot(seq: str, N_fill_value: int = 0.25) -> np.ndarray:
    b = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)  # O(1) 视图
    idx = _LUT[b]                                   # [L]，0..4
    L = idx.size
    
    onehot = np.full((L, 4), 0, dtype=np.float32)  # 先全填 0
    notN = (idx < 4) # 只对 ACGT 做 one-hot
    onehot[notN] = np.eye(4, dtype=np.float32)[idx[notN]]
    isN = (idx == 4) # 对 N 做 N fill
    onehot[isN] = N_fill_value
    return onehot


def strs2onehots(seqs: list[str], N_fill_value: int = 0.25) -> np.ndarray:
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



def random_seq(length: int) -> str:
    bases = np.array(['A', 'C', 'G', 'T'])
    return ''.join(bases[np.random.randint(0, 4, length)])


def random_seqs(length: int, num: int) -> list[str]:
    return [random_seq(length) for _ in range(num)]


def random_onehot(length: int) -> np.ndarray:
    return str2onehot(random_seq(length))


def random_onehots(length: int, num: int) -> np.ndarray:
    return np.array([random_onehot(length) for _ in range(num)])



def crop_seq(seq: str, length: int, crop_position: str = 'center') -> str:
    if len(seq) < length:
        print(f"Warning: input seq length = {len(seq)} is shorter than crop length = {length}, return the original seq")
        return seq

    if crop_position == 'center':
        start = (len(seq) - length) // 2
    elif crop_position == 'left':
        start = 0
    elif crop_position == 'right':
        start = len(seq) - length
    elif crop_position == 'random':
        start = np.random.randint(0, len(seq) - length)
    elif crop_position.isdigit():
        start = int(crop_position)
    else:
        raise ValueError('crop_position must be "center", "left", "right" or "random"')
    cropped_seq = seq[start: start + length]
    return cropped_seq


def crop_seqs(seqs: list[str], length: int, crop_position: str = 'center') -> list[str]:
    return [crop_seq(seq, length, crop_position) for seq in seqs]


# def crop_onehot(onehot: np.ndarray, length: int) -> np.ndarray:
#     assert length < onehot.shape[0], 'crop length must >= sequence length'
#     start = (onehot.shape[0] - length) // 2
#     return onehot[start: start + length]


# def crop_onehots(onehots: np.ndarray, length: int) -> np.ndarray:
#     assert length < onehots.shape[1], 'crop length must >= sequence length'
#     start = (onehots.shape[1] - length) // 2
#     return onehots[:, start: start + length]

def random_genome_seq(genome: Fasta, seq_length: int) -> str:
    if seq_length <= 0:
        raise ValueError('random_genome_seq length must > 0')
    chrom_list = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    chrom = np.random.choice(chrom_list)
    chrom_len = len(genome[chrom])
    
    start = np.random.randint(0, chrom_len - seq_length + 1)
    end = start + seq_length
    seq = str(genome[chrom][start:end]).upper()
    if '>' in seq:
        print(f"Found > in genome[{chrom}][{start}:{end}]: {seq}")
    return seq



def pad_seq(seq: str, padded_length: int, padding_method:str ='N', padding_position: str='both_sides', given_left_seq: str=None, given_right_seq: str=None, genome: Fasta=None) -> str:

    if len(seq) > padded_length:
        print(f"Warning: input seq length = {len(seq)} is longer than padded length = {padded_length}, return the original seq")
        return seq

    padding_len = padded_length - len(seq)

    if padding_position == 'both_sides':
        left_len = padding_len // 2
        right_len = padding_len - left_len
    elif padding_position == 'left':
        left_len = padding_len
        right_len = 0
    elif padding_position == 'right':
        left_len = 0
        right_len = padding_len
    elif padding_position.isdigit():
        left_len = int(padding_position)
        right_len = padding_len - left_len
    elif padding_position == 'random':
        left_len = np.random.randint(0, padding_len)
        right_len = padding_len - left_len
    else:
        raise ValueError('padding_postition must be "both_sides", "left", "right" or a integer')

    if padding_method == 'N':
        left_seq = 'N' * left_len
        right_seq = 'N' * right_len

    elif padding_method == 'random':
        bases = np.array(['A', 'C', 'G', 'T'])
        left_seq = ''.join(bases[np.random.randint(0, 4, left_len)])
        right_seq = ''.join(bases[np.random.randint(0, 4, right_len)])

    elif padding_method == 'genome':
        left_seq = random_genome_seq(genome, left_len) if left_len > 0 else ''
        right_seq = random_genome_seq(genome, right_len) if right_len > 0 else ''

    elif padding_method == 'repeat':
        if left_len == 0:
            left_seq = ''
        else:
            repeats_needed = left_len // len(seq) + 1
            repeated_seq = seq * repeats_needed
            left_seq = repeated_seq[-left_len:]
        if right_len == 0:
            right_seq = ''
        else:
            repeats_needed = right_len // len(seq) + 1
            repeated_seq = seq * repeats_needed
            right_seq = repeated_seq[:right_len]

    elif padding_method == 'given':
        if len(given_left_seq) < left_len or len(given_right_seq) < right_len:
            raise ValueError('given_left_seq and given_right_seq must be at least as long as the padding length')
        left_seq = given_left_seq[-left_len:] if left_len > 0 else ''
        right_seq = given_right_seq[:right_len] if right_len > 0 else ''

    elif padding_method == 'given+N':
        if left_len == 0:
            left_seq = ''
        elif len(given_left_seq) < left_len:
            left_seq = 'N' * (left_len - len(given_left_seq)) + given_left_seq
        else:
            left_seq = given_left_seq[-left_len:]
        if right_len == 0:
            right_seq = ''
        elif len(given_right_seq) < right_len:
            right_seq =  given_right_seq + 'N' * (right_len - len(given_right_seq))
        else:
            right_seq = given_right_seq[:right_len]
        
    else:
        raise ValueError('padding_method must be "N", "random", or "given"')
    
    padded_seq = ''.join([left_seq, seq, right_seq])
    return padded_seq


def pad_seqs(seqs: list[str], *args, **kwargs) -> list[str]:
    return [pad_seq(seq, *args, **kwargs) for seq in seqs]



def crop_or_pad_seq(seq: str, target_length: int, **kwargs) -> str:
    if len(seq) > target_length:
        return crop_seq(seq, target_length, **kwargs)
    elif len(seq) < target_length:
        return pad_seq(seq, target_length, **kwargs)
    else:
        return seq



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
