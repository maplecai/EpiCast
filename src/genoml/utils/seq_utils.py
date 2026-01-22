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


def rc_onehot(onehot: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    '''反向互补'''
    if isinstance(onehot, np.ndarray):
        onehot = np.flip(onehot, axis=(-1,-2))
    elif isinstance(onehot, torch.Tensor):
        onehot = torch.flip(onehot, dims=(-1,-2))
    else:
        raise ValueError('onehot must be a numpy array or a torch tensor.')
    return onehot




# 1) 构建 256 x 4 查表
_LUT_ONEHOT = np.zeros((256, 4), dtype=np.float32)

# A/C/G/T 大小写
for i, base in enumerate(b"ACGT"):
    _LUT_ONEHOT[base, i] = 1.0
    _LUT_ONEHOT[base + 32, i] = 1.0

# N/n → 默认 0.25
for i, base in enumerate(b"N"):
    _LUT_ONEHOT[base, :] = 0.25
    _LUT_ONEHOT[base + 32, :] = 0.25


def seq2onehot(seq: str, N_fill_value: float = 0.25) -> np.ndarray:
    """seq str → onehot [L,4]"""
    b = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    onehot = _LUT_ONEHOT[b]

    # 覆盖 N 填充值
    if N_fill_value != 0.25:
        maskN = (b == 78) | (b == 110)  # N/n
        onehot[maskN] = N_fill_value

    return onehot


def seq2onehot_batch(seqs: list[str], N_fill_value: float = 0.25) -> np.ndarray:
    """
    批量 seq 列表 → onehot [B, L, 4]
    要求所有序列等长，否则请先 padding。
    """

    # ---------------------------
    # 1. 转为 bytes，并存入二维 uint8 array
    # ---------------------------
    B = len(seqs)
    L = len(seqs[0])

    # 将每条序列转成 uint8，堆叠成 [B, L]
    arr = np.empty((B, L), dtype=np.uint8)
    for i, seq in enumerate(seqs):
        arr[i] = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)

    # ---------------------------
    # 2. 查 LUT：结果 [B, L, 4]
    # ---------------------------
    onehot = _LUT_ONEHOT[arr].copy()  # 必须 copy，避免改 LUT

    # ---------------------------
    # 3. 覆盖 N 填充值
    # ---------------------------
    if N_fill_value != 0.25:
        maskN = (arr == 78) | (arr == 110)  # 'N' or 'n'
        onehot[maskN] = N_fill_value

    return onehot



# 全局：ACGT 字节 LUT
_BASE_CODES_NP = np.frombuffer(b"ACGT", dtype=np.uint8)
_N_CODE = np.uint8(ord('N'))

def onehot2seq(onehot: np.ndarray | torch.Tensor) -> str:
    """onehot [L,4] → seq str"""
    if isinstance(onehot, torch.Tensor):
        onehot = onehot.detach().cpu().numpy()

    # argmax + max
    idx = onehot.argmax(axis=1)
    maxv = onehot.max(axis=1)

    L = onehot.shape[0]
    codes = np.full(L, _N_CODE, dtype=np.uint8)

    valid = maxv > 0.5
    codes[valid] = _BASE_CODES_NP[idx[valid]]

    return codes.tobytes().decode("ascii")



def random_seq(length: int) -> str:
    bases = np.array(['A', 'C', 'G', 'T'])
    return ''.join(bases[np.random.randint(0, 4, length)])


def random_onehot(length: int) -> np.ndarray:
    return seq2onehot(random_seq(length))


def crop_seq(
        seq: str, 
        cropped_len: int, 
        crop_position: str = 'center'
    ) -> str:
    seq_length = len(seq)
    if len(seq) < cropped_len:
        print(f"{seq_length = }, {cropped_len = }, return the original seq")
        return seq

    if crop_position == 'center':
        start = (seq_length - cropped_len) // 2
    elif crop_position == 'left':
        start = 0
    elif crop_position == 'right':
        start = seq_length - cropped_len
    elif crop_position == 'random':
        start = np.random.randint(0, seq_length - cropped_len)
    elif crop_position.isdigit():
        start = int(crop_position)
    else:
        raise ValueError('crop_position must be "center", "left", "right" or "random"')
    cropped_seq = seq[start: start + cropped_len]
    return cropped_seq



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


def pad_seq(
        seq: str, 
        padded_len: int, 
        pad_mode: str = 'N', 
        pad_position: str = 'both_sides', 
        pad_left_seq: str = None, 
        pad_right_seq: str = None, 
        genome: Fasta=None
    ) -> str:

    if pad_mode == 'nothing':
        return seq

    if len(seq) > padded_len:
        print(f"Warning: input seq length = {len(seq)} is longer than padded length = {padded_len}, return the original seq")
        return seq

    padding_len = padded_len - len(seq)

    if pad_position == 'both_sides':
        left_len = padding_len // 2
        right_len = padding_len - left_len
    elif pad_position == 'left':
        left_len = padding_len
        right_len = 0
    elif pad_position == 'right':
        left_len = 0
        right_len = padding_len
    elif pad_position.isdigit():
        left_len = int(pad_position)
        right_len = padding_len - left_len
    elif pad_position == 'random':
        left_len = np.random.randint(0, padding_len)
        right_len = padding_len - left_len
    else:
        raise ValueError('padding_postition must be "both_sides", "left", "right" or a integer')

    if pad_mode == 'N':
        left_seq = 'N' * left_len
        right_seq = 'N' * right_len

    elif pad_mode == 'random':
        bases = np.array(['A', 'C', 'G', 'T'])
        left_seq = ''.join(bases[np.random.randint(0, 4, left_len)])
        right_seq = ''.join(bases[np.random.randint(0, 4, right_len)])

    elif pad_mode == 'random_genome':
        left_seq = random_genome_seq(genome, left_len) if left_len > 0 else ''
        right_seq = random_genome_seq(genome, right_len) if right_len > 0 else ''

    elif pad_mode == 'repeat':
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

    # elif pad_mode == 'given':
    #     if left_len <= 0:
    #         left_seq = ''
    #     else:
    #         if left_len <= len(pad_left_seq):
    #             left_seq = pad_left_seq[-left_len:]
    #         else:
    #             left_seq = 'N' * (left_len - len(pad_left_seq)) + pad_left_seq
        
    #     if right_len <= 0:
    #         right_seq = ''
    #     else:
    #         if right_len <= len(pad_right_seq):
    #             right_seq = pad_right_seq[:right_len]
    #         else:
    #             right_seq = pad_right_seq + 'N' * (right_len - len(pad_right_seq))

    elif pad_mode == 'given':
        if left_len == 0:
            left_seq = ''
        elif len(pad_left_seq) < left_len:
            left_seq = 'N' * (left_len - len(pad_left_seq)) + pad_left_seq
        else:
            left_seq = pad_left_seq[-left_len:]
        if right_len == 0:
            right_seq = ''
        elif len(pad_right_seq) < right_len:
            right_seq =  pad_right_seq + 'N' * (right_len - len(pad_right_seq))
        else:
            right_seq = pad_right_seq[:right_len]
        
    else:
        raise ValueError('pad_mode must be "N", "random", or "given"')
    
    padded_seq = ''.join([left_seq, seq, right_seq])
    return padded_seq




import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pyfaidx import Fasta
from multiprocessing import Lock

class GenomicInterval():
    def __init__(
        self,
        genome_path: str,
    ):
        
        self.lock = Lock()
        self.genome_path = genome_path
        self._genome = None

    # lazy load genome, for multiprocessing each process has its own copy of genome
    @property
    def genome(self):
        if self._genome is None:
            self._genome = Fasta(self.genome_path)
        return self._genome


    def get(self, chr, start, end):
        chromosome = self.genome[chr]
        
        # # adjust start and end to window_length
        # if (self.window_length is not None):
        #     mid = (start + end) // 2
        #     start = mid - self.window_length // 2
        #     end = start + self.window_length


        # padding N if outside the chromosome
        left_padding = 0
        if start < 0:
            left_padding = -start
            start = 0
        right_padding = 0
        if end > len(chromosome):
            right_padding = end - len(chromosome)
            end = len(chromosome)
        
        with self.lock:
            seq = chromosome[start:end].seq.upper()
        seq = ('N' * left_padding) + seq + ('N' * right_padding)

        return seq




import time
import numpy as np

if __name__ == "__main__":

    # 基准测试配置
    N = 10240
    L = 1024

    # 预生成同一批序列，保证两个方法对比公平
    seqs = [random_seq(L) for _ in range(N)]

    # 方法 1：逐个 seq 调用 seq2onehot 再 stack
    t0 = time.perf_counter()
    outs = np.stack([seq2onehot(s) for s in seqs])
    t1 = time.perf_counter()
    time_single = t1 - t0
    print("seqs_single shape:", outs.shape)
    print(f"逐个 seq2onehot + stack 用时: {time_single:.4f} 秒")

    # 方法 2：直接调用 batch 版本 seq2onehot_batch
    t0 = time.perf_counter()
    outs = seq2onehot_batch(seqs)
    t1 = time.perf_counter()
    time_batch = t1 - t0
    print("seqs_batch shape:", outs.shape)
    print(f"seq2onehot_batch 用时: {time_batch:.4f} 秒")



    # from joblib import Parallel, delayed
    # t0 = time.perf_counter()
    # outs = Parallel(n_jobs=8, backend="threading")(delayed(seq2onehot)(s) for s in seqs)
    # outs = np.stack(outs, axis=0)
    # t1 = time.perf_counter()
    # time_batch = t1 - t0
    # print("seqs_batch shape:", outs.shape)
    # print(f"joblib 用时: {time_batch:.4f} 秒")

