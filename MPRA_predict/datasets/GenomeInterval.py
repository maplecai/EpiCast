import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *
from pyfaidx import Fasta
from multiprocessing import Lock

class GenomeInterval():
    def __init__(
        self,
        genome_path: str,
        window_length: int = None,
        rc_aug: bool = False,
        shift_aug: bool = False,
        shift_aug_range: tuple[int, int] = None,
        return_aug_info: bool = False,
    ):
        self.lock = Lock()

        # self.genome = Fasta(genome_path)
        self.genome_path = genome_path
        self.window_length = window_length
        self.rc_aug = rc_aug
        self.shift_aug = shift_aug
        self.shift_aug_range = shift_aug_range
        self.return_aug_info = return_aug_info
        
        self._genome = None

    # lazy load genome, for multiprocessing each process has its own copy of genome
    @property
    def genome(self):
        if self._genome is None:
            self._genome = Fasta(self.genome_path)
        return self._genome


    def __call__(self, chr, start, end):
        chromosome = self.genome[chr]
        
        # adjust start and end to window_length
        if (self.window_length is not None):
            mid = (start + end) // 2
            start = mid - self.window_length // 2
            end = start + self.window_length

        # shift augmentation
        if self.shift_aug:
            min_shift, max_shift = self.shift_aug_range
            shift = np.random.randint(min_shift, max_shift + 1)
            start += shift
            end += shift
        else:
            shift = 0

        # padding if outside the chromosome
        left_padding = 0
        right_padding = 0
        if start < 0:
            left_padding = -start
            start = 0
        if end > len(chromosome):
            right_padding = end - len(chromosome)
            end = len(chromosome)

        # N means unknown base
        # . means outside the chromosome
        with self.lock:
            seq = chromosome[start:end].seq.upper()
        seq = ('N' * left_padding) + seq + ('N' * right_padding)

        # reverse complement augmentation
        if self.rc_aug and np.random.rand() < 0.5:
            seq = rc_seq(seq)
            rc = True
        else:
            rc = False

        if self.return_aug_info == False:
            return seq
        else:
            return seq, rc, shift
