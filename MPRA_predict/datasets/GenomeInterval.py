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
    ):
        
        self.lock = Lock()
        self.genome_path = genome_path
        self.window_length = window_length
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

        # padding if outside the chromosome
        left_padding = 0
        if start < 0:
            left_padding = -start
            start = 0
        right_padding = 0
        if end > len(chromosome):
            right_padding = end - len(chromosome)
            end = len(chromosome)

        # N means unknown base, . means outside the chromosome
        with self.lock:
            seq = chromosome[start:end].seq.upper()
        seq = ('N' * left_padding) + seq + ('N' * right_padding)

        return seq
