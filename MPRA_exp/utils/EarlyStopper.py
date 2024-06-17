import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Callable

class EarlyStopping:
    def __init__(
            self,
            monitor: str = None, 
            patience: int = 5, 
            delta: float = 0, 
            mode: str = 'min',
            save_dir: str = './', 
            verbose: bool = False, 
            trace_func: Callable = print, 
            ):
        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_dir = save_dir
        self.verbose = verbose
        self.trace_func = trace_func
        
        self.save_path = f'{self.save_dir}/checkpoint.pt'
        self.counter = 0
        self.stop_flag = False
        self.update_flag = False

        if self.mode =='min':
            self.best_score = np.inf
        elif self.mode =='max':
            self.best_score = -np.inf
        else:
            raise ValueError('mode should be either "min" or "max"')

    # def check(self, score, model=None, save=True):
    #     if self.monitor is not None:
    #         score = score[self.monitor]

    #     if self.mode =='min':
    #         self.update_flag = (score < self.best_score - self.delta)
    #     elif self.mode =='max':
    #         self.update_flag = (score > self.best_score + self.delta)

    #     if self.update_flag == False:
    #         self.counter += 1
    #         if self.verbose:
    #             self.trace_func(f'best score = {self.best_score}, {self.counter} out of {self.patience}')
    #         if self.counter >= self.patience:
    #             self.stop_flag = True
    #     else:
    #         if self.verbose:
    #             self.trace_func(f'best score changed ({self.best_score:.6f} --> {score:.6f}).')
    #         self.best_score = score
    #         self.counter = 0

    #         if save is True and model is not None:
    #             if os.path.isfile(self.save_path):
    #                 os.remove(self.save_path)
    #             self.save_path = f'{self.save_dir}/checkpoint_{score:.6f}.pt'
    #             torch.save(model.state_dict(), self.save_path)
    #             self.trace_func(f'save model to {self.save_path}')



    def check(self, score):
        if self.monitor is not None and type(score) == dict:
            score = score[self.monitor]

        if self.mode =='min':
            self.update_flag = (score < self.best_score - self.delta)
        elif self.mode =='max':
            self.update_flag = (score > self.best_score + self.delta)

        if self.update_flag == False:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'best score = {self.best_score:.6f}, round {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop_flag = True
        else:
            if self.verbose:
                self.trace_func(f'best score changed ({self.best_score:.6f} --> {score:.6f}).')
            self.best_score = score
            self.counter = 0
