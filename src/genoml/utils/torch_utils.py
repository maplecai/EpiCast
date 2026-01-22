import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torchinfo
import subprocess

from typing import Callable
from tqdm import tqdm


def to_device(data: list | tuple | dict | torch.Tensor, device: str | torch.device, non_blocking: bool = False):
    if isinstance(data, list):
        return [to_device(x, device, non_blocking=non_blocking) for x in data]
    elif isinstance(data, tuple):
        return tuple(to_device(x, device, non_blocking=non_blocking) for x in data)
    elif isinstance(data, dict):
        return {k: to_device(v, device, non_blocking=non_blocking) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    else:
        raise TypeError(f'data should be a list, tuple, dict or torch.Tensor, but got {type(data)}')

def dist_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    tensor_list = [torch.zeros_like(tensor, device=tensor.device) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list)
    return tensor_list


# def load_model(model: nn.Module, state_dict, strict=False) -> nn.Module:
#     if 'model_state_dict' in state_dict:
#         model_state_dict = state_dict['model_state_dict']
#     else:
#         model_state_dict = state_dict

#     if 'module' in model_state_dict.keys()[0]:
#         model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
#     # 去掉多卡训练前缀
#     first_key = next(iter(model_state_dict))
#     if first_key.startswith('module.'):
#         model_state_dict = {k.replace('module.', '', 1): v for k, v in model_state_dict.items()}

#     model_dict = model.state_dict()

#     # 只保留匹配的键
#     filtered_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

#     # 打印加载情况（可选）
#     missing_keys = model_dict.keys() - filtered_dict.keys()
#     unexpected_keys = model_state_dict.keys() - model_dict.keys()
#     print(f"Loaded params: {len(filtered_dict)}/{len(model_dict)}")
#     if missing_keys:
#         print(f"Missing keys: {list(missing_keys)[:5]} ...")
#     if unexpected_keys:
#         print(f"Unexpected keys: {list(unexpected_keys)[:5]} ...")

#     # 加载匹配部分
#     model_dict.update(filtered_dict)
#     model.load_state_dict(model_dict, strict=False)

#     return model

# def save_model(model: nn.Module, checkpoint_path: str) -> None:
#     model_state_dict = model.state_dict().copy()
#     model_state_dict = {
#         (k.replace('module.', '') if k.startswith('module.') else k): v
#         for k, v in model_state_dict.items()
#     }
#     torch.save(model_state_dict, checkpoint_path)
#     return


import subprocess
import numpy as np


def get_gpu_info_from_nvidia_smi():
    """Return list of (free_mem_MB, total_mem_MB) for each GPU."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.free,memory.total",
        "--format=csv,noheader,nounits"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    gpu_info = []

    for line in result.stdout.strip().split("\n"):
        free_mem, total_mem = map(float, line.split(","))
        gpu_info.append((free_mem, total_mem))

    return gpu_info


def get_free_gpus(min_memory_mb=40000):
    """Return GPU ids with free memory above the threshold, sorted by free memory descending."""
    gpu_info = get_gpu_info_from_nvidia_smi()

    # list of (gpu_id, free_mem)
    gpu_free = [(idx, info[0]) for idx, info in enumerate(gpu_info)]

    # sort by free memory desc
    gpu_free_sorted = sorted(gpu_free, key=lambda x: x[1], reverse=True)

    # filter and format
    return [f"cuda:{idx}" for idx, free_mem in gpu_free_sorted if free_mem > min_memory_mb]



def get_nums_trainable_params(model:nn.Module) -> int:
    '''
    计算模型的可训练参数数量
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params



class EarlyStopping:
    def __init__(
            self,
            monitor: str = None, 
            patience: int = 5, 
            delta: float = 0, 
            mode: str = 'min',
            saved_root_dir: str = './', 
            verbose: bool = False, 
            trace_func: Callable = print, 
            ):
        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.saved_root_dir = saved_root_dir
        self.verbose = verbose
        self.trace_func = trace_func
        
        self.save_path = f'{self.saved_root_dir}/checkpoint.pt'
        self.counter = 0
        self.stop_flag = False
        self.update_flag = False

        if self.mode == 'min':
            self.best_score = np.inf
        elif self.mode == 'max':
            self.best_score = -np.inf
        else:
            raise ValueError('mode should be either "min" or "max"')

    def check(self, score):
        if self.monitor is not None and type(score) == dict:
            score = score[self.monitor]


        print(score)
        if self.mode == 'min':
            self.update_flag = bool(score < self.best_score - self.delta)
        elif self.mode == 'max':
            self.update_flag = bool(score > self.best_score + self.delta)

        if self.update_flag is False:
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





import math
import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineAnnealingWarmRestarts(LRScheduler):
    """
    Warmup + CosineAnnealingWarmRestarts

    Args:
        optimizer: torch optimizer
        warmup_epochs (float): 线性 warmup 持续的 epoch 数
        T_0 (float): 第一个 cosine 周期长度（不含 warmup 部分）
        eta_min (float): 最小学习率
        T_mult (float): 周期长度倍率，和官方 CosineAnnealingWarmRestarts 一致
        last_epoch (int or float): 起始 epoch
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        T_0,
        eta_min=0.0,
        T_mult=1.0,
        last_epoch=-1,
    ):
        self.warmup_epochs = float(warmup_epochs)
        self.T_0 = float(T_0)
        self.T_mult = float(T_mult)
        self.eta_min = float(eta_min)

        if self.T_0 <= 0:
            raise ValueError("T_0 must be positive.")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0.")

        super().__init__(optimizer, last_epoch)

    # PyTorch 会在 step() 之后调用这个函数来拿当前 lr
    def get_lr(self):
        epoch = float(self.last_epoch)
        return self._compute_lr(epoch)

    def _compute_lr(self, epoch: float):
        # 1) Warmup phase
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            warmup_progress = (epoch + 1.0) / max(1.0, self.warmup_epochs)
            warmup_progress = max(0.0, min(warmup_progress, 1.0))
            return [base_lr * warmup_progress for base_lr in self.base_lrs]

        # 2) Cosine annealing + warm restarts
        t = max(epoch - self.warmup_epochs, 0.0)

        T_i = self.T_0

        if self.T_mult == 1.0:
            t_i = t % T_i
        else:
            # 周期递增：T_i, T_i*T_mult, T_i*T_mult^2, ...
            t_i = t
            while t_i >= T_i:
                t_i -= T_i
                T_i *= self.T_mult

        # 现在 t_i ∈ [0, T_i)
        # 标准 cosine 退火公式：从 base_lr → eta_min 再回到 eta_min
        cos_inner = math.pi * t_i / T_i

        return [
            self.eta_min + (base_lr - self.eta_min) * (1.0 + math.cos(cos_inner)) / 2.0
            for base_lr in self.base_lrs
        ]

