import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchinfo
from typing import Callable
from tqdm import tqdm
from .seq_utils import *
import subprocess

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise TypeError(f'data should be a list, tuple, dict or torch.Tensor, but got {type(data)}')


def dist_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    tensor_list = [torch.zeros_like(tensor, device=tensor.device) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list)
    return tensor_list


# def load_model(model: nn.Module, checkpoint_path: str) -> nn.Module:
#     state_dict = torch.load(checkpoint_path)
#     if 'model_state_dict' in state_dict:
#         model_state_dict = state_dict['model_state_dict']
#     else:
#         model_state_dict = state_dict
#     if 'module' in model_state_dict.keys()[0]:
#         model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
#     model.load_state_dict(model_state_dict)
#     return model



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







def save_model(model: nn.Module, checkpoint_path: str) -> None:
    # checkpoint = {
    #     'config': self.config,
    #     'epoch': self.epoch,
    #     'model_state_dict': self.model.state_dict(),
    #     'optimizer_state_dict': self.optimizer.state_dict(),
    #     }
    # checkpoint_path = os.path.join(checkpoint_dir, f'epoch{self.epoch}.pth')
    model_state_dict = model.state_dict().copy()
    model_state_dict = {
        (k.replace('module.', '') if k.startswith('module.') else k): v
        for k, v in model_state_dict.items()
    }
    torch.save(model_state_dict, checkpoint_path)
    return


def get_free_gpu_ids(min_memory_mb=40000):
    free_list, _ = get_gpu_info_from_nvidia_smi()
    gpu_ids = np.argsort(free_list)[::-1]

    if gpu_ids[0] < min_memory_mb:
        print(f'max free memory = {free_list[gpu_ids[0]]}MB, using gpu {gpu_ids[0]}')

    # free_gpu_ids = [i for i in range(len(free_list)) if free_list[i] > min_memory_mb]
    return gpu_ids


def get_gpu_info_from_nvidia_smi():
    # 执行nvidia-smi命令并读出结果
    cmd = "nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits"
    output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split('\n')
    # 每行形如 "37278, 40960"
    free_list, total_list = [], []
    for line in output:
        free_str, total_str = line.split(',')
        free_list.append(float(free_str))
        total_list.append(float(total_str))
    return free_list, total_list







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





from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts

class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_epochs, T_0, T_mult=1, eta_min=0):
        # super().__init__(optimizer)
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )
        self.warmup_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min(epoch / self.warmup_epochs, 1.0)
        )
        
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch < self.warmup_epochs:
            self.warmup_scheduler.step(epoch)
        else:
            self.cosine_scheduler.step(epoch - self.warmup_epochs)
