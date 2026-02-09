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



def load_model(model: nn.Module, ckpt: dict) -> nn.Module:
    if 'model_state_dict' in ckpt:
        ckpt_dict = ckpt['model_state_dict']
    elif 'model' in ckpt:
        ckpt_dict = ckpt['model']
    else:
        ckpt_dict = ckpt

    # 去掉多卡训练前缀 module
    ckpt_dict = {(k.replace('module.', '') if k.startswith("module.") else k): v for k, v in ckpt_dict.items()}
    
    model_dict = model.state_dict()

    # 只保留匹配的键和形状
    matched_dict = {k: v for k, v in ckpt_dict.items() 
        if k in model_dict and v.size() == model_dict[k].size()}
    
    missing_keys = model_dict.keys() - matched_dict.keys()
    extra_keys = ckpt_dict.keys() - model_dict.keys()

    print(f"number of matched keys: {len(matched_dict)}")
    if missing_keys:
        print(f"number of missing keys: {len(missing_keys)}, {list(missing_keys)} ...")
    if extra_keys:
        print(f"number of extra keys: {len(extra_keys)}, {list(extra_keys)} ...")

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)

    return model


def save_model(model: nn.Module, ckpt_path: str) -> None:
    model_dict = model.state_dict().copy()
    model_dict = {
        (k.replace('module.', '') if k.startswith('module.') else k): v
        for k, v in model_dict.items()
    }
    torch.save(model_dict, ckpt_path)
    return





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


def get_free_gpus(min_memory_mb=20480):
    """Return GPU ids with free memory above the threshold, sorted by free memory descending."""
    gpu_info = get_gpu_info_from_nvidia_smi()

    # list of (gpu_id, free_mem)
    gpus = [(idx, info[0]) for idx, info in enumerate(gpu_info)]

    free_gpus = [gpu for gpu in gpus if gpu[1] > min_memory_mb]

    # sort by free memory desc
    free_gpus_sorted = sorted(free_gpus, key=lambda x: x[1], reverse=True)

    return [f"cuda:{gpu[0]}" for gpu in free_gpus_sorted]



def get_nums_trainable_params(model:nn.Module) -> int:
    '''
    计算模型的可训练参数数量
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

