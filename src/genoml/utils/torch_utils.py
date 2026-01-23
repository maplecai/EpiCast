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

