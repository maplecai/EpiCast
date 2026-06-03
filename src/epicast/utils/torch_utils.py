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



def load_model(model: nn.Module, ckpt_path: str, strict: bool = True) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if "model_state_dict" in ckpt:
        ckpt_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        ckpt_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        ckpt_dict = ckpt["model"]
    else:
        ckpt_dict = ckpt
    
    ckpt_dict = {
        k.removeprefix("module."): v
        for k, v in ckpt_dict.items()
    }

    if strict:
        model.load_state_dict(ckpt_dict, strict=True)
        print(f"Strictly loaded {len(ckpt_dict)} keys.")
        return model

    else:
        model_dict = model.state_dict()

        matched_dict = {}
        missing_keys = []
        extra_keys = []
        shape_mismatch_keys = []

        for k, v in ckpt_dict.items():
            if k not in model_dict:
                extra_keys.append(k)
                continue

            if v.shape != model_dict[k].shape:
                shape_mismatch_keys.append(
                    (k, tuple(v.shape), tuple(model_dict[k].shape))
                )
                continue

            matched_dict[k] = v

        for k in model_dict.keys():
            if k not in matched_dict:
                missing_keys.append(k)

        model.load_state_dict(matched_dict, strict=False)

        print(f"Model keys: {len(model_dict)}")
        print(f"Checkpoint keys: {len(ckpt_dict)}")
        print(f"Matched keys: {len(matched_dict)}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Extra keys in checkpoint: {len(extra_keys)}")
        print(f"Shape mismatch keys: {len(shape_mismatch_keys)}")

        return model



def save_model(model: nn.Module, ckpt_path: str) -> None:
    model_dict = model.state_dict().copy()
    model_dict = {
        k.removeprefix('module.', ''): v
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

    free_gpus = []
    for idx, info in enumerate(gpu_info):
        free_mem = info[0]   # assumes info[0] is free memory in MB
        if free_mem >= min_memory_mb:
            free_gpus.append((idx, free_mem))

    free_gpus.sort(key=lambda x: x[1], reverse=True)
    return [f"cuda:{idx}" for idx, _ in free_gpus]


def get_nums_trainable_params(model:nn.Module) -> int:
    '''
    计算模型的可训练参数数量
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

