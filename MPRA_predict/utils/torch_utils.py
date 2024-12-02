import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        return data.to(device)


def dist_all_gather(tensor):
    tensor_list = [torch.zeros_like(tensor, device=tensor.device) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list)
    return tensor_list


def load_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    if 'model_state_dict' in state_dict:
        model_state_dict = state_dict['model_state_dict']
    else:
        model_state_dict = state_dict
    if 'module' in model_state_dict.keys()[0]:
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    return model


def get_free_gpu_ids(min_memory_mb=40000):
    """Return a list of GPU ids with more than min_memory MB free memory."""
    free_memorys = []
    for i in range(torch.cuda.device_count()):
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        free_memory_mb = free_memory / (1024 ** 2)  # Convert to MB
        free_memorys.append(free_memory_mb)
    
    free_gpus = [i for i in range(len(free_memorys)) if free_memorys[i] > min_memory_mb]
    return free_gpus


def get_free_gpu_id():
    free_memorys = []
    for i in range(torch.cuda.device_count()):
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        free_memory_mb = free_memory / (1024 ** 2)  # Convert to MB
        free_memorys.append(free_memory_mb)
    free_gpu_id = np.argmax(free_memorys)
    return free_gpu_id


def get_nums_trainable_params(model:nn.Module) -> int:
    '''
    计算模型的可训练参数数量
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


# def load_partial_parameters(target_model, source_model_path, prefix_list=None, print_func=print):
#     """
#     Load parameters with specific prefix from a source model file into a target model.

#     Args:
#         target_model (torch.nn.Module): The target model instance to initialize.
#         source_model_path (str): Path to the source model's state_dict.
#         prefix_list (list of str, optional): List of prefixes to filter parameters to load. Default is None, which loads all common parameters.
#         print_func (callable, optional): Function to print log messages. Default is print.
#     """

#     # Load source model parameters
#     source_state_dict = torch.load(source_model_path)
    
#     # Get target model parameters
#     target_state_dict = target_model.state_dict()
    
#     # Initialize parameters
#     common_params = {k: v for k, v in source_state_dict.items() 
#                      if k in target_state_dict and v.size() == target_state_dict[k].size()}
    
#     if prefix_list is None:
#         new_state_dict = common_params
#     else:
#         new_state_dict = {}
#         for k, v in common_params.items():
#             for prefix in prefix_list:
#                 if k.startswith(prefix):
#                     new_state_dict[k] = v
#                     break

#         if new_state_dict:
#             print_func(f'Loading parameters: {list(new_state_dict.keys())} from {source_model_path}')
#         else:
#             print_func(f'No matching parameters found with prefixes {prefix_list}')
        
#     # Update target state dict with the new parameters
#     target_state_dict.update(new_state_dict)
    
#     # Load the updated state dict into the target model
#     target_model.load_state_dict(target_state_dict)
