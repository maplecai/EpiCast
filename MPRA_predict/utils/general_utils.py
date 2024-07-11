import os
import re
import sys
import yaml
import random
import pickle
import argparse
import logging
import logging.config
import numpy as np
import pandas as pd
import subprocess
import torch
import torch.nn as nn
import warnings

from tqdm import tqdm
from typing import Callable
from datetime import datetime
from torch.utils.data import DataLoader, Subset, random_split
from collections import Counter
from torchinfo import summary


def load_txt(file_dir: str) -> list:
    with open(file_dir, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def save_txt(file_dir: str, data: list) -> None:
    with open(file_dir, 'w') as f:
        for d in data:
            f.write(f"{d}\n")
    return

def load_pickle(file_dir: str):
    with open(file_dir, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(file_dir: str, data) -> None:
    with open(file_dir, 'wb') as f:
        data = pickle.dump(data, f)
    return


def init_obj(module, obj_dict:dict, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = init_obj(module, obj_dict, a, b=1)`
        is equivalent to
        `object = module.obj_dict['type'](a, b=1)`
        """
        if obj_dict is None:
            return None
        assert isinstance(obj_dict, dict), "invalid init object dict"
        module_name = obj_dict['type']
        module_args = dict(obj_dict.get('args', {}))
        for k in kwargs:
            if k in module_args:
                logging.debug(f'overwriting kwargs [{k}] in config')
        # assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        # config = yaml.safe_load(f)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['task_name'] = config_path.split('_', 1)[1].split('.')[0]
    return config

def process_config(config: dict) -> dict:
    task_name = config['task_name']
    # root_dir = config['root_dir']
    save_dir = config['save_dir']
    run_id = config.get('run_id', datetime.now().strftime(r'%m%d_%H%M%S')) 

    # make directory for saving checkpoints and log.
    save_dir = os.path.join(save_dir, task_name, run_id)
    # log_dir = os.path.join(save_dir, 'logs')
    log_dir = save_dir
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(log_dir, exist_ok=False)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # update config_dict after write it
    config['save_dir'] = save_dir
    config['log_dir'] = log_dir
    config['checkpoint_dir'] = checkpoint_dir

    # update logging
    loggingConfigDict = config['logger']
    for _, handler in loggingConfigDict['handlers'].items():
        if 'filename' in handler.keys():
            handler['filename'] = os.path.join(log_dir, handler['filename'])
    logging.config.dictConfig(loggingConfigDict)
    
    # save new config file
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(config))
    
    return config


def set_seed(seed:int = 42) -> None:
    '''
    设置随机数种子
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_nums_trainable_params(model:nn.Module) -> int:
    '''
    计算模型的可训练参数数量
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


# def load_and_trim_parameters_from_Sei(trained_model_path, output_channels_list=[]):
#     '''
#     加载Sei模型的参数到MLP模型中并选择部分channels作为输出
#     '''
#     state_dict = torch.load(trained_model_path) 
#     new_state_dict = {k.replace('module.model.', ''): v for k, v in state_dict.items()}

#     mlp_state_dict = {}
#     mlp_state_dict['fc1.weight'] = new_state_dict['classifier.0.weight']
#     mlp_state_dict['fc1.bias'] = new_state_dict['classifier.0.bias']
#     mlp_state_dict['fc2.weight'] = new_state_dict['classifier.2.weight']
#     mlp_state_dict['fc2.bias'] = new_state_dict['classifier.2.bias']

#     # # 20977, 21325 是 HepG2/K562 ENCODE DNase channel
#     # # 20988, 21326 是 HepG2/K562 Roadmap DNase channel
#     # # 4574,  10728 是 HepG2/K562 相关系数最高的 DNase channel
#     mlp_state_dict['fc2.weight'] = mlp_state_dict['fc2.weight'][output_channels_list]
#     mlp_state_dict['fc2.bias'] = mlp_state_dict['fc2.bias'][output_channels_list]

#     return mlp_state_dict


def get_free_gpu_id():
    # 执行nvidia-smi命令获取GPU状态
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True)
    memory_info = result.stdout.strip().split('\n')
    free_gpu_id = np.argmax([int(free_memory) for free_memory in memory_info])

    # index_free_memory = [re.split(r'\s*,\s*', info) for info in memory_info]
    # free_gpu_id = np.argmax([int(free_memory) for index, free_memory in index_free_memory])
    return free_gpu_id