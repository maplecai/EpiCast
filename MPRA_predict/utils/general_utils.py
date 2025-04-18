import os
import sys
import json
import pickle
import argparse
import logging
import logging.config
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from ruamel.yaml import YAML
yaml = YAML()
from icecream import ic

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 3)
pd.set_option('display.float_format', '{:.3f}'.format) 
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

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


def init_obj_2(module, class_name, *args, **kwargs):
    return getattr(module, class_name)(*args, **kwargs)


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
        config = yaml.load(f)
    config['task_name'] = config_path.split('_', 1)[1].split('.')[0]
    return config


def process_config(config: dict) -> dict:
    task_name = config['task_name']
    save_dir = config['save_dir']
    run_id = config.get('run_id', datetime.now().strftime(r'%m%d_%H%M%S')) 

    # make directory for saving checkpoints and log.
    save_dir = os.path.join(save_dir, task_name, run_id)
    os.makedirs(save_dir, exist_ok=True)

    # update config_dict after write it
    config['save_dir'] = save_dir

    # update logging
    loggingConfigDict = config['logger']
    for _, handler in loggingConfigDict['handlers'].items():
        if 'filename' in handler.keys():
            handler['filename'] = os.path.join(save_dir, handler['filename'])
    logging.config.dictConfig(loggingConfigDict)
    
    # save new config file
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    return config

def detect_delimiter(csv_file_path):
    with open(csv_file_path, 'r') as file:
        first_line = file.readline()
        if '\t' in first_line:
            return '\t'
        elif ',' in first_line:
            return ','
        else:
            raise ValueError('delimiter is not , or \t')