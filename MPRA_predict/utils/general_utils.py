import os
import re
import sys
import json
import h5py
import pickle
import argparse
import logging
import logging.config
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from ruamel.yaml import YAML
yaml = YAML()
from icecream import ic
from collections import Counter


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 40)
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

def load_h5(file_dir: str, key: str=None):
    with h5py.File(file_dir, 'r') as f:
        if key:
            data = f[key][:]
        
        else:
            keys = list(f.keys())
            print(f'file {file_dir} has keys: {keys}')
            if len(keys) == 1:
                data = f[keys[0]][:]
            else:
                print('file has more than one key')
                data = {key: f[key][:] for key in keys}
    return data

def save_h5(file_dir: str, data) -> None:
    with h5py.File(file_dir, 'w') as f:
        f.create_dataset('data', data=data)
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
            return ','
            # raise ValueError('delimiter is not , or \t')







class HDF5Writer:
    def __init__(self, file_path, dataset_name='data', data_shape=None, max_samples=None, chunk_size=1024, dtype="float32", compression="gzip"):
        """
        HDF5 增量写入工具
        Args:
            file_path (str): HDF5 文件路径
            dataset_name (str): 数据集名称
            data_shape (tuple): 单个样本的形状，例如 (2048, 305)，可为 None（自动推断）
            max_samples (int, optional): 最大样本数（None 表示动态增长）
            chunk_size (int): 每块的 batch 大小
            dtype (str): 数据类型
            compression (str): 压缩方式，可为 None/gzip/lzf
        """
        self.file = h5py.File(file_path, "a")
        self.dataset_name = dataset_name
        self.data_shape = data_shape
        self.max_samples = max_samples
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.compression = compression

        # 如果已存在，直接复用
        if dataset_name in self.file:
            self.dset = self.file[dataset_name]
            self.index = self.dset.shape[0] if max_samples is None else 0
        # 如果没有但给了 data_shape，则立即创建
        elif self.data_shape is not None:
            self._create_dataset(self.data_shape)
        # 否则延迟到第一次 append 才建
        else:
            self.dset = None
            self.index = 0

    def _create_dataset(self, data_shape):
        """内部函数：根据 data_shape 创建 dataset"""
        self.data_shape = data_shape
        if self.max_samples is None:
            dset = self.file.create_dataset(
                self.dataset_name,
                shape=(0,) + data_shape,
                maxshape=(None,) + data_shape,
                dtype=self.dtype,
                chunks=(self.chunk_size,) + data_shape,
                compression=self.compression
            )
        else:
            dset = self.file.create_dataset(
                self.dataset_name,
                shape=(self.max_samples,) + data_shape,
                dtype=self.dtype,
                chunks=(self.chunk_size,) + data_shape,
                compression=self.compression
            )
        self.dset = dset
        self.index = 0

    def append(self, batch):
        """追加写入 batch 数据"""
        batch = np.asarray(batch, dtype=self.dtype)
        n = batch.shape[0]

        # 如果 dataset 还没建，自动推断 shape
        if self.dset is None:
            self._create_dataset(batch.shape[1:])

        if self.max_samples is None:
            self.dset.resize(self.index + n, axis=0)
        elif self.index + n > self.max_samples:
            raise ValueError("超过最大样本数！")

        self.dset[self.index:self.index+n, ...] = batch
        self.index += n
        self.file.flush()

    def __len__(self):
        return self.index
    

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()







# class H5BatchWriter:
#     """
#     逐批追加写 HDF5：
#       >>> writer = H5BatchWriter("pred.h5", "pred", dtype=np.float16)
#       >>> writersave(batch_pred)
#       >>> writer.close()              # 用完记得关
#     """
#     def __init__(
#             self, 
#             path, 
#             dset_name='data',
#             dtype=np.float32, # torch.float32
#             compression=None
#         ):
#         self.f           = h5py.File(path, "w")
#         self.dset        = None
#         self.dset_name   = dset_name
#         self.dtype       = dtype
#         self.compression = compression
#         self.offset      = 0          # 写入样本计数

#     def save(self, batch_arr):
#         batch_arr = batch_arr.astype(self.dtype, copy=False)
#         bsz       = len(batch_arr)

#         # 第一次来时创建数据集（可无限增长）
#         if self.dset is None:
#             full_shape = (None,) + batch_arr.shape[1:]
#             self.dset  = self.f.create_dataset(
#                 self.dset_name, shape=(0,) + batch_arr.shape[1:],
#                 maxshape=full_shape, chunks=True,
#                 compression=self.compression, dtype=self.dtype)

#         # 扩容 + 写入
#         new_size = self.offset + bsz
#         self.dset.resize(new_size, axis=0)
#         self.dset[self.offset:new_size] = batch_arr
#         self.offset = new_size

#     def flush(self):
#         self.f.flush()

#     def close(self):
#         self.f.flush()
#         self.f.close()