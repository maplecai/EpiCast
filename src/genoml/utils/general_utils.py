import os
import h5py
import logging
import logging.config
import numpy as np
import pandas as pd
import random
import torch
import inspect

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from ruamel.yaml import YAML
yaml = YAML()
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
from hydra.core.hydra_config import HydraConfig

def set_mpl_params():
    mpl_params = {
        # 字体参数
        'font.family': 'Arial',
        'font.size': 12,
        # 数学文本参数
        'mathtext.fontset': 'stix', 
        # 图像参数
        'figure.dpi': 100,
        'figure.figsize': (8, 6),
        # 保存pdf字体可编辑
        'pdf.fonttype': 42,
    }
    plt.rcParams.update(mpl_params)
set_mpl_params()

sns.set_theme(context="talk", style="whitegrid")

def set_print_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 3)
    pd.set_option('display.float_format', '{:.3f}'.format) 
    np.set_printoptions(linewidth=1000, precision=3, formatter={'float': '{: 0.3f}'.format})
set_print_options()

def set_seed(seed:int = 42) -> None:
    '''
    设置随机数种子
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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



def init_obj(module, obj_dict: dict, *args, **kwargs):
    if not obj_dict:
        return None
    if not isinstance(obj_dict, dict):
        raise TypeError("invalid init object dict")

    name = obj_dict["type"]
    module_args = {**obj_dict.get("args", {}), **kwargs}

    if module is None:
        cls = globals()[name]
    else:
        cls = getattr(module, name)

    # --- 新增: 如果是函数，直接返回函数本身 ---
    if inspect.isfunction(cls):
        return cls

    # --- 原逻辑：类实例化 ---
    return cls(*args, **module_args)



def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)
    cfg['config_name'] = os.path.basename(config_path)
    return cfg


def process_config(cfg: dict) -> dict:
    config_name = cfg['config_name']
    saved_root_dir = cfg['saved_root_dir']
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')

    saved_dir = os.path.join(saved_root_dir, config_name, run_id)
    os.makedirs(saved_dir, exist_ok=True)
    cfg['saved_dir'] = saved_dir
    
    if isinstance(cfg['logger'], OmegaConf):
        logging_cfg = OmegaConf.to_container(cfg['logger'], resolve=True)
        logging.config.dictConfig(logging_cfg)
    else:
        logging.config.dictConfig(cfg['logger'])
    # save modified config file
    with open(os.path.join(saved_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    
    return cfg



def process_config_hydra(cfg: DictConfig) -> DictConfig:
    with open_dict(cfg):
        cfg.config_name = HydraConfig.get().job.config_name
    run_id = datetime.now().strftime(r"%m%d_%H%M%S")
    saved_dir = os.path.join(cfg.saved_root_dir, cfg.config_name, run_id)
    os.makedirs(saved_dir, exist_ok=True)

    with open_dict(cfg):
        cfg.run_id = run_id
        cfg.saved_dir = saved_dir

    # with open(os.path.join(saved_dir, "config.yaml"), "w") as f:
    #     yaml.dump(cfg_to_save, f)

    with open(os.path.join(saved_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    return cfg



def detect_delimiter(csv_file_path):
    with open(csv_file_path, 'r') as file:
        first_line = file.readline()
        if '\t' in first_line:
            return '\t'
        elif ',' in first_line:
            return ','
        else:
            return ','


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




def resolve_config_paths(cfg, root):
    """
    Recursively convert relative paths in cfg to absolute paths.
    Only keys containing 'path' or 'dir' will be processed.
    """
    if isinstance(cfg, dict):
        out = {}
        for k, v in cfg.items():
            # recursion for nested structure
            if isinstance(v, (dict, list)):
                out[k] = resolve_config_paths(v, root)
                continue

            # None remains None
            if v is None:
                out[k] = None
                continue

            # process only path-like keys
            key_lower = k.lower()
            if "path" in key_lower or "dir" in key_lower:
                if isinstance(v, (str, Path)):
                    p = Path(v).expanduser()
                    if not p.is_absolute():
                        p = (root / p).resolve()
                    out[k] = str(p)
                else:
                    # leave non-string path fields untouched (safe fallback)
                    out[k] = v
            else:
                out[k] = v
        return out

    elif isinstance(cfg, list):
        return [resolve_config_paths(item, root) for item in cfg]

    else:
        return cfg
