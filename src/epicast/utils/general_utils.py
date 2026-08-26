import os
import h5py
import pickle
import logging
import logging.config
import numpy as np
import pandas as pd
import random
import torch
import inspect
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from ruamel.yaml import YAML
yaml = YAML()
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
from hydra.core.hydra_config import HydraConfig



def set_seed(seed:int = 42) -> None:
    '''
    Set the random seeds.
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
                data = {key: f[key][:] for key in keys}
    return data

def save_h5(file_dir: str, data) -> None:
    with h5py.File(file_dir, 'w') as f:
        f.create_dataset('data', data=data)
    return


def load_pickle(file_dir: str):
    with open(file_dir, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(file_dir: str, data) -> None:
    with open(file_dir, 'wb') as f:
        pickle.dump(data, f)



def init_obj(
    module: object | None,
    obj_dict: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Instantiate a class or return a function from a config dict.

    Return type is ``Any`` because resolution is dynamic (class vs function vs
    missing config). Call sites that need a concrete type should annotate or
    cast, e.g. ``loss: nn.Module = init_obj(...)``.
    """
    if not obj_dict:
        return None
    if not isinstance(obj_dict, dict):
        raise TypeError("inval init object dict")

    name = obj_dict["type"]
    module_args = {**obj_dict.get("args", {}), **kwargs}

    if module is None:
        cls = globals()[name]
    else:
        cls = getattr(module, name)

    if inspect.isfunction(cls):
        return cls

    return cls(*args, **module_args)



def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)
    cfg['config_name'] = Path(config_path).stem
    return cfg


def process_config(cfg: dict) -> dict:
    config_name = cfg['config_name']

    saved_root_dir = Path(cfg['saved_root_dir'])
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    saved_dir = (saved_root_dir / config_name / run_id)
    saved_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a run of the same second
    cfg["saved_dir"] = str(saved_dir)
    
    if isinstance(cfg['logger'], OmegaConf):
        logging_cfg = OmegaConf.to_container(cfg['logger'], resolve=True)
    else:
        logging_cfg = cfg['logger']
    
    for _, handler in logging_cfg['handlers'].items():
        if 'filename' in handler.keys():
            handler['filename'] = os.path.join(saved_dir, handler['filename'])

    logging.config.dictConfig(logging_cfg)   

    # save modified config
    with (saved_dir / "config.yaml").open("w", encoding="utf-8") as f:
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



def detect_delimiter(file_path):
    if file_path.endswith('.csv'):
        sep = ','
    elif file_path.endswith('.tsv'):
        sep = '\t'
    else:
        raise ValueError(f'{file_path} not endswith .csv or .tsv')
    return sep



import h5py
import numpy as np


class H5Writer:
    def __init__(
        self,
        path,
        datasets_shape,
        total_size=None,
        chunk_size=1024,
        dtype=np.float32,
        compression="gzip",
    ):
        """
        Args:
            path: path of the h5 file
            datasets_shape: dict
                e.g. {"DNase": (305,), "ATAC": (167,), "TF": (1617,)}
            total_size:
                - int: old mode, preallocate a fixed length
                - None: new mode, grow the first axis on every append
            chunk_size: batch axis of a chunk
            dtype: data type
            compression: compression method
        """
        self.path = path
        self.datasets_shape = datasets_shape
        self.total_size = total_size
        self.chunk_size = int(chunk_size)
        self.dtype = np.dtype(dtype)
        self.compression = compression

        self.f = None
        self.datasets = {}
        self.index = 0
        self._init_datasets()

    @property
    def num_written(self):
        return self.index
    
    def _init_datasets(self):
        self.f = h5py.File(self.path, "a")

        # older files: recover the index from the attribute first
        self.index = int(self.f.attrs.get("num_written", 0))

        for name, sample_shape in self.datasets_shape.items():
            self.datasets_shape[name] = sample_shape
            self.datasets[name] = self._get_or_create_ds(name, sample_shape)

        return self

    def _get_or_create_ds(self, name, sample_shape):
        if name in self.f:
            ds = self.f[name]

            # check the shape apart from the batch axis
            if ds.shape[1:] != sample_shape:
                raise ValueError(
                    f"{name} shape mismatch: existing {ds.shape[1:]} vs expected {sample_shape}"
                )
            if ds.dtype != self.dtype:
                raise ValueError(f"{name} dtype mismatch: {ds.dtype} vs {self.dtype}")

            return ds

        # old mode: fixed capacity
        if self.total_size is not None:
            init_size = int(self.total_size)
            maxshape = (init_size, *sample_shape)
        else:
            # new mode: unbounded append
            init_size = 0
            maxshape = (None, *sample_shape)

        chunks = (max(1, self.chunk_size), *sample_shape)

        return self.f.create_dataset(
            name,
            shape=(init_size, *sample_shape),
            maxshape=maxshape,
            dtype=self.dtype,
            chunks=chunks,
            compression=self.compression,
            shuffle=True,
        )

    def _ensure_capacity(self, end):
        for name, ds in self.datasets.items():
            cur_n = ds.shape[0]
            if end <= cur_n:
                continue

            # fixed capacity: raise on overflow, as before
            if self.total_size is not None:
                raise ValueError(
                    f"Write exceeds total_size: trying to write up to {end}, "
                    f"but dataset capacity is {cur_n}"
                )

            # unbounded append: grow as needed
            ds.resize((end, *ds.shape[1:]))

    def write(self, data_dict, flush=True):
        """
        Args:
            data_dict: dict
                e.g. {
                    "DNase": np.ndarray(shape=(b, 305)),
                    "ATAC": np.ndarray(shape=(b, 167)),
                    "TF": np.ndarray(shape=(b, 1617)),
                }
        """
        if not data_dict:
            return

        # check the keys
        expected_keys = set(self.datasets.keys())
        input_keys = set(data_dict.keys())
        if input_keys != expected_keys:
            missing = expected_keys - input_keys
            extra = input_keys - expected_keys
            raise ValueError(f"Dataset keys mismatch. missing={missing}, extra={extra}")

        # every value must share the batch size
        b = None
        for name, arr in data_dict.items():
            arr = np.asarray(arr)
            if arr.ndim < 1:
                raise ValueError(f"{name} must have batch dimension")
            if arr.shape[1:] != self.datasets[name].shape[1:]:
                raise ValueError(
                    f"{name} sample shape mismatch: got {arr.shape[1:]}, "
                    f"expected {self.datasets[name].shape[1:]}"
                )
            if b is None:
                b = arr.shape[0]
            elif arr.shape[0] != b:
                raise ValueError(f"Batch size mismatch in {name}: got {arr.shape[0]}, expected {b}")

        end = self.index + b
        self._ensure_capacity(end)

        for name, arr in data_dict.items():
            self.datasets[name][self.index:end] = np.asarray(arr, dtype=self.dtype)

        self.index = end
        self.f.attrs["num_written"] = self.index

        if flush:
            self.f.flush()

    def flush(self):
        if self.f is not None:
            self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.attrs["num_written"] = self.index
            self.f.flush()
            self.f.close()
            self.f = None

    def __len__(self):
        return self.index

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()






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





def remove_nan(*arrays):
    """
    Drop the rows (along axis 0) that contain NaN in any of the given
    1D/2D/3D arrays, keeping the intersection of the valid rows.

    Parameters
    ----------
    *arrays : np.ndarray
        Any number of 1D / 2D / 3D arrays of equal length along axis 0.

    Returns
    -------
    tuple of np.ndarray
        The filtered arrays, in the order they were given.

    Raises
    ------
    ValueError
        If nothing is given, an array is not 1-3D, or the lengths differ.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")

    arrays = [np.asarray(arr) for arr in arrays]

    # check the dimensions
    for i, arr in enumerate(arrays):
        if arr.ndim not in (1, 2, 3):
            raise ValueError(
                f"arrays[{i}] has ndim={arr.ndim}, but only 1D/2D/3D arrays are supported."
            )

    # check that axis 0 has the same length everywhere
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"All arrays must have the same length on axis 0, got {lengths}")

    n = lengths[0]
    mask = np.ones(n, dtype=bool)

    for arr in arrays:
        if arr.ndim == 1:
            valid = ~np.isnan(arr)
        else:
            # 2D: axis=1; 3D: axis=(1,2)
            valid = ~np.isnan(arr).any(axis=tuple(range(1, arr.ndim)))
        mask &= valid

    return tuple(arr[mask] for arr in arrays)