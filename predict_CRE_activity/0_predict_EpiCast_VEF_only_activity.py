import sys
import joblib
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from omegaconf import DictConfig, OmegaConf


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))
from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import resolve_paths



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--VEF_path", type=str, required=True)

    # 先解析已声明参数，允许未知参数保留
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def load_cfg(args, unknown_args):
    # 读取配置文件
    cfg = OmegaConf.load(args.config)
    # 处理未声明参数：去掉开头的 "--"
    dotlist = [u.lstrip("-") for u in unknown_args]
    # 融合进 config
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def main():
    args, unknown = parse_args()
    cfg = load_cfg(args, unknown)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict = resolve_paths(cfg_dict, ROOT_DIR)
    cfg = OmegaConf.create(cfg_dict)

    cell_types = cfg.cell_types
    assays = cfg.assays

    model_path = Path(args.model_path)
    model = joblib.load(model_path)
    # model = joblib.load("Gosai_MPRA_Sei_VEF_MLP.joblib")

    VEF_df = pd.read_csv(args.VEF_path, sep="\t")

    all_columns = [f"{ct}_{assay}" for ct in cell_types for assay in assays]

    # 把 (samples × (cell_types × assays)) 取出来
    X_full = VEF_df[all_columns].to_numpy()

    n_samples = VEF_df.shape[0]
    n_cell_types = len(cell_types)
    n_assays = len(assays)

    # reshape 为 (samples * cell_types, assays)
    X_flat = X_full.reshape(n_samples * n_cell_types, n_assays)

    # ---- 预测 ----
    y_pred_flat = model.predict(X_flat)

    # ---- reshape 回去 ----
    # 变成 (samples, cell_types)
    y_pred = y_pred_flat.reshape(n_samples, n_cell_types)
    print(y_pred.shape)

    output_path = str(Path(cfg.saved_dir) / args.output_name)
    np.save(output_path, y_pred)
    print(f'save to {output_path}')


if __name__ == '__main__':
    main()
