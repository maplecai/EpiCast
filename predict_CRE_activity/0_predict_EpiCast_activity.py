import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))
from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import *

from jsonargparse import ArgumentParser
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def get_pred(model, dataloader, device='cuda'):
    model = model.to(device).eval()
    preds = []
    for batch in tqdm(dataloader):
        batch = utils.to_device(batch, device)
        out = model(batch)
        out = out.cpu().numpy()
        preds.append(out)
    preds = np.concatenate(preds, axis=0)
    return preds


def get_parser():
    parser = ArgumentParser()

    # 让 --config 正常加载 YAML
    parser.add_argument('--config', action='config')

    # 脚本需要但 YAML 里没有的字段
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_name', type=str, default='pred.npy')

    # 顶层字段
    parser.add_argument('--distributed', type=bool)
    parser.add_argument('--gpu_ids')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--num_log_steps', type=int)
    parser.add_argument('--num_valid_epochs', type=int)
    parser.add_argument('--scheduler_interval', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--load_saved_model', type=bool)
    parser.add_argument('--saved_dir', type=str)

    # dict blocks
    parser.add_argument('--total_dataset', type=None)
    parser.add_argument('--total_dataset.args.seq_file_path', type=str)
    parser.add_argument('--total_dataset.args.epi_file_path', type=str)

    parser.add_argument('--model', type=None)
    parser.add_argument('--optimizer', type=None)
    parser.add_argument('--lr_scheduler', type=None)
    parser.add_argument('--loss_func', type=None)
    parser.add_argument('--metric_funcs', type=None)
    parser.add_argument('--early_stopper', type=None)
    parser.add_argument('--logger', type=None)

    return parser



def resolve_paths(cfg, root=ROOT_DIR):
    """递归把 cfg 中所有 path/dir 自动转为绝对路径"""
    if isinstance(cfg, dict):
        new_cfg = {}
        for k, v in cfg.items():
            if isinstance(v, (dict, list)):
                new_cfg[k] = resolve_paths(v, root)
            else:
                if v is None:
                    new_cfg[k] = None
                else:
                    # 只处理包含 path / dir 的字段
                    if any(x in k.lower() for x in ["path", "dir"]):
                        p = Path(v)
                        if not p.is_absolute():
                            p = root / p
                        new_cfg[k] = str(p)
                    else:
                        new_cfg[k] = v
        return new_cfg

    elif isinstance(cfg, list):
        return [resolve_paths(x, root) for x in cfg]

    else:
        return cfg



def main():
    parser = get_parser()
    cfg = parser.parse_args()
    config = cfg.as_dict()
    config = resolve_paths(config)


    # # 改为 ROOT_DIR 下的路径
    # saved_dir = str(ROOT_DIR / config['saved_dir'])
    saved_dir = config['saved_dir']
    device = config['device']
    output_name = config['output_name']

    # --------------------
    # model
    # --------------------
    model = utils.init_obj(models, config['model'])
    saved_model_path = str(Path(saved_dir) / 'checkpoint.pth')
    state_dict = torch.load(saved_model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)

    total_dataset = utils.init_obj(datasets, config['total_dataset'])

    total_loader = DataLoader(
        dataset=total_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    # --------------------
    # predict
    # --------------------
    preds = get_pred(model, total_loader, device)
    output_path = str(Path(saved_dir) / output_name)
    np.save(output_path, preds)
    print(f'save to {output_path}')


if __name__ == '__main__':
    main()
