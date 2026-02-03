import os
import sys
import torch
import argparse
import torch.utils.data
from tqdm import tqdm
import numpy as np
from pathlib import Path

from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from genoml import models, datasets, metrics, utils


@torch.no_grad()
def run_inference(
    model: torch.nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    device: str|torch.device, 
    output_path: str|Path,
):
    model.eval()
    model = model.to(device)
    pred_list = []
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        batch = utils.to_device(batch, device)
        pred = model(batch)
        pred_list.append(pred.detach())

    pred_list = torch.cat(pred_list).cpu().numpy()
    np.save(output_path, pred_list)
    torch.cuda.empty_cache()
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--saved_dir', type=str, default=None, help='saved folder dir')
    parser.add_argument('-c', '--config_path', type=str, default=None, help='config file path, default is saved_dir/config.yaml')
    parser.add_argument('-de', '--device', type=str, default='cuda:0', help='device')
    parser.add_argument('-o', '--output_name', type=str, default='pred.npy', help='output file name')

    # 通用覆盖：支持多次传入，格式 key=value，支持深层路径
    # 例如: --override total_dataset.args.data_path=/xx --override batch_size=16
    parser.add_argument(
        '--override',
        action='append',
        default=[],
        help="Override config with dotlist. Repeatable. e.g. --override total_dataset.args.data_path=/data --override batch_size=16"
    )

    args = parser.parse_args()
    saved_dir = Path(args.saved_dir) if args.saved_dir is not None else None
    config_path = Path(args.config_path) if args.config_path is not None else None
    device = args.device
    output_name = args.output_name

    if config_path is None:
        if saved_dir is None:
            raise ValueError("Either --config_path or --saved_dir must be provided.")
        config_path = saved_dir / 'config.yaml'
        print(f'use saved config: {config_path}')
    else:
        print(f'use new config: {config_path}')

    # 1) 用 OmegaConf 读配置（替代 utils.load_config）
    cfg = OmegaConf.load(str(config_path))

    # 3) 合并通用覆盖
    if args.override:
        override_cfg = OmegaConf.from_dotlist(args.override)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # 4) 如果你后续代码需要普通 dict（例如 utils.init_obj 期望 dict），转回去
    config = OmegaConf.to_container(cfg, resolve=True)

    model = utils.init_obj(models, config['model'])
    saved_model_path = saved_dir / 'checkpoint.pth'

    state_dict = torch.load(str(saved_model_path), map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    model = model.to(device)

    total_dataset = utils.init_obj(datasets, config['total_dataset'])
    total_loader = utils.init_obj(
        torch.utils.data,
        config['val_loader'],
        dataset=total_dataset, 
        sampler=None,
    )

    output_path = saved_dir / output_name
    run_inference(model, total_loader, device, output_path)


if __name__ == '__main__':
    main()
