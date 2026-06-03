import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))
from epicast import models, datasets, metrics, utils
from epicast.utils import *

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--seq_path', type=str, required=True)
    parser.add_argument('--VEF_path', type=str, required=True)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config = resolve_paths(config, ROOT_DIR)


    # # 改为 ROOT_DIR 下的路径
    # saved_dir = str(ROOT_DIR / config['saved_dir'])
    saved_dir = config['saved_dir']
    device = args.device
    output_name = args.output_name

    cwd = Path.cwd()
    config['total_dataset']['args']['seq_file_path'] = cwd / args.seq_path
    config['total_dataset']['args']['epi_file_path'] = cwd / args.VEF_path

    # --------------------
    # model
    # --------------------
    model = utils.init_obj(models, config['model'])
    saved_model_path = str(Path(saved_dir) / 'checkpoint.pth')
    # state_dict = torch.load(saved_model_path)
    state_dict = torch.load(saved_model_path, map_location=device)
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
