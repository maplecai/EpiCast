import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
from genoml import models, datasets, metrics, utils

@torch.no_grad()
def get_preds(model, dataloader, device, reverse_comp=False):
    model.eval()
    model = model.to(device)
    preds = []
    for i, batch in enumerate(tqdm(dataloader)):
        batch = utils.to_device(batch, device)
        if not reverse_comp:
            pred = model(batch)
        else:
            pred1 = model(batch)
            batch['seq'] = batch['seq'].flip(dims=[1,2])
            pred2 = model(batch)
            pred = (pred1 + pred2) / 2
        preds.append(pred.detach())
    preds = torch.cat(preds, dim=0).cpu().numpy()
    torch.cuda.empty_cache()
    return preds


def main(args):
    saved_dir = args.saved_dir
    config_path = args.config_path
    dataset_config_path = args.dataset_config_path
    output_name = args.output_name
    # data_path = args.data_path
    reverse_comp = args.reverse_comp

    device = utils.get_free_gpus()[0]

    saved_dir = Path(saved_dir)

    if config_path is None:
        config_path = saved_dir / 'config.yaml'
        print(f'use saved config: {config_path}')
    else:
        print(f'use new config: {config_path}')

    config = utils.load_config(config_path)

    dataset_config = utils.load_config(dataset_config_path)

    if dataset_config_path is not None:
        print(f'use new dataset: {dataset_config_path}')
        config['total_dataset'] = dataset_config['total_dataset']


    model = utils.init_obj(models, config['model'])
    ckpt_pattern = re.compile(r"checkpoint_epoch=(\d+)_pearson=([0-9.]+)\.pth$")
    ckpts = [p for p in Path(saved_dir).iterdir() if p.is_file() and ckpt_pattern.fullmatch(p.name)]
    saved_model_path = max(ckpts, key=lambda p: int(ckpt_pattern.fullmatch(p.name).group(1)))
    print(f'use saved model: {saved_model_path}')

    state_dict = torch.load(str(saved_model_path), weights_only=False)
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

    preds = get_preds(model, total_loader, device, reverse_comp)
    output_path = saved_dir / output_name
    np.save(output_path, preds)
    print(f"saved: {output_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--saved_dir',    type=str, default=None, help='saved folder dir',)
    parser.add_argument('-c', '--config_path',  type=str, default=None, help='config file path, default is saved_dir/config.yaml',)
    parser.add_argument('-dc', '--dataset_config_path',  type=str, default=None,)
    parser.add_argument('-o', '--output_name',  type=str, default='preds.npy', help='output file name',)
    # parser.add_argument('-d', '--data_path',    type=str, default=None, help='data file path',)
    parser.add_argument('-rc','--reverse_comp', type=str, default=False)
    args = parser.parse_args()
    main(args)
