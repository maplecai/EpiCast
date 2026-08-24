import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
from pathlib import Path
from epicast import models, datasets, metrics, utils

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
    ckpt_path = args.ckpt_path
    config_path = args.config_path
    device = args.device
    output_name = args.output_name
    data_path = args.data_path
    dataset_config_path = args.dataset_config_path
    reverse_comp = args.reverse_comp

    if config_path is None and ckpt_path is not None:
        config_path = Path(ckpt_path).parent.parent / 'config.yaml'
    elif config_path is not None and ckpt_path is None:
        ckpt_path = Path(config_path).parent / 'checkpoints' / 'best.pth'
    elif config_path is None and ckpt_path is None:
        raise ValueError("Either config_path or ckpt_path must be provided")

    config_path = Path(config_path).resolve()
    ckpt_path = Path(ckpt_path).resolve()
    run_dir = config_path.parent

    if device == 'auto':
        device = utils.get_free_gpus()[0]
    

    config = utils.load_config(str(config_path))
    print(f"[load] config: {config_path}")

    if dataset_config_path is not None:
        dataset_config = utils.load_config(str(dataset_config_path))
        config['total_dataset'] = dataset_config['total_dataset']
        print(f"[load] dataset config: {dataset_config_path}")
    elif data_path is not None:
        config['total_dataset']['args']['data_path'] = data_path
        print(f"[load] data: {data_path}")

    model = utils.init_obj(models, config['model'])

    state_dict = torch.load(str(ckpt_path), weights_only=False)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    print(f"[load] ckpt: {ckpt_path}")
    model = model.to(device)

    total_dataset = utils.init_obj(datasets, config['total_dataset'])
    total_loader = utils.init_obj(
        torch.utils.data,
        config['val_loader'],
        dataset=total_dataset, 
        sampler=None,
    )

    preds = get_preds(model, total_loader, device, reverse_comp)
    output_path = run_dir / output_name
    np.save(output_path, preds)
    print(f"[save] {output_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--ckpt_path',    type=str, default=None, help='model checkpoint .pth path')
    parser.add_argument('-c', '--config_path',  type=str, default=None, help='config file path; default is config.yaml in the same directory as ckpt',)
    parser.add_argument('-de','--device',       type=str, default='auto', help='device',)
    parser.add_argument('-o', '--output_name',  type=str, default='preds.npy', help='output file name (written under ckpt directory)',)
    parser.add_argument('-d', '--data_path',    type=str, default=None, help='data file path',)
    parser.add_argument('-dc','--dataset_config_path', type=str, default=None, help='override total_dataset from this yaml (e.g. castillo)',)
    parser.add_argument('-rc','--reverse_comp', type=str, default=False)
    args = parser.parse_args()
    main(args)
