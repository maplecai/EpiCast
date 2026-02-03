import os
import sys
import torch
import argparse
import torch.utils.data
from tqdm import tqdm
import numpy as np
from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent))
from genoml import models, datasets, metrics, utils


@torch.no_grad()
def run_inference(model, test_loader, device):
    model.eval()
    model = model.to(device)
    pred_list = []
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        batch = utils.to_device(batch, device)
        pred = model(batch)
        pred_list.append(pred.detach())

    pred = torch.cat(pred_list).cpu().numpy()
    torch.cuda.empty_cache()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--saved_dir',    type=str, default=None, help='saved folder dir',)
    parser.add_argument('-c', '--config_path',  type=str, default=None, help='config file path, default is saved_dir/config.yaml',)
    parser.add_argument('-de','--device',       type=str, default='cuda', help='device',)
    parser.add_argument('-o', '--output_name',  type=str, default='pred.npz', help='output file name',)
    parser.add_argument('-d', '--data_path',    type=str, default=None, help='data file path',)
    parser.add_argument('--seq_file_path',      type=str, default=None,)
    parser.add_argument('--epi_file_path',      type=str, default=None,)

    args = parser.parse_args()

    saved_dir = args.saved_dir
    config_path = args.config_path
    device = args.device
    output_name = args.output_name
    data_path = args.data_path
    seq_file_path = args.seq_file_path
    epi_file_path = args.epi_file_path

    saved_dir = Path(saved_dir)
    config_path = Path(config_path)
    # if output_name is None:
    #     output_name = 'pred.npz'


    if config_path is None:
        config_path = saved_dir / 'config.yaml'
        print(f'use saved config: {config_path}')
    else:
        print(f'use new config: {config_path}')

    config = utils.load_config(str(config_path))

    if data_path is not None:
        print(f'use new data: {data_path}')
        config['total_dataset']['args']['data_path'] = data_path
    if seq_file_path is not None:
        print(f'use new seq file: {seq_file_path}')
        config['total_dataset']['args']['seq_file_path'] = seq_file_path
    if epi_file_path is not None:
        print(f'use new epi file: {epi_file_path}')
        config['total_dataset']['args']['epi_file_path'] = epi_file_path

    model = utils.init_obj(models, config['model'])
    saved_model_path = saved_dir / 'checkpoint.pth'

    state_dict = torch.load(str(saved_model_path))
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
    pred = run_inference(model, total_loader, device)

    # target_name = cell_types + '_pred'
    cell_types = config['total_dataset']['args']['cell_types']
    target_name = np.array([f"{ct}_pred" for ct in cell_types], dtype=object)
    output_path = saved_dir / output_name
    np.savez(output_path, pred=pred, target_name=target_name)

    print(f"saved: {output_path} (keys: pred, target_name)")


    # np.save(output_path, pred)

if __name__ == '__main__':
    main()
