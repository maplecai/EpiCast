import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MPRA_predict import models, datasets, metrics, utils


@torch.no_grad()
def run_inference(model, test_loader, device, output_path):
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
    parser.add_argument('-s', '--saved_dir', type=str, default=None, help='saved folder dir',)
    parser.add_argument('-c', '--config_path', type=str, default=None, help='config file path, default is saved_dir/config.yaml',)
    parser.add_argument('-de', '--device', type=str, default='cuda:0', help='device',)
    parser.add_argument('-o', '--output_name', type=str, default='pred.npy', help='output file name',)

    parser.add_argument('-d', '--data_path', type=str, default=None, help='data file path',)
    parser.add_argument('--seq_file_path', type=str, default=None,)
    parser.add_argument('--epi_file_path', type=str, default=None,)

    args = parser.parse_args()
    saved_dir = args.saved_dir
    config_path = args.config_path
    device = args.device
    output_name = args.output_name

    data_path = args.data_path
    seq_file_path = args.seq_file_path
    epi_file_path = args.epi_file_path


    if config_path is None:
        config_path = os.path.join(saved_dir, 'config.yaml')
        print(f'use saved config: {config_path}')
    else:
        print(f'use new config: {config_path}')
        pass

    config = utils.load_config(config_path)

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
    saved_model_path = os.path.join(saved_dir, 'checkpoint.pth')
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
    
    output_path = os.path.join(saved_dir, output_name)
    
    run_inference(model, total_loader, device, output_path)



if __name__ == '__main__':
    main()
