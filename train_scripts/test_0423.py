import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MPRA_predict import models, datasets, metrics, utils
from train_0415 import Trainer


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-s', '--saved_dir', type=str, default=None, help='saved folder dir',)
    args.add_argument('-c', '--config_path', type=str, default=None, help='config folder path, default is saved_dir/config.yaml',)
    
    args = args.parse_args()
    saved_dir = args.saved_dir
    config_path = args.config_path

    if config_path is None:
        config_path = os.path.join(saved_dir, 'config.yaml')
        print(f'use saved config: {config_path}')
    else:
        print(f'use new config: {config_path}')
        pass
    # config_path = 'configs/config_0311_SirajMPRA_test_1_cell_type.yaml'

    config = utils.load_config(config_path)
    config['save_dir'] = saved_dir
    config['load_saved_model'] = True
    config['saved_model_path'] = os.path.join(saved_dir, 'checkpoint.pth')

    trainer = Trainer(config)
    total_dataset = utils.init_obj(datasets, config['total_dataset'])
    total_loader = DataLoader(
        dataset=total_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    trainer.test(total_loader)
