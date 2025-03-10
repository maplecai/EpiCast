import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MPRA_predict import models, datasets, metrics, utils
from train_0306 import Trainer


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '-s', '--saved_dir', type=str, default=None, 
        help='saved folder dir',
        )
    args = args.parse_args()
    saved_dir = args.saved_dir

    config_path = 'configs/config_0306_SirajMPRA_test.yaml'
    config = utils.load_config(config_path)
    config['save_dir'] = saved_dir
    config['saved_model_path'] = os.path.join(saved_dir, 'checkpoint.pth')

    trainer = Trainer(config)

    test_dataset = utils.init_obj(datasets, config['test_dataset'])
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    trainer.test(test_loader)
    torch.cuda.empty_cache()
