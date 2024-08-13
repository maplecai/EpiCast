import os
import sys
import yaml
import argparse
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
import torch
from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data
import torchinfo

from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import *


class Trainer:
    def __init__(self, config):
        # setup seed and distribute
        self.config = config
        utils.set_seed(config['seed'])
        logging.config.dictConfig(config['logger'])
        self.logger = logging.getLogger()

        self.distribute = False # config['distribute']
        if self.distribute:
            dist.init_process_group(backend='nccl', init_method='env://')
            self.local_rank = dist.get_rank()
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
            self.logger.info(
                f"Start DDP training on rank {self.local_rank}, {self.device}.")
        else:
            self.local_rank = 0
            free_gpu_id = utils.get_free_gpu_id()
            self.device = torch.device(f'cuda:{free_gpu_id}')
            torch.cuda.set_device(self.device)
            self.logger.info(
                f"Start non-distributed training on rank {self.local_rank}, {self.device}.")

        if self.local_rank == 0:
            self.log = self.logger.info
        else:
            self.log = self.logger.debug

        # setup dataloader
        self.train_cell_types = config['train_cell_types']
        self.valid_cell_types = config['valid_cell_types']

        self.test_dataset = utils.init_obj(
            datasets, 
            config['total_dataset'])
        
        self.test_loader = utils.init_obj(
            torch.utils.data, 
            config['data_loader'], 
            dataset=self.test_dataset, 
            shuffle=False)
            
        self.model = utils.init_obj(models, config['model'])

        if config.get('load_saved_model', False) == True:
            saved_model_path = config['saved_model_path']
            state_dict = torch.load(saved_model_path)
            self.model.load_state_dict(state_dict)
            self.log(f"load saved model from {saved_model_path}")

        self.model = self.model.to(self.device)

    def test(self, test_loader=None):
        torch.set_grad_enabled(False)
        # 代替with torch.no_grad()，避免多一层缩进，和train缩进一样，方便复制

        if test_loader is None:
            test_loader = self.test_loader

        y_pred_list = []

        self.model.eval()
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, disable=(self.local_rank != 0))):
            inputs = to_device(inputs, self.device)
            labels = to_device(labels, self.device)
            out = self.model(inputs)
            y_pred_list.append(out.detach())

        y_pred_list = torch.cat(y_pred_list).cpu().numpy()
        output_file_name = self.config.get('output_file_name', 'test_pred.npy')
        output_file_path = os.path.join(self.config['save_dir'], output_file_name)
        np.save(output_file_path, y_pred_list)
        torch.set_grad_enabled(True)
        return y_pred_list


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config_path', type=str, default=None,
                      help='config file path',)
    args = args.parse_args()
    config_path = args.config_path

    config = utils.load_config(config_path)
    # config = utils.process_config(config)

    trainer = Trainer(config)
    trainer.test()
