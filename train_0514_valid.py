import os
import re
import sys
import yaml
import argparse
import logging
import subprocess
import numpy as np
import pandas as pd

from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch import nn
from torch import distributed
from torch import distributed  as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary

sys.path.append('/home/hxcai/cell_type_specific_CRE')
import MPRA_exp.models as models
import MPRA_exp.datasets as datasets
import MPRA_exp.metrics as metrics
import MPRA_exp.utils as utils


class Trainer:
    def __init__(self, config):

        self.config = config
        utils.set_seed(config['seed'])

        logging.config.dictConfig(config['logger'])
        self.logger = logging.getLogger()

        self.distribute = False

        if self.distribute:
            distributed.init_process_group(backend='nccl', init_method='env://')
            self.local_rank = distributed.get_rank()
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
            self.logger.info(f"Start DDP training on rank {self.local_rank}, {self.device}.")
            
        else:
            self.local_rank = 0
            free_gpu_id = self.get_free_gpu()
            self.device = torch.device(f'cuda:{free_gpu_id}')
            torch.cuda.set_device(self.device)
            self.logger.info(f"Start non-distributed training on rank {self.local_rank}, {self.device}.")

        if self.local_rank == 0:
            self.log = self.logger.info
        else:
            self.log = self.logger.debug


        self.test_dataset = utils.init_obj(datasets, config['test_dataset'])
        self.test_loader = utils.init_obj(torch.utils.data, config['data_loader'], dataset=self.test_dataset, shuffle=False)
        self.logger.info(f'len(test_dataset) = {len(self.test_dataset)}')
        self.logger.info(f'len(test_loader) = {len(self.test_loader)}')

        self.model = utils.init_obj(models, config['model'])


        chechpoint_dir = os.path.join(config['save_dir'], 'checkpoints')
        saved_model_path_list = [os.path.join(chechpoint_dir, f) for f in os.listdir(chechpoint_dir)]
        latest_model = max(saved_model_path_list, key=os.path.getmtime)
        self.log(f"load saved model from {latest_model}")
        state_dict = torch.load(latest_model)
        if self.distribute:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)


    def test(self):
        torch.set_grad_enabled(False) # 代替with torch.no_grad()，避免缩进，和train缩进一样方便复制

        device = self.device
        y_true_list = []
        y_pred_list = []

        self.model.eval()
        for batch_idx, (x, y) in enumerate(tqdm(self.test_loader, disable=(self.local_rank != 0))):
            x, y = x.to(device), y.to(device)
            out = self.model(x)
            y_true_list.append(y.detach())
            y_pred_list.append(out.detach())

        y_true_list = torch.cat(y_true_list).cpu()
        y_pred_list = torch.cat(y_pred_list).cpu()

        torch.set_grad_enabled(True)

        save_dir = self.config['save_dir']
        np.save(os.path.join(save_dir, f'y_pred_list.npy'), y_pred_list.numpy())
        np.save(os.path.join(save_dir, f'y_true_list.npy'), y_true_list.numpy())

        return None




    def get_free_gpu(self):
        # 执行nvidia-smi命令获取GPU状态
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        memory_info = result.stdout.strip().split('\n')
        free_gpu_id = np.argmax([int(free_memory) for free_memory in memory_info])

        # index_free_memory = [re.split(r'\s*,\s*', info) for info in memory_info]
        # free_gpu_id = np.argmax([int(free_memory) for index, free_memory in index_free_memory])
        return free_gpu_id


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
