import os
import sys
import yaml
import argparse
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch import nn
from torch import distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary

sys.path.append('/home/hxcai/cell_type_specific_CRE')
import MPRA_exp.models as models
import MPRA_exp.datasets as datasets
import MPRA_exp.metrics as metrics
import MPRA_exp.utils as utils

class Trainer_DDP:
    def __init__(self, config):

        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '23456'

        self.config = config
        
        utils.set_seed(config['seed'])
        logging.config.dictConfig(config['logger'])
        self.logger = logging.getLogger()

        if config['DDP'] is True:
            # local_rank = config['local_rank']
            distributed.init_process_group(backend='nccl', init_method='env://')
            self.local_rank = distributed.get_rank()

            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.logger.info(f"Start DDP on rank {self.local_rank}, cuda {self.device}.")

            # self.n_gpu = config['n_gpu']
            # if config.get('device_ids') is not None:
            #     self.device = config['device_ids'][local_rank]
            # else:
            #     self.device = self.local_rank
            # distributed.init_process_group(backend="nccl", init_method="env://", world_size=self.n_gpu, rank=self.local_rank)
            # torch.cuda.set_device(self.device)
            # self.logging.info(f"Start DDP on rank {self.local_rank}, cuda {self.device}.")
            
        else:
            self.local_rank = 0
            self.device = config['device']
            self.logger.info(f"Start training on cuda {self.device}.")
            
        if self.local_rank == 0:
            self.log_func = self.logger.info
        else:
            self.log_func = self.logger.debug

        self.cell_type_list = self.config['train_dataset_1']['args']['output_columns']

        if config.get('random_data_split', False) is True:
            total_dataset = utils.init_obj(datasets, config['train_dataset_1'])
            indices_list = np.arange(len(total_dataset))
            train_indices, valid_indices, test_indices = utils.split_dataset(indices_list, train_valid_test_ratio=config['train_valid_test_ratio'])
        else:
            train_indices, valid_indices, test_indices = None, None, None

        self.train_dataset = utils.init_obj(datasets, config['train_dataset_1'], selected_indices=train_indices)
        self.valid_dataset = utils.init_obj(datasets, config['valid_dataset_1'], selected_indices=valid_indices)
        self.train_sampler = DistributedSampler(self.train_dataset)
        self.valid_sampler = DistributedSampler(self.valid_dataset)
        self.train_loader = utils.init_obj(torch.utils.data, config['data_loader'], dataset=self.train_dataset, sampler=self.train_sampler)
        self.valid_loader = utils.init_obj(torch.utils.data, config['data_loader'], dataset=self.valid_dataset, sampler=self.valid_sampler)

        self.model = utils.init_obj(models, config['model']).to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank])

        if config.get('load_saved_model', False) is True:
            self.model.load_state_dict(torch.load(config['saved_model_path']))
            for name, param in self.model.named_parameters():
                if name in config.get('freeze_parameters_list', []):  # 冻结某层的参数
                    param.requires_grad = False

        self.loss_func = utils.init_obj(metrics, config['loss_func'])
        self.metric_func_list = [utils.init_obj(metrics, m) for m in config.get('metric_func_list', [])]
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = utils.init_obj(torch.optim, config['optimizer'], trainable_params)

        if 'lr_scheduler' in config:
            self.lr_scheduler = utils.init_obj(torch.optim.lr_scheduler, config['lr_scheduler'], self.optimizer)#constant lr
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(factor=1.0)

        # if 'early_stopper' in config:
        #     self.early_stopper = utils.init_obj(utils, config['early_stopper'], trace_func=self.logger.info)
        # else:
        #     self.early_stopper = utils.EarlyStopping(patience=np.inf)



    def train(self):
        config = self.config

        num_epochs = config['num_epochs']
        batch_size = config['data_loader']['args']['batch_size']
        num_valid_epochs = config['num_valid_epochs']
        save_model = config.get('save_model', False)
        train_steps = len(self.train_loader)
        valid_steps = len(self.valid_loader)

        if self.local_rank == 0:
            self.logger.info(yaml.dump(config))
            ct, x, y = next(iter((self.train_loader)))
            self.logger.info(summary(self.model, input_data=[ct, x], verbose=0))
            self.logger.info(f'len(train_dataset) = {len(self.train_dataset)}, len(valid_dataset) = {len(self.valid_dataset)}')
            self.logger.info(f'num_epochs = {num_epochs}, batch_size = {batch_size}')
            self.logger.info(f'train_steps = {train_steps}, valid_steps = {valid_steps}')
            self.logger.info(f'start training')

        for epoch in range(num_epochs):
            self.train_sampler.set_epoch(epoch)
            self.valid_sampler.set_epoch(epoch)
            # if (self.local_rank == 0) and (epoch == 0):
            # 训练之前先验证一次
            if (epoch == 0):
                self.valid_epoch(-1, self.valid_loader)

            self.train_epoch(epoch, self.train_loader)
            
            # if (self.local_rank == 0) and (epoch % num_valid_epochs == 0):
            if (epoch % num_valid_epochs == 0):
                self.valid_epoch(epoch, self.valid_loader)

                if (self.local_rank == 0) and (save_model is True):
                    self.save_model(epoch)

                # if early_stopper is not None:
                #     # early_stopper.check(valid_loss, model)
                #     early_stopper.check(score, model)
                #     if early_stopper.stop_flag is True:
                #         break
                    
        if self.local_rank == 0:
            self.logger.info(f'finish training.')

        distributed.destroy_process_group()


    def save_model(self, epoch):
        checkpoint_dir = self.config.get('checkpoint_dir', None)

        checkpoint = {
            'config': self.config,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }

        checkpoint_path = os.path.join(checkpoint_dir, f'epoch{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f'save model at {checkpoint_path}')



    def train_epoch(self, epoch, train_loader=None):
        device = self.device
        scheduler_interval = self.config['scheduler_interval']
        num_log_steps = self.config['num_log_steps']
        train_steps = len(train_loader)

        self.model.train()
        train_loss = 0
        y_true_list = []
        y_pred_list = []
        ct_list = []
        for batch_idx, batch_data in enumerate(tqdm(train_loader)):
            ct, x, y = batch_data
            ct, x, y = ct.to(device), x.to(device), y.to(device)
            out = self.model(ct, x)
            loss = self.loss_func(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()
            
            if scheduler_interval == 'step':
                self.lr_scheduler.step(epoch + batch_idx/train_steps)
            
            train_loss += loss.item()
            ct_list.extend(ct.cpu().detach())
            y_true_list.extend(y.cpu().detach())
            y_pred_list.extend(out.cpu().detach())
            if num_log_steps != 0 and batch_idx % num_log_steps == 0:
                self.logger.debug(f'local_rank = {self.local_rank}, epoch = {epoch:3}, batch_idx = {batch_idx:3}, train_loss = {loss.item():.6f}')

        if scheduler_interval == 'epoch':
            self.lr_scheduler.step()

        train_loss = train_loss / train_steps
        ct_list = torch.tensor(ct_list)
        y_true_list = torch.cat(y_true_list)
        y_pred_list = torch.cat(y_pred_list)

        self.log_func(f'local_rank = {self.local_rank}, epoch = {epoch:3}, train_loss = {train_loss:.6f}')
        
        for ct_idx, ct in enumerate(self.cell_type_list):
            y_true_list_0 = y_true_list[torch.where(ct_list == ct_idx)]
            y_pred_list_0 = y_pred_list[torch.where(ct_list == ct_idx)]
            for metric_func in self.metric_func_list:
                score = metric_func(y_true_list_0, y_pred_list_0)
                self.log_func(f'local_rank = {self.local_rank}, epoch = {epoch:3}, '
                              f'ct = {ct}, train_{type(metric_func).__name__} = {score:.6f}')


    def valid_epoch(self, epoch, valid_loader=None):
        torch.set_grad_enabled(False) # 代替with torch.no_grad()，避免缩进，和train缩进一样方便复制

        device = self.device
        valid_steps = len(valid_loader)

        # with torch.no_grad():
        self.model.eval()
        valid_loss = 0
        y_true_list = []
        y_pred_list = []
        ct_list = []
        for batch_idx, batch_data in enumerate(valid_loader):
            ct, x, y = batch_data
            ct, x, y = ct.to(device), x.to(device), y.to(device)
            out = self.model(ct, x)
            loss = self.loss_func(out, y)

            valid_loss += loss.item()
            ct_list.extend(ct.cpu().detach())
            y_true_list.extend(y.cpu().detach())
            y_pred_list.extend(out.cpu().detach())
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
        valid_loss = valid_loss / valid_steps
        ct_list = torch.tensor(ct_list)
        y_true_list = torch.cat(y_true_list)
        y_pred_list = torch.cat(y_pred_list)

        for ct_idx, ct in enumerate(self.cell_type_list):
            y_true_list_0 = y_true_list[torch.where(ct_list == ct_idx)]
            y_pred_list_0 = y_pred_list[torch.where(ct_list == ct_idx)]
            for metric_func in self.metric_func_list:
                score = metric_func(y_true_list_0, y_pred_list_0)
                self.log_func(f'local_rank = {self.local_rank}, epoch = {epoch:3}, '
                              f'ct = {ct}, valid_{type(metric_func).__name__} = {score:.6f}')

        torch.set_grad_enabled(True)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config_path', type=str, default=None,
                      help='config file path',)
    args = args.parse_args()
    config_path = args.config_path

    config = utils.load_config(config_path)
    config = utils.process_config(config)

    trainer = Trainer_DDP(config)
    trainer.train()
    # mp.spawn(main, args=(config, ), nprocs=config['n_gpu'])
    # main(config)
