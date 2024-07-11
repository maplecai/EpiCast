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
import torchinfo

from MPRA_exp import models
from MPRA_exp import datasets
from MPRA_exp import metrics
from MPRA_exp import utils
from MPRA_exp.utils import *


class Trainer:
    def __init__(self, config):

        self.config = config
        utils.set_seed(config['seed'])
        logging.config.dictConfig(config['logger'])
        self.logger = logging.getLogger()

        self.distribute = config['distribute']

        if self.distribute:
            distributed.init_process_group(backend='nccl', init_method='env://')
            self.local_rank = distributed.get_rank()
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


        self.train_cell_types = config['train_cell_types']
        self.valid_cell_types = config['valid_cell_types']
        self.logger.info(f'train_cell_types = {self.train_cell_types}')
        self.logger.info(f'valid_cell_types = {self.valid_cell_types}')

        # single task
        self.train_dataset = utils.init_obj(
            datasets, 
            config['train_dataset'], 
            cell_types=self.train_cell_types)
        self.valid_dataset = utils.init_obj(
            datasets, 
            config['valid_dataset'], 
            cell_types=self.valid_cell_types)
        
        if not self.distribute:
            self.train_loader = utils.init_obj(
                torch.utils.data, 
                config['data_loader'], 
                dataset=self.train_dataset, 
                shuffle=True)
            self.valid_loader = utils.init_obj(
                torch.utils.data, 
                config['data_loader'], 
                dataset=self.valid_dataset, 
                shuffle=False)
        else:
            self.train_loader = utils.init_obj(
                torch.utils.data, 
                config['data_loader'], 
                dataset=self.train_dataset, 
                sampler=DistributedSampler(self.train_dataset), 
                shuffle=True)
            self.valid_loader = utils.init_obj(
                torch.utils.data, 
                config['data_loader'], 
                dataset=self.valid_dataset, 
                sampler=DistributedSampler(self.valid_dataset), 
                shuffle=False)
            
        self.logger.info(f'len(train_dataset) = {len(self.train_dataset)}')
        self.logger.info(f'len(valid_dataset) = {len(self.valid_dataset)}')
        self.logger.info(f'len(train_loader) = {len(self.train_loader)}')
        self.logger.info(f'len(valid_loader) = {len(self.valid_loader)}')

        self.model = utils.init_obj(models, config['model'])

        if config.get('load_saved_model', False) == True:
            state_dict = torch.load(config['saved_model_path'])
            self.model.load_state_dict(state_dict)

        if self.distribute:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank], 
                find_unused_parameters=False)
        else:
            self.model = self.model.to(self.device)

        self.loss_func = utils.init_obj(metrics, config['loss_func'])
        self.metric_func_list = [
            utils.init_obj(metrics, m) for m in config.get('metric_func_list', [])]
        self.metric_names = [m['type'] for m in config.get('metric_func_list', [])]
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = utils.init_obj(torch.optim, config['optimizer'], trainable_params)

        if 'lr_scheduler' in config:
            self.lr_scheduler = utils.init_obj(
                torch.optim.lr_scheduler, 
                config['lr_scheduler'], 
                self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, 
                factor=1.0)

        if 'early_stopper' in config:
            self.early_stopper = utils.init_obj(
                utils, 
                config['early_stopper'], 
                save_dir=os.path.join(config['save_dir'], 'checkpoints'), 
                trace_func=self.log)
        else:
            self.early_stopper = utils.EarlyStopping(patience=np.inf)


    def train(self):
        config = self.config

        num_epochs = config['num_epochs']
        batch_size = config['data_loader']['args']['batch_size']
        num_valid_epochs = config['num_valid_epochs']
        # num_save_epochs = config['num_save_epochs']
        # save_model = config['save_model']

        if self.local_rank == 0:
            self.logger.debug(yaml.dump(config))
            (inputs, labels) = next(iter((self.train_loader)))
            self.logger.info(torchinfo.summary(
                self.model, 
                input_data=[to_device(inputs, self.device)], 
                verbose=0, 
                depth=5))
            self.logger.info(f'num_epochs = {num_epochs}')
            self.logger.info(f'batch_size = {batch_size}')
            self.logger.info(f'start training')

        for epoch in range(num_epochs):
            self.epoch = epoch
            if self.distribute:
                self.train_loader.set_epoch(epoch)
                self.valid_loader.set_epoch(epoch)

            # 训练之前先验证一次
            if (epoch == 0):
                self.valid_epoch(self.valid_loader)

            self.log(f'train on epoch {epoch}')
            self.train_epoch(self.train_loader)
            
            if ((epoch+1) % num_valid_epochs == 0):
                self.log(f'valid on epoch {epoch}')
                self.valid_epoch(self.valid_loader)

                if (self.early_stopper is not None):
                    score = self.score_df.loc[self.train_cell_types, 'Pearson'].mean()
                    self.early_stopper.check(score)

                    if self.early_stopper.update_flag == True:
                        if self.local_rank == 0:
                            self.save_model()
                    if self.early_stopper.stop_flag == True:
                        break

        self.log(f'local_rank = {self.local_rank:1}, finish training.')

        if self.distribute:
            dist.destroy_process_group()


    def train_epoch(self, train_loader):
        device = self.device
        scheduler_interval = self.config.get('scheduler_interval', 'epoch')
        num_log_steps = self.config.get('num_log_steps', 0)
        train_steps = len(train_loader)
        train_loss = 0

        self.model.train()
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, disable=(self.local_rank != 0))):
            inputs = to_device(inputs, device)
            labels = to_device(labels, device)
            out = self.model(inputs)
            loss = self.loss_func(out, labels)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()
            
            if scheduler_interval == 'step':
                self.lr_scheduler.step(self.epoch + batch_idx/train_steps)
            
            train_loss += loss.item()
            if num_log_steps != 0 and batch_idx % num_log_steps == 0:
                self.logger.debug(
                    f'local_rank = {self.local_rank}, epoch = {self.epoch:3}, '
                    f'batch_idx = {batch_idx:3}, train_loss = {loss.item():.6f}')

        if scheduler_interval == 'epoch':
            self.lr_scheduler.step()

        train_loss = train_loss / train_steps
        self.log(
            f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, train_loss = {train_loss:.6f}')

        return


    def valid_epoch(self, valid_loader):
        torch.set_grad_enabled(False)
        # 代替with torch.no_grad()，避免多一层缩进，和train缩进一样方便复制

        device = self.device
        valid_steps = len(valid_loader)
        valid_loss = 0
        y_true_list = []
        y_pred_list = []

        self.model.eval()
        for batch_idx, (inputs, labels) in enumerate(tqdm(valid_loader, disable=(self.local_rank != 0))):
            inputs = to_device(inputs, device)
            labels = to_device(labels, device)
            out = self.model(inputs)
            loss = self.loss_func(out, labels)
            valid_loss += loss
            y_true_list.append(labels.detach())
            y_pred_list.append(out.detach())

        y_true_list = torch.cat(y_true_list)
        y_pred_list = torch.cat(y_pred_list)

        if self.distribute:
            y_true_list = self.dist_all_gather(y_true_list).cpu()
            y_pred_list = self.dist_all_gather(y_pred_list).cpu()
        else:
            y_true_list = y_true_list.cpu()
            y_pred_list = y_pred_list.cpu()

        valid_loss = valid_loss / valid_steps
        if self.distribute:
            dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        self.log(f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, valid_loss = {valid_loss:.6f}')


        if self.local_rank == 0:
            # self.log(f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, valid_loss = {loss_list.mean():.6f}')
            self.score_df = pd.DataFrame(index=self.valid_cell_types, columns=self.metric_names)

            for idx, cell_type in enumerate(self.valid_cell_types):
                log_message = f'cell_type = {cell_type:6}'
                indice = (self.valid_dataset.df['cell_type'] == cell_type)
                y_true_list_0 = y_true_list[indice]
                y_pred_list_0 = y_pred_list[indice]
                
                for metric_func in self.metric_func_list:
                    metric_name = type(metric_func).__name__
                    score = metric_func(y_pred_list_0, y_true_list_0)
                    log_message += f', {metric_name} = {score:.6f}'
                    self.score_df.loc[cell_type, metric_name] = score
                self.log(log_message)
        torch.set_grad_enabled(True)

        return


    def dist_all_gather(self, tensor):
        tensor_list = [torch.zeros_like(tensor, device=tensor.device) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, tensor)
        tensor_list = torch.cat(tensor_list)
        return tensor_list
    

    def save_model(self):
        checkpoint_dir = self.config.get('checkpoint_dir', None)

        # checkpoint = {
        #     'config': self.config,
        #     'epoch': self.epoch,
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     }
        checkpoint = self.model.module.state_dict() if self.distribute else self.model.state_dict()

        # checkpoint_path = os.path.join(checkpoint_dir, f'epoch{self.epoch}.pth')
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f'save model at {checkpoint_path}')



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config_path', type=str, default=None,
                      help='config file path',)
    args = args.parse_args()
    config_path = args.config_path

    config = utils.load_config(config_path)
    config = utils.process_config(config)

    trainer = Trainer(config)
    trainer.train()
