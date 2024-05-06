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
from torch import distributed  as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary

sys.path.append('/home/hxcai/cell_type_specific_CRE')
import MPRA_exp.models as models
import MPRA_exp.datasets as datasets
import MPRA_exp.metrics as metrics
import MPRA_exp.utils as utils

def dist_all_gather(tensor):
    tensor_list = [torch.zeros_like(tensor, device=tensor.device) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list)
    return tensor_list


class Trainer_DDP:
    def __init__(self, config):

        self.config = config
        utils.set_seed(config['seed'])
        logging.config.dictConfig(config['logger'])
        self.logger = logging.getLogger()

        self.distribute = config['distribute']

        if config['distribute'] == True:
            distributed.init_process_group(backend='nccl', init_method='env://')
            self.local_rank = distributed.get_rank()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')

            if self.local_rank == 0:
                self.log = self.logger.info
            else:
                self.log = self.logger.debug
            self.log(f"Start DDP on rank {self.local_rank}, {self.device}.")
            
        else:
            self.local_rank = 0
            self.device = config['device']
            self.log = self.logger.info
            self.log(f"Start non-distribute training on cuda {self.device}.")


        self.task_names = config['task_names']
        self.num_tasks = len(self.task_names)

        self.selected_train_datasets_idx = config['selected_train_datasets_idx']
        self.selected_valid_datasets_idx = config['selected_valid_datasets_idx']

        self.log(f'selected_train_datasets_idx = {self.selected_train_datasets_idx}')
        self.log(f'selected_valid_datasets_idx = {self.selected_valid_datasets_idx}')

        self.train_datasets = [
            utils.init_obj(datasets, config['train_datasets'][i]) 
            for i in range(len(config['train_datasets']))
            if config['train_datasets'][i]['args']['task_idx'][0] in config['selected_train_datasets_idx']]
        
        self.valid_datasets = [
            utils.init_obj(datasets, config['valid_datasets'][i]) 
            for i in range(len(config['valid_datasets']))
            if config['valid_datasets'][i]['args']['task_idx'][0] in config['selected_valid_datasets_idx']]
        
        if self.distribute:
            self.train_distributed_samplers = [DistributedSampler(dataset) for dataset in self.train_datasets]
            self.valid_distributed_samplers = [DistributedSampler(dataset) for dataset in self.valid_datasets]

            self.train_loaders = [utils.init_obj(torch.utils.data, config['data_loader'], 
                dataset=self.train_datasets[i], sampler=self.train_distributed_samplers[i])
                for i in range(len(self.train_datasets))]
            self.valid_loaders = [utils.init_obj(torch.utils.data, config['data_loader'], 
                dataset=self.valid_datasets[i], sampler=self.valid_distributed_samplers[i]) 
                for i in range(len(self.valid_datasets))]
        
        else:
            self.train_loaders = [utils.init_obj(torch.utils.data, config['data_loader'], 
                dataset=self.train_datasets[i])
                for i in range(len(self.train_datasets))]
            self.valid_loaders = [utils.init_obj(torch.utils.data, config['data_loader'], 
                dataset=self.valid_datasets[i]) 
                for i in range(len(self.valid_datasets))]
        
        self.train_loader = utils.init_obj(datasets, config['multi_task_data_loader'], dataloaders=self.train_loaders)
        self.valid_loader = utils.init_obj(datasets, config['multi_task_data_loader'], dataloaders=self.valid_loaders)

        self.logger.info(f'len(train_dataset) = {[len(dataset) for dataset in self.train_datasets]}')
        self.logger.info(f'len(valid_dataset) = {[len(dataset) for dataset in self.valid_datasets]}')
        self.logger.info(f'len(train_loader) = {len(self.train_loader)}')
        self.logger.info(f'len(valid_loader) = {len(self.valid_loader)}')

        self.model = utils.init_obj(models, config['model']).to(self.device)

        if self.distribute:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        if config.get('load_saved_model', False) == True:
            state_dict = torch.load(config['saved_model_path'])
            if self.distribute:
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            # self.model.load_state_dict(torch.load(config['saved_model_path']))

        self.loss_func = utils.init_obj(metrics, config['loss_func'])
        self.metric_func_list = [utils.init_obj(metrics, m) for m in config.get('metric_func_list', [])]
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = utils.init_obj(torch.optim, config['optimizer'], trainable_params)

        if 'lr_scheduler' in config:
            #constant lr
            self.lr_scheduler = utils.init_obj(torch.optim.lr_scheduler, config['lr_scheduler'], self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(factor=1.0)

        if 'early_stopper' in config:
            self.early_stopper = utils.init_obj(utils, config['early_stopper'], 
                                                save_dir=os.path.join(config['save_dir'], 'checkpoints'), trace_func=self.log)
        else:
            self.early_stopper = utils.EarlyStopping(patience=np.inf)


    def train(self):
        config = self.config

        num_epochs = config['num_epochs']
        batch_size = config['data_loader']['args']['batch_size']
        num_valid_epochs = config['num_valid_epochs']
        num_save_epochs = config['num_save_epochs']
        save_model = config['save_model']

        if self.local_rank == 0:
            self.logger.debug(yaml.dump(config))
            (task_idx, cell_idx, output_idx), (x, y) = next(iter((self.train_loader)))
            self.logger.info(summary(self.model, input_data=[x.to(self.device), cell_idx.to(self.device), output_idx.to(self.device)], verbose=0, depth=5))
            self.logger.info(f'num_epochs = {num_epochs}')
            self.logger.info(f'batch_size = {batch_size}')
            self.logger.info(f'start training')

        for epoch in range(num_epochs):
            self.epoch = epoch
            self.train_loader.set_epoch(epoch)
            self.valid_loader.set_epoch(epoch)

            # 训练之前先验证一次
            if (epoch == 0):
                self.log(f'valid_dataset')
                self.valid_epoch(self.valid_loader)

            self.train_epoch(self.train_loader)
            
            if ((epoch+1) % num_valid_epochs == 0):
                self.log(f'valid_dataset')
                valid_loss = self.valid_epoch(self.valid_loader)

                if (self.early_stopper is not None):
                    if self.distribute:
                        self.early_stopper.check(valid_loss, self.model.module, save=(self.local_rank == 0))
                    else:
                        self.early_stopper.check(valid_loss, self.model, save=(self.local_rank == 0))
                    if self.early_stopper.stop_flag == True:
                        break

        self.log(f'local_rank = {self.local_rank:1}, finish training.')

        if self.distribute:
            dist.destroy_process_group()


    def train_epoch(self, train_loader=None):
        train_loader.set_cycle(True)
        train_loader.set_mode('min')
        device = self.device
        scheduler_interval = self.config['scheduler_interval']
        train_steps = len(train_loader)
        train_loss = 0

        self.model.train()
        for batch_idx, batch_data in enumerate(tqdm(train_loader, disable=(self.local_rank != 0))):
            (task_idx, cell_idx, output_idx), (x, y) = batch_data
            cell_idx, output_idx, x, y = cell_idx.to(device), output_idx.to(device), x.to(device), y.to(device)
            out = self.model(x, cell_idx, output_idx)
            loss = self.loss_func(out, y, output_idx)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()
            
            if scheduler_interval == 'step':
                self.lr_scheduler.step(self.epoch + batch_idx/train_steps)
            
            train_loss += loss.item()
            # if num_log_steps != 0 and batch_idx % num_log_steps == 0:
            #     self.logger.debug(f'local_rank = {self.local_rank}, epoch = {epoch:3}, 
            # batch_idx = {batch_idx:3}, train_loss = {loss.item():.6f}')

        if scheduler_interval == 'epoch':
            self.lr_scheduler.step()

        train_loss = train_loss / train_steps
        self.log(f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, train_loss = {train_loss:.6f}')

        return train_loss

    def valid_epoch(self, valid_loader=None):
        torch.set_grad_enabled(False) # 代替with torch.no_grad()，避免缩进，和train缩进一样方便复制
        valid_loader.set_cycle(False)
        valid_loader.set_mode('min')

        device = self.device
        valid_steps = len(valid_loader)
        valid_loss = 0
        loss_list = []
        task_idx_list = []
        y_true_list = []
        y_pred_list = []

        self.model.eval()
        for batch_idx, batch_data in enumerate(tqdm(valid_loader, disable=(self.local_rank != 0))):
            (task_idx, cell_idx, output_idx), (x, y) = batch_data
            task_idx, cell_idx, output_idx, x, y = \
                task_idx.to(device), cell_idx.to(device), output_idx.to(device), x.to(device), y.to(device)
            out = self.model(x, cell_idx, output_idx)
            loss = self.loss_func(out, y, output_idx)

            valid_loss += loss
            loss_list.append(self.loss_func(out, y, output_idx, reduction='none').detach())
            task_idx_list.append(task_idx.detach())
            y_true_list.append(y.detach())
            y_pred_list.append(out.detach())

        # dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        # valid_loss = valid_loss.item() / (valid_steps * dist.get_world_size())
        # self.log(f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, valid_loss = {valid_loss:.6f}')
        
        loss_list = torch.cat(loss_list)
        task_idx_list = torch.cat(task_idx_list)
        y_true_list = torch.cat(y_true_list)
        y_pred_list = torch.cat(y_pred_list)

        if self.local_rank == 0:
            if self.distribute:
                loss_list = dist_all_gather(loss_list).cpu()
                task_idx_list = dist_all_gather(task_idx_list).cpu()
                y_true_list = dist_all_gather(y_true_list).cpu()
                y_pred_list = dist_all_gather(y_pred_list).cpu()
            else:
                loss_list = loss_list.cpu()
                task_idx_list = task_idx_list.cpu()
                y_true_list = y_true_list.cpu()
                y_pred_list = y_pred_list.cpu()
            
            valid_loss = loss_list.mean()

            self.log(f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, valid_loss = {valid_loss:.6f}')

            for task_idx in self.selected_valid_datasets_idx:
                task_name = self.task_names[task_idx]
                loss_list_0 = loss_list[torch.where(task_idx_list == task_idx)]
                y_true_list_0 = y_true_list[torch.where(task_idx_list == task_idx)]
                y_pred_list_0 = y_pred_list[torch.where(task_idx_list == task_idx)]
                log_message = f'task_name = {task_name:10}, loss = {loss_list_0.mean():.6f}'
                # self.log(log_message)
                # log_message = ''
                for metric_func in self.metric_func_list:
                    score = metric_func(y_pred_list_0, y_true_list_0)
                    log_message += f', {type(metric_func).__name__} = {score:.6f}'
                self.log(log_message)

        torch.set_grad_enabled(True)
        return valid_loss


    # def save_model(self):
    #     checkpoint_dir = self.config.get('checkpoint_dir', None)

    #     checkpoint = {
    #         'config': self.config,
    #         'epoch': self.epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         }

    #     checkpoint_path = os.path.join(checkpoint_dir, f'epoch{self.epoch}.pth')
    #     torch.save(checkpoint, checkpoint_path)
    #     self.logger.debug(f'save model at {checkpoint_path}')



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
