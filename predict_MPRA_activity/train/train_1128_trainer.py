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
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data
import torchinfo

from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import *


class Trainer:
    def __init__(
            self,
            config,
            rank,
            logger,
            model,
            loss_func,
            metric_funcs,
            optimizer,
            device,
            train_dataset,
            valid_dataset,
            lr_scheduler=None,
            early_stopper=None,
    ):
        self.config = config
        self.rank = rank
        self.logger = logger
        self.model = model
        self.loss_func = loss_func
        self.metric_funcs = metric_funcs
        self.optimizer = optimizer
        self.device = device

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr_scheduler = lr_scheduler
        self.early_stopper = early_stopper
        
        self.log = self.logger.info if self.rank == 0 else self.logger.log
        self.distributed = config['distributed']

        self.train_sampler = DistributedSampler(self.train_dataset)
        self.valid_sampler = DistributedSampler(self.valid_dataset)

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config['batch_size'], 
            sampler=self.train_sampler if self.distributed else None,
            num_workers=1, 
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(0),)
        self.valid_loader = DataLoader(
            self.valid_dataset, 
            batch_size=config['batch_size'], 
            sampler=self.valid_sampler if self.distributed else None,
            num_workers=1, 
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(0),)
    
        self.cell_types = config['cell_types']


    def train(self):
        config = self.config
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        num_valid_epochs = config['num_valid_epochs']
        
        self.log(f'len(train_dataset) = {len(self.train_dataset)}')
        self.log(f'len(valid_dataset) = {len(self.valid_dataset)}')
        self.log(f'len(train_loader) = {len(self.train_loader)}')
        self.log(f'len(valid_loader) = {len(self.valid_loader)}')
        self.log(f'num_epochs = {num_epochs}')
        self.log(f'batch_size = {batch_size}')
        self.log(f'learnrate = {config["optimizer"]["args"]["lr"]}')
        self.log(f'start training')

        inputs, labels = next(iter(self.train_loader))
        inputs = to_device(inputs, self.device)
        self.log(
            torchinfo.summary(
                self.model, 
                input_data=[inputs], 
                verbose=0, 
                depth=10,))

        for epoch in range(-1, num_epochs):
            self.epoch = epoch
            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            # valid one epoch before training
            if (epoch == -1):
                self.log(f'valid on epoch {epoch}')
                self.valid_epoch(epoch)
                continue

            self.log(f'train on epoch {epoch}')
            self.train_epoch(epoch)
            
            if ((epoch+1) % num_valid_epochs == 0):
                self.log(f'valid on epoch {epoch}')
                self.valid_epoch(epoch)

                if (self.early_stopper is not None):
                    valid_pearson = self.metric_df.loc[self.cell_types, 'Pearson'].mean()
                    self.log(f'epoch = {epoch}, valid_pearson = {valid_pearson:.6f}, check for early stopping')
                    # we should not use valid_loss to check early stopping
                    self.early_stopper.check(valid_pearson)

                    if self.early_stopper.update_flag == True:
                        if self.rank == 0:
                            self.save_model()
                    if self.early_stopper.stop_flag == True:
                        break

        self.log(f'rank = {self.rank:1}, finish training.')

        if self.distributed:
            dist.destroy_process_group()


    def train_epoch(self, epoch):
        scheduler_interval = self.config.get('scheduler_interval', 'epoch')
        num_log_steps = self.config.get('num_log_steps', 0)
        train_steps = len(self.train_loader)
        train_loss = 0

        self.model.train()
        for batch_idx, (inputs, labels) in enumerate(tqdm(self.train_loader, disable=(self.rank != 0))):
            inputs = to_device(inputs, self.device)
            labels = to_device(labels, self.device)
            out = self.model(inputs)
            loss = self.loss_func(out, labels)
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10) # clip gradient
            self.optimizer.step()
            
            if scheduler_interval == 'step':
                self.lr_scheduler.step(self.epoch + batch_idx/train_steps)
            
            train_loss += loss.item()
            if num_log_steps != 0 and batch_idx % num_log_steps == 0:
                self.logger.debug(
                    f'rank = {self.rank}, epoch = {self.epoch:3}, '
                    f'batch_idx = {batch_idx:3}, train_loss = {loss.item():.6f}')

        if scheduler_interval == 'epoch':
            self.lr_scheduler.step()

        train_loss = train_loss / train_steps
        self.log(f'rank = {self.rank:1}, epoch = {self.epoch:3}, train_loss = {train_loss:.6f}')

    def valid_epoch(self, epoch):
        torch.set_grad_enabled(False)
        # 代替with torch.no_grad()，避免多一层缩进，和train缩进一样方便复制

        valid_steps = len(self.valid_loader)
        valid_loss = 0
        y_pred_list = []

        self.model.eval()
        for batch_idx, (inputs, labels) in enumerate(tqdm(self.valid_loader, disable=(self.rank != 0))):
            inputs = to_device(inputs, self.device)
            labels = to_device(labels, self.device)
            out = self.model(inputs)
            loss = self.loss_func(out, labels)
            valid_loss += loss
            y_pred_list.append(out.detach())

        valid_loss = valid_loss / valid_steps
        self.log(f'rank = {self.rank:1}, epoch = {self.epoch:3}, valid_loss = {valid_loss:.6f}')

        y_pred_list = torch.cat(y_pred_list)
        y_pred_list = y_pred_list.cpu()

        y_true_list = self.valid_dataset.labels
        assert y_pred_list.shape == y_true_list.shape, f'y_pred_list.shape = {y_pred_list.shape}, y_true_list.shape = {y_true_list.shape}'

        for idx, cell_type in enumerate(self.cell_types):
            log_message = f'cell_type = {cell_type:6}'
            if len(y_true_list.shape) == 1:
                indice = (self.valid_dataset.df['cell_type'] == cell_type)
                y_true_list_0 = y_true_list[indice]
                y_pred_list_0 = y_pred_list[indice]
            elif len(y_true_list.shape) == 2:
                y_true_list_0 = y_true_list[:, idx]
                y_pred_list_0 = y_pred_list[:, idx]
            else:
                raise ValueError(f'y_true_list.shape = {y_true_list.shape}')
            
            for metric_func in self.metric_funcs:
                metric_name = type(metric_func).__name__
                score = metric_func(y_pred_list_0, y_true_list_0)
                log_message += f', {metric_name} = {score:.6f}'
                self.metric_df.loc[cell_type, metric_name] = score
            self.log(log_message)
        torch.set_grad_enabled(True)


    def test(self, test_loader):
        torch.set_grad_enabled(False)
        y_pred_list = []
        self.model.eval()
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, disable=(self.rank != 0))):
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

    
    def save_model(self):
        checkpoint = {
            'config': self.config,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }
        checkpoint = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        checkpoint_dir = self.config['save_dir']
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f'save model at {checkpoint_path}')
        return


def load_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    if 'model_state_dict' in state_dict:
        model_state_dict = state_dict['model_state_dict']
    else:
        model_state_dict = state_dict
    if 'module' in model_state_dict.keys()[0]:
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    return model


def main_worker(rank, config):
    # 设置随机种子
    utils.set_seed(config['seed'] + rank)

    # 设置logger
    logging.config.dictConfig(config['logger'])
    logger = logging.getLogger()

    # 设置分布式训练
    distributed = config['distributed']
    gpu_ids = config['gpu_ids']
    world_size = len(config['gpu_ids'])
    device = torch.device(f'cuda:{gpu_ids[rank]}')
    torch.cuda.set_device(device)
    port = random.randint(10000, 20000)
    dist.init_process_group(
        backend='nccl',
        init_method=f"tcp://localhost:{port}",
        world_size=world_size,
        rank=rank,
    )
    logger.info(f"Process {rank}/{world_size} initialized on device {device}.")

    # 建立模型
    model = utils.init_obj(models, config['model'])
    if config.get('load_saved_model', False):
        model = load_model(model, config['saved_model_path'])
        logger.info(f"Process {rank}: Loaded saved model from {config['saved_model_path']}.")

    # 使用 DDP 包装模型
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model.to(device), device_ids=[gpu_ids[rank]], find_unused_parameters=False)

    # 数据和优化器
    train_dataset = utils.init_obj(datasets, config['train_dataset'])
    valid_dataset = utils.init_obj(datasets, config['valid_dataset'])

    loss_func = utils.init_obj(metrics, config['loss_func'])
    metric_funcs = [utils.init_obj(metrics, m) for m in config.get('metric_funcs', [])]
    optimizer = utils.init_obj(torch.optim, config['optimizer'], filter(lambda p: p.requires_grad, model.parameters()))
    lr_scheduler = utils.init_obj(torch.optim.lr_scheduler, config['lr_scheduler'], optimizer)
    early_stopper = utils.init_obj(utils, config['early_stopper'], save_dir=config['save_dir'])

    # 开始训练
    trainer = Trainer(
        config, rank, logger, model, loss_func, metric_funcs, optimizer, device,
        train_dataset, valid_dataset, lr_scheduler, early_stopper
    )
    trainer.train()

    dist.destroy_process_group()


def main(config):
    if config['distributed']:
        world_size = len(config['gpu_ids'])
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=main_worker, args=(rank, config))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        main_worker(0, config)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config_path', type=str, default=None,
                      help='config file path',)
    args = args.parse_args()
    config = utils.load_config(args.config_path)
    config = utils.process_config(config)

    main(config)
