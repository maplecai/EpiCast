import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchinfo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MPRA_predict import models, datasets, metrics, utils

class Trainer:
    def __init__(self, config):
        self.config = config
        utils.set_seed(config['seed'])

        # setup logger
        logging.config.dictConfig(config['logger'])
        self.logger = logging.getLogger()

        # setup distributed training
        self.distributed = config['distributed']
        if not self.distributed:
            self.local_rank = 0
            self.gpu_id = config['gpu_ids'][0]
            self.device = torch.device(f'cuda:{self.gpu_id}')
            torch.cuda.set_device(self.device)
            self.logger.info(f"Start non DDP training on rank {self.local_rank}, {self.device}.")
        else:
            dist.init_process_group(backend='nccl', init_method='env://')
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.gpu_id = config['gpu_ids'][self.local_rank]
            self.device = torch.device(f'cuda:{self.gpu_id}')
            torch.cuda.set_device(self.device)
            self.logger.info(f"Start DDP training on rank {self.local_rank}, {self.device}.")


        if self.local_rank == 0:
            self.log = self.logger.info
        else:
            self.log = self.logger.debug

        # setup dataloader
        self.train_dataset = utils.init_obj(
            datasets, 
            config['train_dataset'])
        self.valid_dataset = utils.init_obj(
            datasets, 
            config['valid_dataset'])
        
        if not self.distributed:
            self.train_loader = DataLoader(
                dataset=self.train_dataset, 
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=1,
                pin_memory=True,
                drop_last=False)
            self.valid_loader = DataLoader(
                dataset=self.valid_dataset, 
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False)
        else:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.valid_sampler = DistributedSampler(self.valid_dataset, shuffle=False)
            self.train_loader = DataLoader(
                dataset=self.train_dataset, 
                batch_size=config['batch_size'] // config['world_size'],
                sampler=self.train_sampler,
                num_workers=1,
                pin_memory=True,
                drop_last=False)
            self.valid_loader = DataLoader(
                dataset=self.valid_dataset, 
                batch_size=config['batch_size'] // config['world_size'],
                sampler=self.valid_sampler,
                num_workers=1,
                pin_memory=True,
                drop_last=False)
            
        # setup model
        self.model = utils.init_obj(models, config['model'])

        if config.get('load_saved_model', False) == True:
            saved_model_path = config['saved_model_path']
            state_dict = torch.load(saved_model_path)
            self.model.load_state_dict(state_dict)
            self.log(f"load saved model from {saved_model_path}")


        # if config.get('freeze_layers', False) == True:
        #     self.log(f"freeze conv layers")
        #     for name, param in self.model.named_parameters():
        #         if name.startwith('conv_layers'):
        #             param.requires_grad = False
        #         elif name.startwith('linear_layers'):
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = True

        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.gpu_id], 
                find_unused_parameters=False)
        
        # setup training
        self.loss_func = utils.init_obj(metrics, config['loss_func'])
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = utils.init_obj(torch.optim, config['optimizer'], trainable_params)
        self.lr_scheduler = utils.init_obj(
                torch.optim.lr_scheduler, 
                config['lr_scheduler'], 
                self.optimizer)
        self.early_stopper = utils.init_obj(
                utils, 
                config['early_stopper'], 
                save_dir=os.path.join(config['save_dir']), 
                trace_func=self.log)
        
        # setup metrics
        self.cell_types = config['cell_types']
        self.metric_funcs = [
            utils.init_obj(metrics, m) for m in config.get('metric_funcs', [])]
        self.metric_names = [
            m['type'] for m in config.get('metric_funcs', [])]
        self.metric_df = pd.DataFrame(
            index=self.cell_types, 
            columns=self.metric_names)


    def train(self):
        config = self.config
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        num_valid_epochs = config['num_valid_epochs']
        
        if self.local_rank == 0:
            sample = next(iter(self.train_loader))
            sample = utils.to_device(sample, self.device)
            self.log(torchinfo.summary(
                self.model, 
                input_data=[sample], 
                verbose=0, 
                depth=5))
            
        self.log(f'cell_types = {self.cell_types}')
        self.log(f'len(train_dataset) = {len(self.train_dataset)}')
        self.log(f'len(valid_dataset) = {len(self.valid_dataset)}')
        self.log(f'len(train_loader) = {len(self.train_loader)}')
        self.log(f'len(valid_loader) = {len(self.valid_loader)}')
        self.log(f'num_epochs = {num_epochs}')
        self.log(f'batch_size = {batch_size}')
        self.log(f'start training')

        for epoch in range(num_epochs):
            self.epoch = epoch
            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            # valid one epoch before training
            if (epoch == 0):
                self.valid_epoch()

            self.log(f'train on epoch {epoch}')
            self.train_epoch()
            
            if ((epoch+1) % num_valid_epochs == 0):
                self.log(f'valid on epoch {epoch}')
                self.valid_epoch()

                if (self.early_stopper is not None):
                    valid_pearson = self.metric_df.loc[self.cell_types, 'Pearson'].mean()
                    self.log(f'epoch = {epoch}, valid_pearson = {valid_pearson:.6f}, check for early stopping')
                    # use valid_pearson instread of valid_loss to check early stopping
                    self.early_stopper.check(valid_pearson)

                    if self.early_stopper.update_flag == True:
                        if self.local_rank == 0:
                            self.save_model()
                    if self.early_stopper.stop_flag == True:
                        break

        self.log(f'local_rank = {self.local_rank:1}, finish training.')

        if self.distributed:
            dist.destroy_process_group()


    def train_epoch(self, train_loader=None):
        if train_loader is None:
            train_loader = self.train_loader
        scheduler_interval = self.config.get('scheduler_interval', 'epoch')
        num_log_steps = self.config.get('num_log_steps', 0)
        train_steps = len(train_loader)
        train_loss = 0

        self.model.train()
        for batch_idx, sample in enumerate(tqdm(train_loader, disable=(self.local_rank != 0))):
            sample = utils.to_device(sample, self.device)
            pred = self.model(sample)
            label = sample['label']
            loss = self.loss_func(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
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
        self.log(f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, train_loss = {train_loss:.6f}')


    @torch.no_grad()
    def valid_epoch(self, valid_loader=None):
        if valid_loader is None:
            valid_loader = self.valid_loader

        valid_steps = len(valid_loader)
        valid_loss = 0
        idx_list = []
        pred_list = []
        label_list = []

        self.model.eval()
        for batch_idx, sample in enumerate(tqdm(valid_loader, disable=(self.local_rank != 0))):
            sample = utils.to_device(sample, self.device)
            idx = sample['idx']
            pred = self.model(sample)
            label = sample['label']
            loss = self.loss_func(pred, label)
            valid_loss += loss
            idx_list.append(idx.detach())
            pred_list.append(pred.detach())
            label_list.append(label.detach())

        valid_loss = valid_loss / valid_steps
        if self.distributed:
            dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
            valid_loss = valid_loss / self.config['world_size']
        self.log(f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, valid_loss = {valid_loss:.6f}')

        idx_list = torch.cat(idx_list, dim=0)
        pred_list = torch.cat(pred_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        if self.distributed:
            idx_list = utils.dist_all_gather(idx_list).cpu()
            pred_list = utils.dist_all_gather(pred_list).cpu()
            label_list = utils.dist_all_gather(label_list).cpu()
        else:
            idx_list = idx_list.cpu()
            pred_list = pred_list.cpu()
            label_list = label_list.cpu()

        # sort pred by sample idx
        sorted_idx, sort_indices = torch.sort(idx_list)
        idx_list = sorted_idx
        pred_list = pred_list[sort_indices]
        label_list = label_list[sort_indices]

        for i, cell_type in enumerate(self.cell_types):
            log_message = f'cell_type = {cell_type:6}'
            if len(label_list.shape) == 1:
                indice = (self.valid_dataset.df['cell_type'] == cell_type)
                label_list_0 = label_list[indice]
                pred_list_0 = pred_list[indice]
            elif len(label_list.shape) == 2:
                label_list_0 = label_list[:, i]
                pred_list_0 = pred_list[:, i]
            else:
                raise ValueError(f'label_list.shape = {label_list.shape}')
            
            for metric_func in self.metric_funcs:
                metric_name = type(metric_func).__name__
                score = metric_func(pred_list_0, label_list_0)
                log_message += f', {metric_name} = {score:.6f}'
                self.metric_df.loc[cell_type, metric_name] = score
            self.log(log_message)


    @torch.no_grad()
    def test(self, test_loader):
        # only single gpu
        self.model.eval()
        idx_list = []
        pred_list = []
        label_list = []
        for batch_idx, sample in enumerate(tqdm(test_loader, disable=(self.local_rank != 0))):
            sample = utils.to_device(sample, self.device)
            idx = sample['idx']
            pred = self.model(sample)
            label = sample['label']
            idx_list.append(idx.detach())
            pred_list.append(pred.detach())
            label_list.append(label.detach())

        idx_list = torch.cat(idx_list).cpu().numpy()
        pred_list = torch.cat(pred_list).cpu().numpy()
        label_list = torch.cat(label_list).cpu().numpy()

        save_file_path = os.path.join(self.config['save_dir'], f'test_pred.npy')
        np.save(save_file_path, pred_list)


    def save_model(self):
        checkpoint_dir = self.config['save_dir']

        # checkpoint = {
        #     'config': self.config,
        #     'epoch': self.epoch,
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     }
        # checkpoint_path = os.path.join(checkpoint_dir, f'epoch{self.epoch}.pth')
        
        checkpoint = self.model.module.state_dict() if self.distributed else self.model.state_dict()
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

    # # 设置分布式环境变量 （如果使用torchrun,则不需要手动设置）
    # if config['distributed']:
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '13579'

    trainer = Trainer(config)
    trainer.train()
