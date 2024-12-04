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
import torch.multiprocessing as mp

class Trainer:
    def __init__(self, config, rank=0):
        # setup seed and distribute
        self.config = config
        utils.set_seed(config['seed'])
        logging.config.dictConfig(config['logger'])
        self.logger = logging.getLogger()

        self.distribute = config['distribute']
        self.gpu_ids = config['gpu_ids']
        self.world_size = len(self.gpu_ids)
        self.rank = rank
        
        if self.distribute == False:
            self.device = torch.device(f'cuda:{self.gpu_ids[0]}')
            torch.cuda.set_device(self.device)
            self.logger.info(
                f"Start non-distributed training on rank {self.rank}, {self.device}.")
        else:
            dist.init_process_group(
                backend='nccl', 
                init_method="tcp://localhost:12355",
                world_size=self.world_size,
                rank=self.rank)
            self.device = torch.device(f'cuda:{self.gpu_ids[self.rank]}')
            torch.cuda.set_device(self.device)
            self.logger.info(
                f"Start DDP training on rank {self.rank}, {self.device}.")

        if self.rank == 0:
            self.log = self.logger.info
        else:
            self.log = self.logger.debug

        # setup dataloader
        self.cell_types = config['cell_types']

        if config.get('train', False):
            config['batch_size'] = config['global_batch_size'] // self.world_size

            self.train_dataset = utils.init_obj(
                datasets, 
                config['train_dataset'])
            if self.distribute == False:
                self.train_loader = DataLoader(
                    self.train_dataset, 
                    batch_size=config['batch_size'], 
                    shuffle=True, 
                    num_workers=1, 
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(0),)
                # self.train_loader = utils.init_obj(
                #     torch.utils.data, 
                #     config['data_loader'], 
                #     dataset=self.train_dataset, 
                #     shuffle=True,
                #     worker_init_fn=seed_worker,
                #     generator=torch.Generator().manual_seed(0),)
                # self.valid_loader = utils.init_obj(
                #     torch.utils.data, 
                #     config['data_loader'], 
                #     dataset=self.valid_dataset, 
                #     shuffle=False,
                #     worker_init_fn=seed_worker,
                #     generator=torch.Generator().manual_seed(0),)
            else:
                self.train_sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=len(self.gpu_ids),
                    rank=self.rank,
                    shuffle=True,)
                self.train_loader = DataLoader(
                    self.train_dataset,
                    sampler=self.train_sampler,
                    batch_size=config['batch_size'],
                    num_workers=1, 
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(0),)
                # self.train_loader = utils.init_obj(
                #     torch.utils.data, 
                #     config['data_loader'], 
                #     dataset=self.train_dataset, 
                #     sampler=self.train_sampler,
                #     worker_init_fn=seed_worker,
                #     generator=torch.Generator().manual_seed(0),)
                # self.valid_loader = utils.init_obj(
                #     torch.utils.data, 
                #     config['data_loader'], 
                #     dataset=self.valid_dataset, 
                #     shuffle=False,
                #     worker_init_fn=seed_worker,
                #     generator=torch.Generator().manual_seed(0),)
            self.valid_dataset = utils.init_obj(
                datasets, 
                config['valid_dataset'])
            self.valid_loader = DataLoader(
                self.valid_dataset, 
                batch_size=config['global_batch_size'], 
                shuffle=False, 
                num_workers=1, 
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(0),)
        else:
            self.test_dataset = utils.init_obj(
                datasets, 
                config['test_dataset'])
            self.test_loader = DataLoader(
                self.test_dataset, 
                batch_size=config['global_batch_size'], 
                shuffle=False,
                num_workers=1, 
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(0),)
            
        self.model = utils.init_obj(models, config['model'])

        if config.get('load_saved_model', False):
            saved_model_path = config['saved_model_path']
            state_dict = torch.load(saved_model_path)
            if 'model_state_dict' in state_dict:
                model_state_dict = state_dict['model_state_dict']
            else:
                model_state_dict = state_dict
            self.model.load_state_dict(model_state_dict)
            self.log(f"load saved model from {saved_model_path}")

        if config.get('load_partial_saved_model', False):
            load_partial_parameters(self.model, config['saved_model_path'], None, self.log)
            self.log(f"load partial saved model from {config['saved_model_path']}")

        if config.get('freeze_layers', False):
            self.log(f"freeze conv layers")
            for name, param in self.model.named_parameters():
                if name.startswith('conv_layers'):
                    param.requires_grad = False
                elif name.startswith('linear_layers'):
                    param.requires_grad = True
                else:
                    param.requires_grad = True
        if not self.distribute:
            self.model = self.model.to(self.device)
        else:
            # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(
                self.model.to(self.device), 
                device_ids=[self.gpu_ids[self.rank]], 
                find_unused_parameters=False)

        self.loss_func = utils.init_obj(metrics, config['loss_func'])
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = utils.init_obj(torch.optim, config['optimizer'], trainable_params)
        
        self.metric_func_list = [
            utils.init_obj(metrics, m) for m in config.get('metric_func_list', [])]
        self.metric_names = [
            m['type'] for m in config.get('metric_func_list', [])]
        self.metric_df = pd.DataFrame(
            index=self.cell_types, 
            columns=self.metric_names)
        
        self.lr_scheduler = utils.init_obj(
            torch.optim.lr_scheduler, 
            config['lr_scheduler'], 
            self.optimizer)

        self.early_stopper = utils.init_obj(
            utils, 
            config['early_stopper'], 
            save_dir=os.path.join(config['save_dir']), 
            trace_func=self.log)

    def train(self):
        config = self.config
        num_epochs = config['num_epochs']
        # batch_size = config['data_loader']['args']['batch_size']
        batch_size = config['batch_size']
        num_valid_epochs = config['num_valid_epochs']
            
        self.log(f'cell_types = {self.cell_types}')
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
        self.log(torchinfo.summary(
            self.model, 
            input_data=[inputs], 
            verbose=0, 
            depth=10,))

        for epoch in range(-1, num_epochs):
            self.epoch = epoch
            if self.distribute:
                self.train_sampler.set_epoch(epoch)

            # valid one epoch before training
            if (epoch == -1):
                self.log(f'valid on epoch {epoch}')
                self.valid_epoch(self.valid_loader)
                continue

            self.log(f'train on epoch {epoch}')
            self.train_epoch(self.train_loader)
            
            if ((epoch+1) % num_valid_epochs == 0):
                self.log(f'valid on epoch {epoch}')
                self.valid_epoch(self.valid_loader)

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

        if self.distribute:
            dist.destroy_process_group()

    def train_epoch(self, train_loader):
        scheduler_interval = self.config.get('scheduler_interval', 'epoch')
        num_log_steps = self.config.get('num_log_steps', 0)
        train_steps = len(train_loader)
        train_loss = 0

        self.model.train()
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, disable=(self.rank != 0))):
            inputs = to_device(inputs, self.device)
            labels = to_device(labels, self.device)
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
                    f'rank = {self.rank}, epoch = {self.epoch:3}, '
                    f'batch_idx = {batch_idx:3}, train_loss = {loss.item():.6f}')

        if scheduler_interval == 'epoch':
            self.lr_scheduler.step()

        train_loss = train_loss / train_steps
        self.log(f'rank = {self.rank:1}, epoch = {self.epoch:3}, train_loss = {train_loss:.6f}')

    def valid_epoch(self, valid_loader):
        torch.set_grad_enabled(False)
        # 代替with torch.no_grad()，避免多一层缩进，和train缩进一样，方便复制

        valid_steps = len(valid_loader)
        valid_loss = 0
        y_pred_list = []

        self.model.eval()
        for batch_idx, (inputs, labels) in enumerate(tqdm(valid_loader, disable=(self.rank != 0))):
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
            
            for metric_func in self.metric_func_list:
                metric_name = type(metric_func).__name__
                score = metric_func(y_pred_list_0, y_true_list_0)
                log_message += f', {metric_name} = {score:.6f}'
                self.metric_df.loc[cell_type, metric_name] = score
            self.log(log_message)
        torch.set_grad_enabled(True)

    def test(self, test_loader):
        torch.set_grad_enabled(False)
        # 代替with torch.no_grad()，避免多一层缩进，和train缩进一样，方便复制

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
            'model_state_dict': self.model.state_dict() if self.distribute else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }
        checkpoint = self.model.module.state_dict() if self.distribute else self.model.state_dict()
        
        checkpoint_dir = self.config['save_dir']
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f'save model at {checkpoint_path}')
        return

def dist_all_gather(tensor):
    tensor_list = [torch.zeros_like(tensor, device=tensor.device) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list)
    return tensor_list

def main_worker(rank, config):
    trainer = Trainer(config, rank)
    trainer.train()


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config_path', type=str, default=None,
                      help='config file path',)
    args = args.parse_args()

    config = utils.load_config(args.config_path)
    config = utils.process_config(config)
    
    if config['distribute'] == True:
        # mp.spawn(main_worker, args=(config,), nprocs=len(gpu_ids))
        processes = []
        for rank in range(len(config['gpu_ids'])):
            p = mp.Process(target=main_worker, args=(rank, config))
            p.start()     
            processes.append(p)
        for p in processes:
            p.join()

    else:
        main_worker(0, config)


if __name__ == '__main__':
    main()
