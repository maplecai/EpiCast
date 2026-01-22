import os
import re
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from ruamel.yaml import YAML
from io import StringIO

import torch
import torchinfo
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, R2Score, PearsonCorrCoef

from varlen_genomics import models, datasets, utils, metrics


class WdsTrainer:
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
            if config['device'] == 'auto':
                self.device = utils.get_free_gpus()[0]
            elif isinstance(config['device'], list):
                self.device = config['device'][0]
            else:
                self.device = config['device']
            torch.cuda.set_device(self.device)
            self.logger.info(f"Start non DDP training on rank {self.local_rank}, {self.device}.")
        else:
            dist.init_process_group(backend='nccl', init_method='env://')
            self.local_rank = int(os.environ["LOCAL_RANK"])

            if config['device'] == 'auto':
                # auto 时默认按 local_rank 编号
                self.device = f"cuda:{self.local_rank}"
            elif isinstance(config['device'], list):
                # 例如 ["cuda:0", "cuda:1", ...]
                self.device = config['device'][self.local_rank]
            else:
                raise ValueError('DDP device should be a list or "auto"')
            torch.cuda.set_device(self.device)
            self.logger.info(f"Start DDP training on rank {self.local_rank}, {self.device}.")

        if self.local_rank == 0:
            self.log = self.logger.info
        else:
            self.log = self.logger.debug
        
        yaml = YAML()
        stream = StringIO()
        yaml.dump(self.config, stream)
        self.log(stream.getvalue())

        # setup dataset and dataloader
        self.train_dataset = utils.init_obj(datasets, config['train_dataset'])
        self.valid_dataset = utils.init_obj(datasets, config['valid_dataset'])

        self.train_loader = utils.init_obj(torch.utils.data, config['train_loader'], dataset=self.train_dataset)
        self.valid_loader = utils.init_obj(torch.utils.data, config['valid_loader'], dataset=self.valid_dataset)



        # setup model
        self.model = utils.init_obj(models, config['model'])

        if config.get('load_saved_model', False) == True:
            checkpoint_path = config['saved_model_path']
            self.load_checkpoint(checkpoint_path)

            # pretrained_dict = torch.load(saved_model_path)
            # # missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            # # print("Missing keys:", missing_keys)
            # # print("Unexpected keys:", unexpected_keys)
            # # self.log(f"load saved model from {saved_model_path}")

            # model_dict = self.model.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items()
            #                 if k in model_dict and v.shape == model_dict[k].shape}
            # model_dict.update(pretrained_dict)
            # self.model.load_state_dict(model_dict)

        if config.get("freeze_parameters", False):
            self.log("freeze parameters")
            patterns = config.get("freeze_patterns", [])  # 改名更明确
            regex_list = [re.compile(p) for p in patterns]

            for name, param in self.model.named_parameters():
                for regex in regex_list:
                    if regex.search(name):   # 正则匹配
                        param.requires_grad = False
                        self.log(f"freeze parameter {name} (matched by {regex.pattern})")
                        break


        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model, 
                device_ids=[int(self.device.split(':')[-1])],
                find_unused_parameters=False,
            )
        
        # setup training
        if 'transformer_args' in self.config['optimizer']:
            transformer_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'transformer' in name:
                        transformer_params.append(param)
                    else:
                        other_params.append(param)
            param_groups = [
                {
                    'params': transformer_params,
                    **config['optimizer']['transformer_args'],
                },
                {
                    'params': other_params,
                    **config['optimizer']['args'],
                },
            ]
            self.optimizer = utils.init_obj(
                torch.optim,
                config['optimizer'],
                param_groups
            )

        else:
            trainable_params = [param for param in self.model.parameters() if param.requires_grad]
            self.optimizer = utils.init_obj(
                torch.optim, 
                config['optimizer'], 
                trainable_params
            )

        self.loss_func = utils.init_obj(
            metrics, 
            config['loss_func']
        )

        self.lr_scheduler = utils.init_obj(
            utils, 
            config['lr_scheduler'], 
            self.optimizer
        )
        self.early_stopper = utils.init_obj(
            utils, 
            config['early_stopper'], 
            saved_root_dir=os.path.join(config['saved_root_dir']), 
            trace_func=self.log
        )
        
        self.metric_df = pd.DataFrame(columns=['mse', 'r2', 'pearson'])

        self.metrics = MetricCollection({
            "mse": MeanSquaredError(num_outputs=5313, sync_on_compute=True),
            "r2": R2Score(num_outputs=5313, sync_on_compute=True),
            "pearson": PearsonCorrCoef(num_outputs=5313, sync_on_compute=True),
        }).to(self.device)

    def train(self):
        config = self.config
        max_epochs = config['max_epochs']
        batch_size = config['batch_size']
        epochs_per_valid = config['epochs_per_valid']

        self.log(f'max_epochs = {max_epochs}, batch_size = {batch_size}, epochs_per_valid = {epochs_per_valid}')
        self.log(f'start training')

        if self.local_rank == 0:
            sample = next(iter(self.train_loader))
            sample = utils.to_device(sample, self.device)
            self.log(torchinfo.summary(
                self.model, 
                input_data=[sample], 
                verbose=0, 
                depth=5,
                col_names=["input_size", "output_size", "num_params"],
                row_settings=["var_names"],
            ))

        for epoch in range(max_epochs):
            self.epoch = epoch

            # # valid one epoch before training
            # if (epoch == 0):
            #     self.valid_epoch()

            self.log(f'train on epoch {epoch}')
            self.train_epoch()
            
            if ((epoch+1) % epochs_per_valid == 0):
                self.log(f'valid on epoch {epoch}')
                self.valid_epoch()

                if (self.local_rank == 0) and (self.early_stopper is not None):
                    self.early_stopper.check(self.metric_df.loc[self.epoch, 'pearson'])
                    if self.early_stopper.update_flag is True:
                        self.save_checkpoint(self.epoch, 0)
                    if self.early_stopper.stop_flag == True:
                        break

        self.log(f'local_rank = {self.local_rank:1}, finish training.')

        if self.distributed:
            dist.destroy_process_group()


    def train_epoch(self):
        steps_per_log = self.config.get('steps_per_log', 0)
        train_steps = 0
        train_loss = 0

        self.model.train()
        for batch_idx, sample in enumerate(tqdm(self.train_loader, disable=(self.local_rank != 0))):
            sample = utils.to_device(sample, self.device)
            pred = self.model(sample)
            target = sample['target']
            loss = self.loss_func(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            train_steps += 1
            train_loss += loss.item()
            if steps_per_log != 0 and batch_idx % steps_per_log == 0:
                self.logger.debug(
                    f'local_rank = {self.local_rank}, epoch = {self.epoch}, '
                    f'batch_idx = {batch_idx:3}, train_batch_loss = {loss.item():.6f}')

        self.lr_scheduler.step()

        train_loss = train_loss / train_steps
        self.log(f'local_rank = {self.local_rank}, epoch = {self.epoch}, train_loss = {train_loss:.6f}')


    @torch.no_grad()
    def valid_epoch(self):

        valid_steps = 0
        valid_loss = 0

        self.model.eval()
        self.metrics.reset()
        for batch_idx, sample in enumerate(tqdm(self.valid_loader, disable=(self.local_rank != 0))):
            sample = utils.to_device(sample, self.device)
            pred = self.model(sample)
            target = sample['target']
            loss = self.loss_func(pred, target)
            valid_steps += 1
            valid_loss += loss.item()

            B, L, C = pred.shape
            pred = pred.reshape(B*L, C)
            target = target.reshape(B*L, C)
            self.metrics.update(pred, target)

        valid_loss = valid_loss / valid_steps
        # if self.distributed:
        #     dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        #     valid_loss = valid_loss / self.config['world_size']
        self.log(f'local_rank = {self.local_rank}, epoch = {self.epoch}, valid_loss = {valid_loss:.6f}')

        self.results = self.metrics.compute()
        if self.local_rank == 0:
            for name, val in self.results.items():
                score = val.mean().cpu().numpy()
                self.metric_df.loc[self.epoch, name] = score
                self.log(f"local_rank = {self.local_rank}, epoch = {self.epoch}, {name:9} = {score:.6f}")

    @torch.no_grad()
    def test(self):
        self.model.eval()
        pred_list = []
        target_list = []
        for batch_idx, sample in enumerate(tqdm(self.test_loader, disable=(self.local_rank != 0))):
            sample = utils.to_device(sample, self.device)
            pred = self.model(sample)
            target = sample['target']
            pred_list.append(pred.detach())
            target_list.append(target.detach())

        pred_list = torch.cat(pred_list).cpu().numpy()
        target_list = torch.cat(target_list).cpu().numpy()

        save_file_path = os.path.join(self.config['saved_root_dir'], f'test_pred.npy')
        np.save(save_file_path, pred_list)
        torch.cuda.empty_cache()
        return


    def save_checkpoint(self, epoch, step, filename="checkpoint.pt"):
        """Save model/optimizer/lr_scheduler states.

        Args:
            epoch: current epoch
            step: current global step
            filename: save file name
        """

        # only the main process (rank0) saves checkpoint
        if self.distributed and self.local_rank != 0:
            return

        save_path = os.path.join(self.config['saved_root_dir'], filename)

        # If using DDP, model is wrapped in model.module
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        state = {
            "epoch": epoch,
            "step": step,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if hasattr(self, "lr_scheduler") else None,
            "config": self.config,
        }

        torch.save(state, save_path)

        self.log(f"Checkpoint saved to {save_path}")



    def load_checkpoint(self, filename="checkpoint.pt", load_optimizer=True, load_lr_scheduler=True):
        """
        Load model/optimizer/lr_scheduler states.

        Args:
            filename (str): checkpoint file name.
            load_optimizer (bool): whether to load optimizer state.
            load_lr_scheduler (bool): whether to load lr scheduler state.

        Returns:
            epoch (int), step (int)
        """

        load_path = os.path.join(self.config['saved_root_dir'], filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path, map_location="cpu")

        # 如果是 DDP，则参数实际在 model.module 中
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(checkpoint["model"])

        if load_optimizer and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if load_lr_scheduler and "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] is not None:
            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)

        self.log(f"Checkpoint loaded from {load_path} (epoch={epoch}, step={step})")

        return epoch, step