import os
import logging
import numpy as np
import pandas as pd

import torch
import torchinfo
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from ruamel.yaml import YAML
from io import StringIO

from varlen_genomics import models, datasets, utils, metrics


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
            if self.config['gpu_ids'] == 'auto':
                # self.gpu_id = utils.get_free_gpu_ids()[0]
                self.gpu_id = 0
            else:
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
        
        yaml = YAML()
        stream = StringIO()
        yaml.dump(self.config, stream)
        self.log(stream.getvalue())

        # setup dataloader
        self.train_dataset = utils.init_obj(
            datasets, 
            config['train_dataset'],
        )
        self.valid_dataset = utils.init_obj(
            datasets, 
            config['valid_dataset'],
        )
        
        # Setup collate function for mixed-length training if needed
        self.token_size = config.get('token_size', 128)
        use_mixed_length = config.get('use_mixed_length', False)
        if use_mixed_length:
            collate_fn = datasets.create_length_bucketed_collate_fn(token_size=self.token_size)
        else:
            collate_fn = None
        
        if not self.distributed:
            train_loader_config = config['train_loader'].copy()
            train_loader_args = train_loader_config.get('args', {}).copy()
            if collate_fn is not None:
                train_loader_args['collate_fn'] = collate_fn
            train_loader_config['args'] = train_loader_args
            self.train_loader = utils.init_obj(
                torch.utils.data,
                train_loader_config,
                dataset=self.train_dataset,
            )

            valid_loader_config = config['valid_loader'].copy()
            valid_loader_args = valid_loader_config.get('args', {}).copy()
            if collate_fn is not None:
                valid_loader_args['collate_fn'] = collate_fn
            valid_loader_config['args'] = valid_loader_args
            self.valid_loader = utils.init_obj(
                torch.utils.data,
                valid_loader_config,
                dataset=self.valid_dataset,
            )
            
        else:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.valid_sampler = DistributedSampler(self.valid_dataset, shuffle=False)
            train_loader_config = config['train_loader'].copy()
            train_loader_args = train_loader_config.get('args', {}).copy()
            if collate_fn is not None:
                train_loader_args['collate_fn'] = collate_fn
            train_loader_config['args'] = train_loader_args
            self.train_loader = utils.init_obj(
                torch.utils.data,
                train_loader_config,
                dataset=self.train_dataset,
                sampler=self.train_sampler,
            )
            valid_loader_config = config['valid_loader'].copy()
            valid_loader_args = valid_loader_config.get('args', {}).copy()
            if collate_fn is not None:
                valid_loader_args['collate_fn'] = collate_fn
            valid_loader_config['args'] = valid_loader_args
            self.valid_loader = utils.init_obj(
                torch.utils.data,
                valid_loader_config,
                dataset=self.valid_dataset,
                sampler=self.valid_sampler,
            )
            
        # self.log(f'{len(self.train_dataset) = }')
        # self.log(f'{len(self.valid_dataset) = }')
        # self.log(f'{len(self.train_loader) = }')
        # self.log(f'{len(self.valid_loader) = }')
        # self.batch_size = config['batch_size']
        # self.log(f'{self.batch_size = }')


        # setup model
        self.model = utils.init_obj(models, config['model'])

        if config.get('load_saved_model', False) == True:
            saved_model_path = config['saved_model_path']
            pretrained_dict = torch.load(saved_model_path)
            # missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            # print("Missing keys:", missing_keys)
            # print("Unexpected keys:", unexpected_keys)
            # self.log(f"load saved model from {saved_model_path}")

            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)


        if config.get('freeze_parameters', False) == True:
            self.log(f"freeze parameters")
            freezed_key_words = config.get('freezed_key_words', [])
            for name, param in self.model.named_parameters():
                for word in freezed_key_words:
                    if word in name:
                        param.requires_grad = False
                        self.log(f"freeze parameter {name}")


        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model, 
                device_ids=[self.gpu_id], 
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
            save_dir=os.path.join(config['save_dir']), 
            trace_func=self.log
        )
        
        from torchmetrics import MetricCollection
        from torchmetrics.regression import MeanSquaredError, R2Score, PearsonCorrCoef
        self.metrics = MetricCollection({
            "mse": MeanSquaredError(num_outputs=5313),
            "r2": R2Score(num_outputs=5313),
            "pearson": PearsonCorrCoef(num_outputs=5313),
        }).to(self.device)
        self.metric_df = pd.DataFrame(columns=['mse', 'r2', 'pearson'])
        # if self.distributed:''
        #     for m in self.metrics.values():
        #         m.sync_dist = True
        # # setup metrics
        # self.metric_funcs = [utils.init_obj(metrics, m) for m in config.get('metric_funcs', [])]
        # self.metric_names = [m['type'] for m in config.get('metric_funcs', [])]
        # self.metric_df = pd.DataFrame(columns=self.metric_names)


    def train(self):
        config = self.config
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        epochs_per_valid = config['epochs_per_valid']

        self.log(f'num_epochs = {num_epochs}, batch_size = {batch_size}, epochs_per_valid = {epochs_per_valid}')
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

        # # Model info is already logged during initialization
        # if self.local_rank == 0:
        #     skip_model_summary = config.get('skip_model_summary', True)  # Default to True
        #     if not skip_model_summary:
        #         self.log("Attempting to get sample for model summary (this may take a while)...")
        #         try:
        #             sample = next(iter(self.train_loader))
        #             sample = utils.to_device(sample, self.device)
        #             self.log(torchinfo.summary(
        #                 self.model, 
        #                 input_data=[sample], 
        #                 verbose=0, 
        #                 depth=5,
        #                 col_names=["input_size", "output_size", "num_params"],
        #                 row_settings=["var_names"],
        #             ))
        #         except Exception as e:
        #             self.log(f"Warning: Could not get sample for model summary: {e}. Training will proceed.")
        #     else:
        #         self.log("Skipping model summary (skip_model_summary=True). Training will proceed.")


        for epoch in range(num_epochs):
            self.epoch = epoch
            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            # valid one epoch before training
            if (epoch == 0):
                self.valid_epoch()

            self.log(f'train on epoch {epoch}')
            self.train_epoch()
            
            if ((epoch+1) % epochs_per_valid == 0):
                self.log(f'valid on epoch {epoch}')
                self.valid_epoch()

                if (self.local_rank == 0) and (self.early_stopper is not None):
                    self.early_stopper.check(self.metric_df.loc[self.epoch, 'pearson'])
                    if self.early_stopper.update_flag is True:
                        self.save_model()
                    if self.early_stopper.stop_flag == True:
                        break

        self.log(f'local_rank = {self.local_rank:1}, finish training.')

        if self.distributed:
            dist.destroy_process_group()


    def save_model(self):
        if self.local_rank == 0:
            checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint.pth')
            utils.save_model(self.model, checkpoint_path)
            self.log(f'save model at {checkpoint_path}')


    def train_epoch(self, train_loader=None):
        if train_loader is None:
            train_loader = self.train_loader
        num_log_steps = self.config.get('num_log_steps', 0)
        train_steps = 0
        train_loss = 0

        self.model.train()
        for batch_idx, sample in enumerate(tqdm(train_loader, disable=(self.local_rank != 0))):
            sample = utils.to_device(sample, self.device)
            pred = self.model(sample)

            # 看一下有没有真的边长
            if batch_idx < 10 or batch_idx % 100 == 0:  # 防刷屏
                seq = sample['seq']
                self.log(
                    f"[train] epoch={self.epoch}, batch={batch_idx}, "
                    f"seq.shape={tuple(seq.shape)}"
                )

            target = sample['target']
            loss = self.loss_func(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            train_steps += 1
            train_loss += loss.item()
            if num_log_steps != 0 and batch_idx % num_log_steps == 0:
                self.logger.debug(
                    f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, '
                    f'batch_idx = {batch_idx:3}, train_loss = {loss.item():.6f}')

        self.lr_scheduler.step()

        train_loss = train_loss / train_steps
        self.log(f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, train_loss = {train_loss:.6f}')

    @torch.no_grad()
    def valid_epoch(self, valid_loader=None):
        if valid_loader is None:
            valid_loader = self.valid_loader

        valid_steps = 0
        valid_loss = 0

        self.model.eval()
        self.metrics.reset()
        for batch_idx, sample in enumerate(tqdm(valid_loader, disable=(self.local_rank != 0))):
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
        self.log(f'local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, valid_loss = {valid_loss:.6f}')

        self.results = self.metrics.compute()
        for name, val in self.results.items():
            score = val.mean().cpu().numpy()
            self.metric_df.loc[self.epoch, name] = score
            self.log(f"local_rank = {self.local_rank:1}, epoch = {self.epoch:3}, {name} = {score:.6f}")

    @torch.no_grad()
    def test(self, test_loader):
        self.model.eval()
        pred_list = []
        target_list = []
        for batch_idx, sample in enumerate(tqdm(test_loader, disable=(self.local_rank != 0))):
            sample = utils.to_device(sample, self.device)
            pred = self.model(sample)
            target = sample['target']
            pred_list.append(pred.detach())
            target_list.append(target.detach())

        pred_list = torch.cat(pred_list).cpu().numpy()
        target_list = torch.cat(target_list).cpu().numpy()

        save_file_path = os.path.join(self.config['save_dir'], f'test_pred.npy')
        np.save(save_file_path, pred_list)
        torch.cuda.empty_cache()
        return
