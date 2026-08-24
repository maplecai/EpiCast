import os
import re
import sys
import logging
import logging.config
import numpy as np
import pandas as pd

from io import StringIO
from tqdm import tqdm
from pathlib import Path
from ruamel.yaml import YAML
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torchinfo
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, R2Score, PearsonCorrCoef

from epicast import models, datasets, metrics, training, utils


class Trainer:
    def __init__(self, config):
        self.config = config
        utils.set_seed(config['seed'])

        # setup logger
        logging.config.dictConfig(config['logger'])
        self.logger = logging.getLogger()

        # setup distributed training
        self.setup_distributed()

        if self.local_rank == 0:
            self.log = self.logger.info
            self.debug = self.logger.debug
        else:
            self.log = self.logger.debug
            self.debug = self.logger.debug

        # log config
        yaml = YAML()
        stream = StringIO()
        yaml.dump(self.config, stream)
        self.log(stream.getvalue())

        # runtime states
        self.start_epoch = 0
        self.epoch = -1
        self.step = 0

        # setup dataloader
        self.build_dataloader()

        # setup model
        self.build_model()

        # freeze parameters
        if config.get("freeze_parameters", False):
            self.log("freeze parameters")
            patterns = config.get("freeze_patterns", [])
            regex_list = [re.compile(p) for p in patterns]

            for name, param in self.model.named_parameters():
                for regex in regex_list:
                    if regex.search(name):
                        param.requires_grad = False
                        self.log(f"freeze parameter {name} (matched by {regex.pattern})")
                        break

        # setup training
        self.build_optimizer()

        self.loss_func = utils.init_obj(
            metrics,
            config['loss_func']
        )

        self.lr_scheduler = utils.init_obj(
            training,
            config['lr_scheduler'],
            self.optimizer
        )
        self.early_stopper = utils.init_obj(
            training,
            config['early_stopper'],
            saved_dir=os.path.join(config['saved_dir']),
            trace_func=self.log
        )

        # resume training if needed
        if config.get("resume", False):
            resume_path = config["resume_path"]
            last_epoch, last_step = self.load_checkpoint(
                chechpoint_path=resume_path,
                load_optimizer=True,
                load_lr_scheduler=True,
            )
            self.start_epoch = last_epoch + 1
            self.step = last_step
            self.log(f"Resume training from epoch={self.start_epoch}, step={self.step}")

        self.cell_types = config.get('train_cell_types', config.get('cell_types', []))
        self.metric_df = pd.DataFrame(columns=['mse', 'r2', 'pearson'])
        # self.metrics = MetricCollection({
        #     "mse": MeanSquaredError(num_outputs=len(self.cell_types), sync_on_compute=True),
        #     "r2": R2Score(num_outputs=len(self.cell_types), multioutput="raw_values", sync_on_compute=True),
        #     "pearson": PearsonCorrCoef(num_outputs=len(self.cell_types), sync_on_compute=True),
        # }).to(self.device)

    def setup_distributed(self):
        config = self.config

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
            self.logger.info(f"Start training (non DDP) on rank {self.local_rank}, {self.device}.")
        else:
            dist.init_process_group(backend='nccl', init_method='env://')
            self.local_rank = int(os.environ["LOCAL_RANK"])

            if config['device'] == 'auto':
                self.device = f"cuda:{self.local_rank}"
            elif isinstance(config['device'], list):
                self.device = config['device'][self.local_rank]
            else:
                raise ValueError('DDP device should be a list or "auto"')

            torch.cuda.set_device(self.device)
            self.logger.info(f"Start training (DDP) on rank {self.local_rank}, {self.device}.")

    def build_dataloader(self):
        config = self.config

        self.train_dataset = utils.init_obj(
            datasets,
            config['train_dataset'],
        )
        self.val_dataset = utils.init_obj(
            datasets,
            config['val_dataset'],
        )

        if not self.distributed:
            self.train_loader = utils.init_obj(
                torch.utils.data,
                config['train_loader'],
                dataset=self.train_dataset,
            )

            self.val_loader = utils.init_obj(
                torch.utils.data,
                config['val_loader'],
                dataset=self.val_dataset,
            )

        else:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)

            self.train_loader = utils.init_obj(
                torch.utils.data,
                config['train_loader'],
                dataset=self.train_dataset,
                sampler=self.train_sampler,
            )
            self.val_loader = utils.init_obj(
                torch.utils.data,
                config['val_loader'],
                dataset=self.val_dataset,
                sampler=self.val_sampler,
            )

    def build_model(self):
        config = self.config

        self.model = utils.init_obj(models, config['model'])

        # only load model weights, not full training state
        if config.get('load_saved_model', False):
            utils.load_model(self.model, config['saved_model_path'], strict=config.get('strict', True))
            self.log(f"Loaded saved model weights from {config['saved_model_path']}")

        self.model = self.model.to(self.device)

        if self.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[int(self.device.split(':')[-1])],
                find_unused_parameters=False,
            )

    def build_optimizer(self):
        opt_cfg = self.config["optimizer"]
        groups_cfg = opt_cfg.get("param_groups", []) or []

        named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

        # no grouping
        if not groups_cfg:
            self.optimizer = utils.init_obj(torch.optim, opt_cfg, [p for _, p in named_params])
            return

        # compile regex for each group
        compiled = []
        for g in groups_cfg:
            pats = g.get("contains")
            if not pats:
                continue
            rx = re.compile("|".join(map(re.escape, pats)))
            compiled.append((rx, g.get("args", {}) or {}))

        assigned = set()
        param_groups = []

        # build explicit groups in order (first match wins)
        for rx, g_args in compiled:
            ps = [p for n, p in named_params if id(p) not in assigned and rx.search(n)]
            if ps:
                assigned.update(map(id, ps))
                param_groups.append({"params": ps, **g_args})

        # remaining -> optimizer global defaults
        rest = [p for _, p in named_params if id(p) not in assigned]
        if rest:
            param_groups.append({"params": rest})

        self.optimizer = utils.init_obj(torch.optim, opt_cfg, param_groups)

    def train(self):
        config = self.config
        max_epochs = config['max_epochs']
        epochs_per_val = config['epochs_per_val']
        batch_size = self.train_loader.batch_size

        self.log(f'cell_types = {self.cell_types}')
        self.log(f'len(train_dataset) = {len(self.train_dataset)}')
        self.log(f'len(val_dataset) = {len(self.val_dataset)}')
        self.log(f'len(train_loader) = {len(self.train_loader)}')
        self.log(f'len(val_loader) = {len(self.val_loader)}')
        self.log(f'max_epochs = {max_epochs}')
        self.log(f'batch_size = {batch_size}')

        if self.local_rank == 0:
            sample = self.model.expect_input_sample
            sample = utils.to_device(sample, self.device)
            self.log(torchinfo.summary(
                self.model,
                input_data=[sample],
                verbose=0,
                depth=5,
                col_names=["input_size", "output_size", "num_params"],
                row_settings=["var_names"],
            ))

        self.log('start training ...')

        # keep original behavior: validate once before training only when starting from scratch
        if self.start_epoch == 0:
            self.epoch = -1
            self.val_epoch()

        for epoch in range(self.start_epoch, max_epochs):
            self.epoch = epoch

            if self.distributed:
                self.train_sampler.set_epoch(epoch)

            self.log(f'train on epoch {epoch}')
            self.train_epoch()

            if ((epoch + 1) % epochs_per_val == 0):
                self.log(f'val on epoch {epoch}')
                self.val_epoch()

                if self.local_rank == 0:
                    self.save_checkpoint(self.epoch, self.step, 'last.pth')

                    if self.early_stopper is not None:
                        score = self.metric_df.loc[self.epoch, 'pearson']
                        self.early_stopper.check(score)

                        if self.early_stopper.update_flag is True:
                            self.save_checkpoint(
                                self.epoch,
                                self.step,
                                f'checkpoint_epoch={epoch}_pearson={self.early_stopper.best_score:.6f}.pth',
                            )
                            self.save_checkpoint(self.epoch, self.step, 'best.pth')

                        if self.early_stopper.stop_flag is True:
                            break

        if self.local_rank == 0:
            self.save_checkpoint(self.epoch, self.step, 'last.pth')

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

            self.step += 1
            train_steps += 1
            train_loss += loss.item()

            if steps_per_log != 0 and batch_idx % steps_per_log == 0:
                self.logger.debug(
                    f'local_rank = {self.local_rank}, epoch = {self.epoch}, '
                    f'batch_idx = {batch_idx:3}, step = {self.step}, train_batch_loss = {loss.item():.6f}'
                )

        self.lr_scheduler.step()

        train_loss = train_loss / train_steps
        self.log(f'local_rank = {self.local_rank}, epoch = {self.epoch}, train_loss = {train_loss:.6f}')

    @torch.no_grad()
    def val_epoch(self):
        val_steps = 0
        val_loss = 0

        self.model.eval()
        # self.metrics.reset()

        preds, targets = [], []
        for batch_idx, sample in enumerate(tqdm(self.val_loader, disable=(self.local_rank != 0))):
            sample = utils.to_device(sample, self.device)
            pred = self.model(sample)
            target = sample['target']
            loss = self.loss_func(pred, target)

            val_steps += 1
            val_loss += loss.item()
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
            # self.metrics.update(pred, target)

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        
        val_loss = val_loss / val_steps
        self.metric_df.loc[self.epoch, 'val_loss'] = val_loss
        self.log(f'local_rank = {self.local_rank}, epoch = {self.epoch}, val_loss = {val_loss:.6f}')
        
        if self.local_rank == 0:
            for metric in ["pearson"]:
                if metric == "pearson":
                    metric_fn = metrics.pearson
                for i, cell_type in enumerate(self.cell_types):
                    pred = preds[:, i]
                    target = targets[:, i]
                    r = metric_fn(pred, target)
                    self.metric_df.loc[self.epoch,f'{cell_type}_{metric}'] = r
                    self.log(f"local_rank = {self.local_rank}, epoch = {self.epoch}, {cell_type}_{metric} = {r:.6f}")
                cols = [f"{cell_type}_{metric}" for cell_type in self.cell_types]
                self.metric_df.loc[self.epoch, 'pearson'] = self.metric_df.loc[self.epoch, cols].mean()
                
        # self.results = self.metrics.compute()

        # if self.local_rank == 0:
        #     for name, val in self.results.items():
        #         val_mean = val.mean().item()
        #         self.log(
        #             f"local_rank = {self.local_rank}, epoch = {self.epoch}, {name} mean = {val_mean:.6f}"
        #         )
        #         self.metric_df.loc[self.epoch, f"{name}"] = val_mean

        #     for name, val in self.results.items():
        #         val = val.detach().cpu()
        #         for i, ct in enumerate(self.cell_types):
        #             value_i = val[i].item()
        #             if name == 'pearson':
        #                 self.log(f"local_rank = {self.local_rank}, epoch = {self.epoch}, {name} {ct} = {value_i:.6f}")
        #             else:
        #                 self.debug(f"local_rank = {self.local_rank}, epoch = {self.epoch}, {name} {ct} = {value_i:.6f}")

    def _resolve_checkpoint_path(self, chechpoint_path):
        path = Path(chechpoint_path)
        if path.is_file():
            return str(path)

        saved_dir = Path(self.config["saved_dir"])
        for candidate in (
            saved_dir / "checkpoints" / path.name,
            saved_dir / path.name,
            saved_dir / "checkpoints" / path,
        ):
            if candidate.is_file():
                return str(candidate)
        return str(path)

    def save_checkpoint(self, epoch, step, filename="checkpoint.pth"):
        """
        Save model/optimizer/lr_scheduler states.

        Args:
            epoch: current epoch
            step: current global step
            filename: save file name
        """
        if self.distributed is True and self.local_rank != 0:
            return

        checkpoint_dir = Path(self.config["saved_dir"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = checkpoint_dir / filename

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

    def load_checkpoint(self, chechpoint_path="checkpoint.pth", load_optimizer=True, load_lr_scheduler=True):
        """
        Load model/optimizer/lr_scheduler states.

        Args:
            chechpoint_path (str): checkpoint file path.
            load_optimizer (bool): whether to load optimizer state.
            load_lr_scheduler (bool): whether to load lr scheduler state.

        Returns:
            epoch (int), step (int)
        """
        chechpoint_path = self._resolve_checkpoint_path(chechpoint_path)
        checkpoint = torch.load(chechpoint_path, map_location=self.device)

        if "model" in checkpoint:
            model_state_dict = checkpoint["model"]
        else:
            model_state_dict = checkpoint

        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(model_state_dict)

        if load_optimizer and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if load_lr_scheduler and "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] is not None:
            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)

        self.log(f"Checkpoint loaded from {chechpoint_path} (epoch={epoch}, step={step})")
        return epoch, step