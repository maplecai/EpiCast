import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import sys
import pandas as pd

from torchinfo import summary
from tqdm import tqdm
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter

sys.path.append('/home/hxcai/cell_type_specific_CRE')
import MPRA_exp.models as models
import MPRA_exp.datasets as datasets
import MPRA_exp.metrics as metrics
import MPRA_exp.utils as utils


def train_epoch(config, logger, model, train_data_loader, epoch, loss_func, metric_func_list, optimizer, lr_scheduler):
    device = config['device']
    scheduler_interval = config['scheduler_interval']
    num_log_steps = config['num_log_steps']
    train_steps = len(train_data_loader)

    model.train()
    train_loss = 0
    y_true = []
    y_pred = []
    for batch_idx, (x, y) in enumerate(tqdm(train_data_loader)):
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = loss_func(out, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        if scheduler_interval == 'step':
            lr_scheduler.step(epoch + batch_idx/train_steps)
        
        y_true.extend(y.cpu().detach())
        y_pred.extend(out.cpu().detach())
        if num_log_steps != 0 and batch_idx % num_log_steps == 0:
            logger.debug(f'epoch = {epoch:3}, batch_idx = {batch_idx:3}, train_loss = {loss.item():.6f}')
            # logger.debug(f'epoch = {epoch:3}, batch_idx = {batch_idx:3}, out = {out.item():.6f}, y = {y.item():.6f}')
            # for metric_func in metric_funcs:
            #     score = metric_func(out, y)
            #     logger.info(f'epoch = {epoch:3}, train_{type(metric_func).__name__} = {score:.6f}')
    train_loss = train_loss / train_steps
    logger.info(f'epoch = {epoch:3}, train_loss = {train_loss:.6f}')
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    for metric_func in metric_func_list:
        score = metric_func(y_true, y_pred)
        logger.info(f'epoch = {epoch:3}, train_{type(metric_func).__name__} = {score:.6f}')

        if scheduler_interval == 'epoch':
            lr_scheduler.step()



def valid_epoch(config, logger, model, valid_data_loader, epoch, loss_func, metric_func_list):
    device = config['device']
    valid_steps = len(valid_data_loader)

    with torch.no_grad():
        model.eval()
        valid_loss = 0
        y_true = []
        y_pred = []
        for batch_idx, (x, y) in enumerate(valid_data_loader):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = loss_func(out, y)

            y_true.extend(y.cpu().detach())
            y_pred.extend(out.cpu().detach())
            valid_loss += loss.item() / valid_steps

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        logger.info(f'epoch = {epoch:3}, valid_loss = {valid_loss:.6f}')
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        for metric_func in metric_func_list:
            score = metric_func(y_true, y_pred)
            logger.info(f'epoch = {epoch:3}, valid_{type(metric_func).__name__} = {score:.6f}')



def main(config):
    seed = config['seed']
    utils.set_seed(seed)
    logging.config.dictConfig(config['logger'])
    logger = logging.getLogger()
    device = config['device']

    if config.get('random_data_split', False) is True:
        total_dataset_1 = utils.init_obj(datasets, config['train_dataset_1'])
        index_list_1 = np.arange(len(total_dataset_1))
        train_index_1, valid_index_1, test_index_1 = utils.split_dataset(index_list_1, train_valid_test_ratio=config['train_valid_test_ratio'])
    else:
        train_index_1, valid_index_1, test_index_1 = None, None, None

    train_dataset_1 = utils.init_obj(datasets, config['train_dataset_1'], selected_index=train_index_1)
    valid_dataset_1 = utils.init_obj(datasets, config['valid_dataset_1'], selected_index=valid_index_1)
    train_data_loader_1 = utils.init_obj(torch.utils.data, config['data_loader'], dataset=train_dataset_1, shuffle=True)
    valid_data_loader_1 = utils.init_obj(torch.utils.data, config['data_loader'], dataset=valid_dataset_1, shuffle=False)

    # train_dataset_2 = utils.init_obj(datasets, config['train_dataset_2'], selected_index=train_index_1)
    # valid_dataset_2 = utils.init_obj(datasets, config['valid_dataset_2'], selected_index=valid_index_1)
    # train_data_loader_2 = utils.init_obj(torch.utils.data, config['data_loader'], dataset=train_dataset_2, shuffle=True)
    # valid_data_loader_2 = utils.init_obj(torch.utils.data, config['data_loader'], dataset=valid_dataset_2, shuffle=False)

    model = utils.init_obj(models, config['model']).to(device)
    if config.get('load_saved_model', False) is True:
        model.load_state_dict(torch.load(config['saved_model_path']))
        for name, param in model.named_parameters():
            if name in config.get('freeze_parameters_list', []):  # 冻结某层的参数
                param.requires_grad = False

    loss_func = utils.init_obj(metrics, config['loss_func'])
    metric_func_list = [utils.init_obj(metrics, m) for m in config.get('metric_func_list', [])]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = utils.init_obj(torch.optim, config['optimizer'], trainable_params)

    if 'lr_scheduler' in config:
        lr_scheduler = utils.init_obj(torch.optim.lr_scheduler, config['lr_scheduler'], optimizer)#constant lr
    else:
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(factor=1.0)

    if 'early_stopper' in config:
        early_stopper = utils.init_obj(utils, config['early_stopper'], trace_func=logger.info)
    else:
        early_stopper = utils.EarlyStopping(patience=np.inf)

    # train and valid
    device = config['device']
    num_epochs = config['num_epochs']
    num_log_steps = config['num_log_steps']
    num_valid_epochs = config['num_valid_epochs']
    scheduler_interval = config.get('scheduler_interval', 'epoch')

    batch_size = config['data_loader']['args']['batch_size']
    train_steps = len(train_data_loader_1)
    valid_steps = len(valid_data_loader_1)

    logger.info(yaml.dump(config))
    x, y = next(iter((train_data_loader_1)))
    logger.info(summary(model, x.shape, verbose=0))
    logger.info(f'len(train_dataset_1) = {len(train_dataset_1)}, len(valid_dataset_1) = {len(valid_dataset_1)}')
    # logger.info(f'len(valid_dataset_2) = {len(valid_dataset_2)}')
    logger.info(f'num_epochs = {num_epochs}, batch_size = {batch_size}')
    logger.info(f'train_steps = {train_steps}, valid_steps = {valid_steps}')
    logger.info(f'start training')

    for epoch in range(num_epochs):
        # train
        if epoch != 0:
            train_epoch(config, logger, model, train_data_loader_1, epoch, loss_func, metric_func_list, optimizer, lr_scheduler)
            # train_epoch(config, logger, model, train_data_loader_2, epoch, loss_func, metric_func_list, optimizer, lr_scheduler)
        # valid
        if (epoch % num_valid_epochs == 0):
            valid_epoch(config, logger, model, valid_data_loader_1, epoch, loss_func, metric_func_list)
            # valid_epoch(config, logger, model, valid_data_loader_2, epoch, loss_func, metric_func_list)

            if config.get('save_model', False) is True:
                checkpoint = {
                    'config': config,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }

                checkpoint_path = os.path.join(config['checkpoint_dir'], f'epoch{epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
                logger.debug(f'save model at {checkpoint_path}')

            # if early_stopper is not None:
            #     # early_stopper.check(valid_loss, model)
            #     early_stopper.check(score, model)
            #     if early_stopper.stop_flag is True:
            #         break

    logger.info(f'finish training.')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config_path', type=str, default=None,
                      help='config file path',)
    args = args.parse_args()
    config_path = args.config_path

    config = utils.load_config(config_path)
    config = utils.process_config(config)
    main(config)
