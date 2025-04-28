import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from MPRA_predict import models, datasets, metrics, utils


# @torch.no_grad()
# def run_inference(model, test_loader, device, output_dir, output_name='pred.npy'):
#     # only single gpu
#     model.eval()
#     model = model.to(device)
#     pred_list = []
#     label_list = []
#     for batch_idx, sample in enumerate(tqdm(test_loader)):
#         sample = utils.to_device(sample, device)
#         pred = model(sample)
#         label = sample['label']
#         pred_list.append(pred.detach())
#         label_list.append(label.detach())

#     pred_list = torch.cat(pred_list).cpu().numpy()
#     label_list = torch.cat(label_list).cpu().numpy()

#     save_file_path = os.path.join(output_dir, output_name)
#     np.save(save_file_path, pred_list)
#     torch.cuda.empty_cache()
#     return pred_list, label_list





def mask_feature(feature, mask):
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=feature.dtype, device=feature.device)
    else:
        mask = mask.to(dtype=feature.dtype, device=feature.device)

    if feature.dim() == 2:
        mask = mask.view(1, -1)
    elif feature.dim() == 3:
        mask = mask.view(1, 1, -1)
    else:
        raise ValueError(f'Unsupported feature.dim() = {feature.dim()}')
    return feature * mask




@torch.no_grad()
def run_inference(model, test_loader, device, output_dir, output_name='pred.npy', mask=None):
    model.eval()
    model = model.to(device)
    pred_list = []
    label_list = []
    for batch_idx, sample in enumerate(tqdm(test_loader)):
        if mask is not None:
            sample['feature'] = mask_feature(sample['feature'], mask)

        sample = utils.to_device(sample, device)
        pred = model(sample)
        label = sample['label']
        pred_list.append(pred.detach())
        label_list.append(label.detach())

    pred_list = torch.cat(pred_list).cpu().numpy()
    label_list = torch.cat(label_list).cpu().numpy()

    save_file_path = os.path.join(output_dir, output_name)
    np.save(save_file_path, pred_list)
    torch.cuda.empty_cache()
    return pred_list, label_list




def main():
    args = argparse.ArgumentParser()
    args.add_argument('-s', '--saved_dir', type=str, default=None, help='saved folder dir',)
    args.add_argument('-c', '--config_path', type=str, default=None, help='config folder path, default is saved_dir/config.yaml',)
    args.add_argument('-d', '--device', type=str, default='cuda:0', help='device',)
    
    args = args.parse_args()
    saved_dir = args.saved_dir
    config_path = args.config_path
    device = args.device

    if config_path is None:
        config_path = os.path.join(saved_dir, 'config.yaml')
        print(f'use saved config: {config_path}')
    else:
        print(f'use new config: {config_path}')
        pass
    # config_path = 'configs/config_0311_SirajMPRA_test_1_cell_type.yaml'

    config = utils.load_config(config_path)

    model = utils.init_obj(models, config['model'])
    saved_model_path = os.path.join(saved_dir, 'checkpoint.pth')
    state_dict = torch.load(saved_model_path)

    # # 替换 key 名称
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     if "cls_embedding_layer" in k:
    #         new_key = k.replace("cls_embedding_layer", "feature_embedding_layer")
    #         new_state_dict[new_key] = v
    #     else:
    #         new_state_dict[k] = v
    # torch.save(new_state_dict, saved_model_path)

    model.load_state_dict(state_dict)
    model = model.to(device)

    total_dataset = utils.init_obj(datasets, config['total_dataset'])
    total_loader = DataLoader(
        dataset=total_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    
    # run_inference(model, total_loader, device, saved_dir)


    mask = torch.tensor([1,0,0,0])
    run_inference(model, total_loader, device, saved_dir, output_name='test_pred_mask1000.npy', mask=mask)



if __name__ == '__main__':
    main()
