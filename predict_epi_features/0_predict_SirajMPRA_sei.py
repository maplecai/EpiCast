import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import *

def get_pred(model, test_data_loader, device='cuda'):
    y_pred = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_data_loader)):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch['seq']
            else:
                x = batch
            x = x.to(device)

            # if i == 1: 

            #     from torch.profiler import profile, record_function, ProfilerActivity
            #     with profile(
            #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #         record_shapes=True,
            #         profile_memory=True,
            #         with_stack=True
            #     ) as prof:
            #         with record_function("model_inference"):
            #             output = model(x)
            #     print(prof.key_averages().table(
            #         sort_by="cuda_time_total",  # 可改为 cpu_time_total / self_cpu_time_total 等
            #         row_limit=20
            #     ))
            #     break

            output = model(x)
            y_pred.append(output.detach().cpu().numpy())
    #         del batch, x, output  # 清理内存
    torch.cuda.empty_cache()
    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred



if __name__ == '__main__':

    set_seed(0)

    # print("PyTorch version:", torch.__version__)
    # print("CUDA version:", torch.version.cuda)
    # print("cuDNN version:", torch.backends.cudnn.version())
    # print(torch.__config__.show())
    torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = True

    device = f'cuda:0'
    model_path = f'pretrained_models/Sei/sei.pth'
    data_path = f'data/SirajMPRA/SirajMPRA_562654.csv'
    # output_path = f'predict_epi_features/outputs/SirajMPRA_Sei_prediction_zero_padding.npy'
    output_dir = f'predict_epi_features/outputs'
    output_path = f'{output_dir}/test.npy'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'cannot find {output_dir}, creating {output_dir}')
    if os.path.exists(output_path):
        print(f'already exists {output_path}, exit')
        exit()
    
    print(f'predicting {output_path}')

    model = models.Sei()
    model_state_dict = torch.load(model_path)
    model_state_dict = {k.replace('module.model.', ''): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)


#     config_str = '''
# model:
#   type: MyBasset
#   args:
#     input_seq_length: 4096
#     output_dim: 1
#     sigmoid: false
#     squeeze: false

#     conv_channels_list: [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
#       256, 256, 256, 256, 256, 256]
#     conv_kernel_size_list: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
#     conv_padding_list: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#     pool_kernel_size_list: [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
#     conv_dropout_rate: 0.2

#     linear_channels_list: [1024, 1024]
#     linear_dropout_rate: 0.5

#     '''
#     config = yaml.load(config_str)
#     model = init_obj(models, config['model'])
#     model = model.to(device)
    

    dataset = datasets.SeqDataset(
        data_path=data_path,
        seq_column='seq', 
        crop=False,
        padding=True,
        padding_method='N',
        padded_length=4096,
        N_fill_value=0.25)


    test_data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    pred = get_pred(model, test_data_loader, device)
    np.save(output_path, pred)



    # num_splits = 10    # num_splits是你要分割的部分数
    # split_size = len(dataset) // num_splits
    # for i in range(num_splits):
    #     start_idx = i * split_size
    #     end_idx = (i + 1) * split_size if i != num_splits - 1 else len(dataset)

    #     subset = Subset(dataset, range(start_idx, end_idx))
    #     subloader = DataLoader(subset, batch_size=6, shuffle=False, num_workers=0)
    #     y_pred = get_pred(model, subloader, device)
    #     np.save(f'{output_path}_{i}.npy', np.array(y_pred))
