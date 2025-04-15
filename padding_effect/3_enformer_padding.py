import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import *
from MPRA_predict.models import enformer_pytorch


def get_pred(model, test_data_loader, device='cuda'):
    model = model.to(device)
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
            output = model(x)
            output = output['human'][:, 447:449]
            y_pred.append(output.detach().cpu().numpy())
            del batch, x, output  # 清理内存
    y_pred = np.concatenate(y_pred, axis=0)
    torch.cuda.empty_cache()
    return y_pred




# def get_pred_total(model, dataset, device, output_path):
#     test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
#     pred = get_pred(model, test_data_loader, device)
#     np.save(output_path, pred)
#     return



# # split to many parts, predict and save, in order to save memory
# def get_pred_split(model, dataset, device, output_path, num_splits):

#     split_size = len(dataset) // num_splits  # num_splits是你要分割的部分数
#     # 分割数据集
#     for i in range(num_splits):
#         start_idx = i * split_size
#         end_idx = (i + 1) * split_size if i != num_splits - 1 else len(dataset)

#         subset = Subset(dataset, range(start_idx, end_idx))
#         subloader = DataLoader(subset, batch_size=4, shuffle=False, num_workers=0)
#         y_pred = get_pred(model, subloader, device)
#         np.save(f'{output_path}_{i}.npy', np.array(y_pred))
#     return




if __name__ == '__main__':

    set_seed(0)
    device = f'cuda:0'
    model_path = f'pretrained_models/enformer_weights'
    data_path = f'pretrained_models/Enformer_data/enformer_sequences_test.csv'
    # data_path = f'../data/SirajMPRA/SirajMPRA_562654.csv'

    model = enformer_pytorch.from_pretrained(model_path)





    # for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
    #     for seed in range(5):
    #         set_seed(seed)
            
    #         output_path = f'padding_effect/outputs/Enformer_pred_crop_{cropped_length}_genome_{seed}.npy'
    #         if os.path.exists(output_path):
    #             print(f'already exists {output_path}, skip')
    #             continue
    #         print(f'predicting {output_path}')

    #         dataset = datasets.SeqDataset(
    #             data_path=data_path,
    #             seq_column='seq', 
    #             crop=True,
    #             crop_position='center',
    #             cropped_length=cropped_length,
    #             padding=True,
    #             padding_position='both',
    #             padded_length=196608,

    #             # # N
    #             # padding_method='N',
    #             # # zero
    #             # padding_method='N',
    #             # N_fill_value=0,
    #             # # random
    #             # padding_method='random',
    #             # genome
    #             padding_method='genome',
    #             genome=Fasta('data/genome/hg38.fa')
    #             # # repeat
    #             # padding_method='repeat',
    #         )
            
    #         test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
    #         pred = get_pred(model, test_data_loader, device)
    #         # pred = pred[:, 447:449]
    #         np.save(output_path, pred)






    # for cropped_length in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]:
    #     for seed in range(5):
    #         set_seed(seed)
            
    #         output_path = f'padding_effect/outputs/Enformer_pred_crop_{cropped_length}_random_{seed}.npy'
    #         if os.path.exists(output_path):
    #             print(f'already exists {output_path}, skip')
    #             continue
    #         print(f'predicting {output_path}')

    #         dataset = datasets.SeqDataset(
    #             data_path=data_path,
    #             seq_column='seq', 
    #             crop=True,
    #             crop_position='center',
    #             cropped_length=cropped_length,
    #             padding=True,
    #             padding_position='both',
    #             padded_length=196608,

    #             # # N
    #             # padding_method='N',
    #             # # zero
    #             # padding_method='N',
    #             # N_fill_value=0,
    #             # random
    #             padding_method='random',
    #             # # genome
    #             # padding_method='genome',
    #             # genome=Fasta('data/genome/hg38.fa')
    #             # # repeat
    #             # padding_method='repeat',
    #         )
            
    #         test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
    #         pred = get_pred(model, test_data_loader, device)
    #         # pred = pred[:, 447:449]
    #         np.save(output_path, pred)










    for cropped_length in [8192]:
        output_path = f'padding_effect/outputs/Enformer_pred_crop_{cropped_length}_N_new_bs_3_2.npy'
        if os.path.exists(output_path):
            print(f'already exists {output_path}, skip')
            continue
        print(f'predicting {output_path}')

        dataset = datasets.SeqDataset(
            data_path=data_path,
            seq_column='seq', 
            crop=True,
            crop_position='center',
            cropped_length=cropped_length,
            padding=True,
            padding_position='both',
            padded_length=196608,

            # N
            padding_method='N',
            N_fill_value=0.25,
            # # zero
            # padding_method='N',
            # N_fill_value=0,
            # # random
            # padding_method='random',
            # genome
            # padding_method='genome',
            # genome=Fasta('data/genome/hg38.fa')
            # # repeat
            # padding_method='repeat',
        )
        
        test_data_loader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=1)
        pred = get_pred(model, test_data_loader, device)
        # pred = pred[:, 447:449]
        np.save(output_path, pred)


