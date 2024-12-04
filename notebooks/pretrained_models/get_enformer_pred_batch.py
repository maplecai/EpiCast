import numpy as np
import torch
import torchinfo
import sys
sys.path.append('..')

from tqdm import tqdm
from torch.utils.data import DataLoader
from enformer_pytorch import Enformer, from_pretrained
from MPRA_predict.utils import *
from MPRA_predict.datasets import SeqLabelDataset

# def get_pred(model, test_data_loader, device):
#     model = model.to(device)
#     y_pred = []
#     with torch.no_grad():
#         model.eval()
#         for (x, y) in tqdm(test_data_loader):
#             x = x['seq'].to(device)
#             # print(x.shape)
#             x_rc = onehots_reverse_complement(x).to(device)
#             pred_1 = model(x)['human']
#             pred_2 = model(x_rc)['human']
#             pred = (pred_1 + pred_2)/2
#             y_pred.extend(pred.cpu().detach().numpy())
#     y_pred = np.array(y_pred)
#     return y_pred


# trained_model_path = 'Enformer'
# device = 'cuda:3'
# model = from_pretrained(trained_model_path, target_length=2).to(device).eval()
# dataset = SeqLabelDataset(data_path='/home/hxcai/cell_type_specific_CRE/data/SirajMPRA/SirajMPRA_total.csv',
#                           seq_column='seq', padding=True, padded_len=196_608, N_fill_value=0)
# test_data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
# y_pred = get_pred(model, test_data_loader, device)

# np.save(f'data/Enformer_Siraj_pred.npy', y_pred)



def get_pred(model, test_data_loader, device, save_interval, save_prefix):
    model = model.to(device)
    y_pred = []
    batch_count = 0  # 记录批次数
    part_num = 0     # 记录保存部分的编号
    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(test_data_loader):
            x = x['seq'].to(device)
            pred = model(x)['human']
            # x_rc = onehots_reverse_complement(x).to(device)
            # pred_2 = model(x_rc)['human']
            # pred = (pred + pred_2) / 2
            y_pred.extend(pred.cpu().detach().numpy())

            batch_count += 1
            if batch_count % save_interval == 0:
                # 保存当前的部分预测结果
                np.save(f'{save_prefix}{part_num}.npy', np.array(y_pred))
                print(f'Saved part {part_num} at batch {batch_count}')
                part_num += 1
                y_pred = []  # 清空已保存的数据

        # 保存最后的预测结果（如果有剩余）
        if len(y_pred) > 0:
            np.save(f'{save_prefix}{part_num}.npy', np.array(y_pred))
            print(f'Saved final part {part_num}')

pretrained_model_path = 'Enformer'
target_length = 2
device = 'cuda:3'

set_seed(2)
model = from_pretrained(pretrained_model_path, use_tf_gamma=True, target_length=target_length).to(device)
# model = Enformer.from_pretrained(pretrained_model_path, target_length=target_length).to(device)
torchinfo.summary(model, input_size=(1, 200, 4), depth=5)
dataset = SeqLabelDataset(data_path='/home/hxcai/cell_type_specific_CRE/data/SirajMPRA/SirajMPRA_total.csv',
                          seq_column='seq', padding=True, padded_len=256, padding_mode='random', N_fill_value=0)
test_data_loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
y_pred = get_pred(model, test_data_loader, device, save_interval=10, save_prefix='data/Enformer_Siraj_pred_part')




# def get_embedding(model, test_data_loader, device='cuda'):
#     model = model.to(device)
#     y_true = []
#     y_pred = []
#     embedding = []
#     with torch.no_grad():
#         model.eval()
#         for (x, y) in tqdm(test_data_loader):
#             x = x.to(device)
#             x_rc = onehots_reverse_complement(x).to(device)
#             pred_1, emb_1 = model(x, return_embeddings = True)
#             pred_2, emb_2 = model(x_rc, return_embeddings = True)
#             pred = (pred_1['human'] + pred_2['human']) / 2
#             emb  = (emb_1 + emb_2) / 2
#             # x = x.to(device)
#             # pred, emb = model(x, return_embeddings = True)
#             # pred = pred['human']
#             # print(y.shape, pred.shape, emb.shape)

#             y_true.extend(y.cpu().detach().numpy())
#             y_pred.extend(pred.cpu().detach().numpy())
#             embedding.extend(emb.cpu().detach().numpy())

#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     embedding = np.array(embedding)
#     return y_true, y_pred, embedding
