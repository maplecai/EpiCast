import torch
import torch.nn.functional as F
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('/home/hxcai/cell_type_specific_CRE')
from MPRA_exp.datasets import SeqLabelDataset
from MPRA_exp.utils import *
from enformer_pytorch import Enformer, from_pretrained


def get_pred(model, test_data_loader, device='cuda'):
    model = model.to(device)
    y_pred = []
    with torch.no_grad():
        model.eval()
        for (x, y) in tqdm(test_data_loader):
            x = x.to(device)
            x_rc = onehots_reverse_complement(x).to(device)
            pred_1 = model(x)['human']
            pred_2 = model(x_rc)['human']
            pred = (pred_1 + pred_2)/2
            y_pred.extend(pred.cpu().detach().numpy())
    y_pred = np.array(y_pred)
    return y_pred


trained_model_path = 'enformer_pretrained'
model = from_pretrained(trained_model_path, target_length=2).cuda()
model.eval()
dataset = SeqLabelDataset(seq_exp_path='/home/hxcai/cell_type_specific_CRE/data/GosaiMPRA/GosaiMPRA_total.csv',
                          input_column='seq', seq_pad_len=196_608, N_fill_value=0)
test_data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
y_pred = get_pred(model, test_data_loader)

np.save(f'data/enformer_y_pred.npy', y_pred)


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


# trained_model_path = 'enformer_pretrained'
# model = from_pretrained(trained_model_path, target_length=2).cuda()

# dataset = SeqLabelDataset(table_dir='/home/hxcai/cell_type_specific_CRE/data/GosaiMPRA/GosaiMPRA_total.csv',
#                           input_column='nt_sequence', output_column=None, seq_pad_len=196_608, N_fill_value=0, selected_index=np.arange(100000))
# test_data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
# y_true, y_pred, embedding = get_embedding(model, test_data_loader)
# print(f'{y_true.shape=}, {y_pred.shape=}, {embedding.shape=}')

# result_dir = './data'
# np.save(f'{result_dir}/enformer_y_pred.npy', y_pred)
# np.save(f'{result_dir}/enformer_embedding.npy', embedding)
