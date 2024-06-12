import torch
import torch.nn.functional as F
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('/home/hxcai/cell_type_specific_CRE')
from MPRA_exp.datasets import SeqLabelDataset
from MPRA_exp.utils import *
from sei_pretrained.sei import Sei


def get_pred(model, test_data_loader, device='cuda'):
    model = model.to(device)
    y_pred = []
    with torch.no_grad():
        model.eval()
        for (x, y) in tqdm(test_data_loader):
            x = x.to(device)
            x_rc = onehots_reverse_complement(x).to(device)
            pred_1 = model(x)
            pred_2 = model(x_rc)
            pred = (pred_1 + pred_2) / 2
            y_pred.extend(pred.cpu().detach().numpy())
    y_pred = np.array(y_pred)
    return y_pred


trained_model_path = 'sei_pretrained/sei.pth'
model = Sei()
model.eval()
state_dict = torch.load(trained_model_path) 
new_state_dict = {k.replace('module.model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

dataset = SeqLabelDataset(seq_exp_path='/home/hxcai/cell_type_specific_CRE/data/SirajMPRA/SirajMPRA_len200.csv',
                          input_column='seq', seq_pad_len=4096)
test_data_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

y_pred = get_pred(model, test_data_loader)

np.save(f'data/Sei_Siraj_pred.npy', y_pred)


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
#             # pred = (model(x) + model(x_rc))/2

#             pred_1, emb_1 = model.get_embedding(x)
#             pred_2, emb_2 = model.get_embedding(x_rc)
#             pred = (pred_1 + pred_2) / 2
#             emb  = (emb_1 + emb_2) / 2

#             y_true.extend(y.cpu().detach().numpy())
#             y_pred.extend(pred.cpu().detach().numpy())
#             embedding.extend(emb.cpu().detach().numpy())

#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     embedding = np.array(embedding)
#     return y_true, y_pred, embedding


# trained_model_path = 'sei_pretrained/sei.pth'
# model = Sei().eval()
# state_dict = torch.load(trained_model_path) 
# new_state_dict = {k.replace('module.model.', ''): v for k, v in state_dict.items()}
# model.load_state_dict(new_state_dict)

# dataset = SeqLabelDataset(seq_exp_path='/home/hxcai/cell_type_specific_CRE/data/SirajMPRA/SirajMPRA_len200.csv', input_column='seq', seq_pad_len=4096)
# test_data_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

# y_true, y_pred, embedding = get_embedding(model, test_data_loader)

# np.save(f'data/Sei_Siraj_pred.npy', y_pred)
# np.save(f'data/Sei_Siraj_embedding.npy', embedding)
