import torch
import torch.nn.functional as F
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('/home/hxcai/cell_type_specific_CRE')
from MPRA_exp.datasets import SeqLabelDataset
from MPRA_exp.utils import *
from sei_framework.model.sei import Sei


def get_embedding(model, test_data_loader, device='cuda'):
    model = model.to(device)
    y_true = []
    y_pred = []
    embedding = []
    with torch.no_grad():
        model.eval()
        for (x, y) in tqdm(test_data_loader):
            x = x.to(device)
            x_rc = onehots_reverse_complement(x).to(device)
            # pred = (model(x) + model(x_rc))/2

            pred_1, emb_1 = model.get_embedding(x)
            pred_2, emb_2 = model.get_embedding(x_rc)
            pred = (pred_1 + pred_2) / 2
            emb  = (emb_1 + emb_2) / 2

            y_true.extend(y.cpu().detach().numpy())
            y_pred.extend(pred.cpu().detach().numpy())
            embedding.extend(emb.cpu().detach().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    embedding = np.array(embedding)
    return y_true, y_pred, embedding


trained_model_path = '../sei_framework/model/sei.pth'
model = Sei().eval()
state_dict = torch.load(trained_model_path) 
new_state_dict = {k.replace('module.model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

dataset = SeqLabelDataset(seq_exp_path='/home/hxcai/cell_type_specific_CRE/data/GosaiMPRA/GosaiMPRA_len200.csv', input_column='seq', seq_pad_len=4096, subset_range=[0.5, 1])
test_data_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

y_true, y_pred, embedding = get_embedding(model, test_data_loader)

# def save_in_batches(data, batch_size, file_prefix):
#     num_batches = len(data) // batch_size + 1
#     for i in range(num_batches):
#         batch_data = data[i*batch_size:(i+1)*batch_size]
#         np.save(f'{file_prefix}_batch_{i}.npy', batch_data)

# save_in_batches(y_pred, batch_size=100000, file_prefix='data/sei_pred')
# save_in_batches(embedding, batch_size=100000, file_prefix='data/sei_embedding')

np.save(f'data/sei_pred.npy', y_pred)
np.save(f'data/sei_embedding.npy', embedding)
