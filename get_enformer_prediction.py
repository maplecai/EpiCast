import torch
import torch.nn.functional as F
import sys
import numpy as np

sys.path.append('/home/hxcai/cell_type_specific_CRE')
from MPRA_exp.datasets import SeqLabelDataset
from MPRA_exp.utils import *
# from sei_framework.model.sei import Sei
from enformer_pytorch import Enformer, seq_indices_to_one_hot, from_pretrained


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
            pred_1, emb_1 = model(x, return_embeddings = True)
            pred_2, emb_2 = model(x_rc, return_embeddings = True)
            pred = (pred_1['human'] + pred_2['human']) / 2
            emb  = (emb_1 + emb_2) / 2

            # x = x.to(device)
            # pred, emb = model(x, return_embeddings = True)
            # pred = pred['human']
            # print(y.shape, pred.shape, emb.shape)

            y_true.extend(y.cpu().detach().numpy())
            y_pred.extend(pred.cpu().detach().numpy())
            embedding.extend(emb.cpu().detach().numpy())
            

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    embedding = np.array(embedding)
    return y_true, y_pred, embedding



trained_model_path = '../enformer/enformer_pretrained'
result_dir = './data'

model = from_pretrained(trained_model_path, target_length=2).cuda()

dataset = SeqLabelDataset(table_dir='../data/lentiMPRA/joint_library_table_s.csv', input_column='seq', output_column=None, seq_pad_len=196_608, N_fill_value=0)
test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False)
y_true, y_pred, embedding = get_embedding(model, test_data_loader)
print(f'{y_true.shape=}, {y_pred.shape=}, {embedding.shape=}')

np.save(f'{result_dir}/enformer_y_true.npy', y_true)
np.save(f'{result_dir}/enformer_y_pred.npy', y_pred)
np.save(f'{result_dir}/enformer_embedding.npy', embedding)
