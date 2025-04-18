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
    torch.cuda.empty_cache()
    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred



if __name__ == '__main__':

    set_seed(0)

    device = f'cuda:0'
    model_path = f'pretrained_models/enformer_weights'

    # data_path = f'data/SirajMPRA/SirajMPRA_562654.csv'
    data_path = f'data/GosaiMPRA/GosaiMPRA_my_processed_data_len200_norm.csv'

    # output_path = f'outputs/SirajMPRA_Enformer_no_padding.npy'
    output_path = f'predict_epi_features/outputs/GosaiMPRA_Enformer_zero_padding.npy'
    # output_path = f'outputs/SirajMPRA_Enformer_N_padding.npy'

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'cannot find {output_dir}, creating {output_dir}')
    if os.path.exists(output_path):
        print(f'already exists {output_path}, exit')
        exit()
    print(f'predicting {output_path}')

    # model = from_pretrained(model_path, target_length=2, use_tf_gamma=False)
    model = models.enformer_pytorch.from_pretrained(model_path)
    model = model.to(device)

    dataset = datasets.SeqDataset(
        data_path=data_path,
        seq_column='seq', 
        crop=False,
        padding=True,
        padding_method='N',
        padded_length=196608,
        N_fill_value=0)
        # padding = False,
        # N_fill_value = 0.25,
    
    test_data_loader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=4)
    pred = get_pred(model, test_data_loader, device)
    np.save(output_path, pred)



# MPRA_UPSTREAM  = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
# MPRA_DOWNSTREAM= 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'
