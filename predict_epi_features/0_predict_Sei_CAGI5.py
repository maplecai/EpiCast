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

@torch.no_grad()
def get_pred(model, test_data_loader, device='cuda', writer: H5BatchWriter=None, flush_every=10):
    model = model.to(device)
    y_pred = []
    model.eval()
    for i, batch in enumerate(tqdm(test_data_loader)):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch['seq']
        else:
            x = batch
        x = x.to(device)
        output = model(x)

        output = output.detach().cpu().numpy()
        writer.save(output)
        if (i+1) % flush_every == 0:
            writer.flush()
    writer.flush()
    writer.close()



if __name__ == '__main__':

    set_seed(0)

    # data_path = f'data/Gosai_MPRA/Gosai_MPRA_my_processed_data.csv'
    # output_path = f'predict_epi_features/outputs/Gosai_MPRA_Sei_pred_800_float32.h5'

    # data_path = f'data/Agarwal_MPRA/Agarwal_MPRA_joint_56k.csv'
    # output_path = f'predict_epi_features/outputs/Agarwal_MPRA_Sei_pred_200.h5'

    data_path = f'data/CAGI5_MPRA/CAGI5_MPRA.csv'
    output_path = f'predict_epi_features/outputs/CAGI5_MPRA_Sei_pred.h5'


    device = f'cuda:3'
    model_path = f'data/Sei/resources/sei.pth'

    output_dir = os.path.dirname(output_path)
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

    dataset = datasets.SeqDataset(
        data_path=data_path,
        seq_column='seq', 

        crop=False,
        # crop=True,
        # cropped_length=200,

        padding=True,
        padding_method='N',
        padded_length=4096,
        N_fill_value=0.25,
    )
    test_data_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    writer = H5BatchWriter(output_path)
    pred = get_pred(model, test_data_loader, device, writer, 100)
    # np.save(output_path, pred)
