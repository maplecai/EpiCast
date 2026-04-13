import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchinfo
from tqdm import tqdm

import boda
from genoml import models, datasets, utils, metrics
from torch.utils.data import DataLoader

@torch.no_grad()
def get_preds(model, dataloader, device='cuda', reverse_comp=False):
    model.eval()
    model = model.to(device)
    preds = []
    for i, batch in enumerate(tqdm(dataloader)):
        seq = batch['seq']
        seq = seq.to(device)
        seq = seq.permute(0, 2, 1)
        if not reverse_comp:
            pred = model(seq)
        else:
            pred1 = model(seq)
            pred2 = model(seq.flip(dims=[1,2]))
            pred = (pred1 + pred2) / 2
        preds.append(pred.detach())
    preds = torch.cat(preds, dim=0).cpu().numpy()
    torch.cuda.empty_cache()
    return preds


def main(args):
    model_path = args.model_path
    data_path = args.data_path
    out_path = args.out_path
    device = utils.get_free_gpus()[0]

    
    # # Malinois official tutorial
    # mpra_df = pd.read_csv('data/Gosai_MPRA/41586_2024_8070_MOESM4_ESM.txt', sep='\t', low_memory=False)
    # mpra_df = mpra_df[(mpra_df[['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE']].max(axis=1) < 1.0)]
    # mpra_df = mpra_df[(mpra_df['sequence'].str.len() == 200)]
    # mpra_df['chr'] = 'chr' + mpra_df['chr']
    # mpra_df = mpra_df.rename(columns={'sequence': 'seq', 'K562_log2FC': 'K562', 'HepG2_log2FC': 'HepG2', 'SKNSH_log2FC': 'SK-N-SH'})
    # mpra_df = mpra_df[['IDs', 'chr', 'seq', 'K562', 'HepG2', 'SK-N-SH']]
    # mpra_df = mpra_df.reset_index(drop=True)
    # print(mpra_df.shape)
    # print(mpra_df.columns)
    

    mpra_df = pd.read_csv(data_path, sep='\t')
    print(mpra_df.shape)
    print(mpra_df.columns)

    left_pad_seq = boda.common.constants.MPRA_UPSTREAM
    right_pad_seq = boda.common.constants.MPRA_DOWNSTREAM

    dataset = datasets.SeqDataset(
        data_df = mpra_df,
        seq_column = 'seq',
        pad = True,
        padded_len = 600,
        pad_mode = 'given',
        left_pad_seq = left_pad_seq,
        right_pad_seq = right_pad_seq,
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)




    checkpoint = torch.load(os.path.join(model_path,'torch_checkpoint.pt'), weights_only=False)
    model_module = getattr(boda.model, checkpoint['model_module'])
    model        = model_module(**vars(checkpoint['model_hparams']))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from {checkpoint["timestamp"]} in eval mode')
    model.eval()
    # model = boda.common.utils.model_fn(model_path)

    torchinfo.summary(model, (1, 4, 600))


    preds = get_preds(model, dataloader, device, reverse_comp=True)
    np.save(out_path, preds)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='pretrained_models/malinois/original/artifacts')
    parser.add_argument("--data_path", type=str, default='data/gosai_mpra/gosai_mpra_760679_zs.tsv')
    parser.add_argument("--out_path", type=str, default='outputs/predictions/malinois_original_pred.npy')
    args = parser.parse_args()
    main(args)


    # model_path = 'pretrained_models/malinois/HCT116/artifacts'
    # np.save('outputs/predictions/malinois_hct116_pred.npy', preds)

    # model_path = 'pretrained_models/malinois/A549/artifacts'
    # np.save('outputs/predictions/malinois_a549_pred.npy', preds)