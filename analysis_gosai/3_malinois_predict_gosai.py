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
    device = utils.get_free_gpus()[0]
    
    model = boda.common.utils.model_fn(model_path)
    torchinfo.summary(model, (1, 4, 600))
    
    # # Malinois original dataset process, same results with malinois official tutorial

    # mpra_df = pd.read_csv('data/Gosai_MPRA/41586_2024_8070_MOESM4_ESM.txt', sep='\t', low_memory=False)
    # mpra_df = mpra_df[(mpra_df[['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE']].max(axis=1) < 1.0)]
    # mpra_df = mpra_df[(mpra_df['sequence'].str.len() == 200)]
    # mpra_df['chr'] = 'chr' + mpra_df['chr']
    # mpra_df = mpra_df.rename(columns={'sequence': 'seq', 'K562_log2FC': 'K562', 'HepG2_log2FC': 'HepG2', 'SKNSH_log2FC': 'SK-N-SH'})
    # mpra_df = mpra_df[['IDs', 'chr', 'seq', 'K562', 'HepG2', 'SK-N-SH']]
    # mpra_df = mpra_df.reset_index(drop=True)
    # print(mpra_df.shape)
    # print(mpra_df.columns)
    

    # our processed dataset
    mpra_df = pd.read_csv(data_path, sep='\t')
    print(mpra_df.shape)
    print(mpra_df.columns)

    splits = {}
    splits['total'] = np.ones(len(mpra_df), dtype=bool)
    splits['train'] = ~mpra_df['chr'].isin(['chr7', 'chr13', 'chr19', 'chr21', 'chrX'])
    splits['val'] = mpra_df['chr'].isin(['chr19', 'chr21', 'chrX'])
    splits['test']  = mpra_df['chr'].isin(['chr7', 'chr13'])

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

    preds = get_preds(model, dataloader, device, reverse_comp=True)
    np.save('outputs/predictions/malinois_original_pred.npy', preds)
    # preds_df  = pd.DataFrame(preds, columns=['K562_pred', 'HepG2_pred', 'SK-N-SH_pred'] )

    # for split in ['train', 'val', 'test']:
    #     print(split)
    #     for cell in ['K562', 'HepG2', 'SK-N-SH']:
    #         mask= splits[split]
    #         pred = preds_df.loc[mask, f'{cell}_pred']
    #         true = mpra_df.loc[mask, f'{cell}']
    #         r, p = metrics.pearson(pred, true)
    #         print(f'{cell}, pearsonr: {r:.4f}')
    #         # r, p = metrics.spearman(pred, true)
    #         # print(f'{cell}, spearmanr: {r:.4f}')



    # model_path = 'pretrained_models/malinois/HCT116/artifacts'
    # model = boda.common.utils.model_fn(model_path)
    # preds = get_preds(model, dataloader, device)
    # np.save('outputs/predictions/malinois_hct116_pred.npy', preds)

    # model_path = 'pretrained_models/malinois/A549/artifacts'
    # model = boda.common.utils.model_fn(model_path)
    # preds = get_preds(model, dataloader, device)
    # np.save('outputs/predictions/malinois_a549_pred.npy', preds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='pretrained_models/malinois/original/artifacts')
    parser.add_argument("--data_path", type=str, default='data/Gosai_MPRA/Gosai_MPRA_760679.tsv')
    args = parser.parse_args()
    main(args)