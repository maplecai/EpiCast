import numpy as np
import pandas as pd
import torch
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm
from alphagenome_pytorch import AlphaGenome
from genoml import datasets, utils


if __name__ == "__main__":
    model_path = "alphagenome/model_all_folds.safetensors"
    mpra_path = "data/Gosai_MPRA/Gosai_MPRA_760679.tsv"

    # # pad 0.25
    # output_path = "alphagenome/gosai_ag_pred_760k_not_compressed.h5"
    # pad 0
    output_path = "alphagenome/gosai_ag_pred_760k_pad_0.h5"

    device = 'cuda:1'
    padded_len = 2048
    # padded_len = 16384
    token_len = padded_len // 128
    batch_size = 4
    chunk_size = 1024
    write_per_batches = chunk_size // batch_size
    datasets_shape = {
        "dnase_1": (384,), 
        "dnase": (384,), 
        "atac_1": (256,), 
        "atac": (256,), 
        "chip_tf": (1664,),
        "chip_histone": (1152,),
    }

    if not Path(model_path).exists():
        hf_hub_download('gtca/alphagenome_pytorch', 'model_all_folds.safetensors', local_dir='alphagenome/')
    model = AlphaGenome.from_pretrained(model_path)
    model = model.to(device)

    dataset = datasets.SeqDataset(
        data_path=mpra_path,
        seq_column="seq",
        pad=True,
        pad_mode="N",
        # N_fill_value=0.25, # not alphagenome default, but useful
        N_fill_value=0, 
        padded_len=padded_len,
        # slice_range=(0, 1000),
    )

    writer = utils.H5Writer(
        output_path,
        datasets_shape=datasets_shape,
        total_size=len(dataset),
        chunk_size=chunk_size,
        dtype=np.float32,
        compression=None,
    )

    start_i = writer.start

    subset = torch.utils.data.Subset(dataset, range(start_i, len(dataset)))
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    for i, batch in tqdm(
        enumerate(loader, start=start_i // batch_size),
        initial=start_i // batch_size,
        total=(len(subset) + batch_size - 1) // batch_size,
    ):
        seqs = batch['seq']
        seqs = utils.to_device(seqs, device, non_blocking=True)
        outputs = model.predict(seqs, organism_index=0)

        # DNase, ATAC: 1bp resolution
        start = padded_len // 2 - 100
        end = start + 200

        dnase_1 = outputs['dnase'][1][:, start:end, :].mean((1)).detach().cpu().numpy()
        atac_1 = outputs['atac'][1][:, start:end, :].mean((1)).detach().cpu().numpy()

        # DNase, TF, Histone: 128bp resolution
        start = padded_len // 128 // 2 - 1
        end = start + 2
        dnase_128 = outputs['dnase'][128][:, start:end, :].mean((1)).detach().cpu().numpy()
        atac_128 = outputs['atac'][128][:, start:end, :].mean((1)).detach().cpu().numpy()
        chip_tf = outputs['chip_tf'][128][:, start:end, :].mean((1)).detach().cpu().numpy()
        chip_histone = outputs['chip_histone'][128][:, start:end, :].mean((1)).detach().cpu().numpy()

        writer.write({
            "dnase_1": dnase_1,
            "dnase": dnase_128,
            "atac_1": atac_1,
            "atac": atac_128,
            "chip_tf": chip_tf,
            "chip_histone": chip_histone,
        })
    writer.close()
