import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from alphagenome_pytorch import AlphaGenome
from epicast import datasets, utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mpra_path",
        type=str,
        required=True,
        help="Path to input MPRA dataframe/tsv file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output HDF5 filename",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_path = "alphagenome/model_all_folds.safetensors"
    mpra_path = args.mpra_path
    output_path = args.out

    device = utils.get_free_gpus()[0]
    # valid_len = 200
    # valid_len = 145
    valid_len = 256

    padded_len = 2048
    batch_size = 4
    chunk_size = 1024

    start_1 = (padded_len - valid_len) // 2
    end_1 = start_1 + valid_len
    start_128 = start_1 // 128
    end_128 = (end_1 + 127) // 128

    datasets_shape = {
        "dnase_1": (384,),
        "dnase_128": (384,),
        "atac_1": (256,),
        "atac_128": (256,),
        "chip_tf": (1664,),
        "chip_histone": (1152,),
    }

    if not Path(model_path).exists():
        hf_hub_download(
            "gtca/alphagenome_pytorch",
            "model_all_folds.safetensors",
            local_dir="alphagenome/",
        )

    model = AlphaGenome.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    dataset = datasets.SeqDataset(
        data_path=mpra_path,
        seq_column="seq",
        pad=True,
        pad_mode="N",
        # N_fill_value=0.25,
        N_fill_value=0,
        padded_len=padded_len,
        slice_range=(-1, None),
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
        num_workers=4,
        pin_memory=True,
    )

    with torch.inference_mode():
        for i, batch in tqdm(
            enumerate(loader, start=start_i // batch_size),
            initial=start_i // batch_size,
            total=(len(subset) + batch_size - 1) // batch_size,
        ):
            seqs = batch["seq"]
            seqs = utils.to_device(seqs, device, non_blocking=True)
            outputs = model.predict(seqs, organism_index=0)

            # DNase, ATAC: 1bp resolution
            dnase_1 = outputs["dnase"][1][:, start_1:end_1, :].mean(1).cpu().numpy()
            atac_1 = outputs["atac"][1][:, start_1:end_1, :].mean(1).cpu().numpy()

            # DNase, TF, Histone: 128bp resolution
            dnase_128 = outputs["dnase"][128][:, start_128:end_128, :].mean(1).cpu().numpy()
            atac_128 = outputs["atac"][128][:, start_128:end_128, :].mean(1).cpu().numpy()
            chip_tf = outputs["chip_tf"][128][:, start_128:end_128, :].mean(1).cpu().numpy()
            chip_histone = outputs["chip_histone"][128][:, start_128:end_128, :].mean(1).cpu().numpy()

            writer.write(
                {
                    "dnase_1": dnase_1,
                    "dnase_128": dnase_128,
                    "atac_1": atac_1,
                    "atac_128": atac_128,
                    "chip_tf": chip_tf,
                    "chip_histone": chip_histone,
                }
            )

        writer.close()
