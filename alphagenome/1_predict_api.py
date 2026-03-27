import os
import numpy as np
import torch
from tqdm import tqdm

from alphagenome.models import dna_client
from genoml import utils, datasets


if __name__ == "__main__":
    os.environ["http_proxy"] = "http://127.0.0.1:16789"
    os.environ["https_proxy"] = "http://127.0.0.1:16789"

    with open("alphagenome/_api_key", "r") as f:
        api_key = f.read().strip()
    dna_model = dna_client.create(api_key)

    data_path = "data/Gosai_MPRA/Gosai_MPRA_760679.tsv"
    output_path = "alphagenome/gosai_ag_pred_1k_api_test.h5"

    batch_size = 1024
    chunk_size = 1024
    padded_len = 16384

    requested_outputs = [
        dna_client.OutputType.DNASE,
        dna_client.OutputType.ATAC,
        dna_client.OutputType.CHIP_TF,
        dna_client.OutputType.CHIP_HISTONE,
    ]

    datasets_shape = {
        "dnase": (305,),
        "atac": (167,),
        "chip_tf": (1617,),
        "chip_histone": (1116,),
    }

    dataset = datasets.SeqDataset(
        data_path=data_path,
        seq_column="seq",
        pad=True,
        pad_mode="N",
        N_fill_value=0.25,
        padded_len=padded_len,
        slice_range=(0,1000),
        return_str=True,
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
        persistent_workers=True,
    )

    for i, batch in tqdm(
        enumerate(loader, start=start_i // batch_size),
        initial=start_i // batch_size,
        total=(len(subset) + batch_size - 1) // batch_size,
    ):
        seqs = batch["seq"]

        outputs = dna_model.predict_sequences(
            seqs,
            requested_outputs=requested_outputs,
            ontology_terms=None,
            progress_bar=True,
            max_workers=5,
        )

        # DNase, ATAC: 1bp resolution
        start = padded_len // 2 - 100
        end = start + 200
        dnase = np.stack([
            o.dnase.values[start:end, :].mean(axis=0) for o in outputs
        ])

        atac = np.stack([
            o.atac.values[start:end, :].mean(axis=0) for o in outputs
        ])

        # TF, Histone: 128bp resolution
        start = padded_len // 128 // 2 - 1
        end = start + 2
        chip_tf = np.stack([
            o.chip_tf.values[start:end, :].mean(axis=0) for o in outputs
        ])

        chip_histone = np.stack([
            o.chip_histone.values[start:end, :].mean(axis=0) for o in outputs
        ])

        writer.write({
            "dnase": dnase,
            "atac": atac,
            "chip_tf": chip_tf,
            "chip_histone": chip_histone,
        })
    writer.close()