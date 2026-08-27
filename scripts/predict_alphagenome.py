"""Predict AlphaGenome tracks for a table of sequences.

Each sequence is centred in a 2,048-bp input whose flanks are all-zero one-hot
vectors. DNase is taken from the 1-bp head and averaged over the sequence
itself; DNase, histone ChIP-seq and TF ChIP-seq are also taken from the 128-bp
heads and averaged over the bins that cover the sequence. One h5 dataset per
head is written, which is the layout `paper/analysis/02_extract_ag_vef.py`
reads. Writing is incremental, so re-running the script continues after the
last written row.

The ATAC heads are not written because no VEF is derived from them.

Example:
  python scripts/predict_alphagenome.py \
      --seq_path data/gosai_mpra/gosai_mpra_760679.tsv \
      --out_path data/AlphaGenome/gosai_ag_pred_760k_pad_0.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from alphagenome_pytorch import AlphaGenome
from huggingface_hub import hf_hub_download
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from epicast import datasets, utils  # noqa: E402

head_widths = {"dnase_1": 384, "dnase_128": 384, "chip_histone": 1152, "chip_tf": 1664}
bin_width = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Predict AlphaGenome tracks for CRE sequences.")
    parser.add_argument("--seq_path", type=str, required=True, help="tsv/csv with a sequence column")
    parser.add_argument("--out_path", type=str, required=True, help="output h5 file")
    parser.add_argument("--seq_column", type=str, default="seq")
    parser.add_argument(
        "--weights_path", type=str, default="data/AlphaGenome/model_all_folds.safetensors"
    )
    parser.add_argument("--padded_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device):
    if device != "auto":
        return device
    free_gpus = utils.get_free_gpus()
    return free_gpus[0] if free_gpus else "cpu"


def load_model(weights_path, device):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            "gtca/alphagenome_pytorch",
            weights_path.name,
            local_dir=str(weights_path.parent),
        )
    model = AlphaGenome.from_pretrained(str(weights_path))
    return model.to(device).eval()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    dataset = datasets.SeqDataset(
        data_path=args.seq_path,
        seq_column=args.seq_column,
        pad=True,
        pad_mode="N",
        padded_len=args.padded_len,
        N_fill_value=0,
    )
    seq_lens = dataset.df[args.seq_column].str.len().unique()
    assert len(seq_lens) == 1, f"all sequences must have the same length, got {sorted(seq_lens)}"
    seq_len = int(seq_lens[0])

    start_1 = (args.padded_len - seq_len) // 2
    end_1 = start_1 + seq_len
    start_128 = start_1 // bin_width
    end_128 = (end_1 + bin_width - 1) // bin_width

    writer = utils.H5Writer(
        args.out_path,
        datasets_shape={name: (width,) for name, width in head_widths.items()},
        total_size=len(dataset),
        chunk_size=min(1024, len(dataset)),
        dtype=np.float32,
        compression=None,
    )
    start = writer.num_written
    print(
        f"device={device} sequences={len(dataset)} seq_len={seq_len} "
        f"window_1bp=({start_1},{end_1}) window_128bp=({start_128},{end_128}) resume_from={start}"
    )
    if start == len(dataset):
        print(f"{args.out_path} is already complete")
        return

    model = load_model(args.weights_path, device)
    subset = torch.utils.data.Subset(dataset, range(start, len(dataset)))
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    with torch.inference_mode():
        for batch in tqdm(loader):
            seq = batch["seq"].to(device, non_blocking=True)
            outputs = model.predict(seq, organism_index=0)
            writer.write(
                {
                    "dnase_1": outputs["dnase"][1][:, start_1:end_1, :].mean(1).cpu().numpy(),
                    "dnase_128": outputs["dnase"][bin_width][:, start_128:end_128, :]
                    .mean(1)
                    .cpu()
                    .numpy(),
                    "chip_histone": outputs["chip_histone"][bin_width][:, start_128:end_128, :]
                    .mean(1)
                    .cpu()
                    .numpy(),
                    "chip_tf": outputs["chip_tf"][bin_width][:, start_128:end_128, :]
                    .mean(1)
                    .cpu()
                    .numpy(),
                }
            )
    writer.close()

    print(f"saved {args.out_path} rows={writer.num_written}")


if __name__ == "__main__":
    main()
