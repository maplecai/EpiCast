"""Predict Sei chromatin profiles for a table of sequences.

Each sequence is centred in a 4,096-bp input and the flanks are filled with N,
encoded as 0.25 in all four one-hot channels. The 21,907 predicted
probabilities per sequence are written to an h5 dataset named "data", which is
the layout `paper/analysis/02_extract_sei_vef.py` reads. Writing is
incremental, so re-running the script continues after the last written row.

Example:
  python scripts/predict_sei.py \
      --seq_path data/gosai_mpra/gosai_mpra_760679.tsv \
      --out_path data/Sei/Gosai_MPRA_Sei_pred.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from epicast import datasets, models, utils  # noqa: E402

n_profiles = 21907


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Sei chromatin profiles for CRE sequences.")
    parser.add_argument("--seq_path", type=str, required=True, help="tsv/csv with a sequence column")
    parser.add_argument("--out_path", type=str, required=True, help="output h5 file")
    parser.add_argument("--seq_column", type=str, default="seq")
    parser.add_argument("--weights_path", type=str, default="data/Sei/resources/sei.pth")
    parser.add_argument("--padded_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device):
    if device != "auto":
        return device
    free_gpus = utils.get_free_gpus()
    return free_gpus[0] if free_gpus else "cpu"


def load_model(weights_path, device):
    state_dict = torch.load(weights_path, map_location="cpu")
    # the released checkpoint was saved from a DataParallel training wrapper
    state_dict = {k.replace("module.model.", ""): v for k, v in state_dict.items()}
    model = models.Sei(n_genomic_features=n_profiles)
    # the B-spline basis is rebuilt in __init__ and is absent from the checkpoint
    model.load_state_dict(state_dict, strict=False)
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
        N_fill_value=0.25,
    )
    writer = utils.H5Writer(
        args.out_path,
        datasets_shape={"data": (n_profiles,)},
        total_size=len(dataset),
        chunk_size=min(64, len(dataset)),
        dtype=np.float32,
        compression=None,
    )
    start = writer.num_written
    print(f"device={device} sequences={len(dataset)} resume_from={start}")
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
            pred = model(seq).cpu().numpy()
            writer.write({"data": pred})
    writer.close()

    print(f"saved {args.out_path} rows={writer.num_written}")


if __name__ == "__main__":
    main()
