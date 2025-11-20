import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))

from MPRA_predict import models, datasets, utils
from MPRA_predict.utils import set_seed



@torch.no_grad()
def get_pred(model, dataloader, device='cuda'):
    model = model.to(device).eval()
    preds = []
    for batch in tqdm(dataloader):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch.get('seq')
        else:
            x = batch
        x = x.to(device, non_blocking=True)
        out = model(x)
        out = out.cpu().numpy()
        preds.append(out)
    preds = np.concatenate(preds, axis=0)
    return preds



def predict_Sei(seq_df, device='cuda'):
    model_path = ROOT_DIR / "data/Sei/resources/sei.pth"

    raw_state = torch.load(model_path)
    cleaned_state = {
        k.replace("module.model.", ""): v
        for k, v in raw_state.items()
    }
    
    model = models.Sei()
    model.load_state_dict(cleaned_state, strict=False)
    model = model.to(device)

    dataset = datasets.SeqDataset(
        data_df=seq_df,
        seq_column='seq',
        crop=False,
        # cropped_length=200,
        padding=True,
        padded_length=4096,
        padding_method='N',
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    preds = get_pred(model, loader, device)
    return preds




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", type=str)
    parser.add_argument("-m", "--model", type=str, default="Sei")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    args = parser.parse_args()

    # --- resolve paths ---
    input_path = Path(args.input_path)

    if args.output_path is None:
        output_path = Path("outputs") / f"{input_path.stem}_{args.model}_pred.npy"
    else:
        output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"File exists: {output_path}, exit.")
        return

    print(f"Start prediction, output_path = {output_path}")



    seq_df = pd.read_csv(input_path, sep="\t")
    set_seed(0)

    preds = predict_Sei(seq_df, device=args.device)

    np.save(output_path, preds)
    print("Done.")



if __name__ == "__main__":
    main()

    # 300bp context
    # MPRA_UPSTREAM   = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
    # MPRA_DOWNSTREAM = 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'
