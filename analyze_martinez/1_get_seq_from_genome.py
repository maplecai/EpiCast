#!/usr/bin/env python3
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from genoml.utils import GenomicInterval, rc_seq

GENOME_PATH = Path("../genome/hg38.fa")
INPUT_DIR = Path("data/Martinez_MPRA/len200")
OUTPUT_DIR = Path("data/Martinez_MPRA/")

INPUT_FILENAMES = [
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_fold0.bed",
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_fold1.bed",
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_fold2.bed",
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_fold3.bed",
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_fold4.bed",
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_test.bed",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0_len200.bed",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold1_len200.bed",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold2_len200.bed",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold3_len200.bed",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold4_len200.bed",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_test_len200.bed",
]

ACTIVITY_PREFIX = "Log2Norm_"
BASE_COLS = ["chr", "start", "end", "strand"]

CELL_TYPES = [
    "HEK293",
    "HCT116",
    "HepG2",
    "K562",
    "LNCaP",
    "MCF7",
    "U2OS",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gi = GenomicInterval(str(GENOME_PATH), lock=False)

    for fn in INPUT_FILENAMES:
        input_path = INPUT_DIR / fn
        output_path = OUTPUT_DIR / fn.replace(".bed.gz", ".tsv")

        df = pd.read_csv(input_path, sep="\t", compression="infer")
        df = df.rename(columns={f"{ACTIVITY_PREFIX}{ct}": ct for ct in CELL_TYPES})
        n = len(df)
        seqs = [None] * n
        for i, (chrom, start, end, strand) in enumerate(
            tqdm(df[["chr", "start", "end", "strand"]].itertuples(index=False, name=None), total=n)
        ):
            s = gi.get(str(chrom), int(start), int(end))
            if strand == "-":
                s = rc_seq(s)
            seqs[i] = s

        df["seq"] = seqs

        df.to_csv(output_path, sep="\t", index=False)
        print(f"Wrote: {output_path}  (n={len(df):,})")

if __name__ == "__main__":
    main()
