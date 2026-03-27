import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)

TARGET_LENGTH = 200

INPUT_DIR = Path("data/Martinez_MPRA/raw")
OUTPUT_DIR = Path(f"data/Martinez_MPRA/len{TARGET_LENGTH}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILENAMES = [
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_fold0.bed",
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_fold1.bed",
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_fold2.bed",
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_fold3.bed",
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_fold4.bed",
    # "GSE301246_PARM_Training_Library_v1_pHY3_normalized_test.bed",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold1.bed.gz",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold2.bed.gz",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold3.bed.gz",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold4.bed.gz",
    "GSE301246_PARM_Training_Library_v2_MH4_normalized_test.bed.gz",
]



for filename in INPUT_FILENAMES:
    input_path = INPUT_DIR / filename
    df = pd.read_csv(input_path, sep="\t", compression="gzip")

    seq_len = df["end"] - df["start"] + 1
    mask = (seq_len == TARGET_LENGTH)
    df_target = df.loc[mask].copy()

    out_name = filename.replace(".bed.gz", f"_len{TARGET_LENGTH}.bed")
    out_path = OUTPUT_DIR / out_name

    df_target.to_csv(out_path, sep="\t", index=False)

    print(f"{filename}: total={len(df)}, len{TARGET_LENGTH}={mask.sum()} -> saved: {out_path}")
