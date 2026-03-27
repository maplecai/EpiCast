import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
# from scipy.stats import pearsonr
import pyBigWig
from genoml.metrics import pearson


def get_bw_mean(df: pd.DataFrame, bw_file: str,
                chrom_col="chr", start_col="start", end_col="end",
                desc=None) -> np.ndarray:
    """
    返回每个区间在 bigWig 上的均值信号。优先用 bw.stats(type='mean')，更快更省内存。
    stats 返回 None/[] 时记为 NaN。
    """
    out = np.full(len(df), np.nan, dtype=float)
    with pyBigWig.open(bw_file) as bw:
        it = df[[chrom_col, start_col, end_col]].itertuples(index=False, name=None)
        for i, (chrom, start, end) in enumerate(tqdm(list(it), total=len(df), desc=desc)):
            try:
                # pyBigWig.stats 返回 list，如 [mean]，可能为 [None]
                v = bw.stats(chrom, int(start), int(end), type="mean")
                out[i] = v[0] if v and (v[0] is not None) else np.nan
            except RuntimeError:
                out[i] = np.nan
    return out


# def bedtools_overlap_count(df: pd.DataFrame, bed_b: str,
#                            chrom_col="chr", start_col="start", end_col="end") -> np.ndarray:
#     """
#     bedtools intersect -wa -c 计算每个区间与 bed_b 的重叠条目数。
#     使用 stdin/stdout 管道避免临时文件。
#     """
#     bed_a_str = df[[chrom_col, start_col, end_col]].to_csv(
#         sep="\t", header=False, index=False
#     )

#     cmd = ["bedtools", "intersect", "-a", "stdin", "-b", bed_b, "-wa", "-c"]
#     p = subprocess.run(cmd, input=bed_a_str, text=True, capture_output=True)

#     if p.returncode != 0:
#         raise RuntimeError(f"bedtools failed:\nSTDERR:\n{p.stderr}")

#     # 输出列：chr start end count
#     out_df = pd.read_csv(
#         pd.io.common.StringIO(p.stdout),
#         sep="\t", header=None, names=["chr", "start", "end", "overlap"]
#     )
#     return out_df["overlap"].to_numpy(dtype=int)




df = pd.read_csv("data/Siraj_MPRA/Siraj_MPRA_processed.tsv", sep="\t")
df = df[df['allele'] == 'ref'].reset_index(drop=True)

# # overlap
# df["HepG2_overlap"] = bedtools_overlap_count(df, "data/ATAC/HepG2_ENCFF438JMM_IDR.bed")
# df["K562_overlap"] = bedtools_overlap_count(df, "data/ATAC/K562_ENCFF925CYR_IDR.bed")

df["HepG2_bw_value"] = get_bw_mean(df, "data/ATAC/HepG2_ENCFF664EJT.bigWig", desc="HepG2 bigWig mean")


r = pearson(df["HepG2"], df["HepG2_bw_value"])
print(f"Pearson r={r:.6f}")

sns.jointplot(x="HepG2", y="HepG2_bw_value", data=df, s=2)
plt.savefig("analyze_siraj/figures/HepG2_activity_vs_bw_value.png", dpi=400)
plt.close()


df["HepG2_bw_value_log2"] = np.log2(df["HepG2_bw_value"].astype(float) + 1e-6)

r = pearson(df["HepG2"], df["HepG2_bw_value_log2"])
print(f"Pearson r={r:.6f}")

sns.jointplot(x="HepG2", y="HepG2_bw_value_log2", data=df, s=2)
plt.savefig("analyze_siraj/figures/HepG2_activity_vs_bw_value_log2.png", dpi=400)
plt.close()