from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def write_fasta_from_series(seqs, out_fa: Path) -> None:
    """
    将序列写入FASTA。ID使用 0..n-1，便于后续把 FIMO 的 sequence_name 转为 int。
    """
    out_fa = Path(out_fa)
    records = (
        SeqRecord(Seq(str(s)), id=str(i), description="")
        for i, s in enumerate(seqs)
    )
    SeqIO.write(records, out_fa, "fasta")


def run_fimo_once(meme_motif_file: Path, fasta_file: Path, out_dir: Path) -> Path:
    """
    一次性跑 FIMO，输出到 out_dir，并返回 FIMO 的结果表路径（优先 fimo.tsv，其次 fimo.txt）。
    """
    meme_motif_file = Path(meme_motif_file)
    fasta_file = Path(fasta_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "fimo",
        "--oc", str(out_dir),
        "--skip-matched-sequence",
        "--no-pgc",
        "--max-stored-scores", "10000000",
        str(meme_motif_file),
        str(fasta_file),
    ]
    subprocess.run(cmd, check=True)

    out_tsv = out_dir / "fimo.tsv"
    if out_tsv.exists():
        return out_tsv

    out_txt = out_dir / "fimo.txt"
    if out_txt.exists():
        return out_txt

    raise FileNotFoundError(f"FIMO output not found in {out_dir} (expected fimo.tsv or fimo.txt)")


def load_txt(path: Path) -> list[str]:
    path = Path(path)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_tf_count_matrix(fimo_df: pd.DataFrame, tf_names: list[str]) -> pd.DataFrame:
    """
    生成 sequence x TF 的计数矩阵，并按给定TF列表补齐缺失列。
    """
    if fimo_df.empty:
        return pd.DataFrame(columns=tf_names)

    # 防御：某些情况下输出里可能混入重复表头行
    if "motif_id" in fimo_df.columns:
        fimo_df = fimo_df[fimo_df["motif_id"] != "motif_id"]

    # sequence_name 通常是FASTA记录ID（这里是数字字符串）
    fimo_df["sequence_name"] = fimo_df["sequence_name"].astype(int)

    counts = (
        fimo_df
        .groupby(["sequence_name", "motif_alt_id"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    # 按给定TF顺序对齐，并补齐缺失列
    counts = counts.reindex(columns=tf_names, fill_value=0)
    return counts


def main():
    # ===== 配置区（按需改路径）=====
    tsv_path = Path("data/Siraj_MPRA/Siraj_MPRA_processed.tsv")
    seq_col = "seq"

    out_prefix = "SirajMPRA_total"
    fasta_path = Path(f"{out_prefix}.fa")

    motif_meme = Path("data/JASPAR/JASPAR2026_CORE_non-redundant_pfms_meme.txt")

    fimo_out_dir = Path("Siraj_fimo_out")  # FIMO 输出目录（修复问题1）
    tf_names_path = Path("TF_names.txt")   # 你已有的TF顺序列表
    out_matrix_tsv = Path("data/Siraj_MPRA/Siraj_TF_counts_matrix.tsv")  # 输出矩阵（修复问题2）
    # ==============================

    # 1) 读CSV并写FASTA
    df = pd.read_csv(tsv_path, sep="\t")
    print("n_seqs =", len(df))
    write_fasta_from_series(df[seq_col], fasta_path)

    # 2) 一次性跑FIMO（修复问题1：用 --oc 并返回真实结果表路径）
    fimo_table_path = run_fimo_once(motif_meme, fasta_path, fimo_out_dir)

    # 3) 读FIMO输出
    # comment="#" 用来忽略以 # 开头的注释行（FIMO有时会带）
    fimo_df = pd.read_csv(fimo_table_path, sep="\t", comment="#", low_memory=False)

    # 4) 生成 counts matrix，并按 TF_names.txt 对齐列
    tf_names = load_txt(tf_names_path)
    counts_df = build_tf_count_matrix(fimo_df, tf_names)

    # 5) 确保输出目录存在（修复问题3）
    out_matrix_tsv.parent.mkdir(parents=True, exist_ok=True)

    # 建议保留 sequence_name 作为索引（更可追溯）
    counts_df.to_csv(out_matrix_tsv, index=True, sep="\t")
    print("Saved:", out_matrix_tsv)


if __name__ == "__main__":
    main()