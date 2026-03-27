from __future__ import annotations

import math
import subprocess
from pathlib import Path
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def write_fasta_chunks_from_series(
    seqs,
    out_dir: Path,
    chunk_size: int = 20000,
) -> list[Path]:
    """
    将序列按 chunk 写入多个 FASTA。
    记录ID使用全局 0..n-1（字符串），便于后续把 FIMO 的 sequence_name 转为 int。
    返回所有 chunk fasta 路径。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(seqs)
    if n == 0:
        raise ValueError("No sequences to write.")

    fasta_paths: list[Path] = []
    n_chunks = math.ceil(n / chunk_size)

    for c in range(n_chunks):
        start = c * chunk_size
        end = min((c + 1) * chunk_size, n)
        out_fa = out_dir / f"seqs_chunk_{c:06d}.fa"

        records = (
            SeqRecord(Seq(str(seqs[i])), id=str(i), description="")
            for i in range(start, end)
        )
        SeqIO.write(records, out_fa, "fasta")
        fasta_paths.append(out_fa)

    return fasta_paths


def _fimo_result_table(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_tsv = out_dir / "fimo.tsv"
    if out_tsv.exists():
        return out_tsv
    out_txt = out_dir / "fimo.txt"
    if out_txt.exists():
        return out_txt
    raise FileNotFoundError(
        f"FIMO output not found in {out_dir} (expected fimo.tsv or fimo.txt)"
    )


def run_fimo_parallel(
    meme_motif_file: Path,
    fasta_chunks: list[Path],
    out_dir: Path,
    n_jobs: int = 8,
) -> list[Path]:
    """
    用 GNU parallel 并行跑多个 FIMO（每个 chunk 一个 FIMO 任务）。
    返回每个 chunk 的结果表路径列表。
    """
    meme_motif_file = Path(meme_motif_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_out_dirs = [out_dir / f"chunk_{i:06d}" for i in range(len(fasta_chunks))]
    for d in chunk_out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # 使用 GNU parallel 的双输入列表：{1}=chunk fasta, {2}=chunk outdir
    # 注意：用 bash -lc 以确保能找到 parallel/fimo（取决于环境）
    cmd = [
        "bash",
        "-lc",
        (
            "parallel --jobs {jobs} --halt now,fail=1 "
            "'fimo --oc {out} --skip-matched-sequence --no-pgc --max-stored-scores 10000000 {meme} {fa}' "
            "::: {fas} ::: {outs}"
        ).format(
            jobs=int(n_jobs),
            meme=str(meme_motif_file),
            fas=" ".join(map(lambda p: f"'{str(p)}'", fasta_chunks)),
            outs=" ".join(map(lambda p: f"'{str(p)}'", chunk_out_dirs)),
            # 这两个占位符用于 parallel 内部替换（不要 format 掉）
            out="{2}",
            fa="{1}",
        ),
    ]

    subprocess.run(cmd, check=True)

    result_tables = [_fimo_result_table(d) for d in chunk_out_dirs]
    return result_tables


def load_txt(path: Path) -> list[str]:
    path = Path(path)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_tf_count_matrix(fimo_df: pd.DataFrame, tf_names: list[str]) -> pd.DataFrame:
    """生成 sequence x TF 的计数矩阵，并按给定TF列表补齐缺失列。"""
    if fimo_df.empty:
        return pd.DataFrame(columns=tf_names)

    # 防御：某些情况下输出里可能混入重复表头行
    if "motif_id" in fimo_df.columns:
        fimo_df = fimo_df[fimo_df["motif_id"] != "motif_id"]

    fimo_df["sequence_name"] = fimo_df["sequence_name"].astype(int)

    counts = (
        fimo_df.groupby(["sequence_name", "motif_alt_id"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    counts = counts.reindex(columns=tf_names, fill_value=0)
    return counts


def main():
    # ===== 配置区（按需改路径）=====
    tsv_path = Path("data/Siraj_MPRA/Siraj_MPRA_processed.tsv")
    seq_col = "seq"
    out_prefix = "analyze_siraj/fimo"

    motif_meme = Path("data/JASPAR/JASPAR2026_CORE_non-redundant_pfms_meme.txt")

    # 并行相关
    chunk_fa_dir = Path(f"{out_prefix}_fasta_chunks")
    fimo_out_dir = Path("Siraj_fimo_out_parallel")
    chunk_size = 20000   # 你可以按内存/速度调，比如 5000/10000/20000
    n_jobs = 30           # parallel 并发数

    tf_names_path = Path("TF_names.txt")
    out_matrix_tsv = Path("data/Siraj_MPRA/Siraj_TF_counts_matrix.tsv")
    # ==============================

    df = pd.read_csv(tsv_path, sep="\t")
    print("n_seqs =", len(df))

    # 1) 写多个 chunk FASTA（ID仍为全局0..n-1）
    fasta_chunks = write_fasta_chunks_from_series(df[seq_col], chunk_fa_dir, chunk_size)
    print("n_chunks =", len(fasta_chunks))

    # 2) GNU parallel 并行跑 FIMO（每个chunk一个任务）
    fimo_tables = run_fimo_parallel(motif_meme, fasta_chunks, fimo_out_dir, n_jobs=n_jobs)
    print("n_fimo_tables =", len(fimo_tables))

    # 3) 合并所有 FIMO 输出
    dfs = []
    for p in fimo_tables:
        # comment="#" 忽略注释行
        try:
            dfi = pd.read_csv(p, sep="\t", comment="#", low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if not dfi.empty:
            dfs.append(dfi)
    fimo_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # 4) 生成 counts matrix，并按 TF_names.txt 对齐列
    tf_names = load_txt(tf_names_path)
    counts_df = build_tf_count_matrix(fimo_df, tf_names)

    # 5) 输出
    out_matrix_tsv.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(out_matrix_tsv, index=True, sep="\t")
    print("Saved:", out_matrix_tsv)


if __name__ == "__main__":
    main()