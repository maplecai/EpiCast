import re
from pathlib import Path

import pandas as pd

bundle_root = Path(__file__).resolve().parents[1]
project_root = bundle_root.parent

output_dir = bundle_root / "results/model_track_metadata"

lovo_mark_pattern = re.compile(
    r"_LoVo_(H3K4me3|H3K4me1|H3K27ac|H3K36me3|H3K27me3|H3K9me3|H3K9ac|CTCF|DNase)_"
)


def normalize_assay(assay: str) -> str:
    if assay == "DNASE":
        return "DNase"
    if assay == "ATAC":
        return "ATAC-seq"
    if assay in ("eGFP-CTCF",) or assay.endswith("-CTCF"):
        return "CTCF"
    return assay


def normalize_cell_type(cell_type: str) -> str:
    cell_type = cell_type.strip()
    cell_type = cell_type.replace(";", "/")
    cell_type = re.sub(r"\s+", " ", cell_type)
    return cell_type


def parse_lovo_batch_target(body: str) -> tuple[str, str]:
    match = lovo_mark_pattern.search(body)
    assay = match.group(1) if match else "."
    if " ; " in body:
        parts = body.split(" ; ")
        cell_type = " ; ".join(parts[1:]) if len(parts) > 1 else parts[-1]
    else:
        cell_type = body
    return normalize_assay(assay), normalize_cell_type(cell_type)


def split_enformer_target(target: str) -> tuple[str, str]:
    if target.startswith("ChIP-TF:./"):
        return parse_lovo_batch_target(target[len("ChIP-TF:./") :])

    i = 0
    while i < len(target):
        if target[i] == "/" and (i == 0 or target[i - 1] != "."):
            return target[:i], target[i + 1 :]
        i += 1
    return target, ""


def parse_enformer_assay(assay: str) -> str:
    if assay.startswith("ChIP-Histone:"):
        assay = assay.replace("ChIP-Histone:", "")
    elif assay.startswith("ChIP-TF:"):
        assay = assay.replace("ChIP-TF:", "")
    return normalize_assay(assay)


def parse_enformer_target(target: str) -> tuple[str, str]:
    assay_raw, cell_type = split_enformer_target(target)
    assay = parse_enformer_assay(assay_raw)
    if "ENCODE" in cell_type:
        cell_type = cell_type.replace("ENCODE, biol_", "")
    return assay, normalize_cell_type(cell_type)


def parse_borzoi_description(desc: str) -> tuple[str, str | None]:
    if desc.startswith("CHIP:"):
        _, assay, cell_type = desc.split(":", 2)
        return normalize_assay(assay), normalize_cell_type(cell_type)
    parts = desc.split(":")
    if len(parts) == 2:
        return normalize_assay(parts[0]), normalize_cell_type(parts[1])
    return desc, None


def parse_alphagenome_row(row: pd.Series) -> tuple[str, str]:
    assay = row["Assay title"]
    if assay == "DNase-seq":
        assay = "DNase"
    elif assay == "TF ChIP-seq":
        assay = row["transcription_factor"]
    elif assay == "Histone ChIP-seq":
        assay = row["histone_mark"]
    return normalize_assay(str(assay)), normalize_cell_type(str(row["biosample_name"]))


def parse_sei_row(row: pd.Series) -> tuple[str, str]:
    return normalize_assay(str(row["assay"])), normalize_cell_type(str(row["cell_type"]))


def save_parsed_table(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"[save] {path} {df.shape}")


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sei_path = project_root / "data/Sei/Sei_tracks_info.csv"
    sei_df = pd.read_csv(sei_path)
    print(f"[load] {sei_path} {sei_df.shape}")
    sei_out = sei_df.copy()
    sei_parsed = sei_df.apply(parse_sei_row, axis=1, result_type="expand")
    sei_out["assay"] = sei_parsed[0]
    sei_out["cell_type"] = sei_parsed[1]
    save_parsed_table(sei_out, output_dir / "sei_tracks_parsed.csv")

    enformer_path = project_root / "data/Enformer/model_track_info.tsv"
    enformer_df = pd.read_csv(enformer_path, sep="\t")
    print(f"[load] {enformer_path} {enformer_df.shape}")
    enformer_parsed = enformer_df["target"].apply(parse_enformer_target).apply(pd.Series)
    enformer_parsed.columns = ["assay", "cell_type"]
    enformer_out = pd.concat([enformer_df, enformer_parsed], axis=1)
    save_parsed_table(enformer_out, output_dir / "enformer_tracks_parsed.csv")

    borzoi_path = project_root / "data/Borzoi/targets_human.txt"
    borzoi_df = pd.read_csv(borzoi_path, sep="\t", index_col=0).reset_index()
    print(f"[load] {borzoi_path} {borzoi_df.shape}")
    borzoi_parsed = borzoi_df["description"].apply(parse_borzoi_description).apply(pd.Series)
    borzoi_parsed.columns = ["assay", "cell_type"]
    borzoi_out = pd.concat([borzoi_df, borzoi_parsed], axis=1)
    save_parsed_table(borzoi_out, output_dir / "borzoi_tracks_parsed.csv")

    ag_path = project_root / "data/AlphaGenome/metadata.csv"
    ag_df = pd.read_csv(ag_path)
    print(f"[load] {ag_path} {ag_df.shape}")
    ag_parsed = ag_df.apply(parse_alphagenome_row, axis=1, result_type="expand")
    ag_parsed.columns = ["assay", "cell_type"]
    ag_out = pd.concat([ag_df, ag_parsed], axis=1)
    save_parsed_table(ag_out, output_dir / "alphagenome_tracks_parsed.csv")


if __name__ == "__main__":
    main()
