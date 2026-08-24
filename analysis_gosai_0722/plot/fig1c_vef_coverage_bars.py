from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast.utils.plot_utils import set_mpl_params

bundle_root = Path(__file__).resolve().parents[1]
figures_dir = bundle_root / "figures"

selected_assays = ["DNase", "H3K4me3", "H3K27ac", "CTCF"]
model_order = ["Sei", "Enformer", "Borzoi", "AlphaGenome"]
top_n_assays = 10
selected_bar_color = "#EBCACA"
selected_label_color = "#D62728"
default_bar_color = "#B7D9D3"

metadata_dir = bundle_root / "results/model_track_metadata"
results_dir = bundle_root / "results/fig1c_assay_coverage"

model_files = {
    "Sei": metadata_dir / "sei_tracks_parsed.csv",
    "Enformer": metadata_dir / "enformer_tracks_parsed.csv",
    "Borzoi": metadata_dir / "borzoi_tracks_parsed.csv",
    "AlphaGenome": metadata_dir / "alphagenome_tracks_parsed.csv",
}


def merge_assay(assay) -> str:
    name = str(assay).strip()
    lower = name.lower()
    if lower == "rna" or "rna-seq" in lower:
        return "RNA-seq"
    if "cage" in lower:
        return "CAGE"
    return name


def is_valid_assay(assay) -> bool:
    return str(assay).strip().lower() not in ("", ".", "nan")


def prepare_track_table(track_df: pd.DataFrame) -> pd.DataFrame:
    out = track_df.copy()
    out["assay"] = out["assay"].apply(merge_assay)
    out = out[out["assay"].apply(is_valid_assay)]
    return out.dropna(subset=["cell_type"])


def load_parsed_track_table(model_name: str) -> pd.DataFrame:
    path = model_files[model_name]
    df = pd.read_csv(path)
    print(f"[load] {path} {model_name}")
    return prepare_track_table(df[["assay", "cell_type"]])


def build_coverage_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    count_rows = []
    pct_rows = []
    total_cells = {}
    for model_name in model_order:
        track_df = load_parsed_track_table(model_name)
        n_total = track_df["cell_type"].nunique()
        coverage = track_df.groupby("assay")["cell_type"].nunique().sort_values(ascending=False)
        total_cells[model_name] = n_total
        count_rows.append(coverage)
        pct_rows.append(coverage / n_total * 100.0)

    count_wide = pd.concat(count_rows, axis=1, keys=model_order).fillna(0).astype(int)
    pct_wide = pd.concat(pct_rows, axis=1, keys=model_order).fillna(0.0)
    return count_wide, pct_wide, pd.Series(total_cells)


def plot_one_model_bars(ax, model_name: str, counts: pd.Series) -> None:
    assay_order = (
        counts[counts > 0].sort_values(ascending=False).head(top_n_assays).index.tolist()
    )
    model_counts = counts.loc[assay_order]
    x = np.arange(len(assay_order))
    heights = model_counts.to_numpy(dtype=float)

    colors = [
        selected_bar_color if assay in selected_assays else default_bar_color
        for assay in assay_order
    ]

    ax.bar(
        x,
        heights,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        width=0.85,
    )

    y_pad = max(heights.max() * 0.02, 8)
    for i, count in enumerate(heights):
        ax.text(
            i,
            count + y_pad,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(model_name)
    ax.set_xticks(x + 0.25)
    labels = ax.set_xticklabels(assay_order, rotation=45, ha="right")
    for label, assay in zip(labels, assay_order):
        if assay in selected_assays:
            label.set_color(selected_label_color)
            label.set_fontweight("bold")
    ax.set_ylabel("Number of cell types")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def plot_one_model_figure(
    model_name: str,
    counts: pd.Series,
    ymax: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.9, top=0.9)

    plot_one_model_bars(ax, model_name, counts)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, top_n_assays - 0.4)

    fig.savefig(save_path, dpi=400)
    print(f"[save] {save_path}")
    plt.close(fig)


def plot_coverage_bars(count_wide: pd.DataFrame, figures_dir: Path) -> None:
    set_mpl_params()
    sns.set_theme(style="whitegrid", context="notebook")

    ymax = count_wide.max().max() * 1.12
    model_stem = {
        "Sei": "sei",
        "Enformer": "enformer",
        "Borzoi": "borzoi",
        "AlphaGenome": "alphagenome",
    }
    for model_name in model_order:
        out_fig = figures_dir / f"fig1c_{model_stem[model_name]}_assay_coverage_bars.pdf"
        plot_one_model_figure(model_name, count_wide[model_name], ymax, out_fig)


def main() -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    count_wide, pct_wide, total_cells = build_coverage_tables()
    assays = count_wide.max(axis=1).sort_values(ascending=False).head(top_n_assays).index.tolist()
    count_df = count_wide.loc[assays, model_order].T.astype(int)
    pct_df = pct_wide.loc[assays, model_order].T.astype(float)

    count_path = results_dir / "assay_coverage_count.csv"
    pct_path = results_dir / "assay_coverage_pct.csv"
    total_path = results_dir / "model_total_cell_types.csv"
    count_df.to_csv(count_path)
    pct_df.to_csv(pct_path)
    total_cells.to_csv(total_path, header=["n_cell_types"])

    print(f"[save] {count_path}")
    print(f"[save] {pct_path}")
    print(f"[save] {total_path}")
    print(count_df)

    plot_coverage_bars(count_wide, figures_dir)


if __name__ == "__main__":
    main()
