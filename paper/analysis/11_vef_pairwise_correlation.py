"""Pairwise correlations among the four VEFs, absolute and residual (fig4A / fig4C).

For each VEF source, cell type and pair of assays, the PCC is taken across all CREs,
twice: once on the VEF values themselves and once on the residual VEFs. The residual
reference panel is the three training cell types, the same one used for activity
residuals everywhere else in this bundle.

The point of the figure these numbers feed is that the four VEFs are strongly
collinear in absolute terms, which is what makes the partial correlations of
11_vef_partial_correlation.py necessary; the residual setting shows how much of that
collinearity is shared-across-cell-types rather than assay-specific.

Writes one tidy row per (source, setting, cell type, assay pair). The mean +- SEM over
cell types that the figure shows is computed in plot/fig4ac_vef_correlation_heatmap.py.
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import assays, cell_types, results_dir, train_cell_types, vef_paths

output_dir = results_dir / "vef_pairwise_correlation"
vef_sources = ["sei", "alphagenome"]


def assay_frame(vef_df, assay):
    """CREs x cell types for one assay, columns renamed to plain cell type names."""
    return vef_df[[f"{ct}_{assay}" for ct in cell_types]].set_axis(cell_types, axis=1)


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for source in vef_sources:
        vef_df = pd.read_csv(vef_paths[source], sep="\t")
        print(f"[load] {vef_paths[source]} {vef_df.shape}")

        absolute = {assay: assay_frame(vef_df, assay) for assay in assays}
        residual = {
            assay: frame.sub(frame[train_cell_types].mean(axis=1), axis=0)
            for assay, frame in absolute.items()
        }

        for setting, frames in [("absolute", absolute), ("residual", residual)]:
            for assay_a, assay_b in combinations(assays, 2):
                for cell in cell_types:
                    a = frames[assay_a][cell]
                    b = frames[assay_b][cell]
                    rows.append(
                        {
                            "vef_source": source,
                            "setting": setting,
                            "cell_type": cell,
                            "assay_a": assay_a,
                            "assay_b": assay_b,
                            "n": int((a.notna() & b.notna()).sum()),
                            "pcc": float(a.corr(b)),
                        }
                    )

    table = pd.DataFrame(rows)
    path = output_dir / "vef_pairwise_correlation.csv"
    table.to_csv(path, index=False)
    print(f"[save] {path} {table.shape}")

    summary = table.groupby(["vef_source", "setting"])["pcc"].agg(["min", "mean", "max"])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
