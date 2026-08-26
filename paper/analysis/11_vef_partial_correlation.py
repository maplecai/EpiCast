"""How much of each VEF's link to activity survives conditioning on the other three
VEFs (fig4B / fig4D / fig4E).

For every VEF source, setting and cell type, all four assays get:

  marginal_r  PCC(VEF, activity)
  partial_r   PCC of the two after regressing the other three VEFs out of both
  beta        its coefficient in the standardized four-VEF OLS fit of activity

The three come from the same data, so one table serves all three panels: fig4B reads the
two correlations in the absolute setting, fig4D the same in the residual one, fig4E the
betas of both. The absolute setting is activity and VEFs as they are; the residual one
subtracts the mean over the three training cell types from both, the reference panel used
for residuals everywhere else in this bundle.

The point is that the four VEFs are heavily collinear (see 11_vef_pairwise_correlation.py),
so a marginal correlation says little about which assay carries the signal. This script
generalizes what used to be 11_ctcf_ablation.py, which asked the same question of CTCF
alone; the CTCF row of the absolute setting reproduces it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

from config import assays, cell_types, mpra_path, results_dir, train_cell_types, vef_paths

output_dir = results_dir / "vef_partial_correlation"
vef_sources = ["sei", "alphagenome"]
settings = ["absolute", "residual"]


def residualize(y, covars):
    return sm.OLS(y, sm.add_constant(covars, has_constant="add")).fit().resid


def cell_frame(mpra_df, vef_df, cell_type, setting):
    """Activity and the four VEFs of one cell type, VEF columns named by assay."""
    activity = mpra_df[cell_type]
    vef = pd.DataFrame({assay: vef_df[f"{cell_type}_{assay}"] for assay in assays})

    if setting == "residual":
        activity = activity - mpra_df[train_cell_types].mean(axis=1)
        vef = vef - pd.DataFrame(
            {
                assay: vef_df[[f"{ct}_{assay}" for ct in train_cell_types]].mean(axis=1)
                for assay in assays
            }
        )

    return pd.concat([activity.rename("activity"), vef], axis=1).dropna()


def analyze(df):
    """One row per assay, sharing the single four-VEF fit that fig4E plots."""
    z = df.apply(lambda s: (s - s.mean()) / s.std())
    betas = sm.OLS(z["activity"], sm.add_constant(z[assays], has_constant="add")).fit().params

    rows = []
    for assay in assays:
        others = [a for a in assays if a != assay]
        marginal_r, marginal_p = pearsonr(df[assay], df["activity"])
        partial_r, partial_p = pearsonr(
            residualize(df[assay], df[others]), residualize(df["activity"], df[others])
        )
        rows.append(
            {
                "assay": assay,
                "n": len(df),
                "marginal_r": marginal_r,
                "marginal_p": marginal_p,
                "partial_r": partial_r,
                "partial_p": partial_p,
                "beta": betas[assay],
            }
        )
    return rows


def main():
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] {mpra_path} {mpra_df.shape}")

    rows = []
    for source in vef_sources:
        vef_df = pd.read_csv(vef_paths[source], sep="\t")
        print(f"[load] {vef_paths[source]} {vef_df.shape}")

        for setting in settings:
            for cell_type in cell_types:
                df = cell_frame(mpra_df, vef_df, cell_type, setting)
                for row in analyze(df):
                    rows.append({"vef_source": source, "setting": setting, "cell_type": cell_type, **row})
                    print(
                        f"[{source} {setting} {cell_type} {row['assay']}] "
                        f"r={row['marginal_r']:+.3f} partial={row['partial_r']:+.3f} "
                        f"beta={row['beta']:+.3f}"
                    )

    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "vef_partial_correlation.csv"
    table.to_csv(path, index=False)
    print(f"[save] {path} {table.shape}")

    summary = table.groupby(["vef_source", "setting", "assay"])[
        ["marginal_r", "partial_r", "beta"]
    ].mean()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
