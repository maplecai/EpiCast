"""How much of the CTCF VEF signal survives after conditioning on the other VEFs.

For each VEF source and cell type: the marginal CTCF-activity correlation, the
partial correlation given DNase and given all three other assays, and the
matching standardized OLS coefficients.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

from config import cell_types, mpra_path, results_dir, vef_paths

output_dir = results_dir / "ctcf_ablation"
vef_sources = ["alphagenome", "sei"]


def residualize(y, covars):
    return sm.OLS(y, sm.add_constant(covars, has_constant="add")).fit().resid


def partial_corr(df, x_col, y_col, covars):
    return pearsonr(residualize(df[x_col], df[covars]), residualize(df[y_col], df[covars]))


def standardized_beta(df, y_col, x_cols, target):
    z = df[[y_col] + x_cols].apply(lambda s: (s - s.mean()) / s.std())
    model = sm.OLS(z[y_col], sm.add_constant(z[x_cols], has_constant="add")).fit()
    return model.params[target]


def analyze_cell_type(mpra_df, vef_df, cell_type):
    ctcf = f"{cell_type}_CTCF"
    dnase = f"{cell_type}_DNase"
    h3k4me3 = f"{cell_type}_H3K4me3"
    h3k27ac = f"{cell_type}_H3K27ac"
    other3 = [dnase, h3k4me3, h3k27ac]

    df = pd.concat(
        [mpra_df[[cell_type]], vef_df[[dnase, h3k4me3, h3k27ac, ctcf]]], axis=1
    ).dropna()

    marginal_r, marginal_p = pearsonr(df[ctcf], df[cell_type])
    dnase_r, dnase_p = partial_corr(df, ctcf, cell_type, [dnase])
    all3_r, all3_p = partial_corr(df, ctcf, cell_type, other3)

    return {
        "cell_type": cell_type,
        "n": len(df),
        "marginal_r": marginal_r,
        "marginal_p": marginal_p,
        "partial_r_given_dnase": dnase_r,
        "partial_p_given_dnase": dnase_p,
        "partial_r_given_all3": all3_r,
        "partial_p_given_all3": all3_p,
        "beta_marginal": standardized_beta(df, cell_type, [ctcf], ctcf),
        "beta_given_dnase": standardized_beta(df, cell_type, [ctcf, dnase], ctcf),
        "beta_given_all3": standardized_beta(df, cell_type, [ctcf] + other3, ctcf),
    }


def main():
    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] {mpra_path} {mpra_df.shape}")

    rows = []
    for vef_source in vef_sources:
        vef_path = vef_paths[vef_source]
        vef_df = pd.read_csv(vef_path, sep="\t")
        print(f"[load] {vef_path} {vef_df.shape}")
        for cell_type in cell_types:
            row = analyze_cell_type(mpra_df, vef_df, cell_type)
            row["vef_source"] = vef_source
            rows.append(row)
            print(
                f"[{vef_source} {cell_type}] r={row['marginal_r']:.3f} "
                f"partial|DNase={row['partial_r_given_dnase']:.3f} "
                f"partial|all3={row['partial_r_given_all3']:.3f}"
            )

    result_df = pd.DataFrame(rows)
    result_df = result_df[["vef_source"] + [c for c in result_df.columns if c != "vef_source"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ctcf_ablation.csv"
    result_df.to_csv(out_path, index=False)
    print(f"[save] {out_path} {result_df.shape}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
