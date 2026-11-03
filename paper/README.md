# Reproducing the EpiCast analyses

Install EpiCast and extract the data archive as described in the [project README](../README.md). Run the commands below from the repository root.

`config.py` defines input paths, model runs and evaluation settings; `utils.py` provides shared feature-extraction and evaluation functions. Analysis scripts write tables under `paper/results/`, and plotting scripts write PDFs under `paper/figures/`.

## Evaluation workflow

Using the downloaded activity tables, VEF matrices and checkpoint predictions:

```bash
python paper/analysis/05_train_vef_only_models.py
python paper/analysis/07_eval_regression.py
python paper/analysis/08_eval_classification.py
python paper/analysis/09_eval_retrieval.py
python paper/analysis/14_export_prediction_tables.py
python paper/analysis/15_export_figure_metrics.py
python paper/analysis/12_eval_castillo.py
```

Before running the workflow, add the following entry to `full_sources` in `05_train_vef_only_models.py` to generate the four-assay Sei baselines required by the complete Gosai comparison:

```python
"sei": results_dir / "vef_only/sei_vef",
```

The script already generates DNase-only baselines for all four feature sources and four-assay AlphaGenome baselines. The derived baseline predictions are not included in the data archive.

Additional feature analyses:

```bash
python paper/analysis/11_vef_pairwise_correlation.py
python paper/analysis/11_vef_partial_correlation.py
```

See the [analysis index](analysis/README.md) for data preparation and further dependencies. Generate an individual figure with, for example:

```bash
python paper/plot/fig2bc_activity_metrics.py
```

The [figure index](plot/README.md) maps manuscript panels to scripts.

## Evaluation definitions

- **Gosai:** 760,679 CREs. K562, HepG2 and SK-N-SH are training cell types; HCT116 and A549 are held out. Test chromosomes are chr7 and chr13; validation chromosomes are chr19, chr21 and chrX.
- **Gosai specificity:** residual activity subtracts the mean over the three training cell types. CTS-high and CTS-low use the upper and lower 1% tails within each cell type's measured subset. Evaluation uses test chromosomes and excludes missing measurements.
- **Castillo-Hair:** all 8,152 CREs are evaluated without retraining, across seven cell types. CTS-high exceeds all other evaluated cell types by at least 1 log2FC; CTS-low is at least 1 log2FC below all others. Ranking uses the corresponding predicted gaps. Figure 5 prioritization summaries require at least 20 positive examples per cell-type/task pair.
- **AlphaGenome VEFs:** the paper uses variant B: DNase from the 1-bp head, histone/TF signals rescaled from 128-bp bins, and `log1p(10x)` transformation.

Regression, classification and retrieval results are kept separately. `results/predictions/` contains per-sequence measured and predicted activity; `results/figure_metrics/` contains the aggregated tables used by Figures 2 and 3.
