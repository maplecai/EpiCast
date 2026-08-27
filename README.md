# EpiCast: leveraging virtual epigenomic features to predict episomal regulatory activity across cell types

## Introduction

Designing regulatory sequences for synthetic biology and gene therapy requires understanding how DNA sequences drive cell-type-specific expression, yet existing MPRA-based models typically fail to generalize beyond the cell types used for training.

To address this challenge, we present **EpiCast**, a deep learning framework for predicting episomal cis-regulatory element (CRE) activity across diverse human cell types. We integrate DNA sequence with **virtual epigenomic features (VEFs)**: cell-type-specific regulatory proxies inferred from large-scale genomic sequence-to-function models such as Sei and AlphaGenome. Although episomal DNA lacks native chromatin structure, these model-derived features capture how different cell types are predicted to interpret a given sequence, enabling EpiCast to incorporate contextual regulatory information without requiring MPRA data from every cell type.

Trained on MPRA datasets, EpiCast learns both sequence grammar and cell-type-dependent regulatory logic. As a result, it achieves strong performance within training cell types and generalizes to previously unseen ones, which we evaluate both on held-out cell types of the training MPRA and zero-shot on an independent MPRA.

## Repository layout

```
src/epicast/       model, dataset and training library (installable package)
scripts/           command-line entry points for VEF prediction, training and inference
configs/           training and inference configs of the models used in the paper
paper/             everything needed to reproduce the figures of the paper
  config.py        single source of paths, cell types, model registry and colours
  utils.py         helpers shared by the analysis scripts
  analysis/        numbered pipeline, from raw MPRA data to metric tables
  plot/            one script per figure panel
data/              model track metadata; all other data is distributed separately
```

## Installation

```bash
git clone https://github.com/maplecai/EpiCast.git
cd EpiCast
conda create -n epicast python=3.10
conda activate epicast
pip install -r requirements.txt
pip install -e .
```

After installation the library is importable from any working directory:

```python
import epicast
```

`scripts/predict_alphagenome.py` additionally needs
[alphagenome-pytorch](https://github.com/genomicsxai/alphagenome-pytorch), which requires
Python >= 3.12 and therefore its own environment. It downloads the converted all-folds weights
from <https://huggingface.co/gtca/alphagenome_pytorch> on first use.

## Data

The MPRA label table, the derived VEF matrices, the DHS64 baseline predictions and the
trained checkpoints are deposited at <https://doi.org/10.5281/zenodo.17669740> as a single
`epicast_data.zip`. It mirrors the repository layout, so unpacking it at the repository root
is all the setup there is:

```bash
unzip epicast_data.zip -d /path/to/EpiCast
```

That puts the five tables under `data/gosai_mpra/`, the Castillo MPRA and its VEF matrix
under `data/castillo_mpra/`, the DHS64 baseline under `data/DHS64/` and the four runs of
`paper/config.py` under `saved/`, each with its `config.yaml`, its `checkpoints/best.pth`
and the held-out predictions the figures are computed from. Nothing in the archive collides
with a version controlled file.

The raw track predictions the VEF matrices are derived from are not deposited, 66 GB for Sei
and 12 GB for AlphaGenome; the scripts of the next section regenerate them. Only
`analysis/18` reads them directly, everything else works from the VEF matrices in the
archive. Sei model weights live in their own record, <https://zenodo.org/records/4906997>,
and go to `data/Sei/resources/`.

The only files version controlled under `data/` are the track metadata of the four
sequence-to-function models, because they define which output track of each model becomes
which VEF and are small:

| File | Read by | Origin |
|---|---|---|
| `Gosai_MPRA/metadata.csv` | `analysis/01` | the ENCODE accession, cell type and MPRA library of every count table of the Gosai MPRA, with the download link of each |
| `Sei/Sei_tracks_info.csv` | `analysis/01`, `analysis/02_extract_sei_vef` | track table of Sei, <https://github.com/FunctionLab/sei-framework> |
| `AlphaGenome/metadata.csv` | `analysis/01` | output metadata of AlphaGenome, <https://github.com/google-deepmind/alphagenome> |
| `AlphaGenome/metadata_padded.tsv` | `analysis/02_extract_*_ag_vef`, `analysis/18` | the same metadata including the padding rows, which is what gives the column offset of each output head |
| `Enformer/model_track_info.tsv` | `analysis/01` | target table of Enformer, <https://github.com/google-deepmind/deepmind-research/tree/master/enformer> |
| `Borzoi/targets_human.txt` | `analysis/01` | human target table of Borzoi, <https://github.com/calico/borzoi> |

Everything else is downloaded, and the scripts expect this layout:

```
data/
├── Gosai_MPRA/            raw ENCODE files, input of analysis/01_prepare_gosai_data.py,
│                          downloaded from the links in Gosai_MPRA/metadata.csv
├── gosai_mpra/            the 760,679-row label table and the VEF matrices derived from it
├── castillo_mpra/         the independent MPRA of the zero-shot evaluation, its AlphaGenome
│                          predictions and the VEF matrix derived from them
├── Sei/                   the Sei weights under resources/ and the Sei predictions of the
│                          Gosai CREs
├── AlphaGenome/           the AlphaGenome predictions of the Gosai CREs
├── Enformer/
├── Borzoi/
└── DHS64/                 predictions and biosample metadata of the DHS64 baseline
```

Trained EpiCast checkpoints go under `saved/`, which `paper/config.py` resolves relative to
the repository root.

## Predicting activity for your own sequences

The input is a tsv or csv with a `seq` column. All sequences must have the same length; the
paper uses 200-bp CREs.

**1. Predict epigenomic tracks with a sequence-to-function model.** Each CRE is centred in the
native input window of the model, and the predictions over the CRE itself are written to one
h5 file. Both scripts write incrementally and resume where a previous run stopped.

```bash
python scripts/predict_sei.py \
    --seq_path my_seqs.tsv \
    --out_path my_seqs_sei_pred.h5

python scripts/predict_alphagenome.py \
    --seq_path my_seqs.tsv \
    --out_path my_seqs_ag_pred.h5
```

**2. Turn the raw predictions into a VEF matrix.** `paper/analysis/02_extract_sei_vef.py` and
`paper/analysis/02_extract_ag_vef.py` select the track of every (cell type, assay) pair from
the model track metadata and apply the transform of the paper; the selection and transform
helpers live in `paper/utils.py`. The scripts resolve their input and output paths through
`paper/config.py`, so for a new dataset copy one of them and point `pred_path` at your h5. The
result is a table with one `{cell_type}_{assay}` column per VEF, in the same row order as the
sequences.

**3. Predict activity with a trained EpiCast checkpoint.** Config values can be overridden on
the command line, so the same config that trained a model can be pointed at new sequences:

```bash
python scripts/predict.py \
    --config_path saved/0821_gosai_ag_vef_x10_log1p_dnase1_256/<run>/config.yaml \
    --pred_name my_seqs_pred.npy \
    total_dataset.args.seq_file_path=my_seqs.tsv \
    total_dataset.args.epi_file_path=my_seqs_ag_vef.tsv \
    total_dataset.args.target_column=null
```

The prediction is saved next to the checkpoint, with one column per cell type in
`total_dataset.args.cell_types`. Training a model from scratch uses the same configs:

```bash
python scripts/train.py --config_path configs/0821_gosai_ag_vef_x10_log1p_dnase1_256.yaml
```

## Reproducing the paper

`paper/analysis/` turns data into metric tables under `paper/results/`, and `paper/plot/` turns
those tables into one PDF per figure panel under `paper/figures/`. Both stages read all of
their paths, cell-type names, model registry and colours from `paper/config.py`, so no script
takes positional arguments and every script runs from either the repository root or from
`paper/`.

```bash
conda activate epicast
python paper/analysis/07_eval_regression.py
python paper/plot/fig2bc_activity_metrics.py
```

Only `06_infer_trained_model.py` and `10_predict_castillo_mpra.sh` need a GPU; everything else
is CPU only. `results/` and `figures/` are created on demand and are not version controlled.

### Pipeline

The numbering follows the dependency order, but the pipeline is not strictly serial.

```
01_prepare_gosai_data ──────────────────────────────────┐
01_parse_model_track_metadata                           │
                                                        │
02_extract_sei_vef ─┐                                   │
02_extract_ag_vef ──┼─→ 03_normalize_vef ───────────────┤  VEF matrices
02_extract_castillo_ag_vef ─┘                           │
                                                        ├─→ 11_vef_partial_correlation
                                                        ├─→ 11_vef_pairwise_correlation
                                                        ├─→ 18_vef_assay_selection
                                                        ▼
                                     04_vef_activity_specificity   (prints only)
                                     05_train_vef_only_models
                                     06_infer_trained_model        [GPU]
                                     10_predict_castillo_mpra.sh   [GPU]
                                                        │
        ┌───────────────────────────┬───────────────────┴──┐
        ▼                           ▼                      ▼
07_eval_regression        08_eval_classification     09_eval_retrieval
        └───────────────────────────┴──────────────────────┘
                                    ▼
                        14_export_prediction_tables  →  results/predictions/
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
     15_export_figure_metrics                    12_eval_castillo
        → results/figure_metrics/                  → results/castillo/
```

Steps in one line each:

| Step | What it does |
|---|---|
| `01_prepare_gosai_data` | Malinois-style preprocessing of the Gosai MPRA into a 760,679-row label table; z-scores are estimated on training chromosomes only |
| `01_parse_model_track_metadata` | parses the Sei, Enformer, Borzoi and AlphaGenome track metadata onto one naming scheme |
| `02_extract_*_vef` | reads the sequence-to-function model predictions and extracts the four-assay VEF matrix per dataset |
| `03_normalize_vef` | log1p transform for Enformer and Borzoi; Sei and AlphaGenome are normalized during extraction |
| `04_vef_activity_specificity` | sanity check of VEF-activity and VEF-specificity correlations, prints to stdout |
| `05_train_vef_only_models` | fits the VEF-only baselines that never see sequence |
| `06_infer_trained_model` | runs an EpiCast checkpoint, optionally against another dataset config, which is how the zero-shot predictions are produced |
| `07`, `08`, `09` | regression, CTS classification and top-k retrieval metrics |
| `11_vef_pairwise_correlation` | correlations among the four VEFs |
| `11_vef_partial_correlation` | each VEF against activity after conditioning on the other three, plus standardized OLS coefficients |
| `12_eval_castillo` | zero-shot evaluation on the independent Castillo-Hair MPRA |
| `14_export_prediction_tables` | measured and predicted activity side by side in one self-describing table per model |
| `15_export_figure_metrics` | aggregates the metric tables the figure scripts read |
| `18_vef_assay_selection` | ranks the AlphaGenome assays by biosample coverage and by correlation with activity |

Two conventions are worth knowing before reading the numbers. **Residual activity** always
means the activity of a CRE minus its mean over the three training cell types, which is the
quantity that carries cell-type specificity. **CTS CREs** are defined on the Gosai MPRA by
percentile tails of that residual, and on the Castillo MPRA by an absolute activity gap
against the other evaluated cell types; the two datasets are never mixed.

### Figures

One PDF per panel; colorbars and legends are written as separate files. Composing panels into
a figure, and the panel letters themselves, is done by hand, which is why the script names
keep the panel letters of an earlier draft.

| Panel | Script |
|---|---|
| 1C | `plot/figs1_vef_assay_selection.py` |
| 1D | `plot/fig1c_vef_activity_correlation.py` |
| 1E, 1F | `plot/fig1df_activity_correlation_heatmap.py` |
| 1G | `plot/fig1e_dnase_residual_specificity.py` |
| 2A | `plot/fig2a_epicast_scatter.py` |
| 2B, 2C | `plot/fig2bc_activity_metrics.py` |
| 3A | `plot/fig3a_residual_metrics.py` |
| 3B, 3C | `plot/fig3bc_cts_prioritization.py` |
| 3D, 3E | `plot/fig3de_topk_retrieval.py` |
| 3F, 3G | `plot/fig3fg_topk_activity_profile.py` |
| 4A, 4C | `plot/fig4ac_vef_correlation_heatmap.py` |
| 4B, 4D, 4E | `plot/fig4bde_vef_partial_correlation.py` |
| 5A-5E | `plot/fig5_castillo_metrics.py` |

Panels 1A, 1B, 1H are schematics and are not produced by code. In panels 1D, 4B, 4D, 4E and
5A-5E every point is one cell type, summarized by a bar at the mean and an error bar of one
sample SD; the spread is therefore across cell types, not across CREs.

A few conventions are shared by all figure scripts, and changing one figure should not require
touching another: font sizes come from seaborn's `talk` context and are never set per script,
panel height is 6 inches so that panels can be composed without rescaling, a cell type or a
model keeps the same colour in every figure it appears in, and Sei is always drawn before
AlphaGenome.

## Citation

Under review (RECOMB 2026).

## License

MIT License.

## Contact

maplecai142857@gmail.com
