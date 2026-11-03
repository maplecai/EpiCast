# EpiCast

EpiCast predicts episomal cis-regulatory element (CRE) activity across human cell types by combining DNA sequence with **virtual epigenomic features (VEFs)** derived from Sei or AlphaGenome. VEFs provide cell-type-specific regulatory context, enabling prediction in cell types without matched MPRA training data.

## Installation

```bash
git clone https://github.com/maplecai/EpiCast.git
cd EpiCast
conda create -n epicast python=3.11
conda activate epicast
pip install -r requirements.txt
pip install -e .
```

AlphaGenome track prediction additionally requires [alphagenome-pytorch](https://github.com/genomicsxai/alphagenome-pytorch) in a separate Python ≥3.12 environment.

## Data and pretrained models

Download `epicast_data.zip` from [Zenodo](https://doi.org/10.5281/zenodo.17669740) and extract it into the repository root:

```bash
unzip epicast_data.zip -d .
```

The archive contains MPRA activity tables, VEF matrices, DHS64 baseline predictions, and trained EpiCast and sequence-only checkpoints. Raw sequence-to-function predictions are regenerated separately. Sei prediction requires [Sei weights](https://zenodo.org/records/4906997) under `data/Sei/resources/`; AlphaGenome weights are downloaded on first use.

## Training and prediction

Train the AlphaGenome-VEF model:

```bash
python scripts/train.py --config_path configs/0821_gosai_ag_vef_x10_log1p_dnase1_256.yaml
```

For inference, use a saved run's configuration and checkpoint:

```bash
python scripts/predict.py \
  --config_path saved/0821_gosai_ag_vef_x10_log1p_dnase1_256/0820_155453/config.yaml \
  --pred_name predictions.npy
```

For new CREs, provide a CSV/TSV with a `seq` column of 200-bp sequences and a matching VEF table. Override the input paths when running inference:

```bash
python scripts/predict.py \
  --config_path saved/0821_gosai_ag_vef_x10_log1p_dnase1_256/0820_155453/config.yaml \
  --pred_name my_sequences.npy \
  total_dataset.args.seq_file_path=my_sequences.tsv \
  total_dataset.args.epi_file_path=my_vefs.tsv \
  total_dataset.args.target_column=null
```

The VEF table must use the same row order as the sequence table, with `{cell_type}_{assay}` columns for DNase, H3K4me3, H3K27ac and CTCF. Use the VEF source and preprocessing that match the checkpoint. Predictions are saved under the configuration's `saved_dir`, with columns in `total_dataset.args.cell_types` order.

To generate raw tracks, use `scripts/predict_sei.py` or `scripts/predict_alphagenome.py` with `--seq_path` and `--out_path`. VEF extraction examples are in [paper/analysis](paper/analysis/README.md); adapting them to a new dataset requires changing their input and output paths.

## Reproducing the paper

See the [reproduction guide](paper/README.md) for the analysis workflow and [figure index](paper/plot/README.md) for plotting scripts.

| Directory | Contents |
|---|---|
| `src/epicast/` | Models, datasets and training library |
| `scripts/` | Training, inference and track-prediction entry points |
| `configs/` | Paper training and inference configurations |
| `paper/` | Analysis and figure-generation code |
| `data/` | Track metadata and downloaded datasets |
| `saved/` | Downloaded or locally trained checkpoints |

## Citation and license

Manuscript under review. Code is distributed under the [MIT License](LICENSE).

Contact: [maplecai142857@gmail.com](mailto:maplecai142857@gmail.com).
