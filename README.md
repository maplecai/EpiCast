# EpiCast: leveraging virtual epigenomic features to predict episomal regulatory activity across cell types

## Introduction

![Overview](figures/fig1.png)

Designing regulatory sequences for synthetic biology and gene therapy requires understanding how DNA sequences drive cell-type-specific expression, yet existing MPRA-based models typically fail to generalize beyond the cell types used for training.

To address this challenge, we present **EpiCast**, a deep learning framework for predicting episomal cis-regulatory element (CRE) activity across diverse human cell types. We integrate DNA sequence with **virtual epigenomic features (VEFs)**: cell-type-specific regulatory proxies inferred from large-scale genomic sequence-to-function models such as Sei. Although episomal DNA lacks native chromatin structure, these model-derived features capture how different cell types are predicted to interpret a given sequence, enabling EpiCast to incorporate contextual regulatory information without requiring MPRA data from every cell type.

Trained on MPRA datasets, EpiCast learns both sequence grammar and cell-type-dependent regulatory logic. As a result, it achieves strong performance within training cell types and generalizes robustly to previously unseen ones. We additionally provide a web server that predicts episomal CRE activity across 61 human cell types, supporting applications in regulatory element design, functional genomics, and therapeutic engineering.

## Installation

```bash
git clone https://github.com/maplecai/EpiCast.git
conda create -n mpra python=3.10
conda activate mpra
pip install -r requirements.txt
```

## Data preparation

Download the MPRA datasets from: 
https://zenodo.org/records/17669741 
and place the files under `./data/Gosai_MPRA/`.

Download the Sei model weights from: 
https://zenodo.org/records/4906997 
and place them under `./data/Sei/`.


## Predict CRE activity


Step 1: Predict Sei features
```bash
python predict_CRE_activity/0_predict_Sei_feature.py \
    -i ./data/Random/random_200bp_seqs.csv \
    -o ./data/Random/random_200bp_Sei_pred.npy
```

Step 2: Convert Sei features into VEFs
```bash
python predict_CRE_activity/0_predict_Sei_VEF.py \
    -i ./data/Random/random_200bp_Sei_pred.npy \
    -o ./data/Random/random_200bp_Sei_VEF.tsv
```

Step 3: Predict CRE activity using EpiCast
```bash
python predict_CRE_activity/0_predict_EpiCast_activity.py \
    --config ./configs/config_1120_test.yaml \
    --total_dataset.args.seq_file_path ./data/Random/random_200bp_seqs.csv \
    --total_dataset.args.epi_file_path ./data/Random/random_200bp_Sei_VEF.tsv \
    --output_name random_200bp_EpiCast_pred.npy
```

## Citation

Under review (RECOMB 2026).

## License

MIT License.

## Contact

maplecai142857@gmail.com
