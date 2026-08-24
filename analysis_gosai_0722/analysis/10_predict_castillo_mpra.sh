#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

python_script="analysis_gosai_0722/analysis/06_infer_trained_model.py"
dataset_config="configs/0821_castillo_dataset_N_dnase1.yaml"
output_name="castillo_preds_pad_N.npy"

# EpiCast-AlphaGenome, VEF variant B. This is the run that fig5 evaluates
# (config.epicast_ag_castillo_pred resolves to the output of this line).
python $python_script -c saved/0821_gosai_ag_vef_x10_log1p_dnase1_256/0820_155453/config.yaml -dc $dataset_config -o $output_name

# EpiCast-Sei is intentionally not run here: every castillo_dataset config carries
# an AlphaGenome VEF matrix, so pairing one with the Sei checkpoint would feed the
# model a VEF it was not trained on. A Sei-VEF inference set would have to point at
# data/castillo_mpra/sei_vef.tsv first.
# python $python_script -c saved/0722_gosai_sei_vef_log1p_256/0723_031345/config.yaml -dc $dataset_config -o $output_name

# Earlier variants, kept for reference:
# -dc configs/0820_castillo_dataset_N_log1p128.yaml   # variant C
# -dc configs/0728_castillo_dataset_N.yaml            # variant A, pre-CTCF-fix