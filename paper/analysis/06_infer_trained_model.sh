#!/usr/bin/env bash
set -e

script_dir="$(cd "$(dirname "$0")" && pwd)"
# the -c / -dc / -o arguments below are all relative to the project root
cd "$script_dir/../.."

python_script="$script_dir/06_infer_trained_model.py"

# python $python_script -k saved/0722_gosai_ag_vef_log1p/0722_083528/checkpoints/best.pth
# python $python_script -k saved/0722_gosai_ag_vef_log1p_256/0722_104248/checkpoints/best.pth
# python $python_script -c saved/0722_gosai_sei_vef_log1p/0722_083548/config.yaml
# python $python_script -k saved/0722_gosai_seq_only/0722_130620/checkpoints/best.pth
# python $python_script -k saved/0722_gosai_seq_only_256/0722_160527/checkpoints/best.pth
# python $python_script -k saved/0722_gosai_seq_only_malinois/0722_171043/checkpoints/best.pth

# python $python_script -c saved/0722_gosai_sei_vef_log1p_256/0723_031345/config.yaml
# python $python_script -c saved/0722_gosai_ag_vef_log1p_256_trans3/0723_031448/config.yaml
# python $python_script -c saved/0722_gosai_seq_only_256_5/0723_051843/config.yaml

# EpiCast-AlphaGenome, VEF variant B: the run config.epicast_ag_config points at.
python $python_script -c saved/0821_gosai_ag_vef_x10_log1p_dnase1_256/0820_155453/config.yaml