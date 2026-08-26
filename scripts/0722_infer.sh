#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

#python paper/analysis/06_infer_trained_model.py -k saved/0722_gosai_ag_vef_log1p/0722_083528/checkpoints/best.pth
# python paper/analysis/06_infer_trained_model.py -k saved/0722_gosai_ag_vef_log1p_256/0722_104248/checkpoints/best.pth
python paper/analysis/06_infer_trained_model.py -c saved/0722_gosai_sei_vef_log1p/0722_083548/config.yaml
# python paper/analysis/06_infer_trained_model.py -k saved/0722_gosai_seq_only/0722_130620/checkpoints/best.pth
# python paper/analysis/06_infer_trained_model.py -k saved/0722_gosai_seq_only_256/0722_160527/checkpoints/best.pth
# python paper/analysis/06_infer_trained_model.py -k saved/0722_gosai_seq_only_malinois/0722_171043/checkpoints/best.pth
