#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python scripts/train.py -c configs/0722_gosai_seq_only.yaml
python scripts/train.py -c configs/0722_gosai_seq_only_256.yaml
python scripts/train.py -c configs/0722_gosai_seq_only_malinois.yaml
