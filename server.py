import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))
from MPRA_predict import models, datasets, metrics, utils
from MPRA_predict.utils import *


# file_name = "xxxx.tsv"

df_track = pd.read_json(ROOT_DIR / 'data/Sei/Sei_61_cell_types_tracks.json')
cell_types = df_track.index.tolist()
assays = df_track.columns.tolist()


subprocess.run([
    "python", "0_predict_Sei_feature.py", 
    "-i", "../data/Random/random_200bp_seqs.csv",
    "-o", "../data/Random/random_200bp_Sei_pred.npy"])

subprocess.run([
    "python", "0_predict_Sei_VEF.py", 
    "-i", "../data/Random/random_200bp_Sei_pred.npy",
    "-o", "../data/Random/random_200bp_Sei_VEF.tsv"])

subprocess.run([
    "python", "0_predict_EpiCast_activity.py", 
    "--config", "../configs/config_1120_test.yaml",
    "--total_dataset.args.seq_file_path", "../data/Random/random_200bp_seqs.csv", 
    "--total_dataset.args.epi_file_path", "../data/Random/random_200bp_Sei_VEF.tsv",
    "--output_name", "random_200bp_EpiCast_pred.npy"])
