import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
# from scipy.stats import pearsonr
import pyBigWig
from genoml.metrics import pearson

BSSIDs = ['BSS00558', 'BSS00762', 'BSS00007', 'BSS00492', 'BSS01562']
cell_types = ['K562', 'HepG2', 'A549', 'HCT116', 'SKNSH']
assays = ['DNase', 'H3K4me1', 'H3K4me3', 'H3K9me3', 'H3K27me3', 'H3K27ac', 'H3K36me3', 'H3K9ac', 'H3K4me2', 'CTCF']

import pyBigWig

def get_pval_mean_values(df, bw_file):
    bw_mean_values = []
    bw_reader = pyBigWig.open(bw_file)
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        chr, pos = row['chr'], row['pos']
        start, end = pos - 100, pos + 100
        try:
            values = bw_reader.values(chr, start, end)
            bw_mean_values.append(np.mean(values))
        except RuntimeError as e:
            bw_mean_values.append(np.nan)
            pass
    bw_reader.close()
    bw_mean_values = np.array(bw_mean_values)
    return bw_mean_values



def find_files_with_string(folder_path, search_string):
    matching_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if search_string in file:
                matching_files.append(os.path.join(root, file))
    return matching_files


for i, cell_type in enumerate(cell_types):
    for j, assay in enumerate(assays):
        BSSID = BSSIDs[i]
        bw_files = find_files_with_string("../../data/EpiMap/raw_final", f'{assay}_{BSSID}')

        if len(bw_files) == 1:
            bw_file = bw_files[0]
            print(bw_file)
            bw_mean_values = get_pval_mean_values(df, bw_file)
            df[f'{cell_type}_{assay}_pval'] = bw_mean_values

        else:
            print(f"Error: {cell_type} {assay} {BSSID} has {len(bw_files)} bw files: {bw_files}")
            df[f'{cell_type}_{assay}_pval'] = np.nan


df.to_csv('../../data/SirajMPRA/SirajMPRA_ref_pval.csv', index=False)