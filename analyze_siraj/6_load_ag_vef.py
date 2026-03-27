import pandas as pd
import h5py

path = "alphagenome/siraj_ag_pred.h5"
with h5py.File(path, "a") as f:
    print(f.keys())
    print(f['DNase'].shape)
    