import numpy as np
import pandas as pd

cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']

random_output = np.random.rand(10, 5) # uniform (0,1), 10 seqs, 5 cell types
df = pd.DataFrame(data=random_output, columns=cell_types)
df.to_csv('random_output.csv', index=False)
