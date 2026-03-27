import pandas as pd
import genoml

my_df = pd.read_csv('data/Gosai_MPRA/Gosai_MPRA_reprocessed_0305_len200.tsv', sep='\t')
print('my processed dataset')
print(my_df.shape)
print(my_df.columns)
print(my_df[my_df['seq'].duplicated(keep=False)])
my_df = my_df.drop_duplicates(subset=['seq'])

original_df = pd.read_csv('data/Gosai_MPRA/Gosai_MPRA_nature_0206.tsv', sep='\t')
print('gosai original dataset from nature')
print(original_df.shape)
print(original_df.columns)
print(original_df[original_df['seq'].duplicated(keep=False)])
original_df = original_df.drop_duplicates(subset=['seq'])

merged_df = pd.merge(original_df, my_df, on='seq', how='left')
print('merged dataset')
print(merged_df.shape)
print(merged_df.columns)

cell_types = ['K562', 'HepG2', 'SK-N-SH']
cols_x = [f'{c}_x' for c in cell_types]
cols_y = [f'{c}_y' for c in cell_types]
print(merged_df[cols_y].notna().all(axis=1).sum())

# for cell_type in cell_types:
#     r = genoml.metrics.pearson(merged_df[f'{cell_type}_x'], merged_df[f'{cell_type}_y'])
#     print(cell_type, 'pearson', r)

# merged_df[cell_types] = merged_df[cols_x]
# merged_df = merged_df.drop(columns=cols_x+cols_y)
# print(merged_df.columns)
# merged_df.to_csv('data/Gosai_MPRA/Gosai_MPRA_merged_HCT116_A549_0206.tsv', sep='\t', index=False)


# merged_df = merged_df[merged_df['seq'].str.len() == 200].reset_index(drop=True)
# print(f'after filter len == 200, merged_df.shape = {merged_df.shape}')
# merged_df.to_csv('data/Gosai_MPRA/Gosai_MPRA_merged_HCT116_A549_0206_200bp.tsv', sep='\t', index=False)
