import numpy as np
import pandas as pd
import gzip

from genoml.utils import *

# 下面的代码试图复现malinois preprocessing


metadata_df = pd.read_csv('data/Gosai_MPRA/metadata.csv', comment='#')


def read_fasta_gz_to_dict(fasta_gz_path):
    """
    从 .fasta.gz 文件读取并转换为 {id: sequence} 字典。
    """
    seq_dict = {}
    with gzip.open(fasta_gz_path, "rt") as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    seq_dict[seq_id] = "".join(seq_lines)
                seq_id = line[1:]  # 去掉 '>'
                seq_lines = []
            else:
                seq_lines.append(line)
        # 处理最后一条序列
        if seq_id is not None:
            seq_dict[seq_id] = "".join(seq_lines)
    return seq_dict


def map_data_project(ol):
    ukbb_list = ['OL27', 'OL28', 'OL29', 'OL30', 'OL31', 'OL32', 'OL33']
    gtex_list = ['OL41', 'OL42', 'OL41_42', 'OL41B']
    ol15_list = ['OL15']
    if ol in ukbb_list:
        return 'UKBB'
    elif ol in gtex_list:
        return 'GTEx'
    elif ol in ol15_list:
        return 'OL15'
    else:
        raise ValueError(f'Unknown OL: {ol}')



# # 先把每个 ref_accession 对应的 dic 都读出来（并 expand），存到一个 list 里
# dics = []
# ref_accessions = list(set(metadata_df['ref_accession']))
# print(ref_accessions)

# for ref_accession in ref_accessions:
#     if ref_accession == 'ENCFF443RYE;ENCFF728XQT':  # 跳过组合项
#         continue
#     dic = read_fasta_gz_to_dict(f"data/Gosai_MPRA/raw/{ref_accession}.fasta.gz")
#     # dic = expand_dic(dic)
#     dics.append(dic)

# # 一次性合并：如果遇到“相同 ID 但序列不同”，则丢弃该 ID（最终不保留任何值）
# merged = {}
# conflict_ids = set()

# for dic in dics:
#     for k, v in dic.items():
#         if k in conflict_ids:
#             continue
#         if k not in merged:
#             merged[k] = v
#         else:
#             # 已经见过该 ID，检查是否一致
#             if merged[k] != v:
#                 conflict_ids.add(k)
#                 merged.pop(k, None)  # 放弃该 ID 的所有值

# seq_dict = merged
# print(f"final seq_dict size = {len(seq_dict)}, conflicts dropped = {len(conflict_ids)}")
# print(f"conflicts: {conflict_ids}")



from collections import Counter

cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']

dfs_dict = {}
for cell_type in cell_types:
    dfs = []
    for i, row in metadata_df.iterrows():
        if row['cell_type'] == cell_type:
            ref_accession = row['ref_accession']

            if ref_accession == 'ENCFF443RYE;ENCFF728XQT':
                dict1 = read_fasta_gz_to_dict(f"data/Gosai_MPRA/raw/ENCFF443RYE.fasta.gz")
                dict2 = read_fasta_gz_to_dict(f"data/Gosai_MPRA/raw/ENCFF728XQT.fasta.gz")
                seq_dict = (dict1 | dict2)
            else:
                seq_dict = read_fasta_gz_to_dict(f"data/Gosai_MPRA/raw/{ref_accession}.fasta.gz")

            file_accession = row['file_accession']
            print(file_accession)
            df = pd.read_csv(f"data/Gosai_MPRA/raw/{file_accession}.tsv", sep='\t', low_memory=False)
            df['OL'] = row['project']
            df['data_project'] = map_data_project(row['project'])
            df['seq'] = df['ID'].map(seq_dict)
            dfs.append(df)
    dfs = pd.concat(dfs).reset_index(drop=True)
    dfs_dict[cell_type] = dfs




def filter_nonnatural_oligos(df):
    mask = ~((df['chr'].isna()) & (df['data_project'] == 'OL15'))
    df_filtered = df[mask].copy()
    return df_filtered


def filter_seq_is_none(df):
    mask = (df['seq'].notna())
    df_filtered = df[mask].copy()
    return df_filtered



def filter_plasmid_and_rna_count(df):
    # DNA count >= 20, RNA_count > 0
    DNA_mask_1 = (df['DNA_mean'] >= 5) & (df['data_project']=='OL15') # OL15 4 replicates
    DNA_mask_2 = (df['DNA_mean'] >= 4) & (df['data_project']!='OL15') # other 5 replicates
    RNA_mask = (df['exp_mean'] > 0)
    mask = (DNA_mask_1 | DNA_mask_2) & RNA_mask
    df_filtered = df[mask].copy()
    return df_filtered



def filter_log2fc_6std(df):
    std = df['log2FoldChange'].std()
    mean = df['log2FoldChange'].mean()
    mask = (df['log2FoldChange'] >= (mean - 6 * std))
    df_filtered = df[mask].copy()
    return df_filtered



def merge_ukbb_gtex_ol15(df):
    # 定义优先级映射
    priority_map = {'UKBB': 3, 'GTEx': 2, 'OL15': 1}
    # 给 dataframe 新增一列 priority 便于后续比较
    df['priority'] = df['data_project'].map(priority_map)
    if df['priority'].isna().any():
        print('存在 priority 为空的行!!')
    
    # 构造聚合字典：log2FoldChange 做 mean，其余列都用 'first'
    # 如果你的表有很多列，只关心部分字段，也可以只在聚合字典中声明关心的列。
    agg_dict = {}
    for col in df.columns:
        if col == 'log2FoldChange':
            agg_dict[col] = 'mean'
        elif col == 'lfcSE':
            agg_dict[col] = 'max'
        else:
            # 取第一行，如果同一个 (ID, data_project) 下这些列确实没有冲突
            agg_dict[col] = 'first'

    # 一次分组：对 seq + data_project 分组
    df_agg = df.groupby(['seq', 'data_project'], as_index=False).agg(agg_dict)

    # 对聚合完的数据按优先级降序排；同一个 ID 下，UKBB(3) > GTEx(2) > OL15(1)
    df_agg.sort_values(['seq', 'priority'], ascending=[True, False], inplace=True)

    # 同一个 ID 只保留优先级最高的项目那条记录
    df_final = df_agg.drop_duplicates(subset='seq', keep='first').copy()

    # 不需要 priority 列了，可以删除
    df_final.drop(columns='priority', inplace=True)
    
    return df_final






for cell_type in cell_types:
    print(cell_type)
    df_i = dfs_dict[cell_type]
    print(len(df_i))
    df_i = filter_nonnatural_oligos(df_i)
    print(len(df_i))
    df_i = filter_seq_is_none(df_i)
    print(len(df_i))
    df_i = filter_plasmid_and_rna_count(df_i)
    print(len(df_i))
    df_i = merge_ukbb_gtex_ol15(df_i)
    print(len(df_i))
    df_i = filter_log2fc_6std(df_i)
    print(len(df_i))
    dfs_dict[cell_type] = df_i



prepared = {
    ct: df.drop_duplicates('seq').set_index('seq').add_suffix(f'_{ct}')
    for ct, df in dfs_dict.items()
}

all_meta = pd.concat(
    [df[['seq', 'ID', 'chr', 'pos', 'ref_allele', 'alt_allele', 'allele', 'OL', 'data_project']]
     for df in dfs_dict.values()],
    ignore_index=True
)
meta = all_meta.drop_duplicates('seq').set_index('seq')

merged_df = pd.concat([meta] + list(prepared.values()), axis=1, join='outer')
merged_df = merged_df.reset_index()
print(merged_df.columns)

# lfcSE


merged_df = merged_df.rename(columns={
    'ID': 'id',
    'log2FoldChange_K562': 'K562',
    'log2FoldChange_HepG2': 'HepG2',
    'log2FoldChange_SK-N-SH': 'SK-N-SH',
    'log2FoldChange_HCT116': 'HCT116',
    'log2FoldChange_A549': 'A549',
})

merged_df = merged_df[['seq', 'id', 'chr', 'pos', 'ref_allele', 'alt_allele', 'allele', 'OL', 'data_project', 'K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']]
merged_df['chr'] = 'chr' + merged_df['chr'].astype(str)
merged_df['pos'] = pd.to_numeric(merged_df['pos'], errors='coerce').astype('Int64')
print(merged_df.shape)
merged_df.to_csv('data/Gosai_MPRA/Gosai_MPRA_reprocessed_0203.tsv', sep='\t', index=False)


merged_df = merged_df[merged_df['seq'].str.len() == 200].reset_index(drop=True)
print(f'after filter len == 200, merged_df.shape = {merged_df.shape}')


merged_df = merged_df[(merged_df[['K562', 'HepG2', 'SK-N-SH']].notna().all(axis=1))].reset_index(drop=True)
print(f'after filter K562, HepG2, SK-N-SH not nan, merged_df.shape = {merged_df.shape}')

merged_df = merged_df.sort_values(by=['chr', 'pos']).reset_index(drop=True)

merged_df.to_csv('data/Gosai_MPRA/Gosai_MPRA_reprocessed_0203_filter.tsv', sep='\t', index=False)
