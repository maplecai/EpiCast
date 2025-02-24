import sys
sys.path.append('..')
from MPRA_predict.utils import *
from MPRA_predict.datasets import *

bed_df = pd.read_csv('/home/shared/enformer_data/human/sequences.bed', sep='\t', header=None)
bed_df.columns = ['chr', 'start', 'end', 'split']
bed_df.to_csv('data/enformer_sequences.csv', index=False)
print(bed_df.head())
print(bed_df.shape)

# 处理bigwig文件得到target，失败，未来可以考虑用basenji github提供的脚本

# targets = pd.read_csv('/home/shared/enformer_data/human/targets.txt', sep='\t')
# targets_K562 = targets[targets['description'] == 'DNASE:K562']
# targets_K562

# import pyBigWig
# import numpy as np
# from multiprocessing import Pool
# from tqdm import tqdm

# def init_worker(bw_path):
#     # 在子进程初始化时打开 bigWig 文件，全局变量 bw 将在子进程内可用
#     global bw
#     bw = pyBigWig.open(bw_path)

# def process_row(row):
#     chrom, start, end = row
#     # 1024个bin,每个bin128bp,一共128*1024=131072bp
#     start = start
#     end = end
#     mean_values = bw.stats(chrom, start, end, nBins=1024, type='mean')
#     mean_values = mean_values[64: -64]
#     return mean_values

# def process_track(track_row, bed_df, num_workers):
#     track_index = track_row['index']
#     identifier = track_row['identifier']
#     bigwig_file = f'../../data/Enformer_tracks/downloads/{identifier}.bigWig'

#     # 将 bed_df 转换成一个简单的列表，以便传入 pool.imap
#     rows = [(r.chr, r.start, r.end) for r in bed_df.itertuples()]

#     # 使用多进程
#     # initializer 用于在子进程启动时运行 init_worker，将 bw 对象在子进程内打开
#     with Pool(processes=num_workers, initializer=init_worker, initargs=(bigwig_file,)) as pool:
#         # 使用 imap 异步迭代，配合 tqdm 显示进度条
#         labels = list(tqdm(pool.imap(process_row, rows), total=len(rows)))

#     labels = np.array(labels)
#     np.save(f'data/labels_track_index_{track_index}.npy', labels)

# # 假设你有一个 targets_K562 是 DataFrame，里面有至少四行数据
# # 以及 bed_df 是你要处理的区间表格
# for i in range(4):
#     track_row = targets_K562.iloc[i]
#     process_track(track_row, bed_df, num_workers=28)



# 选择从Enformer tfr文件中提取标签

# @title `get_dataset(organism, subset, num_threads=8)`
import glob
import json
import functools
import tensorflow as tf
import tensorflow_hub as hub

# @title `get_targets(organism)`
def get_targets(organism):
  # targets_txt = f'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{organism}.txt'
    targets_txt = f'/home/shared/enformer_data/{organism}/targets.txt'
    return pd.read_csv(targets_txt, sep='\t')


def organism_path(organism):
    # return os.path.join('gs://basenji_barnyard/data', organism)
    return os.path.join('/home/shared/enformer_data', organism)


def get_dataset(organism, subset, num_threads=8):
    metadata = get_metadata(organism)
    dataset = tf.data.TFRecordDataset(
        tfrecord_files(organism, subset),
        compression_type='ZLIB',
        num_parallel_reads=num_threads)
    dataset = dataset.map(
        functools.partial(deserialize, metadata=metadata),
        num_parallel_calls=num_threads)
    return dataset


def get_metadata(organism):
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join(organism_path(organism), 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def tfrecord_files(organism, subset):
    # Sort the values by int(*).
    return sorted(
        tf.io.gfile.glob(os.path.join(organism_path(organism), 'tfrecords', f'{subset}-*.tfr')), 
        key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target, (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {
        'sequence': sequence,
        'target': target
        }

human_dataset = get_dataset('human', 'test')
# human_dataset = get_dataset('human', '*')

seqs = []
for item in tqdm(human_dataset):
    seqs.append(item['sequence'].numpy())
seqs = np.stack(seqs, axis=0)
print(seqs.shape)
np.save('data/enformer_tfr_sequences_test.npy', seqs)
# np.save('data/enformer_tfr_sequences.npy', seqs)

seqs = np.array(onehots2strs(seqs))
print(seqs.shape)
save_txt('data/enformer_tfr_sequences_test.txt', seqs)
# save_txt('data/enformer_tfr_sequences.txt', seqs)
del(seqs)

targets = []
for item in tqdm(human_dataset):
    targets.append(item['target'].numpy())
targets = np.stack(targets, axis=0)
print(targets.shape)

np.save('data/enformer_tfr_targets_test.npy', targets)
# np.save('data/enformer_tfr_targets.npy', targets)
del(targets)
