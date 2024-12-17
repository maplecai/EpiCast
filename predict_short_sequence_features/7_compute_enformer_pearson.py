import sys
sys.path.append('../..')
from MPRA_predict.utils import *
from MPRA_predict.datasets import *
np.set_printoptions(suppress=True, precision=6)

bed_df = pd.read_csv('data/enformer_data.csv')
print(bed_df.shape)

preds = np.load('data/Enformer_test_pred_196608.npy', mmap_mode='r')
print(preds.shape)

targets = np.load('data/enformer_test_targets.npy', mmap_mode='r')
print(targets.shape)


# 计算每个特征通道的相关性系数
corr = np.zeros(preds.shape[2])
for i in tqdm(range(preds.shape[2])):
    pred = preds[:, :, i].reshape(-1)  # 获取第 c 个特征通道的预测值，并展平成 N*L
    target = targets[:, :, i].reshape(-1)  # 获取第 c 个特征通道的标签值，并展平成 N*L
    r = np.corrcoef(pred, target)[0, 1] # numpy速度快一点
    corr[i] = r

np.save('pearson.npy', corr)