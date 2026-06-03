import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Optional
import time

# from .enformer_web_dataset import EnformerWebDataset

class MixedLengthEnformerDataset(IterableDataset):
    def __init__(
        self,
        base_dataset_args: dict,
        length_list: List[int],
        token_size: int = 128,
        length_weights: Optional[List[float]] = None,
        random_crop: bool = True,
        center_crop: bool = False,
        samples_per_length: int = 32,
    ):
        self.length_list = sorted(length_list)
        self.token_size = token_size
        self.random_crop = random_crop
        self.center_crop = center_crop
        self.samples_per_length = samples_per_length
        
        # 验证长度
        for length in self.length_list:
            if length % token_size != 0:
                raise ValueError(f"Length {length} not multiple of {token_size}")

        # 设置采样权重
        if length_weights is None:
            self.length_weights = np.ones(len(self.length_list)) / len(self.length_list)
        else:
            self.length_weights = np.array(length_weights) / sum(length_weights)

        self.base_dataset_args = base_dataset_args.copy()
        
        # 【优化1】不在初始化时创建子数据集，推迟到迭代开始，避免多进程冲突
        self.dataset = None

    def _get_synced_length(self, step_index: int) -> int:
        """【优化2】同步多线程长度：基于步数产生随机数，确保所有worker步调一致"""
        # 使用step_index作为种子，这样不同worker在同一个chunk内会选到同一个长度
        rs = np.random.RandomState(step_index)
        return rs.choice(self.length_list, p=self.length_weights)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        # 【优化3】只在迭代开始时初始化一次底层数据集
        # 注意：EnformerWebDataset 内部应支持动态设置 cropped_len
        # 如果不支持，这里建议在底层类增加一个 set_cropped_len 方法
        if self.dataset is None:
            from .enformer_web_dataset import EnformerWebDataset
            self.dataset = EnformerWebDataset(**self.base_dataset_args)
        
        base_iterator = iter(self.dataset)
        
        chunk_step = 0
        try:
            while True:
                # 决定当前Chunk的长度
                current_length = self._get_synced_length(chunk_step)
                
                # 更新底层数据集的裁剪长度（假设底层dataset支持动态修改属性）
                # 这样避免了频繁销毁和重建对象的IO开销
                if hasattr(self.dataset, 'cropped_len'):
                    self.dataset.cropped_len = current_length
                
                # 尝试从底层迭代器连续获取数据
                for _ in range(self.samples_per_length):
                    try:
                        sample = next(base_iterator)
                        # 这里可以在sample中注入当前长度信息，确保collate_fn万无一失
                        yield sample
                    except StopIteration:
                        # 【优化4】底层数据读完了，正式退出，结束Epoch
                        return 
                
                chunk_step += 1
                
        except Exception as e:
            # 记录异常并退出
            print(f"Worker {worker_id} error: {e}")
            return

def create_length_bucketed_collate_fn(token_size: int = 128, min_batch_size: int = 1):
    def collate_fn(batch):
        # 过滤空batch
        batch = [s for s in batch if s is not None]
        if len(batch) < min_batch_size:
            return {}

        # 统计长度分布
        length_groups = {}
        for sample in batch:
            l = sample['seq'].shape[0]
            if l not in length_groups: length_groups[l] = []
            length_groups[l].append(sample)

        # 选出样本数最多的长度组（由于我们在Dataset层做了同步，这里通常只会有一种长度）
        main_len = max(length_groups, key=lambda k: len(length_groups[k]))
        val_samples = length_groups[main_len]

        if len(val_samples) < min_batch_size:
            return {}

        # 常规堆叠
        seqs = torch.stack([torch.as_tensor(s['seq']) for s in val_samples])
        targets = torch.stack([torch.as_tensor(s['target']) for s in val_samples])

        return {
            'seq': seqs,
            'target': targets,
            # 'len': main_len
        }
    return collate_fn