from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, List
import torch
# from catalyst.data import DistributedSamplerWrapper
import numpy as np

# class TaskBatchSampler(BatchSampler):
#     """一个按任务组织batch的sampler"""
#     def __init__(self, dataset: ConcatDataset, batch_size: int, drop_last:bool=False):
#         self.datasets = dataset.datasets
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.samplers = [RandomSampler(dataset) for dataset in self.datasets]
#         self.batch_samplers = [BatchSampler(sampler, batch_size, drop_last) for sampler in self.samplers]
        
#     def __iter__(self):
#         for batch_sampler in self.batch_samplers:
#             for batch in batch_sampler:
#                 yield batch
                
#     def __len__(self):
#         if self.drop_last:
#             return sum(len(d) // self.batch_size for d in self.datasets)
#         else:
#             return sum((len(d) + self.batch_size - 1) // self.batch_size for d in self.datasets)



# class DistributedTaskBatchSampler(DistributedSampler):
#     def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
#                  rank: Optional[int] = None, shuffle: bool = True,
#                  seed: int = 0, drop_last: bool = False, batch_size = 10) -> None:
#         super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
#                          drop_last=drop_last)
#         self.batch_size = batch_size

#     def __iter__(self):
#         indices = list(super().__iter__())
#         batch_sampler = TaskBatchSampler(self.dataset, batch_size=self.batch_size, drop_last=self.drop_last, indices=indices)
#         return iter(batch_sampler)

#     def __len__(self) -> int:
#         return self.num_samples//self.batch_size



# class MultiTaskDataLoader:
#     def __init__(self, dataloaders, cycle=True):
#         """
#         初始化 MultiTaskDataLoader。
#         :param dataloaders: 一个 DataLoader 对象的列表，每个 DataLoader 对应一个任务。
#         """
#         self.dataloaders = dataloaders
#         self.cycle = cycle
#         lengths = [len(dataloader) for dataloader in self.dataloaders]
        
#     def __iter__(self):
#         # 为每个 DataLoader 创建一个迭代器
#         self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
#         self.active_iterator_idx = 0  # 当前活跃的迭代器索引
#         self.iterations_done = [False] * len(self.dataloaders)  # 标记每个 DataLoader 是否完成
#         return self

#     def __next__(self):
#         while True:
#             try:
#                 batch = next(self.iterators[self.active_iterator_idx])
#                 # 成功获取数据后，移动到下一个迭代器
#                 self.active_iterator_idx = (self.active_iterator_idx + 1) % len(self.iterators)
#                 return batch
#             except StopIteration:
#                 # 标记此 DataLoader 已完成至少一次迭代
#                 self.iterations_done[self.active_iterator_idx] = True
#                 # 如果所有迭代器都至少完成了一次迭代，停止迭代
#                 if all(self.iterations_done) == True:
#                     raise StopIteration
                
#                 if self.cycle == True:
#                     # 当前 DataLoader 已经迭代完成，重新创建迭代器
#                     self.iterators[self.active_iterator_idx] = iter(self.dataloaders[self.active_iterator_idx])
#                 else:
#                     # 直接移动到下一个迭代器
#                     self.active_iterator_idx = (self.active_iterator_idx + 1) % len(self.iterators)
    
#     def __len__(self):
#         if self.cycle == True:
#             self.length = (max(lengths) + 1) * len(self.dataloaders)
#         else:
#             self.length = sum(lengths)
#         return self.length
    
#     def set_epoch(self, epoch):
#         for loader in self.dataloaders:
#             if hasattr(loader.sampler, 'set_epoch'):
#                 loader.sampler.set_epoch(epoch)

#     def set_cycle(self, cycle):
#          self.cycle = cycle







class MultiTaskDataLoader:
    def __init__(self, dataloaders, mode='max', cycle=True):
        """
        初始化 MultiTaskDataLoader。
        dataloaders: 一个 DataLoader 对象的列表，每个 DataLoader 对应一个任务。
        """
        self.dataloaders = dataloaders
        self.mode = mode
        self.cycle = cycle
        assert mode in ['max','min']

        
    def __iter__(self):
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders] # 为每个 DataLoader 创建一个迭代器
        self.active_iterator_idx = 0  # 当前活跃的迭代器索引
        self.iterations_done = [False] * len(self.dataloaders)  # 标记每个 DataLoader 是否完成
        return self

    def __next__(self):
        while True:
            try:
                batch = next(self.iterators[self.active_iterator_idx])
                # 成功获取数据后，移动到下一个迭代器
                self.active_iterator_idx = (self.active_iterator_idx + 1) % len(self.iterators)
                return batch
            except StopIteration:
                if self.mode == 'min':
                    # min模式下，最短的loader迭代结束后停止
                    raise StopIteration
                    
                elif self.mode == 'max' and self.cycle == True:
                    # 标记此 DataLoader 已完成至少一次迭代
                    self.iterations_done[self.active_iterator_idx] = True
                    # 循环采样，重新创建迭代器
                    self.iterators[self.active_iterator_idx] = iter(self.dataloaders[self.active_iterator_idx])
                    # max, True模式下，最长的loader迭代结束后停止迭代
                    if all(self.iterations_done) == True:
                        raise StopIteration

                elif self.mode == 'max' and self.cycle == False:
                    # 不循环采样，删除这个迭代器，移动到下一个迭代器
                    self.iterators.pop(self.active_iterator_idx)
                    if len(self.iterators) == 0:
                        raise StopIteration
                    self.active_iterator_idx = (self.active_iterator_idx) % len(self.iterators)
    
    def __len__(self):
        lengths = [len(dataloader) for dataloader in self.dataloaders]
        if self.mode == 'min':
            length = min(lengths) * len(self.dataloaders) + np.argmin(lengths)
        elif self.mode == 'max' and self.cycle == True:
            length = max(lengths) * len(self.dataloaders) + np.argmax(lengths)
        elif self.mode == 'max' and self.cycle == False:
                length = sum(lengths)
        return length
    
    def set_epoch(self, epoch):
        for loader in self.dataloaders:
            if hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch)

    def set_cycle(self, cycle):
        self.cycle = cycle
        self.__iter__()

    def set_mode(self, mode):
        self.mode = mode
        self.__iter__()



if __name__ == '__main__':

    from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
    from torch.utils.data.sampler import BatchSampler
    from torch.utils.data.sampler import SequentialSampler

    a = TensorDataset(torch.arange(0, 100))
    b = TensorDataset(torch.arange(500, 600))

    a_loader = DataLoader(a, batch_size=10)
    b_loader = DataLoader(b, batch_size=20)

    print(len(a_loader), len(b_loader))


    c_loader = MultiTaskDataLoader([a_loader, b_loader], mode='max', cycle=True)
    for batch in c_loader:
        print(batch)
    print(len(c_loader))
    print('done')

    c_loader = MultiTaskDataLoader([a_loader, b_loader], mode='max', cycle=False)
    for batch in c_loader:
        print(batch)
    print(len(c_loader))
    print('done')

    c_loader = MultiTaskDataLoader([a_loader, b_loader], mode='min', cycle=True)
    for batch in c_loader:
        print(batch)
    print(len(c_loader))
    print('done')

    c_loader = MultiTaskDataLoader([a_loader, b_loader], mode='min', cycle=False)
    for batch in c_loader:
        print(batch)
    print(len(c_loader))
    print('done')
