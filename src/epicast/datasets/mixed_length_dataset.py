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
        
        # validate the lengths
        for length in self.length_list:
            if length % token_size != 0:
                raise ValueError(f"Length {length} not multiple of {token_size}")

        # sampling weights
        if length_weights is None:
            self.length_weights = np.ones(len(self.length_list)) / len(self.length_list)
        else:
            self.length_weights = np.array(length_weights) / sum(length_weights)

        self.base_dataset_args = base_dataset_args.copy()
        
        # the sub-dataset is built when iteration starts, not here, to avoid clashes between workers
        self.dataset = None

    def _get_synced_length(self, step_index: int) -> int:
        """One length for every worker: the random draw is seeded by the step index."""
        # seeding on step_index makes all workers pick the same length within a chunk
        rs = np.random.RandomState(step_index)
        return rs.choice(self.length_list, p=self.length_weights)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        # the underlying dataset is built once, when iteration starts
        # EnformerWebDataset has to allow cropped_len to be set on the fly
        # if it does not, add a set_cropped_len method there
        if self.dataset is None:
            from .enformer_web_dataset import EnformerWebDataset
            self.dataset = EnformerWebDataset(**self.base_dataset_args)
        
        base_iterator = iter(self.dataset)
        
        chunk_step = 0
        try:
            while True:
                # length of this chunk
                current_length = self._get_synced_length(chunk_step)
                
                # update the crop length of the underlying dataset
                # cheaper than destroying and rebuilding the object every time
                if hasattr(self.dataset, 'cropped_len'):
                    self.dataset.cropped_len = current_length
                
                # pull from the underlying iterator
                for _ in range(self.samples_per_length):
                    try:
                        sample = next(base_iterator)
                        # the current length can be put into the sample, which makes collate_fn safe
                        yield sample
                    except StopIteration:
                        # the underlying dataset is exhausted, so the epoch ends here
                        return 
                
                chunk_step += 1
                
        except Exception as e:
            # log and exit
            print(f"Worker {worker_id} error: {e}")
            return

def create_length_bucketed_collate_fn(token_size: int = 128, min_batch_size: int = 1):
    def collate_fn(batch):
        # drop empty batches
        batch = [s for s in batch if s is not None]
        if len(batch) < min_batch_size:
            return {}

        # length distribution
        length_groups = {}
        for sample in batch:
            l = sample['seq'].shape[0]
            if l not in length_groups: length_groups[l] = []
            length_groups[l].append(sample)

        # the largest length group; the synchronised sampling usually leaves only one
        main_len = max(length_groups, key=lambda k: len(length_groups[k]))
        val_samples = length_groups[main_len]

        if len(val_samples) < min_batch_size:
            return {}

        # plain stacking
        seqs = torch.stack([torch.as_tensor(s['seq']) for s in val_samples])
        targets = torch.stack([torch.as_tensor(s['target']) for s in val_samples])

        return {
            'seq': seqs,
            'target': targets,
            # 'len': main_len
        }
    return collate_fn