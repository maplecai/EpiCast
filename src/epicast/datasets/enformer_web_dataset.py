import webdataset as wds
import numpy as np
import torch
from torch.utils.data import IterableDataset


class EnformerWebDataset(IterableDataset):
    def __init__(
        self,
        shard_pattern,
        shardshuffle=False,
        decode=True,
        seq_key="seq.npy",
        tgt_key="tgt.npy",
        crop=False,
        crop_mode="random",
        pad=False,
        pad_mode="N",
        random_crop=True,
        center_crop=False,
        cropped_len=1024,
        token_size=128,
        batch_size=None,       # batch inside the dataset; None yields single samples
        partial_batches=True,
    ):
        self.shard_pattern = shard_pattern
        self.shardshuffle = shardshuffle
        self.decode_flag = decode
        self.seq_key = seq_key
        self.tgt_key = tgt_key

        self.crop = crop
        self.crop_mode = crop_mode

        if random_crop:  # kept for backwards compatibility, no longer used
            self.crop = True
            self.crop_mode = "random"
        elif center_crop:
            self.crop = True
            self.crop_mode = "center"

        self.cropped_len = cropped_len
        self.token_size = token_size

        self.batch_size = batch_size
        self.partial_batches = partial_batches

        assert not (random_crop and center_crop), \
            "random_crop and center_crop cannot both be True"

    def _crop_seq_and_target(self, seq, tgt):
        # seq.shape = (131072, 4) tgt.shape = (896, 5313)
        # from the original 131072, crop both ends down to 114688, i.e. 896 tokens
        seq = seq[8192:-8192]

        seq_len = seq.shape[0]
        tgt_len = tgt.shape[0]
        seq_token_len = seq_len // self.token_size
        assert tgt_len == seq_token_len
        cropped_token_len = self.cropped_len // self.token_size

        if self.crop_mode == "random":
            start_token = np.random.randint(0, seq_token_len - cropped_token_len + 1)
        elif self.crop_mode == "center":
            start_token = (seq_token_len - cropped_token_len) // 2
        else:
            raise ValueError("crop_mode must be 'random' or 'center'")
        end_token = start_token + cropped_token_len

        start_pos = start_token * self.token_size
        end_pos = end_token * self.token_size

        seq = seq[start_pos:end_pos]
        tgt = tgt[start_token:end_token]

        return seq, tgt

    @staticmethod
    def _batch_to_tensor(x):
        return {
            "seq": torch.from_numpy(x[0]).float(),
            "target": torch.from_numpy(x[1]).float(),
        }

    def _build_wds(self):
        dataset = wds.WebDataset(
            self.shard_pattern,
            shardshuffle=self.shardshuffle,
            handler=wds.warn_and_continue,
            nodesplitter=wds.split_by_node,
        )

        if self.decode_flag:
            dataset = dataset.decode()

        dataset = dataset.to_tuple(self.seq_key, self.tgt_key)

        
        if self.crop:
            dataset = dataset.map(lambda sample: self._crop_seq_and_target(*sample))
        
        if self.batch_size is not None:
            dataset = dataset.batched(self.batch_size, partial=self.partial_batches)

        dataset = dataset.map(self._batch_to_tensor)
        return dataset

    def __iter__(self):
        return iter(self._build_wds())
