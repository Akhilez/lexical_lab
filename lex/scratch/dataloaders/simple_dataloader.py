import json
from os.path import join

import numpy as np
import torch


class SimpleDataLoader:
    def __init__(self, data_root, mode, batch_size, sequence_length):
        """
        Shuffle the shards every epoch
        This can be shared across GPUs
        Track current shard (maybe based on rank and world_size)
        Shuffle documents in the current shard
        Create an (n * s + 1) array
        Track current token index
        """
        self.data_root = data_root
        self.mode = mode  # train|validation|test
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self._max_tokens_this_shard = 0
        self._current_token_index = 0
        self.shard = None
        self.num_tokens_so_far = 0
        self.num_epochs_so_far = -1

        # Load dataset metadata
        with open(join(data_root, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.shard_indices = torch.tensor(metadata[f"{self.mode}_shards"])
        # self.shard_indices.share_memory_()
        self.current_shard = len(self.shard_indices)
        self.finished_current_shard()

    def next_batch(self):
        end = min(self._current_token_index + self.batch_size * self.sequence_length, self._max_tokens_this_shard)
        self.num_tokens_so_far += end - self._current_token_index
        tokens = self.shard[self._current_token_index: end + 1]  # (bs * sl + 1,)
        tokens = torch.from_numpy(tokens).to(torch.int64)  # (bs * sl + 1,)
        xs = tokens[:-1].reshape(-1, self.sequence_length)  # (bs, sl)
        ys = tokens[1:].reshape(-1, self.sequence_length)  # (bs, sl)
        self._current_token_index = end
        if self._current_token_index >= self._max_tokens_this_shard:
            self.finished_current_shard()
        return xs, ys

    def finished_current_shard(self):
        self.current_shard += 1
        if self.current_shard >= len(self.shard_indices):
            self.num_epochs_so_far += 1
            if self.mode == "train":
                self.shard_indices[:] = self.shard_indices[torch.randperm(len(self.shard_indices))]
            self.current_shard = 0
        self.shard = np.load(join(self.data_root, f"tokens_shard_{self.shard_indices[self.current_shard]:03d}.npy"))
        if self.mode == "train":
            self.shard = self._shuffle(self.shard)
        _num_sequences_this_shard = len(self.shard) // self.sequence_length
        self._max_tokens_this_shard = _num_sequences_this_shard * self.sequence_length
        self._current_token_index = 0

    @staticmethod
    def _shuffle(shard):
        bos_token = 1
        bos_indices = np.where(shard == bos_token)[0]
        shard_splits = np.split(shard, bos_indices[1:])
        np.random.shuffle(shard_splits)
        shard = np.concatenate(shard_splits)  # (n,)
        return shard


if __name__ == '__main__':
    dataloader = SimpleDataLoader(
        data_root="/mnt/ssd/data/fineweb-edu-10BT/llama-tokenizer-debug",
        mode="train",  # train|val|test
        batch_size=4,
        sequence_length=128
    )
    x, y = dataloader.next_batch()
    print(x.shape, y.shape)
    print(x[0][-5:])
    print(y[0][-5:])
