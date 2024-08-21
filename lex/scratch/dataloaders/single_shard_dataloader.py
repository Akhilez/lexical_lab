import json
from os.path import join

import numpy as np
import torch


class SingleShardDataLoader:
    """Use this only for debugging"""

    def __init__(self, data_root, mode, batch_size, sequence_length):
        self.mode = mode  # train|validation|test
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # Load dataset metadata
        with open(join(data_root, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        self.shard = np.load(join(data_root, f"tokens_shard_{metadata[f'{mode}_shards'][0]:03d}.npy"))  # (n,)

        if mode == "train":
            self.shard = self._shuffle(self.shard)

        self._num_sequences = len(self.shard) // sequence_length
        if len(self.shard) % sequence_length == 0:
            self._num_sequences -= 1
        self._num_batches = np.ceil(self._num_sequences / batch_size).astype(int)
        self._max_tokens = self._num_sequences * sequence_length + 1

        self._current_batch = 0
        self._current_epoch = 0

    def next_batch(self):
        start_index = self._current_batch * self.batch_size * self.sequence_length
        end_index = min(start_index + self.batch_size * self.sequence_length + 1, self._max_tokens)
        batch = self.shard[start_index: end_index]  # (bs * sl + 1,)
        batch = torch.from_numpy(batch.astype(np.int64))
        xs = batch[:-1].reshape(-1, self.sequence_length)  # (bs, sl)
        ys = batch[1:].reshape(-1, self.sequence_length)  # (bs, sl)

        self._current_batch += 1
        if self._current_batch >= self._num_batches and self.mode == "train":
            self._current_batch = 0
            self._current_epoch += 1
            self.shard = self._shuffle(self.shard)

        return xs, ys

    def __len__(self):
        return self._num_batches

    def reset(self):
        self._current_batch = 0
        self._current_epoch = 0
        if self.mode == "train":
            self.shard = self._shuffle(self.shard)

    @staticmethod
    def _shuffle(shard):
        bos_token = 1
        bos_indices = np.where(shard == bos_token)[0]
        shard_splits = np.split(shard, bos_indices[1:])
        np.random.shuffle(shard_splits)
        shard = np.concatenate(shard_splits)  # (n,)
        return shard


if __name__ == '__main__':
    dataloader = SingleShardDataLoader(
        data_root="/mnt/ssd/data/fineweb-edu-10BT/llama-tokenizer-debug",
        mode="train",  # train|val|test
        batch_size=4,
        sequence_length=128
    )
    print(len(dataloader))
    x, y = dataloader.next_batch()
    print(x.shape, y.shape)
