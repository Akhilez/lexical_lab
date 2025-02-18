import fcntl
import json
from os import fsync, getpid
from os.path import join

import numpy as np
import torch

from lex.config import ROOT


class LockedFile:
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.file = None

    def __enter__(self):
        # Open the file in 'a+' mode (create if doesn't exist, read, and append)
        self.file = open(self.path, self.mode)

        # Lock the file
        fcntl.flock(self.file, fcntl.LOCK_EX)  # Exclusive lock

        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush the file to ensure all data is written
        self.file.flush()
        fsync(self.file.fileno())

        # Unlock the file
        fcntl.flock(self.file, fcntl.LOCK_UN)

        # Close the file
        self.file.close()


class SharedStateManager:
    def __init__(self, all_shard_indices: torch.Tensor, mode: str, rank: int, world_size: int):
        self.all_shard_indices = all_shard_indices
        self.shared_state_path = join(ROOT, f"shared_state_{mode}.json")
        self.world_size = world_size
        self.rank = str(rank)
        self.mode = mode

    def reset(self) -> int:
        """
        If current shard is an integer, then reset the shared state.
        """
        with LockedFile(self.shared_state_path, 'w+') as f:
            f.seek(0)
            try:
                state = json.load(f)
            except json.JSONDecodeError:
                state = {}

            if (
                not state  # File doesn't exist or corrupt
                or state.get("pids", {}).get(self.rank) != getpid()  # If reusing previous run's file.
                or type(state.get("current_shard_index", {}).get(self.rank)) is int  # reset is manually called later.
            ):
                # Reset the state
                state["num_epochs_so_far"] = 0
                state["num_tokens_so_far"] = 0
                state["current_shard_index"] = {str(rank): None for rank in range(self.world_size)}
                state["pids"] = {str(rank): None for rank in range(self.world_size)}  # TODO: Empty PIDs only if they don't match, right?
                state["shards_read"] = {str(rank): [] for rank in range(self.world_size)}
                state["available_shard_indices"] = self.get_all_shard_indices()

            # If len(shards) < world_size, available_shard_indices might be empty.
            if len(state["available_shard_indices"]) == 0:
                state["num_epochs_so_far"] += 1
                state["available_shard_indices"] = self.get_all_shard_indices()
                state["shards_read"] = {str(rank): [] for rank in range(self.world_size)}

            # Now initialize the state for this process
            state["pids"][self.rank] = getpid()
            idx = state["available_shard_indices"].pop(0)
            state["current_shard_index"][self.rank] = idx
            state["shards_read"][self.rank].append(idx)

            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

            # print(f"Reset on {self.rank}: {json.dumps(state, indent=2)}")
        return idx

    def get_next_shard_index(self, num_tokens_read: int) -> int:
        """
        - If no available shard indices:
            - Increment epoch
            - Set available shard indices to shuffled shard indices
            - Reset shards_read
        - Pick the first shard index
        - Set the current shard index to this shard index
        - Add shard index to shards_read
        - Dump the shared state
        """
        with LockedFile(self.shared_state_path, 'r+') as f:
            f.seek(0)
            state = json.load(f)
            if len(state["available_shard_indices"]) == 0:
                state["num_epochs_so_far"] += 1
                state["available_shard_indices"] = self.get_all_shard_indices()
                state["shards_read"] = {str(rank): [] for rank in range(self.world_size)}
            state["num_tokens_so_far"] += num_tokens_read
            idx = state["available_shard_indices"].pop(0)
            state["current_shard_index"][self.rank] = idx
            state["shards_read"][self.rank].append(idx)
            f.seek(0)
            json.dump(state, f, indent=2)

            # Truncate the rest of the file if the new content is shorter
            f.truncate()
        return idx

    def get_all_shard_indices(self):
        if self.mode == "train":
            return self.all_shard_indices[torch.randperm(len(self.all_shard_indices))].tolist()
        else:
            return self.all_shard_indices.tolist()


class ParallelDataLoader:
    def __init__(self, data_root, mode, batch_size, sequence_length, rank, world_size):
        """
        - shared_state.json:
            - Stores the global state of the data loader across all GPUs
            - Contains the following:
                - indices of the available shards in the epoch.
                - PIDs of all the processes
                - Shard indices that each process has read in the epoch so far.
                - Current shard index that each process is reading.
                - Number of epochs, number of tokens read so far.
        - Picking the shard index:
            - Read the shared state file
            - If available shard indices is empty
                - Reset the shard indices and shuffle them
                - Increment the number of epochs
            - Pick the first shard index from available shards in the epoch
            - Dump
                - Remaining shard indices
                - Number of epochs if it has changed
                - Add the number of tokens read.
                - Current shard index
                - Shard index currently picked
                - Shards read by this process so far in the epoch.
        - Shuffle documents in the current shard
        - Create an (n * s + 1) array
        - Track current token index
        """
        # Static members
        self.data_root = data_root
        self.mode = mode  # train|validation|test
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.rank = rank

        with open(join(data_root, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        shard_indices = metadata[f"{self.mode}_shards"]

        # Local state
        self._max_tokens_this_shard = 0
        self._current_token_index = 0
        self.shard = None
        self.shared_state = SharedStateManager(torch.tensor(shard_indices, dtype=torch.int32), mode, rank, world_size)

        self.reset()

    def next_batch(self):
        end = min(self._current_token_index + self.batch_size * self.sequence_length, self._max_tokens_this_shard)
        tokens = self.shard[self._current_token_index: end + 1]  # (bs * sl + 1,)
        tokens = torch.from_numpy(tokens).to(torch.int64)  # (bs * sl + 1,)
        xs = tokens[:-1].reshape(-1, self.sequence_length)  # (bs, sl)
        ys = tokens[1:].reshape(-1, self.sequence_length)  # (bs, sl)
        self._current_token_index = end
        if self._current_token_index >= self._max_tokens_this_shard:
            self.finished_current_shard()
        return xs, ys

    def reset(self):
        """
        Call this if you want to completely wipe the state of the data loader and start from scratch.
        """
        self._max_tokens_this_shard = 0
        self._current_token_index = 0
        self.shard = None

        shard_index = self.shared_state.reset()
        self.load_shard(shard_index)

    def load_shard(self, shard_index):
        self.shard = np.load(join(self.data_root, f"tokens_shard_{shard_index:03d}.npy"))
        if self.mode == "train":
            self.shard = self._shuffle(self.shard)
        _num_sequences_this_shard = len(self.shard) // self.sequence_length
        self._max_tokens_this_shard = _num_sequences_this_shard * self.sequence_length  # Shouldn't there be +1?
        self._current_token_index = 0

    def finished_current_shard(self):
        """
        Call this when the current shard has been exhausted.
        """
        shard_index = self.shared_state.get_next_shard_index(self._current_token_index)
        self.load_shard(shard_index)

    @staticmethod
    def _shuffle(shard):
        bos_token = 1
        bos_indices = np.where(shard == bos_token)[0]
        shard_splits = np.split(shard, bos_indices[1:])
        np.random.shuffle(shard_splits)
        shard = np.concatenate(shard_splits)  # (n,)
        return shard


if __name__ == '__main__':
    import os
    # Single process:
    # python d4_parallel_dataloader.py
    # os.environ["RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"

    # torchrun --standalone --nproc_per_node=2 d4_parallel_dataloader.py
    dataloader = ParallelDataLoader(
        data_root="/mnt/ssd/data/fineweb-edu-10BT/llama-tokenizer-debug",
        mode="train",  # train|validation|test
        batch_size=4,
        sequence_length=128,
        rank = os.environ["RANK"],
        world_size = os.environ["WORLD_SIZE"],
    )
    for i in range(100):
        xs, ys = dataloader.next_batch()
        print(xs.shape, ys.shape)
