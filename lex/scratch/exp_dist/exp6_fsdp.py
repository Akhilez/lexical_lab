"""

# How FSDP works


# How to implement it (changes from vanilla)


# What to expect
- 20-25% slow down of training time. But due to larger batch sizes, total training time might be lower.
- 33-38% GPU memory freed. (wait, that's it? test it)




"""
# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Sequential
from torchvision import datasets, transforms


from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


model = Sequential(
    nn.Linear(28*28, 128),
    nn.GELU(),
    nn.Linear(128, 256),
    nn.GELU(),
    nn.Linear(256, 256),
    nn.GELU(),
    nn.Linear(256, 256),
    nn.GELU(),
    nn.Linear(256, 10),
)

my_auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=100
)
torch.cuda.set_device(rank)


init_start_event = torch.cuda.Event(enable_timing=True)
init_end_event = torch.cuda.Event(enable_timing=True)

model = Net().to(rank)

model = FSDP(model)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
init_start_event.record()
for epoch in range(1, args.epochs + 1):
    train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
    test(model, rank, world_size, test_loader)
    scheduler.step()

init_end_event.record()

if rank == 0:
    print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
    print(f"{model}")
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
init_start_event.record()
for epoch in range(1, args.epochs + 1):
    train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
    test(model, rank, world_size, test_loader)
    scheduler.step()

init_end_event.record()

if rank == 0:
    print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
    print(f"{model}")
if args.save_model:
    # use a barrier to make sure training is done on all ranks
    dist.barrier()
    states = model.state_dict()
    if rank == 0:
        torch.save(states, "mnist_cnn.pt")

cleanup()



