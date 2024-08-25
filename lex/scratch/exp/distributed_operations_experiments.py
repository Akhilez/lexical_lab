"""
We have a few processes in the same group (the default group in this case).
Now, we want the processes to communicate some tensors (not anything else, just tensors) between each other.
Ignore the GPUs. To test distributed operations, we don't need GPUs.
Also ignore torchrun.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()


def test_simple_point_to_point(rank, size):
    print(f"{rank=}, {size=}")
    tensor = torch.zeros(1)
    if rank == 0:
        # Send the tensor to process 1
        # dist.send(tensor=tensor + 1, dst=1)  # Blocking
        # Immediates are non-blocking. Do not modify the tensor before req.wait()
        req = dist.isend(tensor=tensor+1, dst=1)
    else:
        # Receive tensor from process 0
        # dist.recv(tensor=tensor, src=0)  # Blocking
        req = dist.irecv(tensor=tensor, src=0)  # Immediates are non-Blocking
    req.wait()  # Wait for the non-blocking operation to finish
    print('Rank ', rank, ' has data ', tensor[0])


def run_all_reduce(rank, size):
    """ Simple collective communication. """
    group = None  # dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
    assert tensor == torch.tensor(size)


def run_all_gather(rank, size):
    tensor = torch.ones(1) * (rank + 1)
    tensor_list = [torch.zeros(1) for _ in range(size)]
    dist.all_gather(tensor_list, tensor)
    print(f"{rank=}, {tensor_list=}, {tensor=}")
    assert tensor_list == [torch.ones(1) * (i + 1) for i in range(size)]


def run_broadcast(rank, size):
    tensor = torch.ones(1)
    if rank == 0:
        tensor += 1
    dist.broadcast(tensor, src=0)
    print(f"{rank=}, {tensor=}")
    assert tensor == torch.tensor(2)


def run_scatter(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor_list = [torch.ones(1) for _ in range(size)]
        for i in range(size):
            tensor_list[i] += i
        dist.scatter(tensor, tensor_list, src=0)
    else:
        dist.scatter(tensor, src=0)
    print(f"{rank=}, {tensor=}")
    assert tensor == torch.ones(1) * (rank + 1)


def run_gather(rank, size):
    tensor = torch.ones(1) * (rank + 1)
    if rank == 0:
        tensor_list = [torch.zeros(1) for _ in range(size)]
        dist.gather(tensor, tensor_list, dst=0)
        print(f"{rank=}, {tensor=}, {tensor_list=}")
        assert tensor_list == [torch.ones(1) * (i + 1) for i in range(size)]
    else:
        dist.gather(tensor, dst=0)
        print(f"{rank=}, {tensor=}")
        assert tensor == torch.ones(1) * (rank + 1)


def run_reduce(rank, size):
    tensor = torch.tensor(2)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print(f"{rank=}, {tensor=}")

    if rank == 0:
        assert tensor == size * 2
    else:
        # Since the reduce operation is in-place, the value will change for all processes but one random process
        assert tensor != size * 2


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_all_reduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
