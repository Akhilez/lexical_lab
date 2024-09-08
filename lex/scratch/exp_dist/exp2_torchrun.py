"""
Experiments:
- How to only use specific GPUs like 1, 2, 5 only
    - If you set CUDA_VISIBLE_DEVICES=1,2,5 then the GPUs will behave like 0, 1, 2
- Running with torchrun on a single node
    - Worker RANK, LOCAL_RANK and WORLD_SIZE are assigned automatically.
    - Use --standalone for single node training.
    - A Node runs LOCAL_WORLD_SIZE workers which comprise a LocalWorkerGroup. The union of all LocalWorkerGroups in the nodes in the job comprise the WorkerGroup.
    - More info at https://pytorch.org/docs/stable/elastic/run.html
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def manual_init(rank, size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=size)
    torch.cuda.set_device(rank)


def run_without_torchrun(rank, size):
    manual_init(rank, size)

    print(f"{dist.get_rank()=}, {dist.get_world_size()=}")
    x = torch.rand((1000, 1000, 1000), device=f"cuda:{rank}")

    # Sleep to see the GPU utilization
    import time
    time.sleep(10)

    dist.destroy_process_group()


def main_without_torchrun():
    gpus = [0, 4, 5]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    size = len(gpus)

    # size = torch.cuda.device_count()

    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=run_without_torchrun, args=(rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def main_with_torchrun():
    # Test env variables
    print(f"{os.environ['RANK']=}, {os.environ['LOCAL_RANK']=}, {os.environ['WORLD_SIZE']=}")
    print(f"{os.environ['MASTER_ADDR']=}, {os.environ['MASTER_PORT']=}, {os.environ['CUDA_VISIBLE_DEVICES']=}")

    dist.init_process_group("nccl")
    print(f"{dist.get_rank()=}, {dist.get_world_size()=}")

    x = torch.rand((1000, 1000, 1000), device=f"cuda:{dist.get_rank()}")

    # Sleep to see the GPU utilization
    import time
    time.sleep(10)

    dist.destroy_process_group()


if __name__ == '__main__':
    # main_without_torchrun()

    """
    Run with following command:
    CUDA_VISIBLE_DEVICES=0,4,5 torchrun --standalone --nproc_per_node=3 exp2_torchrun.py
    """
    main_with_torchrun()
