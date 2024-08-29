"""
Experiments:
- Test that the model wrapped in DDP is identical on all workers.
- Test that the gradients are synchronized across all workers.
- Test that the model parameters are the same after the optimizer step
- Test the time taken by loss.backward()


- DDP workflow:
    - Initialization:
        Each process loads a model.
        DDP wraps the model.
        The model parameters are broadcasted to ensure all processes start with the same weights.
        backward hooks are registered for all_reduce.
    - Forward Pass:
        Each process computes the forward pass independently.
    - Backward Pass:
        Each process computes gradients locally.
        DDP automatically launches all_reduce on gradients.
        Gradients are synchronized and averaged across all processes.
        The backward and all_reduce operations are overlapped for efficiency.
        Bucketing of gradients is used to reduce the number of all_reduce operations.
    - Parameter Update:
        Each process updates its model parameters using the synchronized gradients.
    - Identical Models:
        All processes now have identical model parameters and proceed to the next iteration.
- ZeRO optimizer

"""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from linetimer import CodeTimer


def test_without_ddp():
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    model = nn.Linear(100, 10, device=rank)
    print(f"Device: {model.weight.device}, Weights: {model.weight.flatten()[:3].tolist()}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    x = torch.rand((100,), device=rank)
    y = model(x)
    loss = y.sum()
    print(f"Device: {loss.device}, Loss: {loss.item()}")

    with CodeTimer("Backward without ddp"):
        loss.backward()
    print(f"Device: {loss.device}, Gradient: {model.weight.grad.flatten()[:3].tolist()}")

    optimizer.step()
    print(f"Device: {model.weight.device}, Weights: {model.weight.flatten()[:3].tolist()}")

    dist.destroy_process_group()


def test_with_ddp():
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    model = nn.Linear(100, 10, device=rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    print(f"Device: {model.module.weight.device}, Weights: {model.module.weight.flatten()[:3].tolist()}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    x = torch.rand((100,), device=rank)
    y = model(x)
    loss = y.sum()
    print(f"Device: {loss.device}, Loss: {loss.item()}")

    with CodeTimer(f"Backward with ddp on rank {rank}"):
        loss.backward()
    print(f"Device: {loss.device}, Gradient: {model.module.weight.grad.flatten()[:3].tolist()}")

    optimizer.step()
    print(f"Device: {model.module.weight.device}, Weights: {model.module.weight.flatten()[:3].tolist()}")

    dist.destroy_process_group()


if __name__ == '__main__':
    # Use torchrun to launch
    # CUDA_VISIBLE_DEVICES=0,5,6 torchrun --standalone --nproc_per_node=3 ddp_trainer.py
    test_without_ddp()
    test_with_ddp()
