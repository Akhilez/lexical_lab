"""
This code trains a model using Distributed Data Parallel.

Use with torchrun:
CUDA_VISIBLE_DEVICES=3,4,5 torchrun --standalone --nproc_per_node=2 t1_ddp_trainer.py

Run in background:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 nohup torchrun --standalone --nproc_per_node=7 t1_ddp_trainer.py > logs/output.log &
"""

import os
from os.path import join
from time import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchsummary import summary
from transformers import LlamaTokenizerFast

from lex.config import get_device
from lex.scratch.dataloaders.d4_parallel_dataloader import ParallelDataLoader
from lex.scratch.models.vanilla_lm import AutoRegressiveLM


# ============ Config =============
batch_size = 256
max_seq_length = 512
data_root = "/mnt/ssd/data/fineweb-edu-10BT/llama-tokenizer"
output_dir = "/home/akhil/code/lexical_lab/lex/scratch/trainer/logs"

vocab_size = 32768  # 32000 to 32768 (2**15) makes it 2ms faster
embedding_size = 256
n_layers = 10
n_heads = 8
lr = 1e-4
max_training_steps = 50000
generate_every = 500
evaluate_every = 500
eval_steps = 10
checkpoint_every = 1000

# DDP
use_ddp = True
if use_ddp:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"{rank=}, {world_size=}")
    print(f"{os.environ.get('CUDA_VISIBLE_DEVICES')=}")
else:
    rank = 0
    world_size=1

# =========== Data =============
enc = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
assert len(enc) <= vocab_size

dataloader_train = ParallelDataLoader(
    data_root=data_root,
    mode="train",  # train|validation|test
    batch_size=batch_size,
    sequence_length=max_seq_length,
    rank=rank,
    world_size=world_size,
)
dataloader_val = ParallelDataLoader(
    data_root=data_root,
    mode="validation",
    batch_size=batch_size,
    sequence_length=max_seq_length,
    rank=rank,
    world_size=world_size,
)

# ============= Model ==============
torch.set_float32_matmul_precision('medium')  # 97ms to 66ms
device = get_device(rank=rank)
raw_model = AutoRegressiveLM(vocab_size, embedding_size, max_seq_length, n_layers, n_heads)
raw_model = torch.compile(raw_model)
raw_model.to(device)
model = raw_model
if use_ddp:
    model = DDP(raw_model, device_ids=[rank], output_device=rank)
optimizer = AdamW(model.parameters(), lr=lr, fused=True)  # fused=True makes it 2ms faster
criterion = torch.nn.CrossEntropyLoss()

# if rank == 0:
#     summary(
#         model,
#         torch.randint(0, vocab_size, (batch_size, max_seq_length), dtype=torch.long, device=device),
#         device=torch.device(device),
#         depth=10,
#     )

# ============= Training ==============
loss_agg = torch.tensor(0, device=device, dtype=torch.bfloat16)
step_time_agg = 0
data_load_time_agg = 0
for step in range(max_training_steps):
    # --------- Train ----------
    start_time = time()
    x, y = dataloader_train.next_batch()  # (b, s)
    data_load_time_agg += (time() - start_time) * 1000  # ms
    start_time = time()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):  # 66ms to 43ms
        yh = model(x.to(device))  # (b, s, v)
    loss = criterion(yh.view(-1, vocab_size), y.to(device).view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step_time_agg += (time() - start_time) * 1000  # ms
    loss_agg += loss.item()

    # ---------- Eval ----------
    if step > 0 and step % evaluate_every == 0 or step == max_training_steps - 1:
        dataloader_val.reset()
        model.eval()
        eval_loss_agg = torch.tensor(0, device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            for eval_step in range(eval_steps):
                x, y = dataloader_val.next_batch()
                yh = model(x.to(device)).to("cpu")
                loss = criterion(yh.view(-1, vocab_size), y.view(-1))
                eval_loss_agg += loss.item()
        model.train()
        if use_ddp:
            dist.reduce(eval_loss_agg, dst=0, op=dist.ReduceOp.AVG)
            dist.reduce(loss_agg, dst=0, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(f"step: {step:03d}, rank: {rank}, train loss: {loss_agg / evaluate_every:.6f} "
                  f"val loss: {eval_loss_agg / eval_steps:.6f} step time: {step_time_agg / evaluate_every:.2f}ms, "
                  f"data load time: {data_load_time_agg / evaluate_every:.2f}ms")
        loss_agg = torch.tensor(0, device=device, dtype=torch.bfloat16)
        step_time_agg = 0
        data_load_time_agg = 0

    # ----------- Generate -----------
    if rank == 0 and step % generate_every == 0 or step == max_training_steps - 1:
        prefix = "Hi, I'm Indian, my name is"
        tokens = [enc.encode(prefix)] * 5  # (5, s)
        tokens = torch.tensor(tokens, dtype=torch.long)
        model.eval()
        generated = raw_model.generate(tokens.to(device), length=20, top_k=50, temperature=1.0)
        generated = generated.tolist()
        for tokens in generated:
            print(f">>> {enc.decode(tokens)}")
        model.train()

    # ------------ Checkpoint -------------
    if rank == 0 and step % checkpoint_every == 0 or step == max_training_steps - 1:
        torch.save(raw_model.state_dict(), join(output_dir, f"checkpoint_{step}.ckpt"))

if use_ddp:
    dist.destroy_process_group()


"""
# 4 GPUs
step: 100, rank: 0, train loss: 39.000000 val loss: 29.625000 step time: 169.98ms, data load time: 0.06ms
step: 200, rank: 0, train loss: 27.375000 val loss: 25.125000 step time: 35.48ms, data load time: 0.05ms
step: 300, rank: 0, train loss: 22.875000 val loss: 21.750000 step time: 33.09ms, data load time: 0.05ms
step: 400, rank: 0, train loss: 19.625000 val loss: 19.000000 step time: 34.29ms, data load time: 0.05ms
step: 500, rank: 0, train loss: 17.250000 val loss: 16.625000 step time: 33.71ms, data load time: 0.07ms
step: 600, rank: 0, train loss: 16.000000 val loss: 14.500000 step time: 34.67ms, data load time: 0.05ms
step: 700, rank: 0, train loss: 13.750000 val loss: 13.000000 step time: 34.34ms, data load time: 0.06ms
step: 800, rank: 0, train loss: 11.500000 val loss: 11.625000 step time: 34.14ms, data load time: 0.06ms

---
Batch size 128->256, seq 256->320 7 GPUs
step: 1000, rank: 0, train loss: 8.125000 val loss: 9.250000 step time: 65.82ms, data load time: 0.06ms


"""