from time import time

import torch.nn
from torch.optim import AdamW
from torchsummary import summary
from transformers import LlamaTokenizerFast

from lex.config import get_device
from lex.scratch.dataloaders.d3_simple_dataloader import SimpleDataLoader
from lex.scratch.models.vanilla_lm import AutoRegressiveLM


# ============ Config =============
batch_size = 128
max_seq_length = 256
data_root = "/mnt/ssd/data/fineweb-edu-10BT/llama-tokenizer"

vocab_size = 32768  # 32000 to 32768 (2**15) makes it 2ms faster
embedding_size = 256
n_layers = 10
n_heads = 8
lr = 1e-4
max_training_steps = 10000
generate_every = 500
evaluate_every = 100
eval_steps = 10

# =========== Data =============
enc = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
assert len(enc) <= vocab_size

dataloader_train = SimpleDataLoader(
    data_root="/mnt/ssd/data/fineweb-edu-10BT/llama-tokenizer",
    mode="train",  # train|validation|test
    batch_size=batch_size,
    sequence_length=max_seq_length
)
dataloader_val = SimpleDataLoader(
    data_root="/mnt/ssd/data/fineweb-edu-10BT/llama-tokenizer",
    mode="validation",  # train|validation|test
    batch_size=batch_size,
    sequence_length=max_seq_length
)

# ============= Model ==============
torch.set_float32_matmul_precision('medium')  # 97ms to 66ms
device = get_device(rank=2)
model = AutoRegressiveLM(vocab_size, embedding_size, max_seq_length, n_layers, n_heads)
model = torch.compile(model)  # 43ms to 29ms
model.to(device)
optimizer = AdamW(model.parameters(), lr=lr, fused=True)  # fused=True makes it 2ms faster
criterion = torch.nn.CrossEntropyLoss()

summary(
    model,
    torch.randint(0, vocab_size, (batch_size, max_seq_length), dtype=torch.long, device=device),
    device=torch.device(device),
    depth=10,
)

# ============= Training ==============
loss_agg = 0
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
        eval_loss_agg = 0
        with torch.inference_mode():
            for eval_step in range(eval_steps):
                x, y = dataloader_val.next_batch()
                x, y = x.to(device), y.to(device)
                yh = model(x)
                loss = criterion(yh.view(-1, vocab_size), y.view(-1))
                eval_loss_agg += loss.item()
        model.train()
        print(f"step: {step:03d} train loss: {loss_agg / evaluate_every:.6f} val loss: {eval_loss_agg / eval_steps:.6f}"
              f" step time: {step_time_agg / evaluate_every:.2f}ms, data load time: {data_load_time_agg / evaluate_every:.2f}ms")
        loss_agg = 0
        step_time_agg = 0
        data_load_time_agg = 0

    # ----------- Generate -----------
    if step % generate_every == 0 or step == max_training_steps - 1:
        prefix = "Hi, I'm Indian, my name is"
        tokens = [enc.encode(prefix)] * 5  # (5, s)
        tokens = torch.tensor(tokens, dtype=torch.long)
        model.eval()
        generated = model.generate(tokens.to(device), length=20, top_k=50, temperature=1.0)
        generated = generated.tolist()
        for tokens in generated:
            print(f">>> {enc.decode(tokens)}")
        model.train()


"""
step: 100 train loss: 38.988750 val loss: 29.887267 step time: 162.50ms, data load time: 0.07ms
step: 200 train loss: 27.733750 val loss: 25.507024 step time: 31.63ms, data load time: 0.06ms
step: 300 train loss: 24.002500 val loss: 22.240289 step time: 31.99ms, data load time: 0.06ms
step: 400 train loss: 20.923750 val loss: 19.414048 step time: 32.20ms
step: 500 train loss: 18.261250 val loss: 17.143362 step time: 31.42ms

step: 1000 train loss: 11.443125 val loss: 11.035467 step time: 31.58ms
step: 1500 train loss: 8.935625 val loss: 8.809173 step time: 31.43ms
step: 2000 train loss: 7.948437 val loss: 7.944898 step time: 31.96ms
step: 2500 train loss: 7.489375 val loss: 7.511491 step time: 30.68ms

---
With weight initialization:
step: 100 train loss: 8.945000 val loss: 7.858323 step time: 186.70ms, data load time: 0.05ms
step: 200 train loss: 7.545000 val loss: 7.430056 step time: 28.03ms, data load time: 0.04ms
step: 300 train loss: 7.265625 val loss: 7.123016 step time: 27.61ms, data load time: 0.05ms
step: 400 train loss: 6.964688 val loss: 6.814813 step time: 28.15ms, data load time: 0.04ms
step: 500 train loss: 6.701875 val loss: 6.610016 step time: 28.15ms, data load time: 0.04ms

step: 1000 train loss: 6.141250 val loss: 6.103240 step time: 28.14ms, data load time: 0.04ms
step: 1500 train loss: 5.850313 val loss: 5.815863 step time: 29.65ms, data load time: 0.04ms
step: 2000 train loss: 5.589688 val loss: 5.563005 step time: 28.33ms, data load time: 0.04ms

---
Ignore INIT_LARGE_STD
step: 100 train loss: 8.945312 val loss: 7.848524 step time: 183.52ms, data load time: 0.05ms
step: 200 train loss: 7.485625 val loss: 7.284136 step time: 28.67ms, data load time: 0.05ms
step: 300 train loss: 7.134375 val loss: 6.978426 step time: 27.87ms, data load time: 0.05ms
step: 400 train loss: 6.820625 val loss: 6.722461 step time: 29.13ms, data load time: 0.05ms
step: 500 train loss: 6.609063 val loss: 6.537150 step time: 28.36ms, data load time: 0.05ms

step: 1000 train loss: 6.074063 val loss: 6.050111 step time: 27.90ms, data load time: 0.04ms
step: 1500 train loss: 5.797813 val loss: 5.746186 step time: 28.56ms, data load time: 0.05ms
step: 2000 train loss: 5.555937 val loss: 5.503580 step time: 27.96ms, data load time: 0.05ms
"""