"""
Experiments:
- How to check GPU memory in code?
- How much GPU memory for
    - Diferent kind of model - feed forward, CNNs,
    - model without gradients
    - model with gradients
    - Weights, inputs, intermediate results.
    - with fp32 and bf16
    - different optimizers SGD, Adam, AdamW

Observations:

Memory allocation:
- Pytorch reserves more memory, but allocates only what's needed.
    This is done so that when more memory is needed,
    it can be allocated quickly instead of costly reservation operations.
- Datatypes:
    - Each float32 tensor requires 4 bytes of memory.
    - bfloat16 --> 2 bytes
    - int32 --> 4 bytes
    - int64 --> 8 bytes
    - uint8 --> 1 byte
    - int8 --> 1 byte
    - uint16 --> 2 bytes
- Memory is allocated in chunks of 512 bytes.
    When a tensor is created, it is allocated in the next available chunk.
    For a float32 tensor of shape (800,), instead of 800 * 4 = 3200 bytes, 3584 (512 * 7) bytes are allocated.
- When the tensor is deleted, or when the variable goes out of scope,
    the memory is immediately deallocated, but still reserved for future use.

More here: https://medium.com/@akhilez/simple-gpu-memory-allocation-experiments-every-ml-engineer-should-do-ad5f3e132c5e
"""
import subprocess

import torch
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt


# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"


def get_gpu_memory_use(gpu_id):
    """
    Returns the amount of memory used on the specified GPU in bytes.

    Parameters:
    gpu_id (int): The ID of the GPU (e.g., 0 for "cuda:0", 1 for "cuda:1").

    Returns:
    int: The amount of memory used on the GPU in bytes.
    """
    try:
        # Run the nvidia-smi command to get memory usage
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader", f"--id={gpu_id}"],
            stdout=subprocess.PIPE,
            text=True
        )

        # Get the used memory in MiB from the result
        used_memory_mib = int(result.stdout.strip())

        # Convert MiB to bytes (1 MiB = 1024 * 1024 bytes)
        used_memory_bytes = used_memory_mib * 1024 * 1024

        return used_memory_bytes

    except Exception as e:
        print(f"Error occurred: {e}")
        return None


device_id = 2
device = f"cuda:{device_id}"


def test_reservation_vs_allocation():
    print(f"Base memory reserved: {torch.cuda.memory_reserved(device_id)}")
    print(f"Base memory allocated: {torch.cuda.memory_allocated(device_id)}")

    # Allocate some memory
    x = torch.randn((1024,), dtype=torch.float32, device=device)
    print(f"Memory after allocation (reserved): {torch.cuda.memory_reserved(device_id)}")
    print(f"Memory after allocation (allocated): {torch.cuda.memory_allocated(device_id)}")

    # Cleanup
    del x
    print(f"Memory after cleanup (reserved): {torch.cuda.memory_reserved(device_id)}")
    print(f"Memory after cleanup (allocated): {torch.cuda.memory_allocated(device_id)}")

    torch.cuda.empty_cache()
    print(f"Memory after empty_cache (reserved): {torch.cuda.memory_reserved(device_id)}")
    print(f"Memory after empty_cache (allocated): {torch.cuda.memory_allocated(device_id)}")


def test_dtype_memory_allocation():
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.uint8, torch.int8,
              torch.uint16]
    memories = []
    for dtype in dtypes:
        base_memory = torch.cuda.memory_allocated(device_id)
        x = torch.ones((1024,), dtype=dtype, device=device)
        memory_after_allocation = torch.cuda.memory_allocated(device_id)
        memories.append((memory_after_allocation - base_memory) // 1024)
        del x
        torch.cuda.empty_cache()
    fig = plt.figure(figsize=(7, 4))
    fig.set_tight_layout(True)
    plt.bar([str(d) for d in dtypes], memories)
    plt.xlabel("Data type")
    plt.ylabel("Bytes per element")
    plt.title("Memory allocation for different data types")
    plt.xticks(rotation=45)
    plt.show()


def test_gpu_allocation_and_cleanup():
    base_memory = torch.cuda.memory_allocated(device_id)
    print(f"Base memory: {base_memory}")
    # Allocate some memory
    x = torch.randn((800,), dtype=torch.float32, device=device)
    # Integers
    # x = torch.randint(0, 10, (1024,), dtype=torch.int8, device=device)
    memory_after_allocation = torch.cuda.memory_allocated(device_id)
    print(f"Memory after allocation: {memory_after_allocation}")
    # Cleanup
    del x
    torch.cuda.empty_cache()
    memory_after_cleanup = torch.cuda.memory_allocated(device_id)
    print(f"Memory after cleanup: {memory_after_cleanup}")


def test_memory_allocation_relationship():
    """
    For different sizes of tensors, check the memory allocated on GPU.
    """
    memories = []
    sizes = 1050
    for i in tqdm(range(sizes)):
        base_memory = torch.cuda.memory_allocated(device_id)
        x = torch.randn((i,), dtype=torch.float32, device=device)
        memory_after_allocation = torch.cuda.memory_allocated(device_id)
        memories.append(memory_after_allocation - base_memory)
        del x
        torch.cuda.empty_cache()
    plt.plot(memories)
    plt.xlabel("Size of float32 tensor")
    plt.ylabel("Memory allocated (bytes)")
    plt.title("Memory allocation for different tensor sizes")
    plt.show()


def test_single_linear_layer_forward_allocation():
    # Disable cublas
    # import os; os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"

    print(f"Base memory: {torch.cuda.memory_allocated(device_id)}")

    model = nn.Linear(256, 250, device=device, dtype=torch.float32)
    print(f"Memory after model allocation: {torch.cuda.memory_allocated(device_id)}")

    x = torch.randn((1, 256,), dtype=torch.float32, device=device)
    print(f"Memory after input allocation: {torch.cuda.memory_allocated(device_id)}")

    y = model(x)
    final_memory = torch.cuda.memory_allocated(device_id)
    print(f"Memory after forward pass: {final_memory}")

    # Memory calculations
    w_mem = len(model.weight.flatten()) * model.weight.dtype.itemsize
    # Get the higher multiple of 512
    w_mem_as_chunks = (w_mem + 511) // 512 * 512
    print(f"{model.weight.shape=}, {w_mem=}, {w_mem_as_chunks=}")

    b_mem = len(model.bias) * model.bias.dtype.itemsize
    b_mem_as_chunks = (b_mem + 511) // 512 * 512
    print(f"{model.bias.shape=}, {b_mem=}, {b_mem_as_chunks=}")

    x_mem = (len(x.flatten()) * x.dtype.itemsize + 511) // 512 * 512
    y_mem = (len(y.flatten()) * y.dtype.itemsize + 511) // 512 * 512
    print(f"{x_mem=}, {y_mem=}")

    total_memory_expected = w_mem_as_chunks + b_mem_as_chunks + x_mem + y_mem

    cublas_workspace_size = 8519680
    memory_with_cublas = total_memory_expected + cublas_workspace_size
    print(f"{total_memory_expected=}, {memory_with_cublas=}")

    assert final_memory == memory_with_cublas

    del model, x, y
    torch.cuda.empty_cache()
    print(f"Memory after cleanup: {torch.cuda.memory_allocated(device_id)}")

    torch._C._cuda_clearCublasWorkspaces()
    print(f"Memory after clearing cublas workspace: {torch.cuda.memory_allocated(device_id)}")


def test_single_linear_layer_backward_allocation():
    print(f"Base memory: {torch.cuda.memory_allocated(device_id)}")

    model = nn.Linear(256, 250, device=device, dtype=torch.float32)
    x = torch.randn((1, 256,), dtype=torch.float32, device=device)
    y = model(x)

    print(f"Memory after forward pass: {torch.cuda.memory_allocated(device_id)}")
    y.sum().backward()
    final_memory = torch.cuda.memory_allocated(device_id)
    print(f"Memory after backward pass: {final_memory}")

    # Memory calculations
    next_chunk = lambda n: (n + 511) // 512 * 512
    units = model.weight.dtype.itemsize  # 4 bytes for float32
    mem = next_chunk(len(model.weight.flatten()) * units)
    mem += next_chunk(len(model.bias) * units)
    print(f"Excepted model memory: {mem}")

    x_mem = next_chunk(len(x.flatten()) * units)
    y_mem = next_chunk(len(y.flatten()) * units)
    print(f"{x_mem=}, {y_mem=}")
    mem += x_mem + y_mem

    # Gradient memory
    w_grad_mem = next_chunk(len(model.weight.grad.flatten()) * units)
    b_grad_mem = next_chunk(len(model.bias.grad.flatten()) * units)
    print(f"{model.weight.grad.shape=}, {w_grad_mem=}")
    print(f"{model.bias.grad.shape=}, {b_grad_mem=}")
    mem += w_grad_mem + b_grad_mem

    mem += 2 * 8519680  # cublas_size doubled
    print(f"Total memory expected: {mem}")
    assert final_memory == mem

    del model, x, y
    torch.cuda.empty_cache()
    print(f"Memory after cleanup: {torch.cuda.memory_allocated(device_id)}")

    torch._C._cuda_clearCublasWorkspaces()
    print(f"Memory after clearing cublas workspace: {torch.cuda.memory_allocated(device_id)}")


def test_cublas_on_multiple_matmul():
    print(f"Base memory: {torch.cuda.memory_allocated(device_id)}")
    x1 = torch.randn((1, 256), dtype=torch.float32, device=device)
    w1 = torch.randn((256, 250), dtype=torch.float32, device=device)
    y1 = torch.matmul(x1, w1)
    print(f"Memory after first matmul: {torch.cuda.memory_allocated(device_id)}")
    # del x1, w1, y1
    # torch.cuda.empty_cache()
    # print(f"Memory after cleanup: {torch.cuda.memory_allocated(device_id)}")

    x2 = torch.randn((1, 256), dtype=torch.float32, device=device)
    # w2 = torch.randn((256, 250), dtype=torch.float32, device=device)
    y2 = torch.matmul(x2, w1)
    print(f"Memory after second matmul: {torch.cuda.memory_allocated(device_id)}")
    # del x2, w2, y2

    del x1, w1, y1, x2, y2  # w2,
    torch.cuda.empty_cache()
    print(f"Memory after cleanup: {torch.cuda.memory_allocated(device_id)}")


def test_execution_speed_without_cublas():
    from time import time

    def run_forward_pass():
        model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 250),
        ).to(device_id)
        x = torch.randn((1, 256)).to(device_id)
        y = model(x)
        return y

    # With cublas enabled:
    # times = []
    # for i in range(1000):
    #     start = time()
    #     y = run_forward_pass()
    #     times.append(time() - start)
    #     del y
    #     torch.cuda.empty_cache()
    # print(f"Time taken with cublas: {sum(times) / len(times)}")  # 0.005375806093215942
    # print(f"memory: {torch.cuda.memory_allocated(device_id)})")

    # Disable cublas
    import os;
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
    times = []
    for i in range(1000):
        start = time()
        y = run_forward_pass()
        times.append(time() - start)
        del y
        torch.cuda.empty_cache()
    print(f"Time taken without cublas: {sum(times) / len(times)}")  # 0.0061202793121337895
    print(f"memory: {torch.cuda.memory_allocated(device_id)})")


def test_multi_layer_forward():
    print(f"Base memory: {torch.cuda.memory_allocated(device_id)}")

    inference_mode = False
    n_layers = 1
    model = nn.Sequential(*[
        nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),  # No trainable params
            nn.Linear(100, 200),
            nn.Sigmoid(),  # No trainable params
        )
        for _ in range(n_layers)
    ]).to(device_id)
    batch_size = 5
    x = torch.randn((batch_size, 200), device=device_id)
    with torch.inference_mode(inference_mode):
        y = model(x)

    final_memory = torch.cuda.memory_allocated(device_id)
    print(f"Memory after forward pass: {final_memory}")

    # Computed memory
    next_chunk = lambda n: (n + 511) // 512 * 512
    mem = 0
    unit = model[0][0].weight.dtype.itemsize
    for block in model:
        for layer in block:
            if isinstance(layer, nn.Linear):
                mem += next_chunk(len(layer.weight.flatten()) * unit)
                mem += next_chunk(len(layer.bias) * unit)
                if not inference_mode:
                    # Gotta store the input
                    mem += next_chunk(layer.in_features * batch_size * unit)
    mem += next_chunk(len(y.flatten()) * unit)
    mem += 8519680  # cublas_size
    if inference_mode:
        mem += next_chunk(len(y.flatten()) * unit)
    print(f"Total memory expected: {mem}")
    assert final_memory == mem


def test_multi_layer_backward():
    print(f"Base memory: {torch.cuda.memory_allocated(device_id)}")

    n_layers = 1
    model = nn.Sequential(*[
        nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),  # No trainable params
            nn.Linear(100, 200),
            nn.Sigmoid(),  # No trainable params
        )
        for _ in range(n_layers)
    ]).to(device_id)
    batch_size = 5
    x = torch.randn((batch_size, 200), device=device_id)
    y = model(x)
    print(f"Memory after forward pass: {torch.cuda.memory_allocated(device_id)}")
    y.sum().backward()
    final_memory = torch.cuda.memory_allocated(device_id)
    print(f"Memory after backward pass: {final_memory}")

    # Computed memory
    next_chunk = lambda n: (n + 511) // 512 * 512
    mem = 0
    unit = model[0][0].weight.dtype.itemsize
    for block in model:
        for layer in block:
            if isinstance(layer, nn.Linear):
                mem += next_chunk(len(layer.weight.flatten()) * unit) * 2   # Weights and gradients
                mem += next_chunk(len(layer.bias) * unit) * 2               # Biases and gradients
                # mem += next_chunk(layer.in_features * batch_size * unit)  # Intermediate tensors are cleared
    mem += next_chunk(len(y.flatten()) * unit)
    mem += 2 * 8519680                                                      # cublas_size doubled
    mem += next_chunk(len(y.flatten()) * unit)
    print(f"Total memory expected: {mem}")
    assert final_memory == mem


def test_multi_op_forward():
    print(f"Base memory: {torch.cuda.memory_allocated(device_id)}")

    x = torch.randn((1, 256), device=device_id, dtype=torch.float32, requires_grad=False)
    # w1 = torch.randn((256, 26), device=device_id, dtype=torch.float32, requires_grad=True)
    # w2 = torch.randn((256, 256), device=device_id, dtype=torch.float32, requires_grad=True)
    layer_norm = nn.LayerNorm(256).to(device_id)
    y = layer_norm(x + 2)

    total_mem = torch.cuda.memory_allocated(device_id)
    print(f"Memory after forward pass: {total_mem}")

    # Memory calculations
    next_chunk = lambda n: (n + 511) // 512 * 512
    x_mem = next_chunk(len(x.flatten()) * x.dtype.itemsize)
    y_mem = next_chunk(len(y.flatten()) * y.dtype.itemsize)
    # w1_mem = 0  # next_chunk(len(w1.flatten()) * w1.dtype.itemsize)
    # w2_mem = next_chunk(len(w2.flatten()) * w2.dtype.itemsize)
    layer_norm_mem = next_chunk(len(layer_norm.weight.flatten()) * layer_norm.weight.dtype.itemsize)
    layer_norm_mem += next_chunk(len(layer_norm.bias.flatten()) * layer_norm.bias.dtype.itemsize)
    cublas_size = 0#8519680
    intermediate_mem = next_chunk(len(x.flatten()) * x.dtype.itemsize)
    total_mem_expected = x_mem + y_mem + cublas_size + intermediate_mem + layer_norm_mem
    print(f"Total memory expected: {total_mem_expected}")
    assert total_mem == total_mem_expected


def test_layer_norm():
    print(f"Base memory: {torch.cuda.memory_allocated(device_id)}")
    x = torch.rand((10,), device=device_id)
    w = torch.rand((10,), requires_grad=True, device=device_id)
    # Layer Norm
    y = (x - x.mean()) / (x.std() + 1e-6) * w
    final_memory = torch.cuda.memory_allocated(device_id)
    print(f"Memory after forward pass: {final_memory}")

    # Memory calculations
    next_chunk = lambda n: (n + 511) // 512 * 512
    mem = next_chunk(len(x.flatten()) * x.dtype.itemsize)
    mem += next_chunk(len(w.flatten()) * w.dtype.itemsize)
    mem += next_chunk(len(y.flatten()) * y.dtype.itemsize)
    mem += next_chunk(len(x.flatten()) * x.dtype.itemsize)  # intermediate
    print(f"Total memory expected: {mem}")
    assert final_memory == mem


def test_single_linear_layer_with_optimizer():
    # Disable cublas
    import os; os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"

    memory_timeline_real = []
    add = lambda e: memory_timeline_real.append({"event": e, "memory": torch.cuda.memory_allocated(device_id)})
    add("baseline")

    in_size = 256
    out_size = 250
    batch_size = 100
    model = nn.Linear(in_size, out_size, device=device, dtype=torch.float32)
    add("model_allocation")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    add("optimizer_init")

    x = torch.randn((batch_size, in_size,), dtype=torch.float32, device=device)
    add("input_allocation")

    def step(n):
        optimizer.zero_grad()
        add(f"optim_zero_grad_{n}")

        y = model(x)
        add(f"forward_{n}")

        y.sum().backward()
        add(f"backward_{n}")

        optimizer.step()
        del y
        add(f"optim_step_{n}")

    for i in range(4):
        step(i + 1)

    # Bar chart with even name on x-axis and total_memory on y-axis
    fig = plt.figure(figsize=(15, 7))
    fig.set_tight_layout(True)
    plt.ylim((0, 1_300_000))
    plt.bar([event["event"] for event in memory_timeline_real], [event["memory"] for event in memory_timeline_real])
    plt.xlabel("Event")
    plt.ylabel("Total memory allocated (bytes)")
    plt.title(f"Memory allocation during training ({type(optimizer)})")
    plt.xticks(rotation=45)
    plt.show()

    """
    Base memory: 0
    Memory after model allocation: 257024
    Memory after optimizer allocation: 257024
     ------- Step -------
    Memory after optimizer zero_grad: 258048
    Memory after forward pass: 259072
    Memory after backward pass: 516096
    Memory after optimizer step: 1030144
     ------- Step -------
    Memory after optimizer zero_grad: 772096
    Memory after forward pass: 773120
    Memory after backward pass: 1030144
    Memory after optimizer step: 1030144
     ------- Step -------
    Memory after optimizer zero_grad: 772096
    Memory after forward pass: 773120
    Memory after backward pass: 1030144
    Memory after optimizer step: 1030144
    
    Final memory: 1029120
    """

    # Memory calculations
    units = model.weight.dtype.itemsize
    memory_timeline = []
    all_keys = ["trainable_params", "input", "output", "gradient", "intermediate_tensors", "optimizer_state"]
    def update_memory(event: str, update: dict):
        prev_state = memory_timeline[-1] if memory_timeline else {k: 0 for k in all_keys}
        new_state = {k: prev_state.get(k, 0) + update.get(k, 0) for k in all_keys}
        new_state["event"] = event
        memory_timeline.append(new_state)
    next_chunk = lambda n: (n + 511) // 512 * 512

    update_memory("baseline", {})

    # Model memory
    model_mem = next_chunk(len(model.weight.flatten()) * units)
    model_mem += next_chunk(len(model.bias) * units)
    update_memory("model_allocation", {"trainable_params": model_mem})
    update_memory("optimizer_init", {})

    # Input memory
    x_mem = next_chunk(len(x.flatten()) * units)
    update_memory("input_allocation", {"input": x_mem})
    update_memory("optim_zero_grad_1", {})

    # Forward
    y_mem = next_chunk(batch_size * out_size * units)
    # Add any intermediate tensors here.
    update_memory("forward_1", {"output": y_mem})  # , "intermediate_tensors": ...})

    # Backward
    grad_mem = next_chunk(len(model.weight.grad.flatten()) * units)
    grad_mem += next_chunk(len(model.bias.grad.flatten()) * units)
    # Clear any intermediate tensors here.
    update_memory("backward_1", {"gradient": grad_mem})  # "intermediate_tensors": ...})

    # Optimizer memory
    if isinstance(optimizer, torch.optim.SGD):
        # SGD has parameters in memory. They are cleared after each step.
        optimizer_mem = 0
    elif isinstance(optimizer, torch.optim.Adam):
        # Adam has parameters and 2 momentum buffers. Parameters are cleared after each step.
        optimizer_mem = 2 * model_mem
    else:
        raise
    update_memory("optim_step_1", {"optimizer_state": optimizer_mem, "output": -y_mem})

    for step in range(2, 5):
        update_memory(f"optim_zero_grad_{step}", {"gradient": -grad_mem})
        update_memory(f"forward_{step}", {"output": y_mem})
        update_memory(f"backward_{step}", {"gradient": grad_mem})
        update_memory(f"optim_step_{step}", {"output": -y_mem})

    # Make totals
    for event in memory_timeline:
        event["total"] = sum([v for v in event.values() if isinstance(v, int)])

    # Plot memory timeline
    fig = plt.figure(figsize=(15, 7))
    fig.set_tight_layout(True)
    plt.ylim((0, 1_300_000))
    plt.bar([event["event"] for event in memory_timeline], [event["total"] for event in memory_timeline])
    plt.xlabel("Event")
    plt.ylabel("Total memory allocated (bytes)")
    plt.title(f"Memory allocation expected ({type(optimizer)})")
    plt.xticks(rotation=45)
    plt.show()

    import pandas as pd
    df = pd.DataFrame(memory_timeline, columns=all_keys + ["event"])
    df.set_index("event", inplace=True, drop=True)
    df.plot(kind='bar', stacked=True, figsize=(15, 7), ylim=(0, 1_300_000), xlabel="Event", ylabel="Total memory allocated (bytes)", title=f"Memory allocation expected ({type(optimizer)})")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

    # Compare the two timelines
    for i, (real, expected) in enumerate(zip(memory_timeline_real, memory_timeline)):
        assert real["memory"] == expected["total"], f"Memory mismatch at {real['event']}: {real['memory']} != {expected['total']}"



if __name__ == "__main__":
    # test_reservation_vs_allocation()
    # test_dtype_memory_allocation()
    # test_memory_allocation_relationship()
    # test_single_linear_layer_allocation()
    # test_mat_mul_memory_allocation()
    # test_single_linear_layer_forward_allocation()
    # test_single_linear_layer_backward_allocation()
    # test_execution_speed_without_cublas()
    # test_cublas_on_multiple_matmul()
    # test_multi_layer_forward()
    test_single_linear_layer_with_optimizer()
    # test_layer_norm()
    # test_multi_layer_backward()
