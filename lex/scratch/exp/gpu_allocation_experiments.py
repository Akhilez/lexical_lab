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
-
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
    # Disable cublas
    # import os; os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
    cublas_size = 8519680

    print(f"Base memory: {torch.cuda.memory_allocated(device_id)}")

    model = nn.Linear(256, 250, device=device, dtype=torch.float32)
    x = torch.randn((1, 256,), dtype=torch.float32, device=device)
    y = model(x)

    print(f"Memory after forward pass: {torch.cuda.memory_allocated(device_id)}")
    y.sum().backward()
    final_memory = torch.cuda.memory_allocated(device_id)
    print(f"Memory after backward pass: {final_memory}")

    # Memory calculations
    w_mem = (len(model.weight.flatten()) * model.weight.dtype.itemsize + 511) // 512 * 512
    b_mem = (len(model.bias) * model.bias.dtype.itemsize + 511) // 512 * 512
    model_mem = w_mem + b_mem
    print(f"Excepted model memory: {model_mem}")

    x_mem = (len(x.flatten()) * x.dtype.itemsize + 511) // 512 * 512
    y_mem = (len(y.flatten()) * y.dtype.itemsize + 511) // 512 * 512
    print(f"{x_mem=}, {y_mem=}")

    # Gradient memory
    w_grad_mem = (len(model.weight.grad.flatten()) * model.weight.grad.dtype.itemsize + 511) // 512 * 512
    b_grad_mem = (len(model.bias.grad.flatten()) * model.bias.grad.dtype.itemsize + 511) // 512 * 512
    print(f"{model.weight.grad.shape=}, {w_grad_mem=}")
    print(f"{model.bias.grad.shape=}, {b_grad_mem=}")

    total_memory_expected = (2 * model_mem) + x_mem + y_mem + (2 * cublas_size)
    print(f"Total memory expected: {total_memory_expected}")

    assert final_memory == total_memory_expected

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


def next_chunk(n):
    return (n + 511) // 512 * 512


def test_multi_layer_forward():
    # Disable cublas
    import os; os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"

    print(f"Base memory: {torch.cuda.memory_allocated(device_id)}")

    inference_mode = False
    n_layers = 5
    model = nn.Sequential(*[
        nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.LayerNorm(256),
            nn.RMSNorm(256),
        )
        for _ in range(n_layers)
    ]).to(device_id)
    x = torch.randn((1, 256), device=device_id)
    with torch.inference_mode(inference_mode):
        y = model(x)

    final_memory = torch.cuda.memory_allocated(device_id)
    print(f"Memory after forward pass: {final_memory}")

    # Computed memory
    mem = 0
    unit = model[0][0].weight.dtype.itemsize
    for block in model:
        for layer in block:
            if isinstance(layer, nn.Linear):
                mem += next_chunk(len(layer.weight.flatten()) * unit)
                mem += next_chunk(len(layer.bias) * unit)
                if not inference_mode:
                    # One matmul with weight
                    mem += next_chunk(layer.out_features * unit)
            elif isinstance(layer, nn.LayerNorm):
                mem += next_chunk(len(layer.weight.flatten()) * unit)
                mem += next_chunk(len(layer.bias) * unit)
                if not inference_mode:
                    # One weight multiplication and one sqrt?
                    mem += 2 * next_chunk(layer.normalized_shape[0] * unit)
            elif isinstance(layer, nn.RMSNorm):
                mem += next_chunk(len(layer.weight.flatten()) * unit)
                if not inference_mode:
                    # One weight, one sqrt, what is 0.5 for?
                    mem += int(next_chunk(layer.normalized_shape[0] * unit) * 2.5)
    mem += next_chunk(len(x.flatten()) * unit)
    if inference_mode:
        mem += next_chunk(len(y.flatten()) * unit)
    print(f"Total memory expected: {mem}")

    # assert final_memory == mem

    del model, x, y
    torch.cuda.empty_cache()
    print(f"Memory after cleanup: {torch.cuda.memory_allocated(device_id)}")


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
    test_multi_layer_forward()
