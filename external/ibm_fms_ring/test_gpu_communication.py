#!/usr/bin/env python3
"""
GPU-to-GPU Communication Benchmark
"""

import torch
import torch.distributed as dist
from torch.distributed import P2POp
import time
import os

# Initialize distributed
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)

# Configuration (matches Ring Attention K/V tensors)
batch = 1
num_heads = 32
block_size = 16384    # Change to test different sizes (2048, 4096, 8192)
head_dim = 128
num_iterations = 100  # Number of ring shifts to test

# Create test tensor (same shape as K or V in Ring Attention)
tensor = torch.randn(batch, num_heads, block_size, head_dim,
                    dtype=torch.float16, device=device)

# Calculate size in GB
tensor_size_bytes = tensor.numel() * tensor.element_size()
tensor_size_gb = tensor_size_bytes / (1024**3)

# Ring shift logic (same as _ring_shift_tensor)
send_to = (rank + 1) % world_size
recv_from = (rank - 1 + world_size) % world_size

recv_buf = torch.empty_like(tensor)

# Warmup
for _ in range(10):
    ops = [
        P2POp(dist.isend, tensor, peer=send_to),
        P2POp(dist.irecv, recv_buf, peer=recv_from)
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

# Benchmark
torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(num_iterations):
    ops = [
        P2POp(dist.isend, tensor, peer=send_to),
        P2POp(dist.irecv, recv_buf, peer=recv_from)
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

torch.cuda.synchronize()
end = time.perf_counter()

# Calculate results
total_time = end - start
time_per_shift = total_time / num_iterations
total_data_transferred = tensor_size_gb * num_iterations
bandwidth_gb_s = total_data_transferred / total_time

# Print results (only from rank 0)
if rank == 0:
    print(f"\n{'='*60}")
    print(f"GPU Communication Benchmark Results")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  GPUs: {world_size}")
    print(f"  Tensor shape: {list(tensor.shape)}")
    print(f"  Tensor size: {tensor_size_gb:.4f} GB")
    print(f"  block_size: {block_size}")
    print(f"\nResults:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Time per shift: {time_per_shift*1000:.2f} ms")
    print(f"  Bandwidth: {bandwidth_gb_s:.2f} GB/s")
    print(f"\nReference speeds:")
    print(f"  NVLink: ~300-600 GB/s")
    print(f"  PCIe Gen3 x16: ~32 GB/s")
    print(f"  PCIe Gen4 x16: ~64 GB/s")
    print(f"{'='*60}\n")

dist.destroy_process_group()