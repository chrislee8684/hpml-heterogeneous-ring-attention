import torch
import torch.distributed as dist
import time
import os
import argparse

from fms.modules.attention import MultiHeadAttention
import fms.modules.ring_attention


dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class SimpleModel(torch.nn.Module):
    def __init__(self, emb_dim, nheads, kvheads, attn_name):
        super().__init__()
        self.attn = MultiHeadAttention(
            emb_dim,
            emb_dim // nheads,
            emb_dim // nheads,
            nheads,
            kvheads,
            use_bias=True,
        )
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(emb_dim * 4, emb_dim),
        )
        self.ln1 = torch.nn.LayerNorm(emb_dim)
        self.ln2 = torch.nn.LayerNorm(emb_dim)
        self.attn_name = attn_name

    def forward(self, x):
        y, _ = self.attn(self.ln1(x), use_cache=True, attn_name=self.attn_name)
        x = x + y
        x = x + self.ff(self.ln2(x))
        return x


def setup(rank, world_size):
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def benchmark(rank, world_size, implementation, seq_len, dtype_str):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    dtype = dtype_map[dtype_str]

    # Model parameters
    emb_dim = 768
    nheads = 12
    kvheads = 12
    batch_size = 1

    model = SimpleModel(emb_dim, nheads, kvheads, implementation).to(rank).to(dtype)

    # Dummy data
    x = torch.randn(batch_size, seq_len // world_size, emb_dim, dtype=dtype).to(rank)

    # Warmup
    for _ in range(5):
        model(x)

    # Benchmark
    n_iterations = 10
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_iterations):
        model(x)
    torch.cuda.synchronize()
    end_time = time.time()

    if rank == 0:
        print(f"Implementation: {implementation}")
        print(f"World Size: {world_size}")
        print(f"Sequence Length (per GPU): {seq_len // world_size}")
        print(f"Data Type: {dtype_str}")
        print(f"Total time for {n_iterations} iterations: {end_time - start_time:.4f}s")
        print(
            f"Throughput: {(n_iterations * batch_size) / (end_time - start_time):.2f} samples/s"
        )
        print(
            f"Latency: {(end_time - start_time) / n_iterations * 1000:.2f} ms/iteration"
        )

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ring Attention Benchmark')
    parser.add_argument('--implementation', type=str, default='ring_attention',
                        choices=['ring_attention', 'ring_attention_pipelined', 'ring_attention_blocking'],
                        help='Attention implementation to benchmark')
    parser.add_argument('--seq_len', type=int, default=4096,
                        help='Sequence length for the benchmark')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'],
                        help='Data type for the benchmark')
    args = parser.parse_args()

    # To run this benchmark, use the following command:
    # torchrun --nproc_per_node=<num_gpus> scripts/benchmark_ring_attention.py --implementation <implementation_name>
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"Starting benchmark on rank {rank} of {world_size} GPUs.")
    if world_size > 1:
        benchmark(rank, world_size, args.implementation, args.seq_len, args.dtype)
    else:
        print("This benchmark is intended for multi-GPU setups. Running on a single GPU.")
        benchmark(0, 1, args.implementation, args.seq_len, args.dtype)
