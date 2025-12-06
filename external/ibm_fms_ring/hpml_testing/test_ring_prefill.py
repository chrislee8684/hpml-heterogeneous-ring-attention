"""
Test for ring attention prefill (pass-KV) with async overlap.

This script is benchmarked by a launcher script that sets up the distributed
environment and throttling.

Run via a launcher script like `run_manual_benchmark.sh`.
"""
import os
import torch
import torch.distributed as dist
import argparse
import time
import math


def main():
    parser = argparse.ArgumentParser(description="Ring Attention Benchmark")
    parser.add_argument("--total_seq_len", type=int, default=16384,
                        help="Total sequence length to distribute across GPUs.")
    parser.add_argument("--proportional_sharding", action="store_true",
                        help="If set, distribute sequence proportionally to GPU throttle; else evenly.")
    
    args = parser.parse_args()
    total_seq_len = args.total_seq_len
    proportional = args.proportional_sharding

    # ---------------------------------------------------------
    # Initialize distributed environment
    # ---------------------------------------------------------
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    throttle_env = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100")
    print(f"[rank{rank}] throttle={throttle_env}%")

    # Import AFTER distributed init
    from fms.distributed.strategy import RingAttentionStrategy
    from fms.distributed.llama_ring import _compute_attention_ring_pass_kv

    # ---------------------------------------------------------
    # Benchmark configuration
    # ---------------------------------------------------------
    batch_size = 1
    nheads = 8
    head_dim = 64
    scale = head_dim ** 0.5
    accum_dtype = torch.float32
    causal = True

    # ---------------------------------------------------------
    # SHARDING LOGIC
    # ---------------------------------------------------------
    if proportional:
        # -------------------------
        # PROPORTIONAL SHARDING
        # -------------------------
        local_throttle = float(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100"))

        # Gather all throttles
        throttle_tensor = torch.tensor([local_throttle], device=device, dtype=torch.float32)
        throttle_list = [torch.zeros_like(throttle_tensor) for _ in range(world_size)]
        dist.all_gather(throttle_list, throttle_tensor)
        throttle_list = [t.item() for t in throttle_list]

        if rank == 0:
            print(f"[calib] per-rank speeds: {throttle_list}")

        total_thr = sum(throttle_list)
        weights = [t / total_thr for t in throttle_list]

        # Compute shard sizes
        sizes = [int(total_seq_len * w) for w in weights]
        sizes[-1] = total_seq_len - sum(sizes[:-1])  # fix rounding

        # Start offsets
        starts = [0] * world_size
        for i in range(1, world_size):
            starts[i] = starts[i - 1] + sizes[i - 1]

        local_seq_len = sizes[rank]
        q_start = starts[rank]

        if rank == 0:
            print(f"[calib] block_lens={sizes}, block_starts={starts}")

        block_size = max(sizes)  # needed for consistent ring padding

        print(f"[rank{rank}] PROPORTIONAL shard start={q_start}, len={local_seq_len}")

    else:
        # -------------------------
        # EVEN SHARDING
        # -------------------------
        base = total_seq_len // world_size
        sizes = [base] * world_size
        sizes[-1] = total_seq_len - base * (world_size - 1)

        starts = [i * base for i in range(world_size)]
        q_start = starts[rank]
        local_seq_len = sizes[rank]

        block_size = base

        print(f"[rank{rank}] EVEN shard start={q_start}, len={local_seq_len}")

    # ---------------------------------------------------------
    # Setup Ring Strategy
    # ---------------------------------------------------------
    strategy = RingAttentionStrategy(group=None)
    strategy.block_size = block_size  # ensures padding matches all ranks

    # ---------------------------------------------------------
    # Allocate Q, K, V tensors for this rank
    # ---------------------------------------------------------
    torch.manual_seed(42 + rank)
    q = torch.randn(batch_size, nheads, local_seq_len, head_dim,
                    device=device, dtype=torch.float16)
    k = torch.randn(batch_size, nheads, local_seq_len, head_dim,
                    device=device, dtype=torch.float16)
    v = torch.randn(batch_size, nheads, local_seq_len, head_dim,
                    device=device, dtype=torch.float16)

    if rank == 0:
        print(f"[rank0] Q/K/V shapes: {q.shape}")

    # ---------------------------------------------------------
    # Correctness check
    # ---------------------------------------------------------
    out = _compute_attention_ring_pass_kv(
        q, k, v, None,
        strategy,
        q_start,
        local_seq_len,
        scale,
        accum_dtype,
        causal
    )

    if rank == 0:
        print(f"[rank0] Initial output sum = {out.sum().item():.4f}")

    torch.cuda.synchronize()

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------
    for _ in range(3):
        _ = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy,
            q_start, local_seq_len,
            scale, accum_dtype, causal
        )
    torch.cuda.synchronize()

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------
    start = time.perf_counter()
    iters = 10

    for _ in range(iters):
        out = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy,
            q_start, local_seq_len,
            scale, accum_dtype, causal
        )

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / iters * 1000

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    if rank == 0:
        print("\n--- Benchmark Summary ---")
        print(f"GPUs:               {world_size}")
        print(f"Total seq_len:      {total_seq_len}")
        print(f"Shard sizes:        {sizes}")
        print(f"Sharding mode:      {'proportional' if proportional else 'even'}")
        print(f"Avg time per call:  {elapsed_ms:.2f} ms")
        print("--- End Summary ---\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
