"""
Test for ring attention prefill (pass-KV) with async overlap.

This script is parameterized to allow for sweeping context lengths and
simulating heterogeneous GPU performance by throttling compute resources
via CUDA MPS.

Run with:
PYTHONPATH=. torchrun --nproc_per_node=2 hpml_testing/test_ring_prefill.py --total_seq_len 16384 --throttle_rank 1 --throttle_percentage 50
"""
import os
import torch
import torch.distributed as dist
import argparse
import time
import math # Import math for math.sqrt

def main():
    parser = argparse.ArgumentParser(description="Ring Attention Benchmark with Throttling")
    parser.add_argument("--total_seq_len", type=int, default=16384, help="Total sequence length to distribute across GPUs.")
    parser.add_argument("--throttle_rank", type=int, default=None, help="The rank of the GPU to throttle. If None, no throttling is applied.")
    parser.add_argument("--throttle_percentage", type=int, default=50, help="The percentage of threads for the throttled GPU.")
    parser.add_argument("--strong_percentage", type=int, default=100, help="The percentage of threads for the non-throttled GPUs.")
    args = parser.parse_args()

    # Initialize distributed (torchrun sets env vars)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # --- Throttling Logic ---
    if args.throttle_rank is not None:
        if "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in os.environ:
            print(f"Rank {rank}: Warning: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE already set to {os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']}. Script-based override might not take effect.")
        
        if rank == args.throttle_rank:
            os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(args.throttle_percentage)
            print(f"Rank {rank}: Setting CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to {args.throttle_percentage}%")
        else:
            os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(args.strong_percentage)
            print(f"Rank {rank}: Setting CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to {args.strong_percentage}%")

    # Import after dist init
    from fms.distributed.strategy import RingAttentionStrategy
    from fms.distributed.llama_ring import _compute_attention_ring_pass_kv

    # Test config
    batch_size = 1
    total_seq_len = args.total_seq_len  # Use seq len from args
    nheads = 8
    head_dim = 64
    
    # Scale factor for attention scores
    scale = head_dim ** 0.5
    accum_dtype = torch.float32 # Use higher precision for accumulation
    causal = True # Assume causal attention

    # Setup strategy
    strategy = RingAttentionStrategy(group=None)
    local_seq_len = total_seq_len // world_size
    strategy.block_size = local_seq_len

    q_start = rank * local_seq_len

    # Create local Q, K, V tensors for this rank
    torch.manual_seed(42 + rank)
    q = torch.randn(batch_size, nheads, local_seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, nheads, local_seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, nheads, local_seq_len, head_dim, device=device, dtype=torch.float16)

    print(f"Rank {rank}: Q shape={q.shape}, K shape={k.shape}, V shape={v.shape}")

    # Run ring attention
    out = _compute_attention_ring_pass_kv(
        q, k, v,
        mask=None, # No specific mask applied for this benchmark
        strategy=strategy,
        q_start=q_start,
        num_valid_tokens=local_seq_len,
        scale=scale,
        accum_dtype=accum_dtype,
        causal=causal
    )

    print(f"Rank {rank}: Output shape={out.shape}, sum={out.sum().item():.4f}")

    # Simple timing test
    torch.cuda.synchronize()

    # Warmup
    for _ in range(3):
        _ = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy, q_start, local_seq_len,
            scale, accum_dtype, causal
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    num_iters = 10
    for _ in range(num_iters):
        out = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy, q_start, local_seq_len,
            scale, accum_dtype, causal
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000

    if rank == 0:
        print(f"\n--- Benchmark Summary ---")
        print(f"GPUs: {world_size}")
        print(f"Total seq_len: {total_seq_len}")
        print(f"Local seq_len: {local_seq_len}")
        if args.throttle_rank is not None:
            print(f"Throttled Rank: {args.throttle_rank} at {args.throttle_percentage}%")
            print(f"Other Ranks: {args.strong_percentage}%")
        print(f"Avg time per call: {elapsed:.2f} ms")
        print(f"--- End Summary ---\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
