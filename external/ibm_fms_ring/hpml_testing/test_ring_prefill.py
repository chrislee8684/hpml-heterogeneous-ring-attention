"""
Ring Attention Benchmark
Q is sharded proportionally, KV is sharded evenly.
"""
import os
import torch
import torch.distributed as dist
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_seq_len", type=int, default=16384)
    parser.add_argument("--proportional_sharding", action="store_true")
    args = parser.parse_args()

    total_seq_len = args.total_seq_len
    proportional = args.proportional_sharding

    # ---------------------------------------------------------
    # Distributed init
    # ---------------------------------------------------------
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Import AFTER dist init
    from fms.distributed.strategy import RingAttentionStrategy
    from fms.distributed.llama_ring import _compute_attention_ring_pass_kv

    # ---------------------------------------------------------
    # Load MPS throttling for Q proportionality
    # ---------------------------------------------------------
    throttle = float(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100"))

    # ---------------------------------------------------------
    # Compute KV EVEN shards
    # ---------------------------------------------------------
    kv_base = total_seq_len // world_size
    kv_sizes = [kv_base] * world_size
    kv_sizes[-1] = total_seq_len - kv_base * (world_size - 1)

    kv_starts = [0] * world_size
    for i in range(1, world_size):
        kv_starts[i] = kv_starts[i - 1] + kv_sizes[i - 1]

    kv_local_len = kv_sizes[rank]
    kv_start = kv_starts[rank]

    # ---------------------------------------------------------
    # Compute Q PROPORTIONAL shards (optional)
    # ---------------------------------------------------------
    if proportional:
        thr_tensor = torch.tensor([throttle], device=device)
        thr_list = [torch.zeros_like(thr_tensor) for _ in range(world_size)]
        dist.all_gather(thr_list, thr_tensor)
        thr_list = [t.item() for t in thr_list]

        total_thr = sum(thr_list)
        q_weights = [t / total_thr for t in thr_list]

        q_sizes = [int(total_seq_len * w) for w in q_weights]
        q_sizes[-1] = total_seq_len - sum(q_sizes[:-1])

        q_starts = [0] * world_size
        for i in range(1, world_size):
            q_starts[i] = q_starts[i - 1] + q_sizes[i - 1]

        q_local_len = q_sizes[rank]
        q_start = q_starts[rank]

    else:
        # Q evenly sharded if proportional flag not set
        q_local_len = kv_local_len
        q_start = kv_start

    # ---------------------------------------------------------
    # Strategy uses KV sizes only
    # ---------------------------------------------------------
    strategy = RingAttentionStrategy(
        block_lens=kv_sizes,          # MUST be KV sizes
        block_size=max(kv_sizes),     # ring padding size
        group=None
    )

    # ---------------------------------------------------------
    # Allocate tensors:
    # Q = proportional
    # K/V = even
    # ---------------------------------------------------------
    batch_size = 1
    nheads = 8
    head_dim = 64
    scale = head_dim ** 0.5
    accum_dtype = torch.float32
    causal = True

    torch.manual_seed(42 + rank)

    q = torch.randn(batch_size, nheads, q_local_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, nheads, kv_local_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, nheads, kv_local_len, head_dim, device=device, dtype=torch.float16)

    # ---------------------------------------------------------
    # Forward run for correctness
    # ---------------------------------------------------------
    out = _compute_attention_ring_pass_kv(
        q, k, v, None,
        strategy,
        q_start,
        q_local_len,
        scale,
        accum_dtype,
        causal
    )

    torch.cuda.synchronize()

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------
    for _ in range(3):
        _compute_attention_ring_pass_kv(
            q, k, v, None,
            strategy,
            q_start,
            q_local_len,
            scale,
            accum_dtype,
            causal
        )
    torch.cuda.synchronize()

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------
    iters = 10
    start = time.perf_counter()

    for _ in range(iters):
        out = _compute_attention_ring_pass_kv(
            q, k, v, None,
            strategy,
            q_start,
            q_local_len,
            scale,
            accum_dtype,
            causal
        )

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000

    if rank == 0:
        print("\n=== Benchmark Summary ===")
        print(f"Q shard sizes: { [q_local_len if i==rank else '...'] }")
        print(f"KV shard sizes: {kv_sizes}")
        print(f"Mode: {'Q-proportional' if proportional else 'even'}")
        print(f"Avg time per call: {elapsed:.2f} ms")
        print("===========================")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
