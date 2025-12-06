#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
import argparse
import time
import math


def main():

    # -------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Ring Attention Benchmark")
    parser.add_argument("--total_seq_len", type=int, default=16000,
                        help="Total sequence length (Q/K/V global length)")
    parser.add_argument("--proportional_sharding", action="store_true",
                        help="Enable proportional *Q* sharding based on GPU throttle")
    args = parser.parse_args()

    # -------------------------------------------------------------
    # Initialize distributed environment
    # -------------------------------------------------------------
    dist.init_process_group("nccl")

    rank     = dist.get_rank()
    world    = dist.get_world_size()
    local_id = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_id)
    device = torch.device(f"cuda:{local_id}")

    # -------------------------------------------------------------
    # Read GPU throttle (MPS)
    # -------------------------------------------------------------
    throttle_env = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100")
    throttle = float(throttle_env) / 100.0
    print(f"[rank{rank}] throttle = {throttle_env}%")

    # -------------------------------------------------------------
    # Load RingAttentionStrategy + compute kernel
    # -------------------------------------------------------------
    from fms.distributed.strategy import RingAttentionStrategy
    from fms.distributed.llama_ring import _compute_attention_ring_pass_kv

    # -------------------------------------------------------------
    # Step 1: Build block sizes for KV (ALWAYS even)
    # -------------------------------------------------------------
    total_len = args.total_seq_len
    base = total_len // world
    remainder = total_len % world

    # KV is always evenly split
    kv_block_lens = [(base + (1 if i < remainder else 0)) for i in range(world)]

    # -------------------------------------------------------------
    # Step 2: Build Q block sizes (EVEN or PROPORTIONAL)
    # -------------------------------------------------------------
    if not args.proportional_sharding:
        # same as KV
        q_block_lens = kv_block_lens[:]
        if rank == 0:
            print("\n=== EVEN SHARDING MODE ===")
    else:
        # Evaluate proportional scaling based on throttle
        if rank == 0:
            print("\n=== PROPORTIONAL SHARDING MODE ===")

        # Gather all throttles to rank 0
        thr_tensor = torch.tensor([throttle], device=device)
        thr_list = [torch.zeros_like(thr_tensor) for _ in range(world)]
        dist.all_gather(thr_list, thr_tensor)
        thr_list = [t.item() for t in thr_list]

        if rank == 0:
            print(f"[calibration] collected throttles = {thr_list}")

        total_thr = sum(thr_list)
        if total_thr <= 0:
            raise RuntimeError("Invalid throttle configuration.")

        # Compute proportional Q lengths
        q_block_lens = [max(1, int(total_len * (thr / total_thr))) for thr in thr_list]

        # Fix rounding error
        diff = total_len - sum(q_block_lens)
        if diff != 0:
            q_block_lens[0] += diff

    # -------------------------------------------------------------
    # Broadcast Q block_lens to all ranks
    # -------------------------------------------------------------
    q_block_lens_tensor = torch.tensor(q_block_lens, device=device)
    dist.broadcast(q_block_lens_tensor, src=0)
    q_block_lens = q_block_lens_tensor.tolist()

    if rank == 0:
        print(f"[Q lengths] {q_block_lens}")
        print(f"[KV lengths] {kv_block_lens}\n")

    # -------------------------------------------------------------
    # Rank-local Q/K/V shard sizes
    # -------------------------------------------------------------
    q_len_local = q_block_lens[rank]
    kv_len_local = kv_block_lens[rank]

    # -------------------------------------------------------------
    # Build RingAttentionStrategy instance
    # -------------------------------------------------------------
    strategy = RingAttentionStrategy(
        block_lens=kv_block_lens,      # KV is always evenly split
        block_size=max(kv_block_lens), # only used for ring padding
        group=None,
    )

    # Compute global Q offset for this rank
    q_start = sum(q_block_lens[:rank])

    # -------------------------------------------------------------
    # Allocate random Q/K/V shards
    # -------------------------------------------------------------
    batch = 1
    nheads = 8
    head_dim = 64

    torch.manual_seed(42 + rank)

    q = torch.randn(batch, nheads, q_len_local, head_dim,
                    device=device, dtype=torch.float16)

    # Note: KV must have *KV* lengths (NOT Q lengths)
    k = torch.randn(batch, nheads, kv_len_local, head_dim,
                    device=device, dtype=torch.float16)
    v = torch.randn(batch, nheads, kv_len_local, head_dim,
                    device=device, dtype=torch.float16)

    if rank == 0:
        print(f"[rank0] shapes: Q={q.shape}, K={k.shape}, V={v.shape}")

    # -------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------
    torch.cuda.synchronize()
    for _ in range(3):
        _ = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy, q_start, q_len_local,
            head_dim ** 0.5, torch.float32, True
        )
    torch.cuda.synchronize()

    # -------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------
    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
        out = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy, q_start, q_len_local,
            head_dim ** 0.5, torch.float32, True
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1000

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------
    if rank == 0:
        mode = "proportional" if args.proportional_sharding else "even"
        print("=== Benchmark Summary ===")
        print(f"Mode: {mode}")
        print(f"Total seq len : {total_len}")
        print(f"Q shard sizes : {q_block_lens}")
        print(f"KV shard sizes: {kv_block_lens}")
        print(f"Avg time per call: {elapsed:.2f} ms")
        print("==========================\n")

    dist.destroy_process_group()



if __name__ == "__main__":
    main()
