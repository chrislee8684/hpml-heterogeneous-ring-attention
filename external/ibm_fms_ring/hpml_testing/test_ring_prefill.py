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
                        help="If set, distribute sequence proportionally to GPU throttle. Otherwise, distribute evenly.")
    
    args = parser.parse_args()
    total_seq_len = args.total_seq_len
    proportional = args.proportional_sharding

    # ---------------------------------------------------------
    # Initialize distributed env (variables set by launcher)
    # ---------------------------------------------------------
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Report throttle setting
    throttle_env = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100")
    print(f"Rank {rank}: Detected CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={throttle_env}%")

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
    # SHARDING LOGIC (NEW)
    # ---------------------------------------------------------

    if proportional:
        # -------------------------------------------
        # PROPORTIONAL SHARDING BASED ON MPS THROTTLE
        # -------------------------------------------
        local_throttle = float(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100"))

        # Gather throttles from all ranks
        throttle_tensor = torch.tensor([local_throttle], device=device, dtype=torch.float32)
        all_throttles = [torch.zeros_like(throttle_tensor) for _ in range(world_size)]
        dist.all_gather(all_throttles, throttle_tensor)
        all_throttles = [t.item() for t in all_throttles]

        if rank == 0:
            print(f"All throttle values: {all_throttles}")

        total_throttle = sum(all_throttles)
        weights = [t / total_throttle for t in all_throttles]

        # Compute sizes
        sizes = [int(total_seq_len * w) for w in weights]
        sizes[-1] = total_seq_len - sum(sizes[:-1])  # fix rounding

        local_seq_len = sizes[rank]

        # Compute sequence start offsets
        start_indices = [0] * world_size
        for i in range(1, world_size):
            start_indices[i] = start_indices[i - 1] + sizes[i - 1]

        q_start = start_indices[rank]

        print(f"Rank {rank}: PROPORTIONAL shard start={q_start}, local_seq_len={local_seq_len}")

        block_size = max(sizes)  # needed for ring padding

    else:
        # -------------------------------------------
        # EVEN SHARDING (DEFAULT BEHAVIOR)
        # -------------------------------------------
        local_seq_len = total_seq_len // world_size
        q_start = rank * local_seq_len
        sizes = [local_seq_len] * world_size
        sizes[-1] = total_seq_len - local_seq_len * (world_size - 1)  # final GPU absorbs remainder

        print(f"Rank {rank}: EVEN shard start={q_start}, local_seq_len={local_seq_len}")

        block_size = local_seq_len

    # ---------------------------------------------------------
    # Setup Ring Strategy
    # ---------------------------------------------------------
    strategy = RingAttentionStrategy(group=None)
    strategy.block_size = block_size

    # ---------------------------------------------------------
    # Create local Q, K, V tensors
    # ---------------------------------------------------------
    torch.manual_seed(42 + rank)
    q = torch.randn(batch_size, nheads, local_seq_len, head_dim,
                    device=device, dtype=torch.float16)
    k = torch.randn(batch_size, nheads, local_seq_len, head_dim,
                    device=device, dtype=torch.float16)
    v = torch.randn(batch_size, nheads, local_seq_len, head_dim,
                    device=device, dtype=torch.float16)

    if rank == 0:
        print(f"Rank {rank}: Q/K/V shapes: {q.shape}")

    # ---------------------------------------------------------
    # Correctness check
    # ---------------------------------------------------------
    out = _compute_attention_ring_pass_kv(
        q, k, v, None, strategy,
        q_start, local_seq_len,
        scale, accum_dtype, causal
    )

    if rank == 0:
        print(f"Rank {rank}: Initial output sum = {out.sum().item():.4f}")

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
    num_iters = 10

    for _ in range(num_iters):
        out = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy,
            q_start, local_seq_len,
            scale, accum_dtype, causal
        )

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / num_iters * 1000

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    if rank == 0:
        print("\n--- Benchmark Summary ---")
        print(f"GPUs:               {world_size}")
        print(f"Total seq_len:      {total_seq_len}")
        print(f"Shard sizes:        {sizes}")
        print(f"Proportional mode:  {proportional}")
        print(f"Avg time per call:  {elapsed_ms:.2f} ms")
        print("--- End Summary ---\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
