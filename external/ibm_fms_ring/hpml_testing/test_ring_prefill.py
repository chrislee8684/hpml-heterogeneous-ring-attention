import os
import torch
import torch.distributed as dist
import argparse
import time
import math  # you don't actually need this now, but ok

# Import after torch / dist but BEFORE usage
from fms.distributed.strategy import RingAttentionStrategy
from fms.distributed.ring_attention import _compute_attention_ring_pass_kv


def main():
    parser = argparse.ArgumentParser(description="Ring Attention Benchmark")
    parser.add_argument(
        "--total_seq_len",
        type=int,
        default=16384,
        help="Total sequence length to distribute across GPUs.",
    )
    args = parser.parse_args()

    # -----------------------------
    # Distributed init
    # -----------------------------
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in os.environ:
        print(f"Rank {rank}: Detected CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']}%\n")

        # Import after dist init
    from fms.distributed.strategy import RingAttentionStrategy
    from fms.distributed.ring_attention import _compute_attention_ring_pass_kv
    # -----------------------------
    # Per-rank "speed" from MPS
    # -----------------------------
    mps_str = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    if mps_str is None:
        local_speed = 1.0
    else:
        local_speed = float(mps_str)

    speed_tensor = torch.tensor([local_speed], device=device)
    all_speeds = [torch.zeros_like(speed_tensor) for _ in range(world_size)]
    dist.all_gather(all_speeds, speed_tensor)

    all_speeds = torch.stack(all_speeds)[:, 0].cpu().tolist()
    if rank == 0:
        print(f"[calib] per-rank speeds: {all_speeds}")

    # -----------------------------
    # Sharding mode: EVEN vs PROP
    # -----------------------------
    total_seq_len = args.total_seq_len
    shard_mode = os.environ.get("SHARD_MODE", "proportional").lower()

    if shard_mode == "even":
        # Ignore speeds, just split tokens evenly
        base = total_seq_len // world_size
        block_lens = [base] * world_size
        remainder = total_seq_len - sum(block_lens)
        # give leftover tokens to lowest ranks (or any policy you like)
        for i in range(remainder):
            block_lens[i] += 1
        if rank == 0:
            print(f"[calib] SHARD_MODE=even, block_lens={block_lens}")
    else:
        # Proportional to speed (what you had before)
        sum_speed = sum(all_speeds)
        tokens_float = [total_seq_len * s / sum_speed for s in all_speeds]
        block_lens = [int(t) for t in tokens_float]
        remainder = total_seq_len - sum(block_lens)

        frac = [(tokens_float[i] - block_lens[i], i) for i in range(world_size)]
        frac.sort(reverse=True)
        for k in range(remainder):
            _, idx = frac[k]
            block_lens[idx] += 1
        if rank == 0:
            print(f"[calib] SHARD_MODE=proportional, block_lens={block_lens}")

    # prefix sums
    block_starts = [0]
    for i in range(world_size - 1):
        block_starts.append(block_starts[-1] + block_lens[i])

    block_lens_tensor = torch.tensor(block_lens, device=device, dtype=torch.long)
    block_starts_tensor = torch.tensor(block_starts, device=device, dtype=torch.long)
    dist.broadcast(block_lens_tensor, src=0)
    dist.broadcast(block_starts_tensor, src=0)

    block_lens = block_lens_tensor.cpu().tolist()
    block_starts = block_starts_tensor.cpu().tolist()

    local_seq_len = block_lens[rank]
    q_start = block_starts[rank]

    # IMPORTANT: construct strategy AFTER we know block_lens
    strategy = RingAttentionStrategy(group=None, block_lens=block_lens)

    # -----------------------------
    # Test config
    # -----------------------------
    batch_size = 1
    nheads = 8
    head_dim = 64
    scale = head_dim ** 0.5
    accum_dtype = torch.float32
    causal = True

    # Local Q/K/V
    torch.manual_seed(42 + rank)
    q = torch.randn(batch_size, nheads, local_seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, nheads, local_seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, nheads, local_seq_len, head_dim, device=device, dtype=torch.float16)

    if rank == 0:
        print(f"Rank {rank}: Q shape={q.shape}, K shape={k.shape}, V shape={v.shape}")

    # One correctness/run check
    out = _compute_attention_ring_pass_kv(
        q, k, v, None, strategy, q_start, local_seq_len, scale, accum_dtype, causal
    )
    if rank == 0:
        print(f"Rank {rank}: Initial output sum={out.sum().item():.4f}")

    # Warmup
    torch.cuda.synchronize()
    for _ in range(3):
        _ = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy, q_start, local_seq_len, scale, accum_dtype, causal
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    num_iters = 10
    for _ in range(num_iters):
        out = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy, q_start, local_seq_len, scale, accum_dtype, causal
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000



    elapsed_tensor = torch.tensor([elapsed], device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    global_max_elapsed = elapsed_tensor.item()

    if rank == 0:
        print("\n--- Benchmark Summary (global) ---")
        print(f"GPUs: {world_size}")
        print(f"Total seq_len: {total_seq_len}")
        print(f"Local seq_len (rank 0): {local_seq_len}")
        print(f"Global max time per call: {global_max_elapsed:.2f} ms")
        print("--- End Summary ---\n")
    # if rank == 0:
    #     print("\n--- Benchmark Summary ---")
    #     print(f"GPUs: {world_size}")
    #     print(f"Total seq_len: {total_seq_len}")
    #     print(f"Local seq_len (rank 0): {local_seq_len}")
    #     print(f"Avg time per call: {elapsed:.2f} ms")
    #     print("--- End Summary ---\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()