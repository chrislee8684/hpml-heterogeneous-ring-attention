"""
Test for ring attention prefill (pass-KV) with async overlap.

Run with: torchrun --nproc_per_node=2 test_ring_prefill.py
"""
import os
import torch
import torch.distributed as dist

def main():
    # Initialize distributed (torchrun sets env vars)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Import after dist init
    from fms.distributed.strategy import RingAttentionStrategy
    from fms.distributed.llama_ring import _compute_attention_ring_pass_kv

    # Test config
    batch_size = 1
    total_seq_len = 1024  # Total sequence split across GPUs
    nheads = 8
    head_dim = 64

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
        mask=None,
        strategy=strategy,
        q_start=q_start,
        num_valid_tokens=local_seq_len,
        scale=head_dim ** 0.5,
        accum_dtype=torch.float32,
        causal=True
    )

    print(f"Rank {rank}: Output shape={out.shape}, sum={out.sum().item():.4f}")

    # Simple timing test
    torch.cuda.synchronize()
    import time

    # Warmup
    for _ in range(3):
        _ = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy, q_start, local_seq_len,
            head_dim ** 0.5, torch.float32, True
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    num_iters = 10
    for _ in range(num_iters):
        out = _compute_attention_ring_pass_kv(
            q, k, v, None, strategy, q_start, local_seq_len,
            head_dim ** 0.5, torch.float32, True
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000

    if rank == 0:
        print(f"GPUs: {world_size}")
        print(f"Total seq_len: {total_seq_len}")
        print(f"Local seq_len: {local_seq_len}")
        print(f"Avg time per call: {elapsed:.2f} ms")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()



