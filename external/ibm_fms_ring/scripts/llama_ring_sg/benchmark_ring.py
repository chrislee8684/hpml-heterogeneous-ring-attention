"""Benchmark script for comparing Ring Attention vs Regular Attention."""

import argparse
import os
import statistics
import time
import csv
import gc
import torch
import torch.distributed as dist
from pathlib import Path

from fms import models
from fms.utils import tokenizers
from fms.distributed.strategy import NoOpStrategy

SUMMARY_HEADERS = ["strategy", "prompt_tokens", "ttft_ms", "avg_decode_ms", "total_time_ms", "comm_time_ms", "compute_comm_ratio"]


def print0(*args, **kwargs):
    if int(os.getenv("RANK", 0)) == 0:
        print(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Ring vs Regular Attention")
    script_path = Path(__file__).resolve()
    repo_dir = script_path.parents[2]
    model_dir = repo_dir.parent / "llama-hf"

    parser.add_argument("--device_type", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--architecture", type=str, default="llama")
    parser.add_argument("--variant", type=str, default="8b")
    parser.add_argument("--model_path", type=str, default=str(model_dir))
    parser.add_argument("--tokenizer", type=str, default=str(model_dir / "tokenizer.model"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, required=True, help="Number of prompt tokens")
    parser.add_argument("--num_decode_tokens", type=int, default=30, help="Number of tokens to decode")
    parser.add_argument("--run_ring_first", action="store_true", default=True)
    parser.add_argument("--no-run_ring_first", dest="run_ring_first", action="store_false")
    parser.add_argument("--summary_csv", type=str, default=None, help="Summary CSV path (appends)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])

    return parser.parse_args()


def setup_model(args, strategy, dtype):
    # For hf_pretrained, don't pass variant or source - let it infer from model_path
    if args.architecture == "hf_pretrained":
        model = models.get_model(
            args.architecture,
            model_path=args.model_path,
            device_type=args.device_type,
            distributed_strategy=strategy,
            data_type=dtype
        )
    else:
        model = models.get_model(
            args.architecture,
            args.variant,
            model_path=args.model_path,
            device_type=args.device_type,
            source="hf",
            distributed_strategy=strategy,
            data_type=dtype
        )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def run_benchmark(model, input_ids, num_decode, label, device, is_ring=False):
    """Run generation benchmark. Returns dict with timing metrics."""
    from fms.distributed.strategy import RingAttentionStrategy

    rank = dist.get_rank() if dist.is_initialized() else 0
    ids = input_ids.clone().to(device)

    # Get strategy for comm timing (ring only)
    strategy = getattr(model, 'distributed_strategy', None)
    if is_ring and isinstance(strategy, RingAttentionStrategy):
        strategy.reset_comm_time()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Prefill (TTFT)
    t0 = time.perf_counter()
    out = model.forward(ids, use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000

    logits, cache = (out[0], out[1]) if isinstance(out, tuple) else (out.logits, out.past_key_value_states)
    last_token = ids[:, -1:]

    # Decode
    decode_times = []
    for _ in range(num_decode):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.forward(last_token, past_key_value_states=cache, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_times.append((time.perf_counter() - t0) * 1000)

        logits, cache = (out[0], out[1]) if isinstance(out, tuple) else (out.logits, out.past_key_value_states)
        last_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    avg_decode_ms = statistics.mean(decode_times)
    total_time_ms = ttft_ms + sum(decode_times)

    # Get comm time and compute ratio (ring only)
    comm_time_ms = 0.0
    compute_comm_ratio = None
    if is_ring and isinstance(strategy, RingAttentionStrategy):
        comm_time_ms = strategy.get_comm_time_ms()
        compute_time_ms = total_time_ms - comm_time_ms
        if comm_time_ms > 0:
            compute_comm_ratio = compute_time_ms / comm_time_ms

    if rank == 0:
        print0(f"\n{label}:")
        msg = f"  TTFT: {ttft_ms:.2f} ms | Avg Decode: {avg_decode_ms:.2f} ms | Total: {total_time_ms:.2f} ms"
        if is_ring:
            msg += f" | Comm: {comm_time_ms:.2f} ms"
            if compute_comm_ratio:
                msg += f" | Compute/Comm: {compute_comm_ratio:.2f}"
        print0(msg)

    return {
        "ttft_ms": ttft_ms, "avg_decode_ms": avg_decode_ms, "total_time_ms": total_time_ms,
        "comm_time_ms": comm_time_ms, "compute_comm_ratio": compute_comm_ratio, "logits": logits
    }


def main():
    args = parse_args()
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    # Initialize distributed
    if world_size > 1 and args.device_type == "cuda":
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device_type)

    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    # Create random input tokens (use hardcoded vocab range to avoid tokenizer loading issues)
    # LLaMA vocab is typically 32000-128256, use safe range
    vocab_size = 128256
    ids = torch.randint(100, vocab_size - 100, (args.batch_size, args.num_tokens), dtype=torch.long, device=device)

    # Synchronize random tokens across ranks
    if world_size > 1:
        dist.broadcast(ids, src=0)

    print0(f"Benchmark: {args.num_tokens} prompt tokens, {args.num_decode_tokens} decode tokens")

    # Define strategies
    strategies = [("Ring", "ring"), ("Regular", NoOpStrategy)]
    if not args.run_ring_first:
        strategies.reverse()

    results = []
    for label, strategy in strategies:
        # Skip Ring if not distributed
        if strategy == "ring" and not dist.is_initialized():
            print0(f"Skipping {label} (requires distributed)")
            continue

        # Regular only runs on rank 0
        is_regular = strategy is NoOpStrategy
        if is_regular and rank != 0:
            if world_size > 1:
                dist.barrier()
            continue

        if args.device_type == "cuda":
            torch.cuda.empty_cache()

        model = setup_model(args, strategy, dtype)
        is_ring = (strategy == "ring")
        result = run_benchmark(model, ids, args.num_decode_tokens, label, device, is_ring=is_ring)
        result["strategy"] = label
        results.append(result)

        del model
        gc.collect()
        if args.device_type == "cuda":
            torch.cuda.empty_cache()

        if world_size > 1:
            dist.barrier()

    # Write summary CSV
    if rank == 0 and args.summary_csv and results:
        file_exists = os.path.exists(args.summary_csv)
        with open(args.summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(SUMMARY_HEADERS)
            for r in results:
                ratio_str = f"{r['compute_comm_ratio']:.2f}" if r['compute_comm_ratio'] else "N/A"
                writer.writerow([r["strategy"], args.num_tokens, f"{r['ttft_ms']:.2f}",
                                f"{r['avg_decode_ms']:.2f}", f"{r['total_time_ms']:.2f}",
                                f"{r['comm_time_ms']:.2f}", ratio_str])

    # Print summary table
    if rank == 0 and results:
        print0(f"\n{'Strategy':<10} {'Tokens':<8} {'TTFT':<10} {'Avg Decode':<12} {'Total':<10} {'Comm':<10} {'Comp/Comm':<10}")
        print0("-" * 70)
        for r in results:
            ratio_str = f"{r['compute_comm_ratio']:.2f}" if r['compute_comm_ratio'] else "N/A"
            print0(f"{r['strategy']:<10} {args.num_tokens:<8} {r['ttft_ms']:<10.2f} {r['avg_decode_ms']:<12.2f} {r['total_time_ms']:<10.2f} {r['comm_time_ms']:<10.2f} {ratio_str:<10}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
