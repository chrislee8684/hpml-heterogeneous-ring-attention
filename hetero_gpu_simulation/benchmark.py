import argparse
import time
import torch
import os

def benchmark_gemm(m, n, k, device, iters=200, warmup=20, dtype=torch.float16):
    """Run GEMM benchmark and return TFLOPS"""
    A = torch.randn((m, k), device=f'cuda:{device}', dtype=dtype)
    B = torch.randn((k, n), device=f'cuda:{device}', dtype=dtype)
    
    # Warmup
    for _ in range(warmup):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate TFLOPS
    flops = 2.0 * m * n * k * iters
    tflops = flops / elapsed / 1e12
    
    return elapsed, tflops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mem_fraction", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()
    
    # Determine actual GPU being used 
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices and visible_devices.isdigit():
        actual_gpu = int(visible_devices)
    else:
        actual_gpu = args.device
    
    # Set memory limit
    if args.mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device=args.device)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(args.device)
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    
    # Run benchmark
    elapsed, tflops = benchmark_gemm(args.m, args.n, args.k, args.device, args.iters, args.warmup, dtype)
    
    # Memory info
    free, total = torch.cuda.mem_get_info(device=args.device)
    print(f"\nGPU {actual_gpu} Benchmark Results:")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {tflops:.2f} TFLOP/s")
    print(f"Avg per iteration: {elapsed/args.iters*1000:.2f} ms")
    print(f"\nMemory:")
    print(f"Free: {free/1024**3:.2f} GiB / Total: {total/1024**3:.2f} GiB")
    print(f"PyTorch allocated: {torch.cuda.memory_allocated(args.device)/1024**3:.2f} GiB")
    print(f"PyTorch reserved: {torch.cuda.memory_reserved(args.device)/1024**3:.2f} GiB")

if __name__ == "__main__":
    main()
