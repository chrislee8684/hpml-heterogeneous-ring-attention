# hpml-heterogeneous-ring-attention

## hetero_gpu_simulation

Simulates heterogeneous GPUs by limiting resources on one of two identical GPUs.

**Setup:** `pip install -r requirements.txt && chmod +x run_benchmark.sh && ./run_benchmark.sh`

Tests GPU performance with and without compute/memory limits to mimic mixed GPU environments. Uses `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` environment variable to limit warp scheduling and `torch.cuda.set_per_process_memory_fraction()` to restrict memory usage.