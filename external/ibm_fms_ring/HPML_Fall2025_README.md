# HPML Fall 2025

## Code Modifications

### 1. `fms/distributed/strategy.py`
Added communication timing to `RingAttentionStrategy` class:
- `_comm_time_ms` variable to accumulate P2P communication time
- `reset_comm_time()` and `get_comm_time_ms()` methods
- Timing instrumentation around `batch_isend_irecv()` calls

### 2. `scripts/llama_ring_sg/benchmark_ring.py`
Benchmark script comparing Ring vs Regular Attention:
- Measures TTFT (prefill), decode time, and compute/comm ratio
- Outputs results to CSV for comparison tables

### 3. `run_all_benchmarks.sh`
Shell script to run benchmarks across token counts (100, 500, 1000, 1500, 2000).

## Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time To First Token (prefill) |
| Avg Decode | Average time per decoded token |
| Comm Time | P2P communication time (Ring only) |
| Compute/Comm Ratio | Compute time / Communication time |

## Usage

```bash
# Full suite
bash run_all_benchmarks.sh

# Single run
torchrun --nproc_per_node=2 scripts/llama_ring_sg/benchmark_ring.py \
    --num_tokens 1000 --summary_csv results.csv
```

## Output

Results saved to `benchmark_results/summary_<timestamp>.csv`
