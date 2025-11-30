# HPML Fall 2025

## Code Modifications

1. `fms/distributed/strategy.py`
    - Modified RingAttentionStrategy to use async P2P communications
2. `fms/distributed/llama_ring.py`
    - Modified ring loop with async overlap and online softmax
3. `temp_testing/`
    - `test_ring_prefill.py`: test for ring attention without model loading
    - `test_ring_prefill.sh`: SLURM script to run test on 2 GPUs
4. `run_all_benchmarks.sh` (doesn't work right now. need to add option to only run prefill and not decode)
    - Runs benchmarks across token counts (256, 512, 1024, 4096, 8192, 16384, 32768, 65536)
    - Measures TTFT (prefill time), decode time, and comm time
    - Outputs results to `benchmark_results/summary_<timestamp>.csv`

## Usage

### First install requirements

```bash
cd hpml-heterogeneous-ring-attention
pip install -e .
```

```bash
# Simple ring attention test (no model)
cd hpml-heterogeneous-ring-attention/external/ibm_fms_ring
torchrun --nproc_per_node=2 hpml_testing/test_ring_prefill.py

# Full benchmark suite with model
sbatch run_all_benchmarks.sh