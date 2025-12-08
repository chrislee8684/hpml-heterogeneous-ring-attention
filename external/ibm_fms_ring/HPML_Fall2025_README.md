# HPML Fall 2025

## Code Modifications

1. `fms/distributed/strategy.py`
    - Modified RingAttentionStrategy to use async P2P communications
    - Added dedicated CUDA stream for communication (`_comm_stream`)
    - `ring_shift_kv_async()` starts P2P transfers on comm stream, returns immediately
    - `ring_shift_kv_wait()` waits for transfers to complete
    - Comm runs on separate stream so it overlaps with compute on default stream

2. `fms/distributed/ring_attention.py`
    - Ring attention loop with async comm/compute overlap
    - Uses online softmax to accumulate attention across KV blocks from different GPUs
    - Diagonal blocks use `F.scaled_dot_product_attention` (FlashAttention when available)
    - Off-diagonal blocks use a custom Triton kernel that computes block-wise softmax statistics (`z`, `l`, `m`) and fuses QKᵀ, softmax, and V into a single pass, with a naive PyTorch fallback for small blocks
    - CUDA event-based timing to measure comm vs compute time per stream (including separate diag / off-diag timings)

3. `fms/distributed/triton_offdiag_block.py`
    - Triton implementation for off-diagonal attention blocks
    - Tiles over queries and keys (BLOCK_Q × BLOCK_K) and returns block-wise softmax stats used by the ring online softmax
    - Enabled only when the block size is large enough; small blocks fall back to the naive PyTorch implementation

4. `scripts/llama_ring_sg/benchmark_ring.py`
    - Benchmarks Ring Attention vs Regular Attention
    - `--disable_flash` flag to use naive attention for fair comparison
    - `--num_tokens` to set prompt length
    - Outputs TTFT, decode time to CSV

5. `hpml_testing/run_all_benchmarks.sh`
    - Runs benchmarks across token counts (256, 512, 1024, 4096)
    - Outputs results to `hpml_testing/benchmark_results/summary_<timestamp>.csv`

6. `hpml_testing/benchmark_ring_heterogeneous.sh`
    - Benchmarks ring attention on heterogeneous GPUs with even vs proportional sharding
    - Reports per-layer comm, diag, and off-diag compute times and supports long-context experiments via the `SEQ_LEN` environment variable

## Usage

### Setup

```bash
# Create conda environment
conda create -n context_parallelism python=3.11
conda activate context

# Install dependencies
cd hpml-heterogeneous-ring-attention
pip install -e .
```

### Running benchmarks

```bash
cd hpml-heterogeneous-ring-attention/external/ibm_fms_ring/hpml_testing


# Runs Ring attention and regular attention and compares both
# File will have to be modified with correct parameters such as model and tokenizer path
sbatch hpml_testing/run_all_benchmarks.sh```