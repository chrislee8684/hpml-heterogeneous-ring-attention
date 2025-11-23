# HPML Fall 2025 

## New Contributions

### 1. GPU Interconnect Speed Measurement (`test_gpu_communication.py`)

Measures GPU-to-GPU communication bandwidth using ring shift operations to characterize interconnect performance.

**Usage:**
```bash
torchrun --nproc_per_node=2 test_gpu_communication.py
```

**Output:**
- Per-GPU timing statistics
- Measured bandwidth vs theoretical limits
- Interconnect type identification

---

### 2. Comprehensive Benchmark Suite (`run_all_benchmarks.sh`)

Automated benchmarking script that compares Ring Attention vs Regular Attention across varying context lengths.

**Usage:**
```bash
bash run_all_benchmarks.sh
```

**Output Files:**
- `benchmark_results/benchmark_results.csv` - Raw data
- `benchmark_results/benchmark_table.md` - Formatted results table
- `benchmark_results/logs/` - Individual run logs

**Metrics Captured:**
- Number Count (input prompt parameter)
- Input Tokens (actual tokenized length)
- GPUs used
- Average time/token for Ring Attention (ms)
- Average time/token for Regular Attention (ms)
- Speedup (Regular/Ring ratio)
- Output correctness validation

---

## Model Configuration

Both benchmarks use:
- **Model:** Meta-Llama-3.1-8B
- **Architecture:** LLaMA with 32 attention heads
- **Precision:** FP16
- **Backend:** NCCL (CUDA)

---

## Setup Instructions

### Prerequisites

1. **GPU Cluster Access**: Ensure you have access to a GPU cluster (e.g., Columbia Insomnia cluster with NVIDIA L40S GPUs)

2. **Install Dependencies**: From the root of the repository, install all required packages:
   ```bash
   pip install -e .
   ```

3. **Download Model and Tokenizer**:
   - Download the Meta-Llama-3.1-8B model from Hugging Face
   - Update the `MODEL_PATH` in `run_all_benchmarks.sh` to point to your model location:
     ```bash
     MODEL_PATH="/path/to/your/llama3/model"
     ```
   - The script expects the model path to include both model weights and tokenizer files

### Running the Benchmark Script

From the root of the repository:

```bash
bash run_all_benchmarks.sh
```

**What the script does:**
1. Creates `benchmark_results/` directory with logs subdirectory
2. Runs benchmarks across multiple context lengths (100, 500, 1000, 1500, 2000 tokens)
3. Tests both Ring Attention and Regular Attention implementations
4. Validates output correctness between implementations
5. Generates performance comparison results

**Note:** The script is currently configured to use 2 GPUs per benchmark (`--nproc_per_node=2`). You can modify this in the script by changing the GPU parameter in the `run_benchmark` function.

---

## Things to be Improved

### Current Benchmark Limitations

1. **KV Cache Not Enabled**

2. **Measuring Prefill + Decode Together (Not Just Decode)**


