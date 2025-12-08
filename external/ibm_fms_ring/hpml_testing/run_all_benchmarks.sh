#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH -t 0:10:00
#SBATCH --gpus-per-node=2
#SBATCH --constraint=a100
#SBATCH --mem=26G
#SBATCH -o ../logs/slurm-%j.out
#SBATCH -e ../logs/slurm-%j.err
#SBATCH --job-name=ring_bench
#SBATCH --nodes=1
set -e
#"/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
#"/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
#"/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
MODEL_PATH="/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmark_results"
NUM_GPUS=2

module load conda/latest
conda activate context_parallelism
module load cuda/12.6

SUMMARY_CSV="$RESULTS_DIR/summary_${TIMESTAMP}.csv"
LOG_FILE="$RESULTS_DIR/run_${TIMESTAMP}.log"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
mkdir -p "$RESULTS_DIR"

# Check NVLink topology 
nvidia-smi topo -m


run_benchmark() {
      local num_tokens=$1
      local num_gpus=$2

      torchrun --nproc_per_node=$num_gpus ../scripts/llama_ring_sg/benchmark_ring.py \
          --architecture llama \
          --variant 3.2-1b \
          --model_path "$MODEL_PATH" \
          --device_type cuda \
          --num_tokens $num_tokens \
          --num_decode_tokens 0 \
          --batch_size 1 \
          --run_ring_first \
          --summary_csv "$SUMMARY_CSV" \
          --disable_flash \
          
  }

echo "Ring Attention Benchmark - $(date)" | tee "$LOG_FILE"

#512 1024 4096 8192 16384 32768 65536
for num_count in 256 512 1024 4096 ; do
    echo "Running: $num_count tokens" | tee -a "$LOG_FILE"
    run_benchmark $num_count $NUM_GPUS 2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $SUMMARY_CSV" | tee -a "$LOG_FILE"
cat "$SUMMARY_CSV"
