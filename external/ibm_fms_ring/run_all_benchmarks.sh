#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0:30:00
#SBATCH --gpus=2
#SBATCH --constraint=a100
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --job-name=ring_bench
set -e

MODEL_PATH="/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmark_results"
NUM_GPUS=2

SUMMARY_CSV="$RESULTS_DIR/summary_${TIMESTAMP}.csv"
LOG_FILE="$RESULTS_DIR/run_${TIMESTAMP}.log"

mkdir -p "$RESULTS_DIR"

run_benchmark() {
    local num_tokens=$1
    local num_gpus=$2

    torchrun --nproc_per_node=$num_gpus scripts/llama_ring_sg/benchmark_ring.py \
        --architecture llama \
        --variant 3.2-1b \
        --model_path "$MODEL_PATH" \
        --tokenizer "$MODEL_PATH" \
        --device_type cuda \
        --num_tokens $num_tokens \
        --num_decode_tokens 30 \
        --batch_size 1 \
        --run_ring_first \
        --summary_csv "$SUMMARY_CSV"
}

echo "Ring Attention Benchmark - $(date)" | tee "$LOG_FILE"

for num_count in 100 500 1000 1500 2000; do
    echo "Running: $num_count tokens" | tee -a "$LOG_FILE"
    run_benchmark $num_count $NUM_GPUS 2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $SUMMARY_CSV" | tee -a "$LOG_FILE"
cat "$SUMMARY_CSV"
