#!/bin/bash
#SBATCH -p gpu                  # Partition (queue)
#SBATCH -t 0:15:00              # Time limit: 15 minutes (very low GPU time)
#SBATCH --gpus=2                # Number of GPUs (need 2 for communication test)
#SBATCH --constraint=a100       # Request specific GPU type
#SBATCH --mem=16G               # Memory per node
#SBATCH -o slurm-%j.out         # Standard output log
#SBATCH -e slurm-%j.err         # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notification
#SBATCH --job-name=ring_attention    # Job name
set -e

MODEL_PATH="/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmark_results"
NUM_GPUS=2

LOG_FILE="$RESULTS_DIR/run_${TIMESTAMP}.log"
CSV_FILE="$RESULTS_DIR/results_${TIMESTAMP}.csv"
MD_FILE="$RESULTS_DIR/results_${TIMESTAMP}.md"

mkdir -p "$RESULTS_DIR"

echo "Starting benchmark at $(date)" | tee "$LOG_FILE"
echo "Logging to: $LOG_FILE"
echo ""

# Run benchmark
torchrun --nproc_per_node=$NUM_GPUS scripts/llama_ring_sg/benchmark_ring.py \
    --architecture llama \
    --variant 3.2-1b \
    --model_path "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --device_type cuda \
    --num_tokens_to_benchmark 30 \
    --batch_size 1 \
    --run_ring_first \
    --csv_output_file "$CSV_FILE" \
    2>&1 | tee -a "$LOG_FILE"

# Generate markdown table
{
    echo "# Benchmark Results - $(date +"%Y-%m-%d %H:%M")"
    echo ""
    echo "| Strategy | Prompt N | TTFT (ms) | Avg Decode (ms) | Total (ms) |"
    echo "|----------|----------|-----------|-----------------|------------|"
    tail -n +2 "$CSV_FILE" | while IFS=',' read strategy n ttft avg total; do
        printf "| %s | %s | %s | %s | %s |\n" "$strategy" "$n" "$ttft" "$avg" "$total"
    done
} > "$MD_FILE"

echo ""
echo "Done. Files:"
echo "  $LOG_FILE"
echo "  $CSV_FILE"
echo "  $MD_FILE"
cat "$MD_FILE"