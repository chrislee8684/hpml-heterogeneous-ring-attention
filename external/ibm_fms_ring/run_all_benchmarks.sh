#!/bin/bash
set -e

MODEL_PATH="/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
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
    --variant 3-8b \
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