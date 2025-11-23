#!/bin/bash
set -e

# Configuration
MODEL_PATH="$HOME/llama3"
TOKENIZER_PATH="$HOME/llama3"
RESULTS_DIR="benchmark_results"
LOGS_DIR="$RESULTS_DIR/logs"
CSV_FILE="$RESULTS_DIR/benchmark_results.csv"

# Setup
mkdir -p "$LOGS_DIR"
echo "Number_Count,Input_Tokens,GPUs,Avg_Time_Token_Ring_ms,Avg_Time_Token_Regular_ms,Speedup,Outputs_Match" > "$CSV_FILE"

# Parse functions
parse_time() { 
    grep -A 3 "Summary for $2:" "$1" | grep "Average time per token:" | awk '{print $5}'
}

parse_input_tokens() { 
    grep "Input Seq length:" "$1" | awk '{print $NF}'
}

check_correctness() { 
    grep -q "AssertionError\|FAILED" "$1" && echo "FAILED" || echo "PASSED"
}

# Run benchmark
run_benchmark() {
    local log="$LOGS_DIR/benchmark_numcount${1}_gpu${2}.log"
    echo "Running: Number Count=$1, GPUs=$2"

    set +e
    torchrun --nproc_per_node=$2 scripts/llama_ring/benchmark_ring.py \
        --architecture llama \
        --variant 3-8b \
        --model_path "$MODEL_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --device_type cuda \
        --num_tokens_to_benchmark 30 \
        --batch_size 1 \
        --run_ring_first \
        --prompt_len $1 \
        > "$log" 2>&1
    set -e

    local ring=$(parse_time "$log" "Ring Attention")
    local reg=$(parse_time "$log" "Regular Attention")
    local input_tokens=$(parse_input_tokens "$log")
    local speedup="N/A"
    
    [[ -n "$ring" && -n "$reg" ]] && speedup=$(echo "scale=2; $reg / $ring" | bc)

    echo "$1,${input_tokens:-N/A},$2,${ring:-ERROR},${reg:-ERROR},$speedup,$(check_correctness "$log")" >> "$CSV_FILE"
    echo "  Input Tokens: ${input_tokens:-N/A} | Ring: ${ring:-ERROR} ms/token | Regular: ${reg:-ERROR} ms/token | Speedup: ${speedup}x"
}

# Run benchmarks (500 1000 1500 2000)
for num_count in 100 500 1000 1500 2000; do 
    run_benchmark $num_count 2
done


# Generate markdown table
{
    echo -e "# Ring Attention Benchmark Results\n\n| Number Count | Input Tokens | GPUs | Ring (ms) | Regular (ms) | Speedup | Match |"
    echo "|--------------|--------------|------|-----------|--------------|---------|-------|"
    
    tail -n +2 "$CSV_FILE" | while IFS=',' read num_count tokens gpu ring reg spd corr; do
        [[ "$ring" != "ERROR" ]] && ring="$ring ms"
        [[ "$reg" != "ERROR" ]] && reg="$reg ms"
        [[ "$spd" != "N/A" ]] && spd="${spd}x"
        echo "| $num_count | $tokens | $gpu | $ring | $reg | $spd | $corr |"
    done
} > "$RESULTS_DIR/benchmark_table.md"

cat "$RESULTS_DIR/benchmark_table.md"
echo -e "\nCSV: $CSV_FILE | Logs: $LOGS_DIR/"