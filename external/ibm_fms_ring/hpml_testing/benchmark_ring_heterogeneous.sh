#!/bin/bash

# This script manually launches two distributed processes for the ring attention
# benchmark, bypassing torchrun to allow for setting per-process environment
# variables for MPS-based GPU throttling.

# Ensure background processes are killed on script exit
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# --- Configuration ---
STRONG_GPU_PERCENTAGE=100
WEAK_GPU_PERCENTAGE=10
SEQ_LEN=32768
# --- End Configuration ---

# Get the directory where this script resides (hpml_testing/)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Set PYTHONPATH to the project root (parent of SCRIPT_DIR)
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

echo "Starting benchmark with total sequence length: $SEQ_LEN"
echo "Rank 0 (Strong GPU) will be set to $STRONG_GPU_PERCENTAGE%"
echo "Rank 1 (Weak GPU) will be set to $WEAK_GPU_PERCENTAGE%"
echo "-----------------------------------------------------"

# Launch Rank 0 (Strong GPU) in the background
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

# Launch Rank 1 (Weak GPU) in the background
export RANK=1
export LOCAL_RANK=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &


# Wait for both background processes to finish
wait

echo "-----------------------------------------------------"
echo "Benchmark finished."
