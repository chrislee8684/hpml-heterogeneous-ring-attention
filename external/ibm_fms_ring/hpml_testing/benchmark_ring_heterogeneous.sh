#!/bin/bash

# This script manually launches two distributed processes for the ring attention
# benchmark, bypassing torchrun to allow for setting per-process environment
# variables for MPS-based GPU throttling.

# Ensure background processes are killed on script exit
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# --- Configuration ---
STRONG_GPU_PERCENTAGE=${STRONG_GPU_PERCENTAGE:-100}
WEAK_GPU_PERCENTAGE=${WEAK_GPU_PERCENTAGE:-10}
SEQ_LEN=${SEQ_LEN:-32768}
# --- End Configuration ---

# Set up environment for torch.distributed
export WORLD_SIZE=2
export MASTER_ADDR='localhost'
export MASTER_PORT='29500'  # Must be free

# Resolve script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

echo "====================================================="
echo "Starting EVEN SHARDING benchmark"
echo "Total sequence length: $SEQ_LEN"
echo "Rank 0 throttle: $STRONG_GPU_PERCENTAGE%"
echo "Rank 1 throttle: $WEAK_GPU_PERCENTAGE%"
echo "====================================================="

# --- EVEN SHARDING RUN ---
# Rank 0 (Strong GPU)
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

# Rank 1 (Weak GPU)
export RANK=1
export LOCAL_RANK=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

wait

echo "====================================================="
echo "EVEN SHARDING benchmark finished"
echo "====================================================="


# Reuse same MASTER_PORT + WORLD_SIZE (two serial runs, no conflict)
echo "====================================================="
echo "Starting PROPORTIONAL SHARDING benchmark"
echo "Total sequence length: $SEQ_LEN"
echo "Rank 0 throttle: $STRONG_GPU_PERCENTAGE%"
echo "Rank 1 throttle: $WEAK_GPU_PERCENTAGE%"
echo "====================================================="

# --- PROPORTIONAL SHARDING RUN ---
# Rank 0 (Strong GPU)
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN \
    --proportional_sharding &

# Rank 1 (Weak GPU)
export RANK=1
export LOCAL_RANK=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN \
    --proportional_sharding &

wait

echo "====================================================="
echo "PROPORTIONAL SHARDING benchmark finished"
echo "====================================================="
echo "All benchmarks completed."
