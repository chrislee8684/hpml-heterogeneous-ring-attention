#!/bin/bash

# This script manually launches two distributed processes for the ring attention
# benchmark, bypassing torchrun to allow for setting per-process environment
# variables for MPS-based GPU throttling.

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# --- Configuration ---
STRONG_GPU_PERCENTAGE=${STRONG_GPU_PERCENTAGE:-100}
WEAK_GPU_PERCENTAGE=${WEAK_GPU_PERCENTAGE:-10}
SEQ_LEN=${SEQ_LEN:-16384}   # safer default than 32768
# --- End Configuration ---

export WORLD_SIZE=2
export MASTER_ADDR='localhost'
export MASTER_PORT='29500'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

########################
# 1) EVEN / "REGULAR"
########################
echo "-------------------------Regular (even sharding)--------------------------"
echo "Starting benchmark with total sequence length: $SEQ_LEN"
echo "Rank 0 (Strong GPU) MPS: $STRONG_GPU_PERCENTAGE%"
echo "Rank 1 (Strong GPU) MPS: $STRONG_GPU_PERCENTAGE%"
echo "--------------------------------------------------------------------"

# Rank 0
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

# Rank 1
export RANK=1
export LOCAL_RANK=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

wait
echo "--------------------------------------------------------------------"
echo "Benchmark Regular (even sharding) finished."
echo

########################
# 2) PROPORTIONAL
########################
echo "-------------------------Proportion (hetero sharding)--------------------------"
echo "Starting benchmark with total sequence length: $SEQ_LEN"
echo "Rank 0 (Strong GPU) MPS: $STRONG_GPU_PERCENTAGE%"
echo "Rank 1 (Weak GPU)   MPS: $WEAK_GPU_PERCENTAGE%"
echo "--------------------------------------------------------------------"

# Rank 0
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

# Rank 1
export RANK=1
export LOCAL_RANK=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

wait
echo "--------------------------------------------------------------------"
echo "Benchmark Proportion (hetero sharding) finished."