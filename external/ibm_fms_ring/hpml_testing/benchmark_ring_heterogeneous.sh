#!/bin/bash

# Kill children when script exits
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# ---------------- CONFIG ----------------
STRONG_GPU_PERCENTAGE=${STRONG_GPU_PERCENTAGE:-100}
WEAK_GPU_PERCENTAGE=${WEAK_GPU_PERCENTAGE:-10}

SEQ_LEN=${SEQ_LEN:-32768}
PROPORTIONAL_SEQ_LEN=${PROPORTIONAL_SEQ_LEN:-16384}   # smaller to avoid OOM
# ----------------------------------------

export WORLD_SIZE=2
export MASTER_ADDR='localhost'
export MASTER_PORT='29500'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

echo "====================================================="
echo "Running EVEN SHARDING benchmark"
echo "SEQ_LEN: $SEQ_LEN"
echo "====================================================="

# -------- EVEN SHARDING --------
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
    python3 "$SCRIPT_DIR/test_ring_prefill.py" \
        --total_seq_len $SEQ_LEN &

export RANK=1
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
    python3 "$SCRIPT_DIR/test_ring_prefill.py" \
        --total_seq_len $SEQ_LEN &

wait
echo "Even sharding complete."
echo "====================================================="


echo "====================================================="
echo "Running PROPORTIONAL SHARDING benchmark"
echo "SEQ_LEN: $PROPORTIONAL_SEQ_LEN  (reduced to avoid OOM)"
echo "====================================================="

# -------- PROPORTIONAL SHARDING --------
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
    python3 "$SCRIPT_DIR/test_ring_prefill.py" \
        --total_seq_len $PROPORTIONAL_SEQ_LEN \
        --proportional_sharding &

export RANK=1
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
    python3 "$SCRIPT_DIR/test_ring_prefill.py" \
        --total_seq_len $PROPORTIONAL_SEQ_LEN \
        --proportional_sharding &

wait
echo "Proportional sharding complete."
echo "====================================================="
echo "All benchmarks finished."
