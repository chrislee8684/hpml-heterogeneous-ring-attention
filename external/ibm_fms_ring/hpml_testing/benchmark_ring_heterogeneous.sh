#!/bin/bash

# Kill background processes on script exit
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# ---------------- CONFIG ----------------
STRONG_GPU_PERCENTAGE=${STRONG_GPU_PERCENTAGE:-100}
WEAK_GPU_PERCENTAGE=${WEAK_GPU_PERCENTAGE:-10}

SEQ_LEN=${SEQ_LEN:-32768}                # Full sequence length for even sharding
PROPORTIONAL_SEQ_LEN=${PROPORTIONAL_SEQ_LEN:-16384}  # Reduced length for proportional run
# ----------------------------------------

export WORLD_SIZE=2
export MASTER_ADDR='localhost'
export MASTER_PORT='29500'  # must be free

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"


echo "====================================================="
echo "Running EVEN SHARDING (2 physical GPUs)"
echo "Seq Len: $SEQ_LEN"
echo "GPU0 throttle: $STRONG_GPU_PERCENTAGE%"
echo "GPU1 throttle: $WEAK_GPU_PERCENTAGE%"
echo "====================================================="

# -------- EVEN SHARDING --------

# Rank 0 → GPU0
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

# Rank 1 → GPU1
export RANK=1
export LOCAL_RANK=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

wait
echo "Even sharding complete."
echo "====================================================="



echo "====================================================="
echo "Running PROPORTIONAL SHARDING (2 physical GPUs)"
echo "Reduced Seq Len: $PROPORTIONAL_SEQ_LEN"
echo "GPU0 throttle: $STRONG_GPU_PERCENTAGE%"
echo "GPU1 throttle: $WEAK_GPU_PERCENTAGE%"
echo "====================================================="

# -------- PROPORTIONAL SHARDING --------

# Rank 0 → GPU0
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $PROPORTIONAL_SEQ_LEN \
    --proportional_sharding &

# Rank 1 → GPU1
export RANK=1
export LOCAL_RANK=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $PROPORTIONAL_SEQ_LEN \
    --proportional_sharding &

wait
echo "Proportional sharding complete."
echo "====================================================="

echo "All benchmarks finished."
