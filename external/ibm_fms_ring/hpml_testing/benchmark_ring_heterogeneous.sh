#!/bin/bash

# Kill background jobs on exit
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# ============================================================
# Input parameter: sequence length
# ============================================================
SEQ_LEN=${1:-20000}

# GPU throttle values (can be overridden externally)
STRONG_GPU_PERCENTAGE=${STRONG_GPU_PERCENTAGE:-100}
WEAK_GPU_PERCENTAGE=${WEAK_GPU_PERCENTAGE:-10}

# ============================================================
# Resolve actual script directory
# ============================================================
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Canonical path to test script
TEST_SCRIPT="$SCRIPT_DIR/test_ring_prefill.py"

# ============================================================
# Sanity check: file must exist
# ============================================================
if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "❌ ERROR: test_ring_prefill.py not found at:"
    echo "   $TEST_SCRIPT"
    echo ""
    echo "Run this to see where your files actually are:"
    echo "   grep -R \"test_ring_prefill.py\" -n ."
    exit 1
fi

echo "Using test script: $TEST_SCRIPT"
echo "-----------------------------------------------------"

# ============================================================
# Distributed configuration
# ============================================================
export WORLD_SIZE=2
export MASTER_ADDR='localhost'
export MASTER_PORT='29500'
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

# ============================================================
# EVEN SHARDING RUN
# ============================================================
echo "====================================================="
echo "Running EVEN SHARDING benchmark"
echo "Seq Len: $SEQ_LEN"
echo "GPU0 throttle: $STRONG_GPU_PERCENTAGE%"
echo "GPU1 throttle: $WEAK_GPU_PERCENTAGE%"
echo "====================================================="

# Rank 0
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
python3 "$TEST_SCRIPT" \
    --total_seq_len "$SEQ_LEN" &

# Rank 1
export RANK=1
export LOCAL_RANK=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
python3 "$TEST_SCRIPT" \
    --total_seq_len "$SEQ_LEN" &

wait

echo "Even sharding benchmark complete."
echo "====================================================="


# ============================================================
# PROPORTIONAL SHARDING RUN
# ============================================================
echo "====================================================="
echo "Running PROPORTIONAL SHARDING benchmark"
echo "Seq Len: $SEQ_LEN"
echo "GPU0 throttle: $STRONG_GPU_PERCENTAGE%"
echo "GPU1 throttle: $WEAK_GPU_PERCENTAGE%"
echo "====================================================="

# Rank 0
export RANK=0
export LOCAL_RANK=0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
python3 "$TEST_SCRIPT" \
    --total_seq_len "$SEQ_LEN" \
    --proportional_sharding &

# Rank 1
export RANK=1
export LOCAL_RANK=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
python3 "$TEST_SCRIPT" \
    --total_seq_len "$SEQ_LEN" \
    --proportional_sharding &

wait

echo "Proportional sharding benchmark complete."
echo "====================================================="
echo "All benchmarks finished."
