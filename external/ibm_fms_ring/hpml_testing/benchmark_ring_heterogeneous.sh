#!/bin/bash
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

STRONG_GPU_PERCENTAGE=${STRONG_GPU_PERCENTAGE:-10}
WEAK_GPU_PERCENTAGE=${WEAK_GPU_PERCENTAGE:-100}
SEQ_LEN=${SEQ_LEN:-131072}

export WORLD_SIZE=2
export MASTER_ADDR='localhost'
export MASTER_PORT='29500'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

########################
# 1) EVEN SHARDING, HETERO SPEEDS (100 / 10)
########################
echo "-------------------------Even sharding (hetero)--------------------------"
echo "Starting benchmark with total sequence length: $SEQ_LEN"
echo "Rank 0 MPS: $STRONG_GPU_PERCENTAGE%"
echo "Rank 1 MPS: $WEAK_GPU_PERCENTAGE%"
echo "--------------------------------------------------------------------"

# Rank 0
export RANK=0
export LOCAL_RANK=0
SHARD_MODE=even CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

# Rank 1
export RANK=1
export LOCAL_RANK=1
SHARD_MODE=even CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

wait
echo "--------------------------------------------------------------------"
echo "Benchmark Even (hetero) finished."
echo

########################
# 2) PROPORTIONAL SHARDING, SAME HETERO SPEEDS (100 / 10)
########################
echo "-------------------------Proportional sharding (hetero)--------------------------"
echo "Starting benchmark with total sequence length: $SEQ_LEN"
echo "Rank 0 MPS: $STRONG_GPU_PERCENTAGE%"
echo "Rank 1 MPS: $WEAK_GPU_PERCENTAGE%"
echo "--------------------------------------------------------------------"

# Rank 0
export RANK=0
export LOCAL_RANK=0
SHARD_MODE=proportional CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

# Rank 1
export RANK=1
export LOCAL_RANK=1
SHARD_MODE=proportional CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
python3 "$SCRIPT_DIR/test_ring_prefill.py" \
    --total_seq_len $SEQ_LEN &

wait
echo "--------------------------------------------------------------------"
echo "Benchmark Proportional (hetero) finished."