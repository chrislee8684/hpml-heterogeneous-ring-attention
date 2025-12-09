#!/bin/bash
set -e

# Kill children on Ctrl+C
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Heterogeneous MPS slowdown settings (same meaning as your existing script)
STRONG_GPU_PERCENTAGE=${STRONG_GPU_PERCENTAGE:-100}
WEAK_GPU_PERCENTAGE=${WEAK_GPU_PERCENTAGE:-10}

# Sequence lengths to test (edit this list as you like)
SEQ_LENS=(256 512 1024 2048 4096 8192 16384 32768 65536)

export WORLD_SIZE=2
export MASTER_ADDR='localhost'
export MASTER_PORT='29500'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR/../:$PYTHONPATH"

TEST_SCRIPT="$SCRIPT_DIR/test_ring_prefill.py"

echo "Using STRONG_GPU_PERCENTAGE=$STRONG_GPU_PERCENTAGE, WEAK_GPU_PERCENTAGE=$WEAK_GPU_PERCENTAGE"
echo

# Table header
printf "%-8s  %-12s  %-12s  %-12s\n" "seq_len" "even_ms" "prop_ms" "prop/even"
echo "-------------------------------------------------------------"

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    # echo "================ SEQ_LEN = $SEQ_LEN ================"

    ########################################
    # 1) EVEN SHARDING
    ########################################
    # echo "[Even] Starting benchmark with total sequence length: $SEQ_LEN"
    # echo "Rank 0 MPS: $STRONG_GPU_PERCENTAGE%"
    # echo "Rank 1 MPS: $WEAK_GPU_PERCENTAGE%"

    TMP_OUT=$(mktemp)

    # Rank 0
    export RANK=0
    export LOCAL_RANK=0
    SHARD_MODE=even CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
    python3 "$TEST_SCRIPT" --total_seq_len "$SEQ_LEN" >> "$TMP_OUT" 2>&1 &

    # Rank 1
    export RANK=1
    export LOCAL_RANK=1
    SHARD_MODE=even CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
    python3 "$TEST_SCRIPT" --total_seq_len "$SEQ_LEN" >> "$TMP_OUT" 2>&1 &

    wait

    # Extract the Global max time line (only printed by rank 0)
    EVEN_TIME=$(grep "Global max time per call" "$TMP_OUT" | awk 'NR==1 {print $(NF-1)}')

    # echo "[Even] Global max time per call: ${EVEN_TIME} ms"
    # echo

    ########################################
    # 2) PROPORTIONAL SHARDING
    ########################################
    : > "$TMP_OUT"  # truncate the temp file

    # echo "[Proportional] Starting benchmark with total sequence length: $SEQ_LEN"
    # echo "Rank 0 MPS: $STRONG_GPU_PERCENTAGE%"
    # echo "Rank 1 MPS: $WEAK_GPU_PERCENTAGE%"

    # Rank 0
    export RANK=0
    export LOCAL_RANK=0
    SHARD_MODE=proportional CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$STRONG_GPU_PERCENTAGE \
    python3 "$TEST_SCRIPT" --total_seq_len "$SEQ_LEN" >> "$TMP_OUT" 2>&1 &

    # Rank 1
    export RANK=1
    export LOCAL_RANK=1
    SHARD_MODE=proportional CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$WEAK_GPU_PERCENTAGE \
    python3 "$TEST_SCRIPT" --total_seq_len "$SEQ_LEN" >> "$TMP_OUT" 2>&1 &

    wait

    PROP_TIME=$(grep "Global max time per call" "$TMP_OUT" | awk 'NR==1 {print $(NF-1)}')

    # echo "[Proportional] Global max time per call: ${PROP_TIME} ms"
    # echo

    rm -f "$TMP_OUT"

    # Compute ratio prop/even (use python for floating division)
    RATIO=$(python3 - <<EOF
even_time = float("${EVEN_TIME}")
prop_time = float("${PROP_TIME}")
print(f"{prop_time / even_time:.3f}")
EOF
)

    # Print table row
    printf "%-8d  %-12s  %-12s  %-12s\n" "$SEQ_LEN" "$EVEN_TIME" "$PROP_TIME" "$RATIO"
done

echo
echo "Done."