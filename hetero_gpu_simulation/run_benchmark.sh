#!/usr/bin/env bash
set -euo pipefail

# Config
MPS_PERCENT_GPU1=30      # GPU 1 compute limit
MEM_FRACTION_GPU1=0.10   # GPU 1 memory limit
SIZE=8192                # Matrix size
ITERS=200                # Iterations

# Check for 2 GPUs
if [ $(nvidia-smi -L | wc -l) -lt 2 ]; then
    echo "Error: Need 2 GPUs"
    exit 1
fi

# Start MPS daemon if needed
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mps_$USER}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/mps_$USER}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

if ! pgrep -u "$USER" -f "nvidia-cuda-mps-control -d" >/dev/null 2>&1; then
    nvidia-cuda-mps-control -d
    sleep 2
fi

echo "GPU 0: Full performance (100% compute, 100% memory)"
echo "GPU 1: Limited to ${MPS_PERCENT_GPU1}% of warp schedulers (compute throttling) and ${MEM_FRACTION_GPU1} of VRAM capacity"

# Run GPU 0 (full performance) in background
(
    unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE || true
    python benchmark.py --device 0 --iters $ITERS --m $SIZE --n $SIZE --k $SIZE
) &
PID0=$!

# Run GPU 1 (limited) in background
(
    export CUDA_VISIBLE_DEVICES=1
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PERCENT_GPU1
    python benchmark.py --device 0 --iters $ITERS --m $SIZE --n $SIZE --k $SIZE \
        --mem_fraction $MEM_FRACTION_GPU1
) &
PID1=$!

# Wait for both
wait $PID0
wait $PID1