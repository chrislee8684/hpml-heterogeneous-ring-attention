#!/usr/bin/env bash
#
# Sweep ring-attention prefill timings across sequence lengths and MPS throttling.
# Uses hpml_testing/benchmark_ring_heterogeneous.sh (2 ranks) and parses the output
# from test_ring_prefill.py for "Avg time per call".

set -euo pipefail

# Power-of-two sequence lengths up to 32K by default
SEQLENS=${SEQLENS:-"32 64 128 256 512 1024 2048 4096 8192 16384 32768"}
# Include 100% for the homogeneous baseline, then throttled weak-GPU percentages
WEAK_PCTS=${WEAK_PCTS:-"100 10 20 40 60 80"}
STRONG_PCT=${STRONG_PCT:-100}

RESULTS_DIR=${RESULTS_DIR:-"hpml_testing/results"}
LOG_DIR="${RESULTS_DIR}/logs"
OUT_CSV=${OUT_CSV:-"${RESULTS_DIR}/mps_sweep_$(date +%Y%m%d_%H%M%S).csv"}

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

tmp_log=$(mktemp)
echo "weak_pct,seq_len,avg_ms" > "${OUT_CSV}"

for weak in ${WEAK_PCTS}; do
  for L in ${SEQLENS}; do
    echo "=== weak=${weak}% seq_len=${L} ==="
    # Run the benchmark with the requested throttling and seq len
    STRONG_GPU_PERCENTAGE=${STRONG_PCT} WEAK_GPU_PERCENTAGE=${weak} SEQ_LEN=${L} \
      bash hpml_testing/benchmark_ring_heterogeneous.sh | tee "${tmp_log}"

    avg_ms=$(grep -oE "Avg time per call: [0-9.]+" "${tmp_log}" | tail -1 | awk '{print $5}')
    if [[ -z "${avg_ms}" ]]; then
      avg_ms="NaN"
    fi
    echo "${weak},${L},${avg_ms}" >> "${OUT_CSV}"

    cp "${tmp_log}" "${LOG_DIR}/weak${weak}_L${L}.log"
  done
done

rm -f "${tmp_log}"
echo "Saved summary CSV to ${OUT_CSV}"
