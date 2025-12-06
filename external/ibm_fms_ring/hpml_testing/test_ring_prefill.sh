#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH -t 0:05:00
#SBATCH --gpus-per-node=2
#SBATCH --mem=16G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --job-name=ring_test
#SBATCH --nodes=1

set -e

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /work/pi_jaimedavila_umass_edu/asadhgauri_umass_edu/hpml/hetero_gpu_inference/hpml-heterogeneous-ring-attention/external/ibm_fms_ring/temp_testing
torchrun --nproc_per_node=2 test_ring_prefill.py