#!/bin/bash
#SBATCH -p gpu-preempt                  # Partition (queue)
#SBATCH -t 0:02:00              # Time limit: 2 minutes
#SBATCH --gpus=2                # Number of GPUs 
#SBATCH --constraint=a100       # Request specific GPU type 
#SBATCH --mem=32G               # Memory per node
#SBATCH -N 1                    # Ensure both GPUs on same node
#SBATCH -o slurm-%j.out         # Output log
#SBATCH -e slurm-%j.err         # Error log
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notification
#SBATCH --job-name=gpu_comm_test    # Job name

# Load required modules
module load conda/latest
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6

# Activate conda environment
conda activate changemamba

nvidia-smi topo -m
# Run the GPU communication test with 2 GPUs
MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT test_gpu_communication.py