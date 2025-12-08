#!/usr/bin/env bash
set -e

# 1) Go to desired workspace
cd /workspace

# 2) Clone repo if not already present
if [ ! -d "hpml-heterogeneous-ring-attention" ]; then
  git clone https://github.com/chrislee8684/hpml-heterogeneous-ring-attention.git
fi

cd hpml-heterogeneous-ring-attention/external/ibm_fms_ring

# 3) Install Python deps
pip install -r requirements.txt

# 4) Download Llama model
huggingface-cli download meta-llama/Llama-3.1-8B \
  --include "original/*" \
  --local-dir /workspace/llama-hf \
  --local-dir-use-symlinks False