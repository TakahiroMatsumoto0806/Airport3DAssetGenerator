#!/bin/bash
# vLLM サーバー起動スクリプト (DGX Spark / aarch64 対応)

VENV="/home/ntt/WorkSpace/Airport3DAssetGenerator/al3dg/.venv"
TORCH_LIB="$VENV/lib/python3.11/site-packages/torch/lib"

# NVIDIA ライブラリパスを収集
NVIDIA_LIBS=$(find "$VENV/lib/python3.11/site-packages/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')

export LD_LIBRARY_PATH="$TORCH_LIB:$NVIDIA_LIBS${LD_LIBRARY_PATH:-}"

source "$VENV/bin/activate"

exec vllm serve /home/ntt/models/Qwen3-VL-32B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8001 \
  "$@"
