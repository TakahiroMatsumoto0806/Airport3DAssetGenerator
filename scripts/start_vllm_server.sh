#!/bin/bash
# vLLM サーバー起動スクリプト (DGX Spark / aarch64 対応)
#
# 使い方:
#   bash scripts/start_vllm_server.sh
#
# 環境変数でカスタマイズ可能:
#   MODELS_DIR         : モデルディレクトリ（デフォルト: ~/models）
#   VLLM_PORT          : vLLM ポート番号（デフォルト: 8001）
#   SERVED_MODEL_NAME  : vLLM が登録する短い識別子（デフォルト: qwen3-vl-32b）
#                        configs/pipeline_config.yaml の vlm.model_name と一致させる
#   VLLM_GPU_UTIL      : GPU メモリ確保率（デフォルト: 0.85）
#                        DGX Spark の統合メモリでは 0.90 だと page cache 残量次第で OOM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- パス解決（環境変数 or デフォルト）----
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
MODEL_PATH="${MODELS_DIR}/Qwen3-VL-32B-Instruct"
VLLM_PORT="${VLLM_PORT:-8001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-32b}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"

# ---- venv のアクティベート ----
VENV="${PROJECT_ROOT}/.venv"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"

    # NVIDIA ライブラリパスを設定（DGX Spark / aarch64）
    TORCH_LIB="${VENV}/lib/python3.11/site-packages/torch/lib"
    NVIDIA_LIBS=$(find "${VENV}/lib/python3.11/site-packages/nvidia" \
        -name "lib" -type d 2>/dev/null | tr '\n' ':')
    export LD_LIBRARY_PATH="${TORCH_LIB}:${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
fi

# ---- モデルパス解決 ----
if [ ! -d "$MODEL_PATH" ]; then
    # フォールバック: Hugging Face Hub から直接参照
    MODEL_PATH="Qwen/Qwen3-VL-32B-Instruct"
    echo "[INFO] ローカルモデルが見つかりません。Hub から参照: ${MODEL_PATH}"
else
    echo "[INFO] モデル: ${MODEL_PATH}"
fi

echo "[INFO] vLLM サーバー起動: port=${VLLM_PORT}, served-model-name=${SERVED_MODEL_NAME}, gpu-util=${VLLM_GPU_UTIL}"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
    --port "${VLLM_PORT}" \
    "$@"
