#!/usr/bin/env bash
# =============================================================================
# AL3DG 環境構築スクリプト
# 対象: DGX Spark (Ubuntu, Grace Blackwell GB10, 128GB unified memory, CUDA 12.x)
# 使用方法: bash scripts/setup_environment.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION="3.11"
MODELS_DIR="${HOME}/models"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
err() { echo "[ERROR] $*" >&2; exit 1; }

# ---- 1. システム依存パッケージ ----
log "=== Step 1: システム依存パッケージのインストール ==="
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libosmesa6-dev \
    freeglut3-dev \
    ffmpeg \
    unzip

# ---- 2. uv のインストール ----
log "=== Step 2: uv のインストール ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.cargo/bin:${PATH}"
    # シェルプロファイルに追記
    echo 'export PATH="${HOME}/.cargo/bin:${PATH}"' >> "${HOME}/.bashrc"
fi
uv --version

# ---- 3. Python 3.11 仮想環境の作成 ----
log "=== Step 3: Python ${PYTHON_VERSION} 仮想環境の作成 ==="
cd "${PROJECT_ROOT}"
uv venv .venv --python "${PYTHON_VERSION}"
source .venv/bin/activate
python --version

# ---- 4. PyTorch (CUDA 12.4 / Blackwell 対応) のインストール ----
log "=== Step 4: PyTorch (CUDA 12.4) のインストール ==="
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# PyTorch インストール確認
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" \
    || err "PyTorch の GPU 確認に失敗しました"

# ---- 5. プロジェクト依存パッケージのインストール ----
log "=== Step 5: プロジェクト依存パッケージのインストール ==="
uv pip install -e ".[dev]"

# ---- 6. OSMesa (headless rendering) 確認 ----
log "=== Step 6: OSMesa (headless rendering) 確認 ==="
python -c "import pyrender; print('pyrender OK')" 2>/dev/null \
    || log "WARNING: pyrender のインポートに失敗 — OSMesa が未設定の可能性があります"

# headless 環境変数
export PYOPENGL_PLATFORM=osmesa
echo 'export PYOPENGL_PLATFORM=osmesa' >> "${HOME}/.bashrc"

# ---- 7. モデル保存ディレクトリ作成 ----
log "=== Step 7: モデル保存ディレクトリ作成 ==="
mkdir -p "${MODELS_DIR}"
log "モデルダウンロード先: ${MODELS_DIR}"

# ---- 8. huggingface-cli ログイン確認 ----
log "=== Step 8: Hugging Face 認証確認 ==="
if ! huggingface-cli whoami &>/dev/null; then
    log "WARNING: Hugging Face にログインしていません"
    log "  モデルダウンロード前に 'huggingface-cli login' を実行してください"
fi

# ---- 9. vLLM サーバー起動スクリプト生成 ----
log "=== Step 9: vLLM 起動スクリプト生成 ==="
cat > "${SCRIPT_DIR}/start_vllm_server.sh" << 'EOF'
#!/usr/bin/env bash
# Qwen3-VL-32B vLLM サーバーを起動する
# DGX Spark では ~65GB (BF16) を使用
set -euo pipefail

MODEL="${HOME}/models/Qwen3-VL-32B-Instruct"
if [ ! -d "$MODEL" ]; then
    MODEL="Qwen/Qwen3-VL-32B-Instruct"
fi

exec vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
EOF
chmod +x "${SCRIPT_DIR}/start_vllm_server.sh"
log "vLLM 起動スクリプト: scripts/start_vllm_server.sh"

# ---- 10. TRELLIS.2 コードのセットアップ ----
# TRELLIS.2 は pip 配布なし。GitHub repo を clone して付属の setup.sh でインストール。
# setup.sh は conda 環境 "trellis2" を作成する。
log "=== Step 10: TRELLIS.2 コードのセットアップ ==="
TRELLIS_CODE_DIR="${HOME}/trellis2"
if [ ! -d "${TRELLIS_CODE_DIR}" ]; then
    log "TRELLIS.2 を clone: ${TRELLIS_CODE_DIR}"
    git clone --depth 1 https://github.com/microsoft/TRELLIS.2 "${TRELLIS_CODE_DIR}"
else
    log "TRELLIS.2 は既に clone 済み: ${TRELLIS_CODE_DIR}"
    git -C "${TRELLIS_CODE_DIR}" pull --ff-only || true
fi

# conda が利用可能な場合のみ自動セットアップを試みる
if command -v conda &>/dev/null; then
    log "TRELLIS.2 setup.sh を実行します..."
    pushd "${TRELLIS_CODE_DIR}" > /dev/null
    bash setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel
    popd > /dev/null
    log "TRELLIS.2 セットアップ完了 (conda 環境: trellis2)"
else
    log "WARNING: conda が見つかりません。TRELLIS.2 のセットアップをスキップします。"
    log "  以下を手動で実行してください:"
    log "    cd ${TRELLIS_CODE_DIR}"
    log "    . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel"
fi

# ---- 完了 ----
log ""
log "=== 環境構築完了 ==="
log ""
log "次のステップ:"
log "  1. source .venv/bin/activate"
log "  2. python scripts/download_models.py   # モデルダウンロード (T-0.2)"
log "  3. python tests/test_gpu_models.py     # GPU 動作確認 (T-0.3)"
log ""
log "TRELLIS.2 を使う際は:"
log "  conda activate trellis2"
