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
    # シェルプロファイルに追記（重複を避ける）
    grep -q 'cargo/bin' "${HOME}/.bashrc" \
        || echo 'export PATH="${HOME}/.cargo/bin:${PATH}"' >> "${HOME}/.bashrc"
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

# headless 環境変数（重複を避ける）
export PYOPENGL_PLATFORM=osmesa
grep -q 'PYOPENGL_PLATFORM' "${HOME}/.bashrc" \
    || echo 'export PYOPENGL_PLATFORM=osmesa' >> "${HOME}/.bashrc"

# ---- 7. モデル保存ディレクトリ作成 ----
log "=== Step 7: モデル保存ディレクトリ作成 ==="
mkdir -p "${MODELS_DIR}"
log "モデルダウンロード先: ${MODELS_DIR}"

# ---- 8. huggingface-cli ログイン確認 ----
# AL3DG_SKIP_MODEL_DOWNLOAD=1 の場合はスキップ（共有ドライブからのオフライン構築時）
log "=== Step 8: Hugging Face 認証確認 ==="
if [ "${AL3DG_SKIP_MODEL_DOWNLOAD:-0}" = "1" ]; then
    log "  AL3DG_SKIP_MODEL_DOWNLOAD=1: HuggingFace ログイン確認をスキップします"
    log "  （setup_from_share.sh 経由のオフラインセットアップ）"
elif ! huggingface-cli whoami &>/dev/null; then
    log "WARNING: Hugging Face にログインしていません"
    log "  モデルダウンロード前に 'huggingface-cli login' を実行してください"
    log "  オフライン環境の場合は AL3DG_SKIP_MODEL_DOWNLOAD=1 を設定してください"
fi

# ---- 9. vLLM サーバー起動スクリプト確認 ----
log "=== Step 9: vLLM 起動スクリプト確認 ==="
chmod +x "${SCRIPT_DIR}/start_vllm_server.sh"
log "vLLM 起動スクリプト: scripts/start_vllm_server.sh"
log "  起動方法: bash scripts/start_vllm_server.sh"

# ---- 10. TRELLIS.2 について ----
# TRELLIS.2 は DGX Spark (aarch64) では動作しない。
# x86_64 + RTX 5090 の別 PC で以下のようにセットアップすること:
#   git clone --depth 1 https://github.com/microsoft/TRELLIS.2 ~/trellis2
#   cd ~/trellis2
#   . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel
#   huggingface-cli download microsoft/TRELLIS.2-4B --local-dir ~/models/TRELLIS.2-4B
log "=== Step 10: TRELLIS.2 ==="
log "  ※ TRELLIS.2 は DGX Spark(aarch64) では動作しません。"
log "  ※ x86_64 別 PC でのセットアップ手順は README.md の「TRELLIS.2 専用 PC」セクションを参照してください。"

# ---- 完了 ----
log ""
log "=== 環境構築完了 ==="
log ""
log "次のステップ:"
log "  1. source .venv/bin/activate"
if [ "${AL3DG_SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    log "  2. python scripts/download_models.py   # モデルダウンロード (T-0.2)"
    log "  3. python tests/test_gpu_models.py     # GPU 動作確認 (T-0.3)"
else
    log "  2. python tests/test_gpu_models.py     # GPU 動作確認 (T-0.3)"
fi
log ""
log "TRELLIS.2（3D 生成）は x86_64 別 PC で実行してください。"
log "  → README.md の「TRELLIS.2 専用 PC」セクションを参照"
