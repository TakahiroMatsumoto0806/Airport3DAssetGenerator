# =============================================================================
# AL3DG Dockerfile
# ベース: NVIDIA PyTorch (CUDA 12.4 / Ubuntu 22.04)
# 対象:  DGX Spark (Grace Blackwell GB10, 128GB unified memory)
# =============================================================================
FROM nvcr.io/nvidia/pytorch:24.08-py3

LABEL maintainer="AL3DG Project"
LABEL description="Airport Luggage 3D Asset Generator pipeline"

# ---- 環境変数 ----
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYOPENGL_PLATFORM=osmesa \
    MODELS_DIR=/models \
    HF_HOME=/models/hf_cache

# ---- システム依存パッケージ ----
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- uv インストール ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# ---- 作業ディレクトリ ----
WORKDIR /workspace/al3dg

# ---- ソースコードのコピー（editable install に src/ が必要なため先にコピー）----
COPY . .

# ---- 依存パッケージのインストール ----
RUN uv pip install --system -e ".[dev]" --no-cache-dir

# ---- モデルディレクトリ ----
RUN mkdir -p ${MODELS_DIR}
VOLUME ["/models"]

# ---- 出力ディレクトリ ----
VOLUME ["/workspace/al3dg/outputs"]

# ---- エントリポイント ----
CMD ["python", "scripts/run_pipeline.py", "--help"]
