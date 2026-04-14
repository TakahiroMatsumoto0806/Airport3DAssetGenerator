#!/usr/bin/env bash
# =============================================================================
# AL3DG オフラインセットアップスクリプト
# 共有ドライブ（/share）からモデルをコピーして環境を構築する。
# Hugging Face へのアクセス不要。
#
# 使い方:
#   bash scripts/setup_from_share.sh <share_dir>
#   bash scripts/setup_from_share.sh --dry-run <share_dir>
#
# 引数:
#   share_dir : 共有ドライブのルートパス
#               例: /mnt/share  /data/al3dg_share
#
# share_dir の期待構造:
#   <share_dir>/
#   ├── models/
#   │   ├── Qwen3-VL-32B-Instruct/   (~63GB)
#   │   └── FLUX.1-schnell/          (~54GB)
#   └── setup_from_share.sh          (このスクリプトのコピー)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${HOME}/models"

# ---- 引数解析 ----
DRY_RUN=false
SHARE_DIR=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -*) echo "[ERROR] 不明なオプション: $arg" >&2; exit 1 ;;
        *) SHARE_DIR="$arg" ;;
    esac
done

if [ -z "$SHARE_DIR" ]; then
    echo "使い方: bash $0 [--dry-run] <share_dir>" >&2
    echo ""
    echo "  share_dir: 共有ドライブのルートパス" >&2
    echo "  例: bash $0 /mnt/share" >&2
    exit 1
fi

SHARE_MODELS_DIR="${SHARE_DIR}/models"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
info() { log "INFO  $*"; }
warn() { log "WARN  $*"; }
err()  { log "ERROR $*" >&2; exit 1; }
dry()  { if $DRY_RUN; then log "[dry-run] $*"; fi; }

# ---- 検証 ----
if $DRY_RUN; then
    info "=== ドライランモード（実際のコピー・インストールは行いません）==="
fi

info "共有ディレクトリ: ${SHARE_DIR}"

if [ ! -d "$SHARE_DIR" ]; then
    err "共有ディレクトリが存在しません: ${SHARE_DIR}"
fi

if [ ! -d "$SHARE_MODELS_DIR" ]; then
    err "models/ サブディレクトリが見つかりません: ${SHARE_MODELS_DIR}"
fi

# ---- モデル一覧 ----
MODELS=(
    "Qwen3-VL-32B-Instruct"
    "FLUX.1-schnell"
)

# ---- Step 1: モデルをコピー ----
info "=== Step 1: モデルコピー → ${MODELS_DIR} ==="
mkdir -p "${MODELS_DIR}"

for model in "${MODELS[@]}"; do
    src="${SHARE_MODELS_DIR}/${model}"
    dst="${MODELS_DIR}/${model}"

    if [ ! -e "$src" ]; then
        warn "共有ディレクトリにモデルが見つかりません（スキップ）: ${src}"
        continue
    fi

    if [ -d "$dst" ] && [ -n "$(ls -A "$dst" 2>/dev/null)" ]; then
        info "  スキップ（既存）: ${dst}"
        continue
    fi

    src_size=$(du -sh "$src" 2>/dev/null | cut -f1 || echo "?")
    info "  コピー中: ${model} (${src_size})"

    if ! $DRY_RUN; then
        if command -v rsync &>/dev/null; then
            rsync -a --info=progress2 "${src}/" "${dst}/"
        else
            warn "  rsync が見つかりません。cp -r でコピーします（進捗表示なし）"
            cp -r "${src}/." "${dst}/"
        fi
        info "  完了: ${dst}"
    else
        dry "rsync (or cp -r) ${src}/ ${dst}/"
    fi
done

# ---- Step 2: 環境構築スクリプトを実行 ----
info "=== Step 2: 依存パッケージのインストール ==="

SETUP_SCRIPT="${PROJECT_ROOT}/scripts/setup_environment.sh"
if [ ! -f "$SETUP_SCRIPT" ]; then
    err "セットアップスクリプトが見つかりません: ${SETUP_SCRIPT}"
fi

if ! $DRY_RUN; then
    # AL3DG_SKIP_MODEL_DOWNLOAD=1 を渡して HF ログイン・モデルDLステップをスキップ
    AL3DG_SKIP_MODEL_DOWNLOAD=1 bash "${SETUP_SCRIPT}"
else
    dry "AL3DG_SKIP_MODEL_DOWNLOAD=1 bash ${SETUP_SCRIPT}"
fi

# ---- 完了 ----
info ""
info "=== セットアップ完了 ==="
info ""
info "次のステップ:"
info "  source ${PROJECT_ROOT}/.venv/bin/activate"
info "  python scripts/run_pipeline.py"
info ""
info "モデル確認:"
for model in "${MODELS[@]}"; do
    dst="${MODELS_DIR}/${model}"
    if [ -d "$dst" ]; then
        size=$(du -sh "$dst" 2>/dev/null | cut -f1 || echo "?")
        info "  ✓ ${model} (${size})"
    else
        warn "  ✗ ${model} — 見つかりません"
    fi
done
