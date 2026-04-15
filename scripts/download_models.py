"""
T-0.2: モデルダウンロードスクリプト（DGX Spark 用）

ダウンロード対象:
  1. black-forest-labs/FLUX.1-schnell             (~12GB, BF16)  ※ HFライセンス同意必須
  2. Qwen/Qwen3-VL-32B-Instruct                   (~65GB, BF16)
  3. laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K    (~2GB)

注意:
  - 3D 生成 (Step 3) は本プロジェクトの対象範囲外のため、3D 生成用モデルは扱わない。
  - FLUX.1-schnell は HuggingFace でのライセンス同意が必要。
    事前に https://huggingface.co/black-forest-labs/FLUX.1-schnell にアクセスして
    同意してから huggingface-cli login を実行すること。

使用方法:
  python scripts/download_models.py                          # flux + qwen + clip
  python scripts/download_models.py --verify-only            # サイズ検証のみ
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path.home() / "models"))

# ---- モデル定義 ----
# (hf_repo_id, local_dirname, 期待最小サイズGB, 説明)
MODELS = {
    "flux": (
        "black-forest-labs/FLUX.1-schnell",
        "FLUX.1-schnell",
        10.0,
        "FLUX.1-schnell (Text-to-Image, ~12GB BF16) — HFライセンス同意必須",
    ),
    "qwen": (
        "Qwen/Qwen3-VL-32B-Instruct",
        "Qwen3-VL-32B-Instruct",
        60.0,
        "Qwen3-VL-32B-Instruct (VLM/LLM 全タスク統一, ~65GB BF16)",
    ),
    "clip": (
        "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        "CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        1.5,
        "OpenCLIP ViT-L/14 (多様性評価用, ~2GB)",
    ),
}


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    logger.debug(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, text=True)


def check_hf_login() -> bool:
    result = subprocess.run(
        ["huggingface-cli", "whoami"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(
            "Hugging Face にログインしていません。\n"
            "  huggingface-cli login\n"
            "を実行してトークンを設定してください。"
        )
        return False
    logger.info(f"HF ログイン確認: {result.stdout.strip()}")
    return True


def download_hf_model(repo_id: str, local_dir: Path) -> bool:
    """huggingface-cli download で再開可能ダウンロード"""
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {repo_id} → {local_dir}")
    result = run(
        [
            "huggingface-cli",
            "download",
            repo_id,
            "--local-dir",
            str(local_dir),
            "--local-dir-use-symlinks",
            "False",
        ],
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"ダウンロード失敗: {repo_id}")
        return False
    return True


def verify_model_size(local_dir: Path, min_gb: float, name: str) -> bool:
    """ディレクトリの合計サイズが期待最小値以上か検証"""
    if not local_dir.exists():
        logger.error(f"[FAIL] {name}: ディレクトリが存在しない → {local_dir}")
        return False

    total_bytes = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
    total_gb = total_bytes / (1024**3)
    ok = total_gb >= min_gb

    status = "OK  " if ok else "FAIL"
    logger.info(f"[{status}] {name}: {total_gb:.1f} GB (期待最小: {min_gb:.1f} GB)")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="AL3DG モデルダウンロードスクリプト (T-0.2)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["flux", "qwen", "clip"],
        help="ダウンロードするモデルを指定 (デフォルト: flux qwen clip)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="ダウンロードせずにサイズ検証のみ実行",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"モデル保存先ディレクトリ (デフォルト: {MODELS_DIR})",
    )
    args = parser.parse_args()

    models_dir: Path = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"モデル保存先: {models_dir}")

    targets = list(MODELS.keys()) if "all" in args.models else args.models

    # ---- ログイン確認（verify-only 以外） ----
    if not args.verify_only:
        if not check_hf_login():
            sys.exit(1)

    results: dict[str, bool] = {}

    for key in targets:
        repo_id, local_dirname, min_gb, description = MODELS[key]
        local_dir = models_dir / local_dirname

        logger.info(f"\n{'='*60}")
        logger.info(f"モデル: {description}")

        if key == "flux":
            logger.warning(
                "FLUX.1-schnell はライセンス同意が必要です。\n"
                "  https://huggingface.co/black-forest-labs/FLUX.1-schnell\n"
                "  にアクセスして同意済みであることを確認してください。"
            )

        if not args.verify_only:
            ok = download_hf_model(repo_id, local_dir)
            if not ok:
                results[key] = False
                continue

        results[key] = verify_model_size(local_dir, min_gb, local_dirname)

    # ---- サマリ ----
    logger.info(f"\n{'='*60}")
    logger.info("ダウンロード/検証 サマリ:")
    all_ok = True
    for key, ok in results.items():
        _, local_dirname, _, _ = MODELS[key]
        status = "✓" if ok else "✗"
        logger.info(f"  {status} {local_dirname}")
        if not ok:
            all_ok = False

    if not all_ok:
        logger.error("一部のモデルでエラーが発生しました。上記ログを確認してください。")
        sys.exit(1)
    else:
        logger.info("\n全モデルのダウンロード/検証が完了しました。")


if __name__ == "__main__":
    main()
