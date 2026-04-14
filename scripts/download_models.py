"""
T-0.2: モデルダウンロードスクリプト（DGX Spark 用）

ダウンロード対象（DGX Spark で実行するもの）:
  1. black-forest-labs/FLUX.1-schnell     (~12GB, BF16)  ※ HFライセンス同意必須
  2. Qwen/Qwen3-VL-32B-Instruct          (~65GB, BF16)
  3. laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K (~2GB)

注意:
  - TRELLIS.2-4B はこのスクリプトのデフォルト対象外。
    TRELLIS.2 は x86_64 + RTX 5090 の別 PC でのみ動作する（DGX Spark / aarch64 非対応）。
    別 PC でダウンロードする場合は --models trellis を明示的に指定すること。
  - FLUX.1-schnell は HuggingFace でのライセンス同意が必要。
    事前に https://huggingface.co/black-forest-labs/FLUX.1-schnell にアクセスして
    同意してから huggingface-cli login を実行すること。

使用方法（DGX Spark）:
  python scripts/download_models.py                          # flux + qwen + clip のみ
  python scripts/download_models.py --verify-only           # サイズ検証のみ

使用方法（x86_64 別 PC — TRELLIS.2 用）:
  python scripts/download_models.py --models trellis
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path.home() / "models"))
TRELLIS_CODE_DIR = Path(os.environ.get("TRELLIS_CODE_DIR", Path.home() / "trellis2"))

# ---- モデル定義 ----
# (hf_repo_id, local_dirname, 期待最小サイズGB, 説明)
MODELS = {
    "flux": (
        "black-forest-labs/FLUX.1-schnell",
        "FLUX.1-schnell",
        10.0,
        "FLUX.1-schnell (Text-to-Image, ~12GB BF16) — HFライセンス同意必須",
    ),
    "trellis": (
        "microsoft/TRELLIS.2-4B",
        "TRELLIS.2-4B",
        20.0,
        "TRELLIS.2-4B (Image-to-3D, ~24GB BF16) — モデルウェイトのみ",
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


def download_trellis_code():
    """
    TRELLIS.2 は pip 配布なし。GitHub repo を clone して setup.sh でインストール。

    手順:
      1. git clone https://github.com/microsoft/TRELLIS.2 ~/trellis2
      2. cd ~/trellis2
      3. . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel

    ※ setup.sh は conda 環境 "trellis2" を作成する。
      本プロジェクトでは conda 環境から trellis2 パッケージを import して使用する。
    """
    if TRELLIS_CODE_DIR.exists():
        logger.info(f"TRELLIS.2 コードは既に存在します: {TRELLIS_CODE_DIR}")
        # 最新にアップデート
        run(["git", "-C", str(TRELLIS_CODE_DIR), "pull", "--ff-only"], check=False)
        return

    logger.info(f"TRELLIS.2 GitHub リポジトリを clone: {TRELLIS_CODE_DIR}")
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/microsoft/TRELLIS.2",
            str(TRELLIS_CODE_DIR),
        ]
    )
    logger.warning(
        "\n"
        "======================================================================\n"
        "TRELLIS.2 コードの clone が完了しました。\n"
        "モデルウェイトとは別に、以下のコマンドでパッケージをインストールしてください:\n"
        "\n"
        f"  cd {TRELLIS_CODE_DIR}\n"
        "  . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel\n"
        "\n"
        "setup.sh は conda 環境 'trellis2' を作成します。\n"
        "======================================================================"
    )


def main():
    parser = argparse.ArgumentParser(
        description="AL3DG モデルダウンロードスクリプト (T-0.2)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["flux", "qwen", "clip"],
        help=(
            "ダウンロードするモデルを指定 "
            "(デフォルト: flux qwen clip — DGX Spark 用)\n"
            "TRELLIS.2 は x86_64 別 PC でのみ使用。"
            "別 PC で実行する場合は --models trellis を指定すること"
        ),
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

            # TRELLIS.2 はコードも clone
            if key == "trellis":
                download_trellis_code()

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
        if "trellis" in targets and not args.verify_only:
            logger.warning(
                f"\nTRELLIS.2 のパッケージインストールを忘れずに:\n"
                f"  cd {TRELLIS_CODE_DIR} && . ./setup.sh --new-env --basic --flash-attn "
                f"--nvdiffrast --nvdiffrec --cumesh --o-voxel"
            )


if __name__ == "__main__":
    main()
