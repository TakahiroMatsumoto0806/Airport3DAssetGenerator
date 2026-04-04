#!/usr/bin/env python3
"""
T-6.2: フルパイプライン実行スクリプト

全ステップを順番に実行する。
DGX Spark での標準的な使用例:

    # 全ステップ実行
    python scripts/run_pipeline.py

    # 設定ファイルを指定
    python scripts/run_pipeline.py --config configs/pipeline_config.yaml

    # 特定ステップのみ実行
    python scripts/run_pipeline.py --steps prompt image image_qa

    # 中断再開なし（全件再実行）
    python scripts/run_pipeline.py --no-resume

注意:
    - TRELLIS.2 は conda 環境 trellis で実行すること:
        conda activate trellis && python scripts/run_pipeline.py --steps mesh
    - Qwen3-VL-32B は別ターミナルで vLLM サーバーを起動しておくこと:
        vllm serve Qwen/Qwen3-VL-32B-Instruct --dtype bfloat16
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from omegaconf import OmegaConf

from src.pipeline import AL3DGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AL3DG フルパイプライン実行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline_config.yaml",
        help="設定ファイルパス (デフォルト: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--steps", "-s",
        nargs="+",
        choices=["prompt", "image", "image_qa", "mesh", "mesh_qa",
                 "mesh_vlm_qa", "physics", "sim_export", "diversity"],
        default=None,
        help="実行するステップ (デフォルト: 設定ファイルの steps セクションに従う)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="中断再開を無効化して全件再実行する",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログレベル (デフォルト: INFO)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ログ設定
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.add(
        "outputs/logs/pipeline_{time:YYYY-MM-DD_HH-mm-ss}.log",
        level="DEBUG",
        rotation="100 MB",
    )

    # 設定読み込み
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"設定ファイルが見つかりません: {cfg_path}")
        return 1

    cfg = OmegaConf.load(str(cfg_path))
    logger.info(f"設定ファイル: {cfg_path}")

    # パイプライン実行
    pipeline = AL3DGPipeline(cfg)
    try:
        results = pipeline.run(
            steps=args.steps,
            resume=not args.no_resume,
        )
    except KeyboardInterrupt:
        logger.warning("パイプラインが中断されました (Ctrl-C)")
        return 130
    except Exception as e:
        logger.exception(f"パイプライン実行エラー: {e}")
        return 1

    # 結果サマリー表示
    logger.info("=" * 60)
    logger.info("パイプライン完了サマリー:")
    for step, result in results.items():
        logger.info(f"  {step}: {result}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
