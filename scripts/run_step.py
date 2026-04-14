#!/usr/bin/env python3
"""
T-6.2: 個別ステップ実行スクリプト

パイプラインの特定ステップのみを実行する。
各ステップを個別にデバッグ・再実行する際に使用。

使用例:
    # プロンプト生成のみ
    python scripts/run_step.py --step prompt

    # メッシュ生成 (TRELLIS.2 conda 環境で実行)
    conda activate trellis
    python scripts/run_step.py --step mesh

    # 物理プロパティ付与（入力ディレクトリを明示）
    python scripts/run_step.py --step physics --input outputs/meshes_approved

利用可能なステップ:
    prompt       T-1.2: プロンプト生成
    image        T-2.1: 画像生成 (FLUX.1-schnell)
    image_qa     T-2.2: 画像 QA (Qwen3-VL-32B)
    mesh         T-3.1: 3D メッシュ生成 (TRELLIS.2)
    mesh_qa      T-3.2: メッシュ QA (trimesh)
    mesh_vlm_qa  T-3.3: VLM マルチビュー QA
    physics      T-4.1: 物理プロパティ付与 (CoACD)
    sim_export   T-4.2: Isaac Sim USD エクスポート
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from omegaconf import OmegaConf

from src.pipeline import AL3DGPipeline

VALID_STEPS = [
    "prompt", "image", "image_qa",
    "mesh", "mesh_qa", "mesh_vlm_qa",
    "physics", "sim_export",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AL3DG 個別ステップ実行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step", "-s",
        required=True,
        choices=VALID_STEPS,
        help="実行するステップ名",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline_config.yaml",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="入力ディレクトリ（省略時は設定ファイルのデフォルト）",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="出力ディレクトリ（省略時は設定ファイルのデフォルト）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="中断再開を無効化",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"設定ファイルが見つかりません: {cfg_path}")
        return 1

    cfg = OmegaConf.load(str(cfg_path))

    # CLI オプションで設定を上書き
    if args.output and args.step == "sim_export":
        OmegaConf.update(cfg, "sim_export.output_dir", args.output)
    if args.output and args.step == "physics":
        OmegaConf.update(cfg, "physics.output_dir", args.output)
    pipeline = AL3DGPipeline(cfg)
    resume = not args.no_resume

    step_method_map = {
        "prompt":      pipeline.run_prompt_generation,
        "image":       lambda: pipeline.run_image_generation(resume=resume),
        "image_qa":    lambda: pipeline.run_image_qa(resume=resume),
        "mesh":        lambda: pipeline.run_mesh_generation(resume=resume),
        "mesh_qa":     lambda: pipeline.run_mesh_qa(resume=resume),
        "mesh_vlm_qa": lambda: pipeline.run_mesh_vlm_qa(resume=resume),
        "physics":     lambda: pipeline.run_physics(resume=resume),
        "sim_export":  lambda: pipeline.run_sim_export(resume=resume),
    }

    logger.info(f"ステップ実行: {args.step}")
    try:
        result = step_method_map[args.step]()
        logger.info(f"完了: {result}")
        return 0
    except KeyboardInterrupt:
        logger.warning("中断されました (Ctrl-C)")
        return 130
    except Exception as e:
        logger.exception(f"ステップ実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
