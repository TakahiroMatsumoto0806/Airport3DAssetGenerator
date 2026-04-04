#!/usr/bin/env python3
"""
T-6.2: 多様性評価レポート生成スクリプト

既存のアセット (outputs/assets_final/) から多様性評価レポートを生成する。
パイプライン全体を再実行せずにレポートだけ更新したい場合に使用。

使用例:
    # デフォルト設定でレポート生成
    python scripts/generate_report.py

    # アセットディレクトリを指定
    python scripts/generate_report.py --assets-dir outputs/assets_final

    # OpenCLIP 埋め込み計算をスキップ（既存 JSON のみ集計）
    python scripts/generate_report.py --no-clip

    # 出力ディレクトリを指定
    python scripts/generate_report.py --output-dir outputs/reports/v2

出力ファイル:
    outputs/reports/diversity_report.html  — HTML レポート
    outputs/reports/diversity_report.json  — JSON サマリー
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from omegaconf import OmegaConf

from src.diversity_evaluator import DiversityEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AL3DG 多様性評価レポート生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline_config.yaml",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="アセットディレクトリ (デフォルト: 設定ファイルの physics.output_dir)",
    )
    parser.add_argument(
        "--renders-dir",
        default="outputs/renders",
        help="レンダリング画像ディレクトリ (デフォルト: outputs/renders)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="レポート出力ディレクトリ (デフォルト: 設定ファイルの diversity.output_dir)",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        default=False,
        help="OpenCLIP 埋め込み計算をスキップする",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="CLIP 計算デバイス (デフォルト: cuda)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="近似重複判定の cosine 類似度閾値 (デフォルト: 0.95)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="CLIP 推論バッチサイズ (デフォルト: 32)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def collect_metadata(assets_dir: Path) -> tuple[list[dict], list[dict]]:
    """assets_final 内の physics.json を全収集する"""
    metadata_list: list[dict] = []
    mesh_info_list: list[dict] = []

    for phys_json in sorted(assets_dir.glob("*/physics.json")):
        with open(phys_json, encoding="utf-8") as f:
            phys = json.load(f)
        metadata_list.append(phys)
        extents = phys.get("scale", {}).get("scaled_extents_mm")
        if extents:
            mesh_info_list.append({
                "scale": {"scaled_extents_mm": extents},
                "luggage_type": phys.get("luggage_type"),
            })

    logger.info(f"  アセット収集: {len(metadata_list)} 件")
    return metadata_list, mesh_info_list


def main() -> int:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # 設定読み込み
    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg_omg = OmegaConf.load(str(cfg_path))
        cfg = OmegaConf.to_container(cfg_omg, resolve=True)
    else:
        logger.warning(f"設定ファイルが見つかりません: {cfg_path}. デフォルト値を使用。")

    assets_dir_str = (
        args.assets_dir
        or cfg.get("physics", {}).get("output_dir", "outputs/assets_final")
    )
    output_dir_str = (
        args.output_dir
        or cfg.get("diversity", {}).get("output_dir", "outputs/reports")
    )

    assets_dir = Path(assets_dir_str)
    output_dir = Path(output_dir_str)

    if not assets_dir.exists():
        logger.error(f"アセットディレクトリが見つかりません: {assets_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = DiversityEvaluator()

    # CLIP 埋め込み計算
    embeddings = None
    image_paths: list[str] = []

    if not args.no_clip:
        renders_dir = Path(args.renders_dir)
        image_paths = sorted(
            str(p) for p in renders_dir.glob("**/*_0.png")
        )
        if image_paths:
            logger.info(f"  CLIP 埋め込み計算: {len(image_paths)} 枚")
            device = args.device
            evaluator.load_model(device=device)
            try:
                embeddings = evaluator.compute_clip_embeddings(
                    image_paths, batch_size=args.batch_size
                )
            finally:
                evaluator.unload()
        else:
            logger.warning(
                f"  レンダリング画像が見つかりません ({renders_dir})。"
                "  CLIP 埋め込みをスキップします。"
            )

    # メタデータ収集
    metadata_list, mesh_info_list = collect_metadata(assets_dir)

    # size_realism_refs を設定から取得
    size_realism_refs = cfg.get("diversity", {}).get("size_realism_refs")

    # レポート生成
    html_path = evaluator.generate_report(
        output_dir=str(output_dir),
        embeddings=embeddings,
        image_paths=image_paths if image_paths else None,
        metadata_list=metadata_list if metadata_list else None,
        mesh_info_list=mesh_info_list if mesh_info_list else None,
        near_dup_threshold=args.threshold,
    )

    # サイズ現実性チェック
    if mesh_info_list and size_realism_refs:
        realism = evaluator.check_size_realism(mesh_info_list, size_realism_refs)
        realism_path = output_dir / "size_realism.json"
        with open(realism_path, "w", encoding="utf-8") as f:
            json.dump(realism, f, ensure_ascii=False, indent=2)
        logger.info(
            f"  サイズ現実性: 現実的={realism['realistic']}, "
            f"非現実的={realism['unrealistic']}, 未知={realism['unknown']} "
            f"→ {realism_path}"
        )

    logger.info(f"レポート生成完了: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
