#!/usr/bin/env python3
"""outputs/ バックアップ＆リセットスクリプト

E2E 実行前に outputs/ の内容を outputs_backup/<YYYY-MM-DD_HHMMSS>/ に退避し、
outputs/ を次回実行に必要な最小限のディレクトリ構造にリセットする。
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


# リセット後に再作成するディレクトリ構造（リーフパスのリスト）
RESET_DIRS = [
    "assets_final",
    "images",
    "images_approved",
    "logs",
    "meshes_approved",
    "meshes_raw",
    "prompts",
    "renders",
    "reports",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(
        description="outputs/ をバックアップして初期状態にリセットする"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=project_root / "outputs",
        help="バックアップ元ディレクトリ（デフォルト: <project_root>/outputs）",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=project_root / "outputs_backup",
        help="バックアップ先親ディレクトリ（デフォルト: <project_root>/outputs_backup）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際のコピー・削除を行わず、対象ファイル数のみ表示する",
    )
    return parser.parse_args()


def count_files(directory: Path) -> int:
    """ディレクトリ内の全ファイル数を返す（.gitkeep を含む）。"""
    return sum(1 for _ in directory.rglob("*") if _.is_file())


def check_disk_space(src: Path, dst_parent: Path) -> None:
    """バックアップに必要なディスク空き容量を簡易チェックして警告のみ出す。"""
    try:
        total_size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
        disk = shutil.disk_usage(dst_parent if dst_parent.exists() else dst_parent.parent)
        free_gb = disk.free / (1024**3)
        needed_gb = total_size / (1024**3)
        if needed_gb > free_gb:
            logger.warning(
                f"ディスク空き容量が不足している可能性があります: "
                f"必要 {needed_gb:.2f} GB / 空き {free_gb:.2f} GB"
            )
        else:
            logger.info(
                f"ディスク容量: 必要 {needed_gb:.2f} GB / 空き {free_gb:.2f} GB — OK"
            )
    except Exception as exc:
        logger.warning(f"ディスク容量チェックをスキップしました: {exc}")


def backup_outputs(outputs_dir: Path, backup_parent: Path, dry_run: bool) -> Path:
    """outputs_dir を backup_parent/<timestamp>/ にコピーする。

    同一秒内に複数回呼ばれた場合は末尾に _1, _2 ... を付与して衝突を回避する。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_dest = backup_parent / timestamp

    # 衝突回避：既存フォルダと同名になる場合は連番サフィックスを付与
    if not dry_run:
        counter = 1
        while backup_dest.exists():
            backup_dest = backup_parent / f"{timestamp}_{counter}"
            counter += 1

    file_count = count_files(outputs_dir)
    logger.info(f"バックアップ対象: {outputs_dir} ({file_count} ファイル)")
    logger.info(f"バックアップ先  : {backup_dest}")

    if dry_run:
        logger.info("[dry-run] コピーはスキップされます")
        return backup_dest

    backup_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(outputs_dir, backup_dest, symlinks=True)
    logger.success(f"バックアップ完了: {backup_dest}")
    return backup_dest


def reset_outputs(outputs_dir: Path, dry_run: bool) -> None:
    """outputs_dir を削除・再作成して最小限のディレクトリ構造を構築する。"""
    if dry_run:
        logger.info("[dry-run] outputs/ のリセットはスキップされます")
        logger.info(f"[dry-run] 再作成されるディレクトリ ({len(RESET_DIRS)} 件):")
        for d in RESET_DIRS:
            logger.info(f"  outputs/{d}/")
        return

    # 既存コンテンツを全削除して再作成
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)
    outputs_dir.mkdir(parents=True)
    logger.info(f"outputs/ を削除して再作成しました: {outputs_dir}")

    # サブディレクトリと .gitkeep を再作成
    for rel_path in RESET_DIRS:
        leaf = outputs_dir / rel_path
        leaf.mkdir(parents=True, exist_ok=True)
        gitkeep = leaf / ".gitkeep"
        gitkeep.touch()

    created = [str(outputs_dir / d) for d in RESET_DIRS]
    logger.success(
        f"outputs/ リセット完了: {len(created)} ディレクトリ + .gitkeep を再作成しました"
    )


def main() -> None:
    args = parse_args()
    outputs_dir: Path = args.outputs_dir.resolve()
    backup_parent: Path = args.backup_dir.resolve()

    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}")

    if not outputs_dir.exists():
        logger.error(f"outputs ディレクトリが存在しません: {outputs_dir}")
        sys.exit(1)

    if args.dry_run:
        logger.info("=== ドライラン モード（実際の変更は行いません） ===")

    # ディスク容量チェック（警告のみ）
    if not args.dry_run:
        check_disk_space(outputs_dir, backup_parent)

    # 1. バックアップ
    backup_outputs(outputs_dir, backup_parent, dry_run=args.dry_run)

    # 2. リセット
    reset_outputs(outputs_dir, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("=== ドライラン完了（実際の変更は行われていません） ===")
    else:
        logger.success("全処理完了")


if __name__ == "__main__":
    main()
