"""
ログ管理ユーティリティ
loguru ベースの構造化ログ設定
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_file: Path | None = None,
    level: str = "INFO",
    rotation: str = "100 MB",
) -> None:
    """プロジェクト共通のロガーを設定する"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=level,
            rotation=rotation,
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        )


def get_logger(name: str | None = None):
    return logger.bind(name=name) if name else logger
