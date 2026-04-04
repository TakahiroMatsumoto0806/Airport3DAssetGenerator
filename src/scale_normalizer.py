"""
T-4.1: スケール正規化

GLB メッシュを Franka グリッパー（最大開口 80mm）で把持可能なミニチュアサイズに正規化する。

仕様（configs/material_properties.yaml の miniature_scale セクション）:
  target_short_side_mm: 60.0  → 最大短辺がこの値以下
  max_long_side_mm:    120.0  → 最大長辺もこの値以下
  preserve_aspect_ratio: true → アスペクト比を維持

使用例:
    normalizer = ScaleNormalizer("configs/material_properties.yaml")
    result = normalizer.normalize("outputs/meshes_raw/000001.glb", luggage_type="hard_suitcase")
    # result["scale_factor"]   → 0.00085  (元サイズ→ミニチュアの倍率)
    # result["output_path"]    → "outputs/meshes_raw/000001_scaled.glb"
    # result["dimensions_mm"]  → {"x": 52.0, "y": 35.0, "z": 75.0}
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class ScaleNormalizer:
    """Franka グリッパー把持スケールへのメッシュ正規化"""

    def __init__(self, material_config_path: str = "configs/material_properties.yaml") -> None:
        """
        Args:
            material_config_path: material_properties.yaml のパス（miniature_scale セクション使用）
        """
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(material_config_path)
        ms = cfg.get("miniature_scale", {})

        self.enabled: bool = bool(ms.get("enabled", True))
        self.target_short_side_mm: float = float(ms.get("target_short_side_mm", 60.0))
        self.max_long_side_mm: float = float(ms.get("max_long_side_mm", 120.0))
        self.preserve_aspect_ratio: bool = bool(ms.get("preserve_aspect_ratio", True))

    def normalize(
        self,
        mesh_path: str,
        luggage_type: Optional[str] = None,
        miniature: bool = True,
        output_path: Optional[str] = None,
    ) -> dict:
        """
        メッシュを正規化スケールに変換して保存する。

        TRELLIS.2 の出力メッシュは AABB が [-0.5, 0.5]³ の正規化座標系（単位: m）
        → miniature モードでは Franka 把持可能サイズへ縮小する。

        Args:
            mesh_path:    入力 GLB ファイルパス
            luggage_type: 荷物タイプ（ログ用、スケール計算には不使用）
            miniature:    True でミニチュアスケールに変換（デフォルト True）
            output_path:  出力 GLB パス（None の場合は "{stem}_scaled.glb"）

        Returns:
            {
                "input_path":       str,
                "output_path":      str,
                "scale_factor":     float,   # 適用した倍率
                "original_extents_m":  [x, y, z],  # 変換前バウンディングボックス [m]
                "scaled_extents_mm":   [x, y, z],  # 変換後バウンディングボックス [mm]
                "short_side_mm":    float,
                "long_side_mm":     float,
            }
        """
        import trimesh

        mesh_path = Path(mesh_path)
        if output_path is None:
            output_path = mesh_path.parent / f"{mesh_path.stem}_scaled.glb"
        output_path = Path(output_path)

        # メッシュロード
        loaded = trimesh.load(str(mesh_path), force="scene")
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded.dump()))
        else:
            mesh = loaded

        # 元の AABB サイズ [m]
        extents_m = mesh.extents.copy()  # [x, y, z]
        extents_mm = extents_m * 1000.0

        scale_factor = 1.0

        if miniature and self.enabled:
            # 短辺 = min(extents_mm), 長辺 = max(extents_mm)
            short_mm = float(np.min(extents_mm))
            long_mm = float(np.max(extents_mm))

            if short_mm > 0:
                # 短辺を target_short_side_mm に合わせるスケール
                scale_by_short = self.target_short_side_mm / short_mm
                # 長辺が max_long_side_mm を超えないように制限
                scale_by_long = self.max_long_side_mm / long_mm if long_mm > 0 else scale_by_short
                scale_factor = min(scale_by_short, scale_by_long)

        # スケール適用
        mesh.apply_scale(scale_factor)
        scaled_extents_mm = mesh.extents * 1000.0

        short_side_mm = float(np.min(scaled_extents_mm))
        long_side_mm = float(np.max(scaled_extents_mm))

        # 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path), file_type="glb")

        logger.debug(
            f"  スケール正規化: {mesh_path.name} "
            f"× {scale_factor:.6f} → [{short_side_mm:.1f}, {long_side_mm:.1f}] mm"
        )

        return {
            "input_path": str(mesh_path),
            "output_path": str(output_path),
            "scale_factor": scale_factor,
            "original_extents_m": extents_m.tolist(),
            "scaled_extents_mm": scaled_extents_mm.tolist(),
            "short_side_mm": short_side_mm,
            "long_side_mm": long_side_mm,
        }
