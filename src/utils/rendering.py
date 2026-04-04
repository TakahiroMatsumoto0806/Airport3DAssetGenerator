"""
オフスクリーンレンダリングユーティリティ

pyrender + OSMesa (headless) を使って GLB メッシュを複数視点からレンダリングする。

要件:
  - 512×512 解像度
  - PBR マテリアルを考慮した照明設定
  - 4 方向（正面・右・背面・左）のデフォルト視点

依存パッケージ:
  - pyrender (pip install pyrender)
  - OSMesa: sudo apt install libosmesa6-dev が必要
    pyrender はヘッドレス環境では PYOPENGL_PLATFORM=osmesa が必要

使用例:
    from src.utils.rendering import render_multiview

    image_paths = render_multiview(
        "outputs/meshes_raw/000001.glb",
        output_dir="outputs/renders/000001",
        views=4,
    )
    # → ["outputs/renders/000001/view_0.png", ...]
"""

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


# OSMesa ヘッドレスレンダリングに必要な環境変数
# ディスプレイがない環境（DGX Spark の headless 環境）で必要
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


# デフォルトレンダリング解像度
_RENDER_WIDTH = 512
_RENDER_HEIGHT = 512


def _make_camera_pose(
    azimuth_deg: float,
    elevation_deg: float = 20.0,
    distance: float = 2.0,
) -> np.ndarray:
    """
    球面座標からカメラの pose 行列（4×4）を生成する。

    Args:
        azimuth_deg:   水平角（0°=正面, 90°=右）
        elevation_deg: 仰角（正の値で上方から見下ろし）
        distance:      原点からの距離

    Returns:
        4×4 変換行列（カメラ→ワールド）
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # カメラ位置（球面座標）
    cx = distance * math.cos(el) * math.sin(az)
    cy = distance * math.sin(el)
    cz = distance * math.cos(el) * math.cos(az)

    # Look-at: カメラは原点を向く
    forward = -np.array([cx, cy, cz], dtype=np.float64)
    forward /= np.linalg.norm(forward)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up)
    norm_right = np.linalg.norm(right)
    if norm_right < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = [cx, cy, cz]
    return pose


def render_multiview(
    mesh_path: str,
    output_dir: str,
    views: int = 4,
    width: int = _RENDER_WIDTH,
    height: int = _RENDER_HEIGHT,
    elevation_deg: float = 20.0,
    distance: float = 2.0,
    prefix: str = "view",
) -> list[str]:
    """
    GLB メッシュを複数視点からオフスクリーンレンダリングして PNG として保存する。

    Args:
        mesh_path:     入力 GLB ファイルパス
        output_dir:    レンダリング画像の保存先ディレクトリ
        views:         レンダリング視点数（等間隔に水平に配置）
        width:         画像幅（デフォルト 512）
        height:        画像高さ（デフォルト 512）
        elevation_deg: カメラ仰角（デフォルト 20°）
        distance:      カメラ距離（デフォルト 2.0）
        prefix:        出力ファイル名プレフィックス

    Returns:
        list[str]: 保存された PNG ファイルのパスリスト（views 件）

    Raises:
        ImportError: pyrender がインストールされていない場合
        RuntimeError: メッシュロードまたはレンダリング失敗
    """
    try:
        import pyrender
        import trimesh
    except ImportError as e:
        raise ImportError(
            f"pyrender または trimesh が見つかりません: {e}\n"
            "pip install pyrender を実行してください。\n"
            "OSMesa のインストール: sudo apt install libosmesa6-dev"
        ) from e

    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GLB ロード（trimesh）
    try:
        loaded = trimesh.load(str(mesh_path), force="scene")
    except Exception as e:
        raise RuntimeError(f"メッシュロード失敗 ({mesh_path}): {e}") from e

    # trimesh.Scene → pyrender.Scene 変換
    try:
        scene = pyrender.Scene.from_trimesh_scene(loaded)
    except Exception:
        # force="mesh" にフォールバック
        try:
            mesh_obj = trimesh.load(str(mesh_path), force="mesh", process=False)
            pr_mesh = pyrender.Mesh.from_trimesh(mesh_obj)
            scene = pyrender.Scene()
            scene.add(pr_mesh)
        except Exception as e2:
            raise RuntimeError(f"pyrender シーン作成失敗 ({mesh_path}): {e2}") from e2

    # シーンの AABB を計算して距離をスケール
    try:
        bounds = loaded.bounds if hasattr(loaded, "bounds") else None
        if bounds is not None:
            extent = np.max(bounds[1] - bounds[0])
            if extent > 0:
                distance = extent * distance
    except Exception:
        pass

    # カメラ設定
    camera = pyrender.PerspectiveCamera(yfov=math.radians(45.0), znear=0.01)

    # 照明設定（環境光 + ポイントライト 2 個）
    light_color = np.array([1.0, 1.0, 1.0])
    ambient = pyrender.DirectionalLight(color=light_color, intensity=3.0)
    fill = pyrender.DirectionalLight(color=light_color, intensity=1.5)

    # オフスクリーンレンダラー
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    output_paths: list[str] = []
    azimuths = [i * (360.0 / views) for i in range(views)]

    try:
        for i, az in enumerate(azimuths):
            pose = _make_camera_pose(az, elevation_deg, distance)

            # ノード追加（毎回クリア）
            cam_node = scene.add(camera, pose=pose)

            # 照明を正面と背面から
            light_pose1 = _make_camera_pose(az, 40.0, distance * 1.5)
            light_pose2 = _make_camera_pose(az + 180.0, 20.0, distance * 1.5)
            ln1 = scene.add(ambient, pose=light_pose1)
            ln2 = scene.add(fill, pose=light_pose2)

            # レンダリング
            color, _ = renderer.render(scene)

            # PNG 保存
            img = Image.fromarray(color, "RGB")
            out_path = str(output_dir / f"{prefix}_{i}.png")
            img.save(out_path)
            output_paths.append(out_path)

            # ノード削除（次のビューのために）
            scene.remove_node(cam_node)
            scene.remove_node(ln1)
            scene.remove_node(ln2)

    finally:
        renderer.delete()

    logger.debug(f"  レンダリング完了: {len(output_paths)} 枚 → {output_dir}")
    return output_paths
