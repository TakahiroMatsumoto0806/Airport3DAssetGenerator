"""
オフスクリーンレンダリングユーティリティ

pyrender + OSMesa で GLB メッシュを複数視点からレンダリングする。
PBR マテリアル・UV テクスチャを正しく反映する。

必要環境:
  - PYOPENGL_PLATFORM=osmesa （Python 起動前に設定）
  - sudo apt-get install -y libosmesa6-dev
  - uv pip install pyrender PyOpenGL==3.1.7

使用例:
    import os; os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    from src.utils.rendering import render_multiview

    image_paths = render_multiview(
        "outputs/meshes_approved/000001.glb",
        output_dir="outputs/renders/000001",
        views=4,
    )
"""

import math
import os
from pathlib import Path

import numpy as np
from loguru import logger


# OSMesa (CPU ソフトウェアレンダラ) を強制使用する。
# vLLM が GPU メモリをほぼ占有している環境でも GPU OOM を回避できる。
# pyrender より先に設定する必要があるため、モジュール読み込み時に設定する。
if os.environ.get("PYOPENGL_PLATFORM") != "osmesa":
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    logger.debug("PYOPENGL_PLATFORM を 'osmesa' に自動設定しました")


_RENDER_WIDTH = 512
_RENDER_HEIGHT = 512


def _look_at_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    eye から target を向く 4×4 カメラポーズ行列（ワールド座標系）を返す。
    pyrender は OpenGL 規約: カメラは +Z 方向を「後ろ」、-Z が視線方向。
    """
    f = target - eye
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-8:
        raise ValueError("eye と target が同一点です")
    f = f / f_norm

    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(f, up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])

    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)

    pose = np.eye(4)
    pose[:3, 0] = r      # X: 右
    pose[:3, 1] = u      # Y: 上
    pose[:3, 2] = -f     # Z: カメラ後方（-視線方向）
    pose[:3, 3] = eye    # 位置
    return pose.astype(np.float64)


def render_multiview(
    mesh_path: str,
    output_dir: str,
    views: int = 4,
    width: int = _RENDER_WIDTH,
    height: int = _RENDER_HEIGHT,
    elevation_deg: float = 20.0,
    distance_scale: float = 2.2,
    fov_y_deg: float = 45.0,
    prefix: str = "view",
) -> list[str]:
    """
    GLB メッシュを複数視点から pyrender + OSMesa でオフスクリーンレンダリングし
    PNG として保存する。PBR マテリアル・UV テクスチャを正しく反映する。

    Args:
        mesh_path:       入力 GLB ファイルパス
        output_dir:      レンダリング画像の保存先ディレクトリ
        views:           レンダリング視点数（水平等間隔）
        width:           画像幅（px）
        height:          画像高さ（px）
        elevation_deg:   カメラ仰角（度）
        distance_scale:  バウンディングボックス最大辺に掛けるカメラ距離倍率
        fov_y_deg:       縦方向視野角（度）
        prefix:          出力ファイル名プレフィックス

    Returns:
        list[str]: 保存された PNG ファイルパスのリスト（views 件）
    """
    import pyrender
    import trimesh
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- GLB をシーンとしてロード ---
    try:
        scene_tm = trimesh.load(str(mesh_path), process=False)
    except Exception as e:
        raise RuntimeError(f"GLB ロード失敗 ({mesh_path}): {e}") from e

    # trimesh.Trimesh の場合は Scene にラップする
    if isinstance(scene_tm, trimesh.Trimesh):
        scene_tm = trimesh.Scene({"mesh": scene_tm})

    # --- バウンディングボックスを計算してカメラ距離を決定 ---
    try:
        all_verts = np.concatenate([m.vertices for m in scene_tm.geometry.values()])
    except Exception:
        raise RuntimeError(f"GLB にジオメトリが見つかりません: {mesh_path}")

    bb_min = all_verts.min(axis=0)
    bb_max = all_verts.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    extent = (bb_max - bb_min).max()
    cam_dist = extent * distance_scale

    fov_y = math.radians(fov_y_deg)
    el = math.radians(elevation_deg)
    azimuths = [i * (360.0 / views) for i in range(views)]

    # pyrender の OffscreenRenderer は一度だけ生成して使い回す
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    output_paths: list[str] = []

    try:
        for i, az_deg in enumerate(azimuths):
            az = math.radians(az_deg)

            # カメラ位置（球面座標 → デカルト、Y軸が上）
            eye = np.array([
                center[0] + cam_dist * math.cos(el) * math.sin(az),
                center[1] + cam_dist * math.sin(el),
                center[2] + cam_dist * math.cos(el) * math.cos(az),
            ])

            cam_pose = _look_at_pose(eye, center)

            # pyrender シーンを毎視点ごとに生成（trimesh scene に camera/light を追加）
            try:
                pr_scene = pyrender.Scene.from_trimesh_scene(
                    scene_tm,
                    bg_color=np.array([1.0, 1.0, 1.0, 1.0]),
                )
            except Exception as e:
                raise RuntimeError(f"pyrender シーン生成失敗: {e}") from e

            # カメラ追加
            cam = pyrender.PerspectiveCamera(yfov=fov_y, znear=extent * 0.01, zfar=extent * 10.0)
            pr_scene.add(cam, pose=cam_pose)

            # メインライト（カメラ方向から）
            main_light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
            pr_scene.add(main_light, pose=cam_pose)

            # フィルライト（斜め上から）
            fill_pose = _look_at_pose(
                eye=center + np.array([cam_dist * 0.3, cam_dist * 0.6, cam_dist * 0.3]),
                target=center,
            )
            fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
            pr_scene.add(fill_light, pose=fill_pose)

            # アンビエント代替（正面固定の弱いライト）
            ambient_pose = _look_at_pose(
                eye=center + np.array([0, cam_dist * 0.1, cam_dist]),
                target=center,
            )
            ambient_light = pyrender.DirectionalLight(color=np.ones(3), intensity=0.8)
            pr_scene.add(ambient_light, pose=ambient_pose)

            # レンダリング
            color, _depth = renderer.render(pr_scene)

            out_path = str(output_dir / f"{prefix}_{i}.png")
            Image.fromarray(color, "RGB").save(out_path)
            output_paths.append(out_path)

        logger.debug(f"  pyrender レンダリング完了: {len(output_paths)} 枚 → {output_dir}")
    finally:
        renderer.delete()

    return output_paths
