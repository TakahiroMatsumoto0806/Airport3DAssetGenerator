"""
メッシュ修復ユーティリティ

trimesh + Open3D を使った低レベルのメッシュ修復関数。
MeshQA から呼び出されるが、単独でも使用可能。

使用例:
    from src.utils.mesh_repair import remove_degenerate, fill_holes, fix_normals

    tm = trimesh.load("mesh.glb", force="mesh")
    tm = remove_degenerate(tm)
    tm = fill_holes(tm)
    tm = fix_normals(tm)
    tm.export("mesh_fixed.glb")
"""

from pathlib import Path
from typing import Union

from loguru import logger


def load_mesh(mesh_path: Union[str, Path]):
    """GLB / OBJ ファイルを trimesh でロードし単一 Trimesh を返す"""
    import trimesh

    path = Path(mesh_path)
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.dump())
        if not meshes:
            raise ValueError(f"メッシュオブジェクトが見つかりません: {path}")
        loaded = trimesh.util.concatenate(meshes)
    return loaded


def remove_degenerate(mesh) -> "trimesh.Trimesh":
    """縮退面と重複頂点/面を除去する"""
    import trimesh

    before = len(mesh.faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    after = len(mesh.faces)
    if before != after:
        logger.debug(f"  degenerate/duplicate 除去: {before} → {after} faces")
    return mesh


def fill_holes(mesh) -> "trimesh.Trimesh":
    """開いた穴を埋める"""
    import trimesh

    was_watertight = mesh.is_watertight
    trimesh.repair.fill_holes(mesh)
    now_watertight = mesh.is_watertight
    if not was_watertight and now_watertight:
        logger.debug("  fill_holes: watertight に修復")
    return mesh


def fix_normals(mesh) -> "trimesh.Trimesh":
    """ウィンディングと法線方向を統一する"""
    import trimesh

    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def repair_mesh(
    mesh_path: Union[str, Path],
    output_path: Union[str, Path],
) -> bool:
    """
    メッシュに標準修復パイプラインを適用して保存する。

    修復手順:
      1. degenerate/duplicate 除去
      2. hole filling
      3. normal fix

    Args:
        mesh_path:   入力ファイルパス
        output_path: 出力ファイルパス

    Returns:
        True: 修復成功（出力ファイルを保存）
        False: 修復失敗
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mesh = load_mesh(mesh_path)
        mesh = remove_degenerate(mesh)
        mesh = fill_holes(mesh)
        mesh = fix_normals(mesh)

        ext = output_path.suffix.lower()
        if ext == ".glb":
            mesh.export(str(output_path), file_type="glb")
        elif ext == ".obj":
            mesh.export(str(output_path), file_type="obj")
        else:
            mesh.export(str(output_path))

        return True

    except Exception as e:
        logger.error(f"repair_mesh 失敗 ({Path(mesh_path).name}): {e}")
        return False
