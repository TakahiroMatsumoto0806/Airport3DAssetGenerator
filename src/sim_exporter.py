"""
T-4.2: シミュレータエクスポート

assets_final/ 配下のアセット（visual.glb + collision_*.stl + physics.json）を
MJCF (MuJoCo) 形式と Isaac Sim 用 USD 変換メタデータ JSON に変換する。

MJCF 出力:
  - visual mesh (GLB → OBJ 変換、obj2mjcf を活用)
  - collision meshes (STL)
  - mass, inertia, friction, restitution
  - robosuite MujocoObject 互換フォーマット

Isaac Sim USD メタデータ:
  - visual / collision mesh のパス
  - 物理プロパティ（Isaac Lab MeshConverterCfg 互換形式）
  - 実際の USD 変換は Isaac Sim 環境（Nucleus サーバー接続時）で実行

使用例:
    exporter = SimExporter()

    mjcf_path = exporter.export_mjcf(
        "outputs/assets_final/000001",
        "outputs/assets_final/000001/mjcf",
    )

    usd_meta_path = exporter.export_usd_metadata(
        "outputs/assets_final/000001",
        "outputs/assets_final/000001",
    )

    summary = exporter.export_batch(
        assets_dir="outputs/assets_final",
        output_dir="outputs/assets_final",
        format="both",
    )
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from xml.dom.minidom import parseString

import numpy as np
from loguru import logger


class SimExporter:
    """MJCF + Isaac Sim USD メタデータ エクスポーター"""

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _load_physics(self, asset_dir: Path) -> dict:
        """physics.json を読み込む。なければデフォルト値を返す"""
        physics_path = asset_dir / "physics.json"
        if physics_path.exists():
            with open(physics_path, encoding="utf-8") as f:
                return json.load(f)
        return {
            "mass_kg": 0.5,
            "static_friction": 0.40,
            "dynamic_friction": 0.35,
            "restitution": 0.15,
            "material": "composite_hard_suitcase",
        }

    def _find_collision_stls(self, asset_dir: Path) -> list[Path]:
        """collision_*.stl を昇順で返す"""
        return sorted((asset_dir / "collisions").glob("collision_*.stl"))

    def _glb_to_obj(self, glb_path: Path, output_dir: Path) -> Path:
        """GLB → OBJ に変換する（trimesh を使用）"""
        import trimesh

        output_dir.mkdir(parents=True, exist_ok=True)
        obj_path = output_dir / f"{glb_path.stem}.obj"

        loaded = trimesh.load(str(glb_path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(list(loaded.dump()))

        loaded.export(str(obj_path), file_type="obj")
        return obj_path

    def _compute_inertia(self, mass_kg: float, extents_m: list[float]) -> dict:
        """
        直方体近似で慣性テンソルの対角成分を計算する。

        I_x = 1/12 * m * (y² + z²)
        I_y = 1/12 * m * (x² + z²)
        I_z = 1/12 * m * (x² + y²)
        """
        x, y, z = extents_m
        ixx = (1.0 / 12.0) * mass_kg * (y**2 + z**2)
        iyy = (1.0 / 12.0) * mass_kg * (x**2 + z**2)
        izz = (1.0 / 12.0) * mass_kg * (x**2 + y**2)
        return {"ixx": ixx, "iyy": iyy, "izz": izz, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}

    def _get_extents(self, mesh_path: Path) -> list[float]:
        """メッシュの AABB サイズ [x, y, z] を返す（単位: m）"""
        import trimesh
        loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(list(loaded.dump()))
        return loaded.extents.tolist()

    # ------------------------------------------------------------------
    # MJCF エクスポート
    # ------------------------------------------------------------------

    def export_mjcf(
        self,
        asset_dir: str,
        output_dir: str,
    ) -> str:
        """
        アセットを MJCF XML 形式でエクスポートする。

        robosuite MujocoObject 互換レイアウト:
          <mujoco>
            <asset>
              <mesh name="visual" file="visual.obj"/>
              <mesh name="collision_000" file="collisions/collision_000.stl"/>
              ...
            </asset>
            <worldbody>
              <body name="{asset_id}">
                <freejoint/>
                <inertial mass="..." pos="0 0 0" diaginertia="..."/>
                <geom type="mesh" mesh="visual" class="visual"/>
                <geom type="mesh" mesh="collision_000" class="collision"/>
                ...
              </body>
            </worldbody>
            <default>
              <default class="visual">
                <geom contype="0" conaffinity="0" group="1"/>
              </default>
              <default class="collision">
                <geom contype="1" conaffinity="1" friction="..."/>
              </default>
            </default>
          </mujoco>

        Args:
            asset_dir:  アセットディレクトリ（visual.glb, collisions/, physics.json を含む）
            output_dir: MJCF 出力先ディレクトリ

        Returns:
            str: 出力 MJCF ファイルパス
        """
        asset_dir = Path(asset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        asset_id = asset_dir.name
        physics = self._load_physics(asset_dir)
        mass_kg = float(physics.get("mass_kg", 0.5))
        static_f = float(physics.get("static_friction", 0.40))
        dynamic_f = float(physics.get("dynamic_friction", 0.35))
        restitution = float(physics.get("restitution", 0.15))

        # visual GLB → OBJ 変換
        visual_glb = asset_dir / "visual.glb"
        if not visual_glb.exists():
            raise FileNotFoundError(f"visual.glb が見つかりません: {visual_glb}")

        obj_path = self._glb_to_obj(visual_glb, output_dir)

        # 慣性計算
        try:
            extents = self._get_extents(visual_glb)
        except Exception:
            extents = [0.06, 0.04, 0.08]
        inertia = self._compute_inertia(mass_kg, extents)

        # STL パスリスト（output_dir 相対パス用にコピー）
        collision_stls = self._find_collision_stls(asset_dir)
        collision_dest_dir = output_dir / "collisions"
        collision_dest_dir.mkdir(exist_ok=True)

        import shutil
        copied_stls: list[Path] = []
        for stl in collision_stls:
            dest = collision_dest_dir / stl.name
            if stl != dest:
                shutil.copy2(str(stl), str(dest))
            copied_stls.append(dest)

        # XML 構築
        mujoco = ET.Element("mujoco", model=asset_id)

        # <compiler>
        ET.SubElement(mujoco, "compiler", meshdir=".", angle="radian")

        # <default>
        default = ET.SubElement(mujoco, "default")
        vis_default = ET.SubElement(default, "default", **{"class": "visual"})
        ET.SubElement(vis_default, "geom", contype="0", conaffinity="0", group="1")
        col_default = ET.SubElement(default, "default", **{"class": "collision"})
        ET.SubElement(
            col_default, "geom",
            contype="1", conaffinity="1",
            friction=f"{static_f:.4f} {dynamic_f:.4f} {restitution:.4f}",
        )

        # <asset>
        asset = ET.SubElement(mujoco, "asset")
        ET.SubElement(asset, "mesh", name="visual", file=obj_path.name)
        for stl in copied_stls:
            ET.SubElement(
                asset, "mesh",
                name=stl.stem,
                file=str(Path("collisions") / stl.name),
            )

        # <worldbody>
        worldbody = ET.SubElement(mujoco, "worldbody")
        body = ET.SubElement(worldbody, "body", name=asset_id, pos="0 0 0")
        ET.SubElement(body, "freejoint")
        ET.SubElement(
            body, "inertial",
            mass=f"{mass_kg:.6f}",
            pos="0 0 0",
            diaginertia=f"{inertia['ixx']:.8f} {inertia['iyy']:.8f} {inertia['izz']:.8f}",
        )
        ET.SubElement(body, "geom", type="mesh", mesh="visual", **{"class": "visual"})
        for stl in copied_stls:
            ET.SubElement(body, "geom", type="mesh", mesh=stl.stem, **{"class": "collision"})

        # 整形して保存
        xml_str = parseString(ET.tostring(mujoco, encoding="unicode")).toprettyxml(indent="  ")
        # 先頭の XML 宣言を除去（MuJoCo は宣言不要）
        xml_str = "\n".join(xml_str.split("\n")[1:])

        mjcf_path = output_dir / f"{asset_id}.xml"
        with open(mjcf_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

        logger.info(f"  MJCF エクスポート: {mjcf_path}")
        return str(mjcf_path)

    # ------------------------------------------------------------------
    # Isaac Sim USD メタデータ
    # ------------------------------------------------------------------

    def export_usd_metadata(
        self,
        asset_dir: str,
        output_dir: str,
    ) -> str:
        """
        Isaac Sim 用の USD 変換メタデータを JSON で出力する。

        Isaac Lab MeshConverterCfg 互換形式:
        {
          "asset_id":            str,
          "visual_mesh_path":    str,   # visual.glb への絶対パス
          "collision_mesh_paths": [str],
          "physics": {
            "mass_kg":           float,
            "static_friction":   float,
            "dynamic_friction":  float,
            "restitution":       float,
          },
          "usd_output_path":     str,   # 出力先 .usd ファイルパス（Isaac Sim で生成）
          "conversion_config": {
            "make_instanceable":  true,
            "collision_approximation": "convexHull",
          }
        }

        Args:
            asset_dir:  アセットディレクトリ
            output_dir: メタデータ JSON の出力先

        Returns:
            str: 出力 JSON ファイルパス
        """
        asset_dir = Path(asset_dir).resolve()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        asset_id = asset_dir.name
        physics = self._load_physics(asset_dir)

        visual_glb = asset_dir / "visual.glb"
        collision_stls = self._find_collision_stls(asset_dir)

        metadata = {
            "asset_id": asset_id,
            "visual_mesh_path": str(visual_glb),
            "collision_mesh_paths": [str(p) for p in collision_stls],
            "physics": {
                "mass_kg": float(physics.get("mass_kg", 0.5)),
                "static_friction": float(physics.get("static_friction", 0.40)),
                "dynamic_friction": float(physics.get("dynamic_friction", 0.35)),
                "restitution": float(physics.get("restitution", 0.15)),
            },
            "usd_output_path": str(output_dir / f"{asset_id}.usd"),
            "conversion_config": {
                "make_instanceable": True,
                "collision_approximation": "convexDecomposition",
            },
            "material": physics.get("material", "unknown"),
            "luggage_type": physics.get("luggage_type"),
        }

        json_path = output_dir / f"{asset_id}_usd_meta.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"  USD メタデータ出力: {json_path}")
        return str(json_path)

    # ------------------------------------------------------------------
    # バッチエクスポート
    # ------------------------------------------------------------------

    def export_batch(
        self,
        assets_dir: str,
        output_dir: str,
        format: str = "both",
        resume: bool = True,
    ) -> dict:
        """
        assets_dir 内の全アセットをバッチエクスポートする。

        Args:
            assets_dir: アセットルートディレクトリ（各サブディレクトリがアセット）
            output_dir: 出力ルートディレクトリ
            format:     "mjcf" | "usd" | "both"（デフォルト "both"）
            resume:     True の場合、既に出力済みのアセットをスキップ

        Returns:
            {
                "total":    int,
                "success":  int,
                "failed":   int,
                "results":  list[dict],
            }
        """
        assets_dir = Path(assets_dir)
        output_dir = Path(output_dir)

        # アセットディレクトリ一覧（visual.glb が存在するものだけ対象）
        asset_dirs = sorted(
            d for d in assets_dir.iterdir()
            if d.is_dir() and (d / "visual.glb").exists()
        )
        total = len(asset_dirs)
        logger.info(f"シミュレータエクスポートバッチ開始: {total} 件 → {output_dir}")

        results: list[dict] = []
        success = failed = 0

        for asset_dir in asset_dirs:
            asset_id = asset_dir.name
            asset_output = output_dir / asset_id

            entry: dict = {"asset_id": asset_id, "mjcf_path": None, "usd_meta_path": None,
                           "status": "failed", "error": None}

            # resume チェック
            already_mjcf = (asset_output / "mjcf" / f"{asset_id}.xml").exists()
            already_usd = (asset_output / f"{asset_id}_usd_meta.json").exists()
            if resume:
                if format == "both" and already_mjcf and already_usd:
                    entry["status"] = "skipped"
                    success += 1
                    results.append(entry)
                    continue
                elif format == "mjcf" and already_mjcf:
                    entry["status"] = "skipped"
                    success += 1
                    results.append(entry)
                    continue
                elif format == "usd" and already_usd:
                    entry["status"] = "skipped"
                    success += 1
                    results.append(entry)
                    continue

            try:
                if format in ("mjcf", "both"):
                    mjcf_output = asset_output / "mjcf"
                    entry["mjcf_path"] = self.export_mjcf(str(asset_dir), str(mjcf_output))

                if format in ("usd", "both"):
                    entry["usd_meta_path"] = self.export_usd_metadata(
                        str(asset_dir), str(asset_output)
                    )

                entry["status"] = "success"
                success += 1

            except Exception as e:
                entry["error"] = str(e)
                logger.error(f"  エクスポート失敗 ({asset_id}): {e}")
                failed += 1

            results.append(entry)

            done = success + failed
            if done % 20 == 0 or done == total:
                logger.info(f"  進捗: {done}/{total} (成功={success}, 失敗={failed})")

        logger.info(f"エクスポートバッチ完了: 成功={success}, 失敗={failed}")
        return {"total": total, "success": success, "failed": failed, "results": results}
