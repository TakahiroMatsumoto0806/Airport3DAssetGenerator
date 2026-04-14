"""
T-4.2: シミュレータエクスポート

assets_final/ 配下のアセット（visual.glb + collision_*.stl + physics.json）を
Isaac Sim 用 USDA ファイルおよびメタデータ JSON に変換する。

Isaac Sim USDA 出力:
  - USDA ASCII 形式（pxr 不要、ARM64 Linux でも動作）
  - PhysicsRigidBodyAPI / PhysicsMassAPI / PhysicsMaterialAPI を inline 付与
  - visual.glb をペイロード参照（PBR テクスチャ・UV 保持）
  - collision_*.stl を Mesh prim として trimesh 経由で埋め込み

使用例:
    exporter = SimExporter()

    json_path = exporter.export_usd_metadata(
        "outputs/assets_final/000001",
        "outputs/assets_final/000001",
    )

    summary = exporter.export_batch(
        assets_dir="outputs/assets_final",
        output_dir="outputs/assets_final",
    )
"""

import json
from pathlib import Path

from loguru import logger


class SimExporter:
    """Isaac Sim USDA エクスポーター"""

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

    def _get_extents(self, mesh_path: Path) -> list[float]:
        """メッシュの AABB サイズ [x, y, z] を返す（単位: m）"""
        import trimesh
        loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(list(loaded.dump()))
        return loaded.extents.tolist()

    def _generate_usda(self, asset_dir: Path, output_path: Path, physics: dict) -> None:
        """
        USDA ASCII ファイルを生成する（pxr 不要）。

        ビジュアルメッシュは @./visual.glb@ としてペイロード参照し、
        コリジョンメッシュは trimesh で STL を読み込み Mesh prim として埋め込む。
        物理プロパティは PhysicsRigidBodyAPI / PhysicsMassAPI / PhysicsMaterialAPI で付与する。
        """
        import trimesh

        asset_id = asset_dir.name
        # USD prim 名は数字・ハイフン始まり禁止
        asset_id_safe = asset_id.replace("-", "_")
        if asset_id_safe[0].isdigit():
            asset_id_safe = "asset_" + asset_id_safe

        mass_kg = float(physics.get("mass_kg", 0.5))
        static_friction = float(physics.get("static_friction", 0.40))
        dynamic_friction = float(physics.get("dynamic_friction", 0.35))
        restitution = float(physics.get("restitution", 0.15))

        # コリジョン Mesh prim 文字列を生成
        collision_stls = self._find_collision_stls(asset_dir)
        collision_prim_blocks = []
        for i, stl_path in enumerate(collision_stls):
            mesh = trimesh.load(str(stl_path), force="mesh", process=False)
            pts = ", ".join(
                f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in mesh.vertices
            )
            counts = ", ".join("3" for _ in mesh.faces)
            indices = ", ".join(str(idx) for face in mesh.faces for idx in face)
            block = (
                f'        def Mesh "collision_{i:03d}" (\n'
                f'            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n'
                f"        )\n"
                f"        {{\n"
                f'            uniform token physics:approximation = "convexHull"\n'
                f"            point3f[] points = [{pts}]\n"
                f"            int[] faceVertexCounts = [{counts}]\n"
                f"            int[] faceVertexIndices = [{indices}]\n"
                f"        }}"
            )
            collision_prim_blocks.append(block)

        collisions_section = "\n".join(collision_prim_blocks)

        usda_text = (
            "#usda 1.0\n"
            "(\n"
            f'    defaultPrim = "{asset_id_safe}"\n'
            "    metersPerUnit = 1.0\n"
            '    upAxis = "Y"\n'
            ")\n"
            "\n"
            f'def Xform "{asset_id_safe}" (\n'
            '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n'
            ")\n"
            "{\n"
            f"    float physics:mass = {mass_kg}\n"
            "\n"
            '    def Material "physics_material" (\n'
            '        prepend apiSchemas = ["PhysicsMaterialAPI"]\n'
            "    )\n"
            "    {\n"
            f"        float physics:staticFriction = {static_friction}\n"
            f"        float physics:dynamicFriction = {dynamic_friction}\n"
            f"        float physics:restitution = {restitution}\n"
            "    }\n"
            "\n"
            # payload（lazy loading）でGLBを参照。
            # GITFインポーターを持たないビューアは payload をスキップするため
            # フォーマットエラーを起こさない。
            # Isaac Sim はステージを開く際に payload を全て読み込むため、
            # omni.importer.gltf が PBR テクスチャを UsdPreviewSurface/OmniPBR へ
            # 自動変換し、テクスチャが表示される。
            '    def Xform "visual" (\n'
            "        prepend payload = @./visual.glb@\n"
            "    )\n"
            "    {\n"
            "    }\n"
            "\n"
            '    def Xform "collisions"\n'
            "    {\n"
            f"{collisions_section}\n"
            "    }\n"
            "}\n"
        )

        output_path.write_text(usda_text, encoding="utf-8")
        logger.debug(f"  USDA 生成完了: {output_path} ({len(collision_stls)} コリジョン)")

    # ------------------------------------------------------------------
    # Isaac Sim USD メタデータ
    # ------------------------------------------------------------------

    def export_usd_metadata(
        self,
        asset_dir: str,
        output_dir: str,
    ) -> str:
        """
        Isaac Sim 用の USDA ファイルおよびメタデータ JSON を出力する。

        出力ファイル:
          - {asset_id}_usd_meta.json  … Isaac Lab MeshConverterCfg 互換メタデータ
          - {asset_id}.usda           … Isaac Sim 直接インポート可能な USDA

        Args:
            asset_dir:  アセットディレクトリ（visual.glb / collisions/ / physics.json を含む）
            output_dir: 出力先ディレクトリ

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

        usda_output_path = output_dir / f"{asset_id}.usda"

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
            "usd_output_path": str(usda_output_path),
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

        # USDA ファイル生成
        self._generate_usda(asset_dir, usda_output_path, physics)
        logger.info(f"  USDA ファイル出力: {usda_output_path}")

        return str(json_path)

    # ------------------------------------------------------------------
    # バッチエクスポート
    # ------------------------------------------------------------------

    def export_batch(
        self,
        assets_dir: str,
        output_dir: str,
        resume: bool = True,
    ) -> dict:
        """
        assets_dir 内の全アセットをバッチエクスポートする。

        Args:
            assets_dir: アセットルートディレクトリ（各サブディレクトリがアセット）
            output_dir: 出力ルートディレクトリ
            resume:     True の場合、既に USDA が出力済みのアセットをスキップ

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

            entry: dict = {
                "asset_id": asset_id,
                "usd_meta_path": None,
                "usda_path": None,
                "status": "failed",
                "error": None,
            }

            # resume チェック（USDA が存在すればスキップ）
            already_usda = (asset_output / f"{asset_id}.usda").exists()
            if resume and already_usda:
                entry["status"] = "skipped"
                entry["usda_path"] = str(asset_output / f"{asset_id}.usda")
                success += 1
                results.append(entry)
                continue

            try:
                entry["usd_meta_path"] = self.export_usd_metadata(
                    str(asset_dir), str(asset_output)
                )
                entry["usda_path"] = str(asset_output / f"{asset_id}.usda")
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
