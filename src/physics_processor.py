"""
T-4.1: コリジョン生成 + 物理プロパティ付与

処理フロー:
  1. スケール正規化（ScaleNormalizer）
  2. CoACD で凸分解コリジョンメッシュ生成
  3. material_properties.yaml から物理プロパティを取得・付与（±15% ランダム変動）

使用例:
    processor = PhysicsProcessor("configs/material_properties.yaml")

    result = processor.process_single(
        mesh_path="outputs/meshes_approved/000001.glb",
        output_dir="outputs/assets_final",
        material="polycarbonate",
        luggage_type="hard_suitcase",
    )
    # result["collision_paths"]    → ["outputs/assets_final/000001/collision_0.stl", ...]
    # result["physics"]            → {"mass_kg": 0.23, "static_friction": 0.36, ...}

    summary = processor.process_batch(
        mesh_dir="outputs/meshes_approved",
        metadata_json="outputs/meshes_approved/vlm_qa_results.json",
        output_dir="outputs/assets_final",
    )
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class PhysicsProcessor:
    """CoACD コリジョン生成 + 物理プロパティ付与エンジン"""

    def __init__(
        self,
        material_config_path: str = "configs/material_properties.yaml",
        seed: int = 42,
    ) -> None:
        """
        Args:
            material_config_path: material_properties.yaml のパス
            seed:                 物理プロパティのランダム変動に使う乱数シード
        """
        from omegaconf import OmegaConf

        self._cfg = OmegaConf.load(material_config_path)
        self._materials: dict = dict(self._cfg.get("materials", {}))
        self._category_map: dict = dict(self._cfg.get("category_material_mapping", {}))
        self._default_randomize: float = float(
            self._cfg.get("defaults", {}).get("randomize_range", 0.15)
        )
        self._rng = random.Random(seed)

        from src.scale_normalizer import ScaleNormalizer
        self._normalizer = ScaleNormalizer(material_config_path)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _resolve_material(
        self,
        material: Optional[str],
        luggage_type: Optional[str],
    ) -> str:
        """material が None の場合、luggage_type からデフォルト材質を解決する"""
        if material and material in self._materials:
            return material
        if luggage_type and luggage_type in self._category_map:
            return self._category_map[luggage_type]
        # フォールバック
        return "composite_hard_suitcase"

    def _randomize(self, value: float, range_ratio: float) -> float:
        """値を ±range_ratio の範囲でランダム変動させる"""
        delta = value * range_ratio
        return round(self._rng.uniform(value - delta, value + delta), 6)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def generate_collision(
        self,
        mesh_path: str,
        output_dir: str,
        threshold: float = 0.08,
        max_convex_hull: int = 16,
        max_ch_vertex: int = 256,
        resolution: int = 2000,
        mcts_iterations: int = 150,
    ) -> list[str]:
        """
        CoACD で凸分解コリジョンメッシュを生成し STL として保存する。

        Args:
            mesh_path:        入力 GLB / OBJ ファイルパス
            output_dir:       コリジョン STL の出力先ディレクトリ
            threshold:        凸度閾値（0.01〜1.0、小さいほど細かく分解）
            max_convex_hull:  凸包の最大個数（-1 で無制限）
            max_ch_vertex:    凸包1個あたりの最大頂点数（ポリゴン数の上限）
            resolution:       分解解像度（高いほど精密）
            mcts_iterations:  MCTS 探索反復数（多いほど分解品質が上がる）

        Returns:
            list[str]: 生成されたコリジョン STL ファイルのパスリスト

        Raises:
            ImportError: coacd パッケージが見つからない場合
        """
        try:
            import coacd
        except ImportError as e:
            raise ImportError(
                f"coacd パッケージが見つかりません: {e}\n"
                "pip install coacd でインストールしてください:\n"
                "  uv pip install coacd"
            ) from e

        import trimesh

        mesh_path = Path(mesh_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # メッシュロード（trimesh → coacd 用 Mesh に変換）
        loaded = trimesh.load(str(mesh_path), force="mesh", process=True)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(list(loaded.dump()))

        mesh_coacd = coacd.Mesh(
            vertices=np.array(loaded.vertices, dtype=np.float64),
            indices=np.array(loaded.faces, dtype=np.int32),
        )

        logger.debug(
            f"  CoACD 凸分解: {mesh_path.name} "
            f"(threshold={threshold}, max_hull={max_convex_hull}, "
            f"max_ch_vertex={max_ch_vertex}, resolution={resolution})"
        )
        parts = coacd.run_coacd(
            mesh_coacd,
            threshold=threshold,
            max_convex_hull=max_convex_hull,
            max_ch_vertex=max_ch_vertex,
            resolution=resolution,
            mcts_iterations=mcts_iterations,
        )

        collision_paths: list[str] = []
        for i, part in enumerate(parts):
            part_mesh = trimesh.Trimesh(
                vertices=np.array(part[0]),
                faces=np.array(part[1]),
            )
            stl_path = output_dir / f"collision_{i:03d}.stl"
            part_mesh.export(str(stl_path), file_type="stl")
            collision_paths.append(str(stl_path))

        logger.debug(f"    → {len(collision_paths)} 個の凸パーツ生成")
        return collision_paths

    def assign_properties(
        self,
        mesh_path: str,
        material: str,
        randomize: bool = True,
    ) -> dict:
        """
        材質キーに基づいて物理プロパティを計算する。

        質量は「密度 × メッシュ体積」で算出。
        randomize=True の場合、各値を ±randomize_range でランダム変動させる。

        Args:
            mesh_path: GLB / OBJ ファイルパス（体積計算に使用）
            material:  材質キー（material_properties.yaml の materials キー）
            randomize: True でランダム変動を適用

        Returns:
            {
                "material":         str,
                "density_kg_m3":    float,
                "static_friction":  float,
                "dynamic_friction": float,
                "restitution":      float,
                "volume_m3":        float,
                "mass_kg":          float,
            }
        """
        import trimesh

        if material not in self._materials:
            logger.warning(f"  材質 '{material}' が未定義。composite_hard_suitcase にフォールバック")
            material = "composite_hard_suitcase"

        mat = self._materials[material]
        rr = float(mat.get("randomize_range", self._default_randomize))

        density = float(mat["density_kg_m3"])
        static_f = float(mat["static_friction"])
        dynamic_f = float(mat["dynamic_friction"])
        restitution = float(mat["restitution"])

        if randomize:
            density = self._randomize(density, rr)
            static_f = self._randomize(static_f, rr)
            dynamic_f = self._randomize(dynamic_f, rr)
            restitution = self._randomize(restitution, rr)

        # 体積計算（scaled mesh から）
        loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(list(loaded.dump()))
        volume_m3 = abs(float(loaded.volume)) if loaded.is_watertight else abs(float(loaded.convex_hull.volume))

        mass_kg = density * volume_m3

        return {
            "material": material,
            "density_kg_m3": round(density, 2),
            "static_friction": round(static_f, 4),
            "dynamic_friction": round(dynamic_f, 4),
            "restitution": round(restitution, 4),
            "volume_m3": round(volume_m3, 8),
            "mass_kg": round(mass_kg, 6),
        }

    def process_single(
        self,
        mesh_path: str,
        output_dir: str,
        material: Optional[str] = None,
        luggage_type: Optional[str] = None,
        coacd_threshold: float = 0.08,
        coacd_max_convex_hull: int = 16,
        coacd_max_ch_vertex: int = 256,
        coacd_resolution: int = 2000,
        coacd_mcts_iterations: int = 150,
        randomize: bool = True,
    ) -> dict:
        """
        1 アセットの完全処理:
          スケール正規化 → コリジョン生成 → 物理プロパティ付与

        出力ディレクトリ構造:
          output_dir/{asset_stem}/
            visual.glb          # スケール正規化済みメッシュ
            collision_000.stl   # CoACD 凸パーツ
            collision_001.stl
            ...
            physics.json        # 物理プロパティ

        Args:
            mesh_path:              入力 GLB ファイルパス
            output_dir:             出力ディレクトリ（{asset_stem}/ サブディレクトリを作成）
            material:               材質キー（None の場合は luggage_type から推論）
            luggage_type:           荷物タイプ（material 解決のヒント）
            coacd_threshold:        CoACD 凸度閾値
            coacd_max_convex_hull:  凸包の最大個数
            coacd_max_ch_vertex:    凸包1個あたりの最大頂点数（ポリゴン数の上限）
            coacd_resolution:       分解解像度
            coacd_mcts_iterations:  MCTS 探索反復数
            randomize:              物理プロパティにランダム変動を加えるか

        Returns:
            {
                "asset_id":        str,
                "visual_path":     str,
                "collision_paths": list[str],
                "physics":         dict,
                "scale":           dict,
                "status":          "success" | "failed",
                "error":           str | None,
            }
        """
        mesh_path = Path(mesh_path)
        asset_id = mesh_path.stem
        asset_dir = Path(output_dir) / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)

        result: dict = {
            "asset_id": asset_id,
            "visual_path": None,
            "collision_paths": [],
            "physics": {},
            "scale": {},
            "status": "failed",
            "error": None,
        }

        try:
            # 1. スケール正規化
            visual_path = asset_dir / "visual.glb"
            scale_result = self._normalizer.normalize(
                str(mesh_path),
                luggage_type=luggage_type,
                miniature=True,
                output_path=str(visual_path),
            )
            result["scale"] = scale_result
            result["visual_path"] = str(visual_path)

            # 2. コリジョン生成（スケール後のメッシュを使用）
            collision_dir = asset_dir / "collisions"
            collision_paths = self.generate_collision(
                str(visual_path),
                str(collision_dir),
                threshold=coacd_threshold,
                max_convex_hull=coacd_max_convex_hull,
                max_ch_vertex=coacd_max_ch_vertex,
                resolution=coacd_resolution,
                mcts_iterations=coacd_mcts_iterations,
            )
            result["collision_paths"] = collision_paths

            # 3. 物理プロパティ付与
            resolved_material = self._resolve_material(material, luggage_type)
            physics = self.assign_properties(str(visual_path), resolved_material, randomize)
            result["physics"] = physics

            # physics.json に保存
            physics_json_path = asset_dir / "physics.json"
            with open(physics_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        **physics,
                        "asset_id": asset_id,
                        "luggage_type": luggage_type,
                        "collision_count": len(collision_paths),
                        "scale_factor": scale_result["scale_factor"],
                    },
                    f, ensure_ascii=False, indent=2,
                )

            result["status"] = "success"
            logger.info(
                f"  物理付与完了: {asset_id} "
                f"(material={resolved_material}, mass={physics['mass_kg']:.4f}kg, "
                f"collisions={len(collision_paths)})"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"  物理付与失敗 ({asset_id}): {e}")

        return result

    def process_batch(
        self,
        mesh_dir: str,
        output_dir: str,
        metadata_json: Optional[str] = None,
        coacd_threshold: float = 0.08,
        coacd_max_convex_hull: int = 16,
        coacd_max_ch_vertex: int = 256,
        coacd_resolution: int = 2000,
        coacd_mcts_iterations: int = 150,
        extensions: tuple[str, ...] = (".glb",),
        resume: bool = True,
    ) -> dict:
        """
        ディレクトリ内の全 GLB メッシュをバッチ処理する。

        metadata_json（VLM QA 結果 JSON）があれば detected_type / detected_material を
        自動的に material / luggage_type として使用する。

        Args:
            mesh_dir:               入力メッシュディレクトリ
            output_dir:             出力ルートディレクトリ
            metadata_json:          VLM QA 結果 JSON（mesh_vlm_qa の output_json）
            coacd_threshold:        CoACD 凸度閾値
            coacd_max_convex_hull:  凸包の最大個数
            coacd_max_ch_vertex:    凸包1個あたりの最大頂点数（ポリゴン数の上限）
            coacd_resolution:       分解解像度
            coacd_mcts_iterations:  MCTS 探索反復数
            extensions:             対象拡張子
            resume:                 True の場合、physics.json が既存のアセットをスキップ

        Returns:
            {
                "total":    int,
                "success":  int,
                "failed":   int,
                "results":  list[dict],
            }
        """
        mesh_dir = Path(mesh_dir)
        output_dir = Path(output_dir)

        # VLM QA メタデータ読み込み（material / type 情報）
        vlm_meta: dict[str, dict] = {}
        if metadata_json and Path(metadata_json).exists():
            try:
                with open(metadata_json, encoding="utf-8") as f:
                    for entry in json.load(f).get("results", []):
                        key = Path(entry.get("mesh_path", "")).stem
                        vlm_meta[key] = entry
                logger.info(f"VLM QA メタデータ読み込み: {len(vlm_meta)} 件")
            except Exception as e:
                logger.warning(f"VLM QA メタデータ読み込み失敗: {e}")

        mesh_files = sorted(
            f for f in mesh_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )

        # VLM QA の pass 結果でフィルタリング
        # metadata_json が指定されている場合は pass: True のメッシュのみを処理する。
        # vlm_meta に存在しないメッシュ（vlm_qa 未評価）も除外する。
        if vlm_meta:
            before = len(mesh_files)
            mesh_files = [
                f for f in mesh_files
                if vlm_meta.get(f.stem, {}).get("pass", False)
            ]
            filtered = before - len(mesh_files)
            if filtered > 0:
                logger.info(
                    f"VLM QA フィルタ: {before} 件 → {len(mesh_files)} 件 "
                    f"({filtered} 件を除外: pass=False または未評価)"
                )

        total = len(mesh_files)
        logger.info(f"物理付与バッチ開始: {total} 件 → {output_dir}")

        results: list[dict] = []
        success = failed = 0

        for mesh_path in mesh_files:
            asset_id = mesh_path.stem

            # resume チェック
            if resume and (output_dir / asset_id / "physics.json").exists():
                logger.debug(f"  スキップ（既存）: {asset_id}")
                with open(output_dir / asset_id / "physics.json", encoding="utf-8") as f:
                    existing = json.load(f)
                results.append({**existing, "status": "skipped"})
                success += 1
                continue

            # VLM QA からメタデータ取得
            meta = vlm_meta.get(asset_id, {})
            luggage_type = meta.get("detected_type") or None
            material = meta.get("detected_material") or None

            result = self.process_single(
                str(mesh_path),
                str(output_dir),
                material=material,
                luggage_type=luggage_type,
                coacd_threshold=coacd_threshold,
                coacd_max_convex_hull=coacd_max_convex_hull,
                coacd_max_ch_vertex=coacd_max_ch_vertex,
                coacd_resolution=coacd_resolution,
                coacd_mcts_iterations=coacd_mcts_iterations,
            )
            results.append(result)

            if result["status"] == "success":
                success += 1
            else:
                failed += 1

            done = success + failed
            if done % 20 == 0 or done == total:
                logger.info(f"  進捗: {done}/{total} (成功={success}, 失敗={failed})")

        logger.info(f"物理付与バッチ完了: 成功={success}, 失敗={failed}")
        return {"total": total, "success": success, "failed": failed, "results": results}

    def generate_html_report(
        self,
        output_path: str,
        assets_dir: str = "outputs/assets_final",
        render_dir: str = "outputs/renders",
    ) -> None:
        """物理プロパティ付与結果の HTML レポートを生成する。

        Args:
            output_path: 出力 HTML ファイルパス
            assets_dir:  assets_final ルートディレクトリ
            render_dir:  レンダリング画像ルートディレクトリ
        """
        import base64
        import html as html_mod

        assets_path = Path(assets_dir)
        render_path = Path(render_dir)

        skip_dirs = {"collisions", "isaac", "metadata", "mjcf"}
        entries: list[dict] = []
        for asset_dir in sorted(assets_path.iterdir()):
            if not asset_dir.is_dir() or asset_dir.name in skip_dirs:
                continue
            phys_json = asset_dir / "physics.json"
            if not phys_json.exists():
                continue
            with open(phys_json, encoding="utf-8") as f:
                data = json.load(f)
            entries.append(data)

        def _img_tag(asset_id: str) -> str:
            img_path = render_path / asset_id / f"{asset_id}_0.png"
            if img_path.exists():
                b64 = base64.b64encode(img_path.read_bytes()).decode()
                return f'<img src="data:image/png;base64,{b64}" style="height:80px;width:auto;">'
            return "<span style='color:#aaa'>—</span>"

        def _size_str(dims: dict | None, asset_id: str) -> str:
            if dims:
                return f"{dims.get('H','?')} × {dims.get('W','?')} × {dims.get('D','?')}"
            # フォールバック: visual.glb から計算
            visual = assets_path / asset_id / "visual.glb"
            if visual.exists():
                try:
                    import trimesh
                    loaded = trimesh.load(str(visual), force="mesh", process=False)
                    if isinstance(loaded, trimesh.Scene):
                        loaded = trimesh.util.concatenate(list(loaded.dump()))
                    extents = sorted(loaded.extents * 1000.0, reverse=True)
                    return f"{extents[0]:.1f} × {extents[1]:.1f} × {extents[2]:.1f}"
                except Exception:
                    pass
            return "—"

        rows = []
        for i, d in enumerate(entries, 1):
            asset_id = d.get("asset_id", "")
            img = _img_tag(asset_id)
            size = _size_str(d.get("dimensions_mm"), asset_id)
            row = (
                f"<tr>"
                f"<td>{i}</td>"
                f"<td>{html_mod.escape(asset_id)}.glb</td>"
                f"<td style='text-align:center'>{img}</td>"
                f"<td>{html_mod.escape(str(d.get('material', '—')))}</td>"
                f"<td style='white-space:nowrap'>{size}</td>"
                f"<td>{d.get('mass_kg', '—')}</td>"
                f"<td>{d.get('density_kg_m3', '—')}</td>"
                f"<td>{d.get('static_friction', '—')}</td>"
                f"<td>{d.get('dynamic_friction', '—')}</td>"
                f"<td>{d.get('restitution', '—')}</td>"
                f"</tr>"
            )
            rows.append(row)

        html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Physics Properties Report</title>
<style>
  body {{ font-family: sans-serif; font-size: 13px; padding: 16px; }}
  h1 {{ font-size: 1.2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: middle; }}
  th {{ background: #f0f0f0; text-align: center; }}
  tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>物理プロパティレポート — {len(entries)} 件</h1>
<table>
<thead>
<tr>
  <th>#</th><th>GLBファイル名</th><th>View 0</th><th>推定材質</th>
  <th>サイズ H×W×D [mm]</th><th>重量 (kg)</th><th>密度 (kg/m³)</th>
  <th>静止摩擦</th><th>動摩擦</th><th>反発係数</th>
</tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</body>
</html>"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")
        logger.info(f"物理プロパティレポート生成: {output_path} ({len(entries)} 件)")
