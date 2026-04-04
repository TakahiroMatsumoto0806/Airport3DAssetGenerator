"""
T-3.2: メッシュ品質チェックエンジン

Open3D + trimesh を使ったメッシュ品質チェックと修復。

チェック項目:
  - watertight（閉じたメッシュか）
  - edge/vertex manifold（多様体条件）
  - self-intersection（自己交差）
  - face count（5K〜100K）
  - degenerate faces（縮退面）
  - normal consistency（法線一貫性）
  - bounding box aspect ratio（アスペクト比妥当性）

修復対象:
  - degenerate face 除去
  - duplicate face/vertex 除去
  - hole filling
  - normal fix

使用例:
    qa = MeshQA()
    result = qa.check_single("outputs/meshes_raw/000001.glb")
    # result["pass"] が False なら repair を試みる
    repair_result = qa.repair("outputs/meshes_raw/000001.glb", "outputs/meshes_approved/000001.glb")

    summary = qa.check_batch(
        mesh_dir="outputs/meshes_raw",
        output_json="outputs/meshes_approved/mesh_qa_results.json",
    )
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


# 品質基準（仕様書記載値）
_FACE_COUNT_MIN = 5_000
_FACE_COUNT_MAX = 100_000

# アスペクト比上限（直方体としての妥当性チェック）
_ASPECT_RATIO_MAX = 20.0


class MeshQA:
    """Open3D + trimesh による GLB メッシュ品質チェックエンジン"""

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _load_trimesh(self, mesh_path: str):
        """GLB/OBJ ファイルを trimesh でロード。Scene の場合は dump() で展開"""
        import trimesh

        loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.dump())
            if not meshes:
                raise ValueError(f"メッシュオブジェクトが見つかりません: {mesh_path}")
            loaded = trimesh.util.concatenate(meshes)
        return loaded

    def _load_open3d(self, mesh_path: str):
        """GLB/OBJ ファイルを Open3D でロード"""
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(mesh.vertices) == 0:
            raise ValueError(f"Open3D: メッシュが空です: {mesh_path}")
        return mesh

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def check_single(self, mesh_path: str) -> dict:
        """
        1 つのメッシュに対して品質チェックを実行する。

        Args:
            mesh_path: GLB / OBJ ファイルパス

        Returns:
            dict:
                {
                    "mesh_path":          str,
                    "is_watertight":      bool,
                    "is_edge_manifold":   bool,
                    "is_vertex_manifold": bool,
                    "has_self_intersection": bool,
                    "face_count":         int,
                    "face_count_ok":      bool,   # 5K ≤ faces ≤ 100K
                    "degenerate_count":   int,
                    "is_normal_consistent": bool,
                    "aspect_ratio":       float,
                    "aspect_ratio_ok":    bool,
                    "pass":               bool,
                    "issues":             list[str],
                    "timestamp":          str,
                }
        """
        import trimesh
        import open3d as o3d

        mesh_path = str(mesh_path)
        ts = datetime.now(timezone.utc).isoformat()
        issues: list[str] = []

        result: dict = {
            "mesh_path": mesh_path,
            "is_watertight": False,
            "is_edge_manifold": False,
            "is_vertex_manifold": False,
            "has_self_intersection": False,
            "face_count": 0,
            "face_count_ok": False,
            "degenerate_count": 0,
            "is_normal_consistent": False,
            "aspect_ratio": 0.0,
            "aspect_ratio_ok": False,
            "pass": False,
            "issues": issues,
            "timestamp": ts,
        }

        # --- trimesh ロード ---
        try:
            tm = self._load_trimesh(mesh_path)
        except Exception as e:
            issues.append(f"trimesh ロード失敗: {e}")
            return result

        # face count
        face_count = len(tm.faces)
        result["face_count"] = face_count
        result["face_count_ok"] = _FACE_COUNT_MIN <= face_count <= _FACE_COUNT_MAX
        if not result["face_count_ok"]:
            issues.append(
                f"face_count={face_count} (基準: {_FACE_COUNT_MIN}〜{_FACE_COUNT_MAX})"
            )

        # watertight
        result["is_watertight"] = bool(tm.is_watertight)
        if not result["is_watertight"]:
            issues.append("watertight でない（穴あり）")

        # degenerate faces
        degen = tm.faces[trimesh.triangles.area(tm.triangles) < 1e-12]
        result["degenerate_count"] = len(degen)
        if len(degen) > 0:
            issues.append(f"degenerate faces: {len(degen)}")

        # normal consistency（ウィンディング数が一貫しているか）
        try:
            result["is_normal_consistent"] = bool(tm.is_winding_consistent)
        except Exception:
            result["is_normal_consistent"] = False
        if not result["is_normal_consistent"]:
            issues.append("法線の一貫性なし")

        # bounding box aspect ratio
        try:
            extents = tm.extents  # [x, y, z] サイズ
            max_e = float(np.max(extents))
            min_e = float(np.min(extents))
            if min_e > 0:
                aspect = max_e / min_e
            else:
                aspect = float("inf")
            result["aspect_ratio"] = round(aspect, 3)
            result["aspect_ratio_ok"] = aspect <= _ASPECT_RATIO_MAX
            if not result["aspect_ratio_ok"]:
                issues.append(f"aspect_ratio={aspect:.1f} (上限: {_ASPECT_RATIO_MAX})")
        except Exception as e:
            issues.append(f"aspect_ratio 計算失敗: {e}")

        # self-intersection (trimesh)
        try:
            result["has_self_intersection"] = bool(
                trimesh.repair.broken_faces(tm).size > 0
            ) if not tm.is_watertight else False
        except Exception:
            result["has_self_intersection"] = False

        # --- Open3D でマニフォールド判定 ---
        try:
            o3d_mesh = self._load_open3d(mesh_path)
            result["is_edge_manifold"] = bool(o3d_mesh.is_edge_manifold(allow_boundary_edges=False))
            result["is_vertex_manifold"] = bool(o3d_mesh.is_vertex_manifold())
            if not result["is_edge_manifold"]:
                issues.append("edge manifold でない")
            if not result["is_vertex_manifold"]:
                issues.append("vertex manifold でない")
        except Exception as e:
            issues.append(f"Open3D マニフォールド判定失敗: {e}")

        # --- 総合判定 ---
        # pass 条件: face_count_ok AND watertight AND edge/vertex manifold
        #            AND degenerate=0 AND normal_consistent AND aspect_ratio_ok
        result["pass"] = (
            result["face_count_ok"]
            and result["is_watertight"]
            and result["is_edge_manifold"]
            and result["is_vertex_manifold"]
            and result["degenerate_count"] == 0
            and result["is_normal_consistent"]
            and result["aspect_ratio_ok"]
        )

        return result

    def repair(self, mesh_path: str, output_path: str) -> dict:
        """
        メッシュ修復を試みる。

        修復内容:
          1. degenerate face 除去
          2. duplicate face/vertex 除去
          3. hole filling
          4. normal fix（ウィンディング統一）

        Args:
            mesh_path:   入力 GLB / OBJ
            output_path: 修復後の出力パス

        Returns:
            {
                "before": dict,   # check_single() の修復前結果
                "after":  dict,   # check_single() の修復後結果
                "repaired": bool, # 修復によって pass になったか
            }
        """
        import trimesh

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        before = self.check_single(mesh_path)

        try:
            tm = self._load_trimesh(mesh_path)

            # 1. degenerate faces 除去（nondegenerate_faces() + update_faces()）
            trimesh.repair.fix_winding(tm)
            tm.update_faces(tm.nondegenerate_faces())

            # 2. duplicate vertices/faces 除去
            tm.merge_vertices()
            tm.update_faces(tm.unique_faces())
            tm.remove_unreferenced_vertices()

            # 3. hole filling（trimesh は fill_holes を提供）
            trimesh.repair.fill_holes(tm)

            # 4. normal fix
            trimesh.repair.fix_normals(tm)

            # 保存（trimesh は GLB / OBJ どちらも export 可能）
            ext = output_path.suffix.lower()
            if ext == ".glb":
                tm.export(str(output_path), file_type="glb")
            elif ext == ".obj":
                tm.export(str(output_path), file_type="obj")
            else:
                tm.export(str(output_path))

        except Exception as e:
            logger.error(f"  修復失敗 ({Path(mesh_path).name}): {e}")
            return {
                "before": before,
                "after": before,
                "repaired": False,
                "error": str(e),
            }

        after = self.check_single(str(output_path))

        return {
            "before": before,
            "after": after,
            "repaired": after["pass"],
        }

    def check_batch(
        self,
        mesh_dir: str,
        output_json: str,
        approved_dir: Optional[str] = None,
        extensions: tuple[str, ...] = (".glb", ".obj"),
        attempt_repair: bool = True,
    ) -> dict:
        """
        ディレクトリ内の全メッシュに品質チェックを実行する。

        - 不合格メッシュは修復を試み、合格したものを approved_dir に保存
        - 修復不能なものは failed 扱い
        - 結果を output_json に保存

        Args:
            mesh_dir:       入力メッシュディレクトリ
            output_json:    チェック結果 JSON 保存先
            approved_dir:   合格メッシュのコピー先（None の場合はコピーしない）
            extensions:     対象拡張子
            attempt_repair: 不合格時に修復を試みるか

        Returns:
            {
                "total":    int,
                "passed":   int,
                "repaired": int,
                "failed":   int,
                "results":  list[dict],
            }
        """
        import shutil

        mesh_dir = Path(mesh_dir)
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        if approved_dir:
            approved_dir = Path(approved_dir)
            approved_dir.mkdir(parents=True, exist_ok=True)

        mesh_files = sorted(
            f for f in mesh_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        total = len(mesh_files)
        logger.info(f"メッシュQAバッチ開始: {total} 件 → {output_json}")

        results: list[dict] = []
        passed = repaired = failed = 0

        for idx, mesh_path in enumerate(mesh_files):
            check = self.check_single(str(mesh_path))

            if check["pass"]:
                passed += 1
                check["status"] = "passed"
                if approved_dir:
                    shutil.copy2(str(mesh_path), str(approved_dir / mesh_path.name))

            elif attempt_repair:
                repair_output = (approved_dir or mesh_dir.parent / "meshes_repaired") / mesh_path.name
                if not approved_dir:
                    repair_output.parent.mkdir(parents=True, exist_ok=True)

                repair_result = self.repair(str(mesh_path), str(repair_output))

                if repair_result.get("repaired"):
                    repaired += 1
                    check["status"] = "repaired"
                    check["repair"] = repair_result
                else:
                    failed += 1
                    check["status"] = "failed"
                    check["repair"] = repair_result

            else:
                failed += 1
                check["status"] = "failed"

            results.append(check)

            done = passed + repaired + failed
            if done % 50 == 0 or done == total:
                logger.info(
                    f"  進捗: {done}/{total} "
                    f"(合格={passed}, 修復={repaired}, 不合格={failed})"
                )

        summary = {
            "total": total,
            "passed": passed,
            "repaired": repaired,
            "failed": failed,
            "results": results,
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(
            f"メッシュQA完了: 合格={passed}, 修復={repaired}, 不合格={failed}"
        )
        return summary
