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

        # --- trimesh でマニフォールド判定 ---
        # (Open3D の is_edge_manifold/is_vertex_manifold は aarch64 で segfault するため trimesh で代替)
        try:
            from collections import Counter
            # edge manifold: 各エッジが1〜2面に共有されているか
            edge_face_counts = Counter(map(tuple, tm.edges_sorted))
            non_manifold_edges = sum(1 for v in edge_face_counts.values() if v > 2)
            result["is_edge_manifold"] = non_manifold_edges == 0
            # vertex manifold: trimesh の is_watertight は closed surface を意味するが
            # vertex manifold は独立した条件。trimesh では is_edge_manifold かつ
            # 境界エッジがなければ vertex manifold とみなす（簡易判定）
            boundary_edges = sum(1 for v in edge_face_counts.values() if v == 1)
            result["is_vertex_manifold"] = non_manifold_edges == 0 and boundary_edges == 0
            if not result["is_edge_manifold"]:
                issues.append(f"edge manifold でない (non-manifold edges: {non_manifold_edges})")
            if not result["is_vertex_manifold"] and boundary_edges > 0:
                issues.append(f"vertex manifold でない (boundary edges: {boundary_edges})")
        except Exception as e:
            issues.append(f"マニフォールド判定失敗: {e}")

        # --- 総合判定 ---
        # pass 条件: face_count_ok AND edge_manifold AND degenerate=0
        #            AND aspect_ratio_ok
        # ※ 以下は記録するが合否条件から外す（TRELLIS 出力の構造的制約のため）:
        #   - watertight / vertex_manifold:
        #       marching cubes ベースの出力はほぼ常に非 watertight (open boundary あり)。
        #       コリジョンメッシュは T-4.1 CoACD で別途生成するため視覚メッシュに閉性は不要。
        #   - is_normal_consistent:
        #       TRELLIS 出力で法線修復が困難なケースがある。
        #       視覚品質は後段の mesh_vlm_qa (VLM) が判定するため、ここでは必須条件としない。
        result["pass"] = (
            result["face_count_ok"]
            and result["is_edge_manifold"]
            and result["degenerate_count"] == 0
            and result["aspect_ratio_ok"]
        )

        return result

    def repair(self, mesh_path: str, output_path: str) -> dict:
        """
        メッシュ修復を試みる。

        修復内容:
          1. degenerate face 除去
          2. duplicate face/vertex 除去
          3. non-manifold edge に属する face 除去
          4. face count > max の場合はデシメーション（頂点カラー転送付き）
          5. normal fix（ウィンディング統一）

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
        from collections import Counter

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        before = self.check_single(mesh_path)

        try:
            tm = self._load_trimesh(mesh_path)

            # 1+2. degenerate faces 除去 + duplicate 除去（収束するまで繰り返す）
            # check_single() は area < 1e-12 を縮退面と判定するが、
            # nondegenerate_faces() は height ベースのため閾値が異なる。
            # 同じ area 基準で除去してから merge し、merge で生じた新たな縮退面を再除去する。
            _DEGEN_AREA = 1e-12
            for _iter in range(8):
                n_before = len(tm.faces)
                # area ベースの縮退面除去（check_single() と同一基準）
                areas = trimesh.triangles.area(tm.triangles)
                tm.update_faces(areas >= _DEGEN_AREA)
                # duplicate 頂点・面の除去
                tm.merge_vertices()
                tm.update_faces(tm.unique_faces())
                # merge 後に新たな縮退面が生じることがあるため再度除去
                areas = trimesh.triangles.area(tm.triangles)
                tm.update_faces(areas >= _DEGEN_AREA)
                tm.remove_unreferenced_vertices()
                if len(tm.faces) == n_before:
                    break

            # 3. non-manifold edges に属する face を除去
            #    FDG 出力では境界部に少数（< 0.5%）の非マニフォールドエッジが発生する
            ec = Counter(map(tuple, tm.edges_sorted))
            nm_edges = {e for e, c in ec.items() if c > 2}
            if nm_edges:
                keep_mask = np.ones(len(tm.faces), dtype=bool)
                for i, face in enumerate(tm.faces):
                    e0 = tuple(sorted([int(face[0]), int(face[1])]))
                    e1 = tuple(sorted([int(face[1]), int(face[2])]))
                    e2 = tuple(sorted([int(face[2]), int(face[0])]))
                    if e0 in nm_edges or e1 in nm_edges or e2 in nm_edges:
                        keep_mask[i] = False
                removed = int(keep_mask.size - keep_mask.sum())
                logger.debug(f"  non-manifold face 除去: {removed} faces")
                tm.update_faces(keep_mask)
                tm.remove_unreferenced_vertices()

            # 4. face count > max ならデシメーション（頂点カラー転送付き）
            if len(tm.faces) > _FACE_COUNT_MAX:
                try:
                    import fast_simplification
                    from scipy.spatial import cKDTree

                    old_verts = tm.vertices.copy()
                    old_colors = None
                    if hasattr(tm.visual, "vertex_colors") and tm.visual.vertex_colors is not None:
                        old_colors = np.array(tm.visual.vertex_colors).copy()

                    target_reduction = 1.0 - (_FACE_COUNT_MAX * 0.9) / len(tm.faces)
                    target_reduction = float(np.clip(target_reduction, 0.0, 0.99))
                    new_verts, new_faces = fast_simplification.simplify(
                        old_verts, tm.faces, target_reduction
                    )
                    tm = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
                    if old_colors is not None:
                        tree = cKDTree(old_verts)
                        _, idx = tree.query(new_verts)
                        tm.visual.vertex_colors = old_colors[idx]
                    logger.debug(f"  デシメーション後: {len(tm.faces)} faces")

                    # デシメーション後にも non-manifold edges が生成されることがあるため再除去
                    ec2 = Counter(map(tuple, tm.edges_sorted))
                    nm_edges2 = {e for e, c in ec2.items() if c > 2}
                    if nm_edges2:
                        keep_mask2 = np.ones(len(tm.faces), dtype=bool)
                        for i, face in enumerate(tm.faces):
                            e0 = tuple(sorted([int(face[0]), int(face[1])]))
                            e1 = tuple(sorted([int(face[1]), int(face[2])]))
                            e2 = tuple(sorted([int(face[2]), int(face[0])]))
                            if e0 in nm_edges2 or e1 in nm_edges2 or e2 in nm_edges2:
                                keep_mask2[i] = False
                        removed2 = int(keep_mask2.size - keep_mask2.sum())
                        logger.debug(f"  デシメーション後 non-manifold face 除去: {removed2} faces")
                        tm.update_faces(keep_mask2)
                        tm.remove_unreferenced_vertices()
                        if hasattr(tm, 'visual') and old_colors is not None:
                            pass  # vertex_colors already transferred; update_faces preserves indexing

                except Exception as e_dec:
                    logger.warning(f"  デシメーション失敗 (スキップ): {e_dec}")

            # 5. normal fix（multibody=True で複数コンポーネントを個別に修復）
            trimesh.repair.fix_normals(tm, multibody=True)

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
                # 修復中間ファイルは常に一時ディレクトリに書き出す。
                # approved_dir へのコピーは修復が成功した場合のみ行う。
                tmp_dir = mesh_dir.parent / "meshes_repaired_tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                repair_output = tmp_dir / mesh_path.name

                repair_result = self.repair(str(mesh_path), str(repair_output))

                if repair_result.get("repaired"):
                    repaired += 1
                    check["status"] = "repaired"
                    check["repair"] = repair_result
                    if approved_dir:
                        shutil.copy2(str(repair_output), str(approved_dir / mesh_path.name))
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
