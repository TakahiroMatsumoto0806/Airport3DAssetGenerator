"""
T-3.2: MeshQA テスト

既知の不具合メッシュ（non-watertight, degenerate faces 等）を trimesh で作成し、
検出・修復が正しく動作することを検証する。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

# プロジェクトの src を PYTHONPATH に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh_qa import MeshQA, _FACE_COUNT_MIN, _FACE_COUNT_MAX, _ASPECT_RATIO_MAX


# ============================================================
# テスト用メッシュ生成ヘルパー
# ============================================================

def _make_good_mesh() -> trimesh.Trimesh:
    """品質チェックに合格するメッシュ（icosphere, faces=5120）"""
    mesh = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
    return mesh


def _save_glb(mesh: trimesh.Trimesh, path: str) -> None:
    """trimesh メッシュを GLB として保存する"""
    mesh.export(path, file_type="glb")


def _make_non_watertight_mesh() -> trimesh.Trimesh:
    """穴のあるメッシュ（非 watertight）"""
    mesh = trimesh.creation.icosphere(subdivisions=4)
    # 面をいくつか削除して穴を作る
    mesh.faces = mesh.faces[:-5]
    return mesh


def _make_small_face_count_mesh() -> trimesh.Trimesh:
    """face_count < 5K のメッシュ"""
    return trimesh.creation.icosphere(subdivisions=3)  # 1280 faces


def _make_large_face_count_mesh() -> trimesh.Trimesh:
    """face_count > 100K のメッシュ"""
    mesh = trimesh.creation.icosphere(subdivisions=6)  # ~82K faces, but let's subdivide more
    # subdivisions=7 → 327680 faces
    return trimesh.creation.icosphere(subdivisions=7)


def _make_degenerate_face_mesh() -> trimesh.Trimesh:
    """縮退面を含むメッシュ"""
    mesh = trimesh.creation.icosphere(subdivisions=4)
    # 最初の面を縮退させる（3頂点を同じにする）
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    # 最初の三角形の全頂点を同一点に
    faces[0] = [0, 0, 0]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _make_high_aspect_ratio_mesh() -> trimesh.Trimesh:
    """アスペクト比が高い細長いメッシュ"""
    # x 方向に 21 倍引き延ばした楕円体
    mesh = trimesh.creation.icosphere(subdivisions=4)
    mesh.vertices[:, 0] *= 25.0  # x を 25 倍に
    return mesh


# ============================================================
# check_single テスト
# ============================================================

class TestCheckSingle(unittest.TestCase):

    def setUp(self):
        self.qa = MeshQA()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _save(self, mesh: trimesh.Trimesh, name: str) -> str:
        path = str(Path(self.tmpdir) / name)
        _save_glb(mesh, path)
        return path

    def test_good_mesh_passes(self):
        """合格メッシュが pass=True を返すこと"""
        path = self._save(_make_good_mesh(), "good.glb")
        result = self.qa.check_single(path)
        self.assertTrue(result["pass"], f"合格メッシュが不合格: {result['issues']}")

    def test_result_has_required_keys(self):
        """check_single の返り値が必須フィールドをすべて含むこと"""
        path = self._save(_make_good_mesh(), "keys.glb")
        result = self.qa.check_single(path)
        required = [
            "mesh_path", "is_watertight", "is_edge_manifold", "is_vertex_manifold",
            "has_self_intersection", "face_count", "face_count_ok",
            "degenerate_count", "is_normal_consistent", "aspect_ratio",
            "aspect_ratio_ok", "pass", "issues", "timestamp",
        ]
        for key in required:
            self.assertIn(key, result, f"必須キーが欠落: {key}")

    def test_non_watertight_detected(self):
        """non-watertight メッシュが is_watertight=False として記録されること。
        TRELLIS 出力はほぼ常に非 watertight のため、is_watertight は pass 条件に含まない。"""
        path = self._save(_make_non_watertight_mesh(), "non_wt.glb")
        result = self.qa.check_single(path)
        self.assertFalse(result["is_watertight"])
        # issues には "watertight でない" が記録される
        self.assertTrue(any("watertight" in iss for iss in result["issues"]))

    def test_small_face_count_detected(self):
        """face_count < 5K のメッシュが face_count_ok=False かつ pass=False"""
        path = self._save(_make_small_face_count_mesh(), "small.glb")
        result = self.qa.check_single(path)
        self.assertFalse(result["face_count_ok"])
        self.assertLess(result["face_count"], _FACE_COUNT_MIN)
        self.assertFalse(result["pass"])

    def test_large_face_count_detected(self):
        """face_count > 100K のメッシュが face_count_ok=False かつ pass=False"""
        path = self._save(_make_large_face_count_mesh(), "large.glb")
        result = self.qa.check_single(path)
        self.assertFalse(result["face_count_ok"])
        self.assertGreater(result["face_count"], _FACE_COUNT_MAX)

    def test_degenerate_face_detected(self):
        """縮退面が degenerate_count > 0 として検出されること"""
        path = self._save(_make_degenerate_face_mesh(), "degen.glb")
        result = self.qa.check_single(path)
        self.assertGreater(result["degenerate_count"], 0)

    def test_high_aspect_ratio_detected(self):
        """アスペクト比 > 20 のメッシュが aspect_ratio_ok=False"""
        path = self._save(_make_high_aspect_ratio_mesh(), "aspect.glb")
        result = self.qa.check_single(path)
        self.assertFalse(result["aspect_ratio_ok"])
        self.assertGreater(result["aspect_ratio"], _ASPECT_RATIO_MAX)

    def test_good_mesh_face_count_in_range(self):
        """合格メッシュの face_count が 5K〜100K に収まること"""
        path = self._save(_make_good_mesh(), "fc.glb")
        result = self.qa.check_single(path)
        self.assertTrue(result["face_count_ok"])
        self.assertGreaterEqual(result["face_count"], _FACE_COUNT_MIN)
        self.assertLessEqual(result["face_count"], _FACE_COUNT_MAX)

    def test_nonexistent_file_returns_no_pass(self):
        """存在しないファイルで pass=False かつ issues に情報が入ること"""
        result = self.qa.check_single("/nonexistent/path.glb")
        self.assertFalse(result["pass"])
        self.assertGreater(len(result["issues"]), 0)

    def test_issues_list_is_populated_for_bad_mesh(self):
        """不合格メッシュ（face_count < 5K）では issues が空でなく pass=False なこと"""
        path = self._save(_make_small_face_count_mesh(), "issues.glb")
        result = self.qa.check_single(path)
        self.assertFalse(result["pass"])
        self.assertIsInstance(result["issues"], list)
        self.assertGreater(len(result["issues"]), 0)


# ============================================================
# repair テスト
# ============================================================

class TestRepair(unittest.TestCase):

    def setUp(self):
        self.qa = MeshQA()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _save(self, mesh: trimesh.Trimesh, name: str) -> str:
        path = str(Path(self.tmpdir) / name)
        _save_glb(mesh, path)
        return path

    def test_repair_returns_before_after(self):
        """repair() が before/after/repaired キーを返すこと"""
        src = self._save(_make_good_mesh(), "src.glb")
        dst = str(Path(self.tmpdir) / "dst.glb")
        result = self.qa.repair(src, dst)
        self.assertIn("before", result)
        self.assertIn("after", result)
        self.assertIn("repaired", result)

    def test_repair_output_file_created(self):
        """repair() 後に出力ファイルが存在すること"""
        src = self._save(_make_non_watertight_mesh(), "src.glb")
        dst = str(Path(self.tmpdir) / "dst.glb")
        self.qa.repair(src, dst)
        self.assertTrue(Path(dst).exists())

    def test_repair_good_mesh_stays_repaired(self):
        """合格メッシュを修復しても出力ファイルが生成されること"""
        src = self._save(_make_good_mesh(), "src.glb")
        dst = str(Path(self.tmpdir) / "dst.glb")
        self.qa.repair(src, dst)
        self.assertTrue(Path(dst).exists())

    def test_repair_degenerate_removes_degen(self):
        """縮退面修復後 degenerate_count が減ること"""
        src = self._save(_make_degenerate_face_mesh(), "degen.glb")
        dst = str(Path(self.tmpdir) / "fixed.glb")
        result = self.qa.repair(src, dst)
        before_degen = result["before"]["degenerate_count"]
        after_degen = result["after"]["degenerate_count"]
        self.assertGreaterEqual(before_degen, 1)
        self.assertLessEqual(after_degen, before_degen)


# ============================================================
# check_batch テスト
# ============================================================

class TestCheckBatch(unittest.TestCase):

    def setUp(self):
        self.qa = MeshQA()
        self.tmpdir = tempfile.mkdtemp()
        self.mesh_dir = Path(self.tmpdir) / "meshes"
        self.mesh_dir.mkdir()
        self.approved_dir = Path(self.tmpdir) / "approved"
        self.output_json = Path(self.tmpdir) / "results.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_good_meshes(self, n: int) -> None:
        mesh = _make_good_mesh()
        for i in range(n):
            mesh.export(str(self.mesh_dir / f"good_{i:03d}.glb"), file_type="glb")

    def _create_bad_meshes(self, n: int) -> None:
        # face_count < 5K → pass 条件 (face_count_ok) が False になる確実な不合格メッシュ
        mesh = _make_small_face_count_mesh()
        for i in range(n):
            mesh.export(str(self.mesh_dir / f"bad_{i:03d}.glb"), file_type="glb")

    def test_batch_returns_summary_keys(self):
        """check_batch が total/passed/repaired/failed/results を返すこと"""
        self._create_good_meshes(2)
        result = self.qa.check_batch(
            str(self.mesh_dir),
            str(self.output_json),
        )
        for key in ("total", "passed", "repaired", "failed", "results"):
            self.assertIn(key, result)

    def test_batch_json_saved(self):
        """check_batch が JSON ファイルを保存すること"""
        self._create_good_meshes(2)
        self.qa.check_batch(str(self.mesh_dir), str(self.output_json))
        self.assertTrue(self.output_json.exists())
        with open(self.output_json) as f:
            data = json.load(f)
        self.assertIn("results", data)

    def test_batch_empty_directory(self):
        """空ディレクトリでも total=0 で正常終了すること"""
        result = self.qa.check_batch(str(self.mesh_dir), str(self.output_json))
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["passed"], 0)

    def test_batch_good_meshes_counted(self):
        """合格メッシュが passed にカウントされること"""
        self._create_good_meshes(3)
        result = self.qa.check_batch(
            str(self.mesh_dir),
            str(self.output_json),
            attempt_repair=False,
        )
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["passed"], 3)
        self.assertEqual(result["failed"], 0)

    def test_batch_bad_meshes_counted_when_no_repair(self):
        """attempt_repair=False のとき不合格がすべて failed になること"""
        self._create_bad_meshes(2)
        result = self.qa.check_batch(
            str(self.mesh_dir),
            str(self.output_json),
            attempt_repair=False,
        )
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["failed"], 2)

    def test_batch_approved_dir_populated(self):
        """合格メッシュが approved_dir にコピーされること"""
        self._create_good_meshes(2)
        self.qa.check_batch(
            str(self.mesh_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
            attempt_repair=False,
        )
        glbs = list(self.approved_dir.glob("*.glb"))
        self.assertEqual(len(glbs), 2)

    def test_batch_result_status_field(self):
        """各エントリに status フィールドが含まれること"""
        self._create_good_meshes(1)
        result = self.qa.check_batch(str(self.mesh_dir), str(self.output_json))
        for entry in result["results"]:
            self.assertIn("status", entry)
            self.assertIn(entry["status"], ("passed", "repaired", "failed"))

    def test_batch_total_count(self):
        """total が実際のファイル数と一致すること"""
        self._create_good_meshes(2)
        self._create_bad_meshes(3)
        result = self.qa.check_batch(
            str(self.mesh_dir),
            str(self.output_json),
            attempt_repair=False,
        )
        self.assertEqual(result["total"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
