"""
T-4.2: SimExporter テスト
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import trimesh

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sim_exporter import SimExporter


# ============================================================
# ヘルパー
# ============================================================

def _make_asset_dir(tmpdir: str, asset_id: str = "test_asset",
                    n_collisions: int = 2) -> Path:
    """テスト用アセットディレクトリを作成する"""
    asset_dir = Path(tmpdir) / asset_id
    col_dir = asset_dir / "collisions"
    col_dir.mkdir(parents=True)

    box = trimesh.creation.box(extents=[0.06, 0.04, 0.08])

    # visual.glb
    box.export(str(asset_dir / "visual.glb"), file_type="glb")

    # collision_*.stl
    for i in range(n_collisions):
        box.export(str(col_dir / f"collision_{i:03d}.stl"), file_type="stl")

    # physics.json
    physics = {
        "asset_id": asset_id,
        "material": "polycarbonate",
        "density_kg_m3": 1200.0,
        "static_friction": 0.35,
        "dynamic_friction": 0.30,
        "restitution": 0.20,
        "volume_m3": 1.92e-4,
        "mass_kg": 0.2304,
        "luggage_type": "hard_suitcase",
        "collision_count": n_collisions,
        "scale_factor": 0.001,
    }
    with open(asset_dir / "physics.json", "w") as f:
        json.dump(physics, f)

    return asset_dir


# ============================================================
# export_usd_metadata テスト
# ============================================================

class TestExportUSDMetadata(unittest.TestCase):

    def setUp(self):
        self.exporter = SimExporter()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_json_file_created(self):
        """export_usd_metadata が JSON ファイルを作成すること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        json_path = self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        self.assertTrue(Path(json_path).exists())

    def test_json_has_required_keys(self):
        """USD メタデータ JSON が必須フィールドを持つこと"""
        asset_dir = _make_asset_dir(self.tmpdir)
        json_path = self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        with open(json_path) as f:
            data = json.load(f)
        for key in ("asset_id", "visual_mesh_path", "collision_mesh_paths",
                    "physics", "usd_output_path", "conversion_config"):
            self.assertIn(key, data)

    def test_json_physics_keys(self):
        """USD メタデータの physics フィールドが必須キーを持つこと"""
        asset_dir = _make_asset_dir(self.tmpdir)
        json_path = self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        with open(json_path) as f:
            data = json.load(f)
        for key in ("mass_kg", "static_friction", "dynamic_friction", "restitution"):
            self.assertIn(key, data["physics"])

    def test_json_collision_paths_count(self):
        """コリジョンパスの数が STL ファイル数と一致すること"""
        asset_dir = _make_asset_dir(self.tmpdir, n_collisions=3)
        json_path = self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        with open(json_path) as f:
            data = json.load(f)
        self.assertEqual(len(data["collision_mesh_paths"]), 3)

    def test_json_usd_output_path_ends_with_usda(self):
        """usd_output_path が .usda で終わること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        json_path = self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        with open(json_path) as f:
            data = json.load(f)
        self.assertTrue(data["usd_output_path"].endswith(".usda"))


# ============================================================
# USDA 生成テスト
# ============================================================

class TestUSDAGeneration(unittest.TestCase):

    def setUp(self):
        self.exporter = SimExporter()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_usda_file_created(self):
        """export_usd_metadata が .usda ファイルを生成すること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        json_path = self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        asset_id = asset_dir.name
        usda_path = Path(self.tmpdir) / f"{asset_id}.usda"
        self.assertTrue(usda_path.exists(), f".usda が生成されていない: {usda_path}")

    def test_usda_starts_with_header(self):
        """USDA ファイルの先頭行が '#usda 1.0' であること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        asset_id = asset_dir.name
        usda_path = Path(self.tmpdir) / f"{asset_id}.usda"
        content = usda_path.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("#usda 1.0"), "先頭行が '#usda 1.0' でない")

    def test_usda_contains_default_prim(self):
        """USDA に 'defaultPrim' が含まれること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        asset_id = asset_dir.name
        usda_path = Path(self.tmpdir) / f"{asset_id}.usda"
        content = usda_path.read_text(encoding="utf-8")
        self.assertIn("defaultPrim", content)

    def test_usda_contains_visual_payload(self):
        """USDA に visual.glb の payload 参照が含まれること（references は使わない）"""
        asset_dir = _make_asset_dir(self.tmpdir)
        self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        asset_id = asset_dir.name
        usda_path = Path(self.tmpdir) / f"{asset_id}.usda"
        content = usda_path.read_text(encoding="utf-8")
        self.assertIn("@./visual.glb@", content)
        self.assertIn("prepend payload = @./visual.glb@", content)
        self.assertNotIn("prepend references", content)

    def test_usda_collisions_prim_exists(self):
        """USDA に collisions Xform が存在すること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        asset_id = asset_dir.name
        usda_path = Path(self.tmpdir) / f"{asset_id}.usda"
        content = usda_path.read_text(encoding="utf-8")
        self.assertIn('def Xform "collisions"', content)

    def test_usda_contains_physics_mass(self):
        """USDA に physics.json の mass_kg が反映されること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        asset_id = asset_dir.name
        usda_path = Path(self.tmpdir) / f"{asset_id}.usda"
        content = usda_path.read_text(encoding="utf-8")
        # physics.json の mass_kg = 0.2304
        self.assertIn("0.2304", content)

    def test_usda_contains_collision_mesh(self):
        """USDA にコリジョン Mesh prim が含まれること"""
        asset_dir = _make_asset_dir(self.tmpdir, n_collisions=2)
        self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        asset_id = asset_dir.name
        usda_path = Path(self.tmpdir) / f"{asset_id}.usda"
        content = usda_path.read_text(encoding="utf-8")
        self.assertIn('def Mesh "collision_000"', content)
        self.assertIn('def Mesh "collision_001"', content)

    def test_usda_physics_apis(self):
        """USDA に PhysicsRigidBodyAPI / PhysicsMassAPI / PhysicsMaterialAPI が含まれること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        asset_id = asset_dir.name
        usda_path = Path(self.tmpdir) / f"{asset_id}.usda"
        content = usda_path.read_text(encoding="utf-8")
        self.assertIn("PhysicsRigidBodyAPI", content)
        self.assertIn("PhysicsMassAPI", content)
        self.assertIn("PhysicsMaterialAPI", content)


# ============================================================
# export_batch テスト
# ============================================================

class TestExportBatch(unittest.TestCase):

    def setUp(self):
        self.exporter = SimExporter()
        self.tmpdir = tempfile.mkdtemp()
        self.assets_dir = Path(self.tmpdir) / "assets"
        self.assets_dir.mkdir()
        self.output_dir = Path(self.tmpdir) / "output"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_assets(self, n: int) -> None:
        for i in range(n):
            _make_asset_dir(str(self.assets_dir), asset_id=f"asset_{i:03d}")

    def test_batch_returns_summary_keys(self):
        """export_batch が total/success/failed/results を返すこと"""
        self._create_assets(2)
        result = self.exporter.export_batch(str(self.assets_dir), str(self.output_dir))
        for key in ("total", "success", "failed", "results"):
            self.assertIn(key, result)

    def test_batch_total_count(self):
        """total がアセット数と一致すること"""
        self._create_assets(3)
        result = self.exporter.export_batch(str(self.assets_dir), str(self.output_dir))
        self.assertEqual(result["total"], 3)

    def test_batch_empty_directory(self):
        """visual.glb が存在しないディレクトリは対象外（total=0）"""
        result = self.exporter.export_batch(str(self.assets_dir), str(self.output_dir))
        self.assertEqual(result["total"], 0)

    def test_batch_usd_only(self):
        """USD メタデータが出力されること"""
        self._create_assets(1)
        result = self.exporter.export_batch(
            str(self.assets_dir), str(self.output_dir)
        )
        for entry in result["results"]:
            if entry["status"] == "success":
                self.assertIsNotNone(entry["usd_meta_path"])

    def test_batch_resume_skips(self):
        """resume=True のとき既存アセットをスキップすること"""
        self._create_assets(2)
        self.exporter.export_batch(str(self.assets_dir), str(self.output_dir))
        # 2 回目
        result = self.exporter.export_batch(
            str(self.assets_dir), str(self.output_dir), resume=True
        )
        skipped = [r for r in result["results"] if r["status"] == "skipped"]
        self.assertEqual(len(skipped), 2)


# ============================================================
# SimExporter ヘルパーメソッドテスト
# ============================================================

class TestHelpers(unittest.TestCase):

    def setUp(self):
        self.exporter = SimExporter()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_physics_defaults(self):
        """physics.json が存在しないとき デフォルト値が返ること"""
        asset_dir = Path(self.tmpdir) / "no_physics"
        asset_dir.mkdir()
        result = self.exporter._load_physics(asset_dir)
        self.assertIn("mass_kg", result)
        self.assertGreater(result["mass_kg"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
