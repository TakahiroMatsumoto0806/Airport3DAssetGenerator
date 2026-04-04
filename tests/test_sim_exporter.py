"""
T-4.2: SimExporter テスト
"""

import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
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
# export_mjcf テスト
# ============================================================

class TestExportMJCF(unittest.TestCase):

    def setUp(self):
        self.exporter = SimExporter()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mjcf_file_created(self):
        """export_mjcf が XML ファイルを作成すること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        output_dir = Path(self.tmpdir) / "mjcf"
        mjcf_path = self.exporter.export_mjcf(str(asset_dir), str(output_dir))
        self.assertTrue(Path(mjcf_path).exists())

    def test_mjcf_returns_path_string(self):
        """export_mjcf が文字列パスを返すこと"""
        asset_dir = _make_asset_dir(self.tmpdir)
        result = self.exporter.export_mjcf(str(asset_dir), str(Path(self.tmpdir) / "mjcf"))
        self.assertIsInstance(result, str)

    def test_mjcf_valid_xml(self):
        """出力された MJCF が有効な XML であること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        mjcf_path = self.exporter.export_mjcf(str(asset_dir), str(Path(self.tmpdir) / "mjcf"))
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        self.assertEqual(root.tag, "mujoco")

    def test_mjcf_contains_worldbody(self):
        """MJCF に worldbody 要素が含まれること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        mjcf_path = self.exporter.export_mjcf(str(asset_dir), str(Path(self.tmpdir) / "mjcf"))
        tree = ET.parse(mjcf_path)
        self.assertIsNotNone(tree.getroot().find("worldbody"))

    def test_mjcf_contains_asset(self):
        """MJCF に asset 要素が含まれること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        mjcf_path = self.exporter.export_mjcf(str(asset_dir), str(Path(self.tmpdir) / "mjcf"))
        tree = ET.parse(mjcf_path)
        self.assertIsNotNone(tree.getroot().find("asset"))

    def test_mjcf_contains_inertial(self):
        """MJCF の body に inertial 要素が含まれること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        mjcf_path = self.exporter.export_mjcf(str(asset_dir), str(Path(self.tmpdir) / "mjcf"))
        tree = ET.parse(mjcf_path)
        body = tree.getroot().find(".//body")
        self.assertIsNotNone(body)
        self.assertIsNotNone(body.find("inertial"))

    def test_mjcf_mass_from_physics_json(self):
        """MJCF の inertial mass が physics.json の値と一致すること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        mjcf_path = self.exporter.export_mjcf(str(asset_dir), str(Path(self.tmpdir) / "mjcf"))
        tree = ET.parse(mjcf_path)
        inertial = tree.getroot().find(".//inertial")
        mass = float(inertial.get("mass"))
        self.assertAlmostEqual(mass, 0.2304, places=4)

    def test_mjcf_collision_geoms_present(self):
        """コリジョン geom が STL ファイル数と一致すること"""
        asset_dir = _make_asset_dir(self.tmpdir, n_collisions=3)
        mjcf_path = self.exporter.export_mjcf(str(asset_dir), str(Path(self.tmpdir) / "mjcf"))
        tree = ET.parse(mjcf_path)
        col_geoms = [
            g for g in tree.getroot().findall(".//geom")
            if g.get("class") == "collision"
        ]
        self.assertEqual(len(col_geoms), 3)

    def test_mjcf_visual_geom_present(self):
        """visual geom が 1 つ存在すること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        mjcf_path = self.exporter.export_mjcf(str(asset_dir), str(Path(self.tmpdir) / "mjcf"))
        tree = ET.parse(mjcf_path)
        vis_geoms = [
            g for g in tree.getroot().findall(".//geom")
            if g.get("class") == "visual"
        ]
        self.assertEqual(len(vis_geoms), 1)

    def test_mjcf_no_visual_glb_raises(self):
        """visual.glb が存在しない場合 FileNotFoundError が送出されること"""
        asset_dir = Path(self.tmpdir) / "empty_asset"
        asset_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            self.exporter.export_mjcf(str(asset_dir), str(Path(self.tmpdir) / "mjcf"))

    def test_mjcf_freejoint_present(self):
        """body に freejoint 要素が含まれること（自由落下可能）"""
        asset_dir = _make_asset_dir(self.tmpdir)
        mjcf_path = self.exporter.export_mjcf(str(asset_dir), str(Path(self.tmpdir) / "mjcf"))
        tree = ET.parse(mjcf_path)
        body = tree.getroot().find(".//body")
        self.assertIsNotNone(body.find("freejoint"))


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

    def test_json_usd_output_path_ends_with_usd(self):
        """usd_output_path が .usd で終わること"""
        asset_dir = _make_asset_dir(self.tmpdir)
        json_path = self.exporter.export_usd_metadata(str(asset_dir), self.tmpdir)
        with open(json_path) as f:
            data = json.load(f)
        self.assertTrue(data["usd_output_path"].endswith(".usd"))


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

    def test_batch_mjcf_only(self):
        """format='mjcf' のとき MJCF だけ出力されること"""
        self._create_assets(1)
        result = self.exporter.export_batch(
            str(self.assets_dir), str(self.output_dir), format="mjcf"
        )
        for entry in result["results"]:
            if entry["status"] == "success":
                self.assertIsNotNone(entry["mjcf_path"])

    def test_batch_usd_only(self):
        """format='usd' のとき USD メタデータだけ出力されること"""
        self._create_assets(1)
        result = self.exporter.export_batch(
            str(self.assets_dir), str(self.output_dir), format="usd"
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

    def test_compute_inertia_returns_dict(self):
        """_compute_inertia が ixx/iyy/izz を含む dict を返すこと"""
        result = self.exporter._compute_inertia(0.5, [0.06, 0.04, 0.08])
        for key in ("ixx", "iyy", "izz"):
            self.assertIn(key, result)
            self.assertGreater(result[key], 0.0)

    def test_compute_inertia_symmetry(self):
        """立方体の場合 ixx=iyy=izz であること"""
        result = self.exporter._compute_inertia(1.0, [1.0, 1.0, 1.0])
        self.assertAlmostEqual(result["ixx"], result["iyy"], places=10)
        self.assertAlmostEqual(result["iyy"], result["izz"], places=10)

    def test_load_physics_defaults(self):
        """physics.json が存在しないとき デフォルト値が返ること"""
        asset_dir = Path(self.tmpdir) / "no_physics"
        asset_dir.mkdir()
        result = self.exporter._load_physics(asset_dir)
        self.assertIn("mass_kg", result)
        self.assertGreater(result["mass_kg"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
