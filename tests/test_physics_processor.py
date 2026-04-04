"""
T-4.1: PhysicsProcessor / ScaleNormalizer テスト

CoACD は DGX Spark にしか入らないためモック化。
scale_normalizer と assign_properties / process_single のロジックをユニットテスト。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scale_normalizer import ScaleNormalizer
from src.physics_processor import PhysicsProcessor

# テストで使う material_properties.yaml のパス
_MATERIAL_CFG = str(Path(__file__).parent.parent / "configs" / "material_properties.yaml")


# ============================================================
# テスト用ヘルパー
# ============================================================

def _make_glb(tmpdir: str, name: str = "mesh.glb", size_m: float = 1.0) -> str:
    """unit cube GLB を作成して返す（TRELLIS.2 の標準出力サイズ 1m³ を模倣）"""
    # box の extents が size_m になるよう作成
    box = trimesh.creation.box(extents=[size_m, size_m * 0.6, size_m * 1.4])
    path = str(Path(tmpdir) / name)
    box.export(path, file_type="glb")
    return path


# ============================================================
# ScaleNormalizer テスト
# ============================================================

class TestScaleNormalizer(unittest.TestCase):

    def setUp(self):
        self.normalizer = ScaleNormalizer(_MATERIAL_CFG)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_normalize_returns_required_keys(self):
        """normalize() が必須フィールドをすべて返すこと"""
        src = _make_glb(self.tmpdir, size_m=1.0)
        result = self.normalizer.normalize(src)
        for key in ("input_path", "output_path", "scale_factor",
                    "original_extents_m", "scaled_extents_mm",
                    "short_side_mm", "long_side_mm"):
            self.assertIn(key, result, f"必須キーが欠落: {key}")

    def test_output_file_created(self):
        """normalize() 後に出力ファイルが存在すること"""
        src = _make_glb(self.tmpdir, size_m=1.0)
        result = self.normalizer.normalize(src)
        self.assertTrue(Path(result["output_path"]).exists())

    def test_short_side_within_target(self):
        """ミニチュアモードで短辺が target_short_side_mm 以下になること"""
        src = _make_glb(self.tmpdir, size_m=1.0)
        result = self.normalizer.normalize(src, miniature=True)
        self.assertLessEqual(
            result["short_side_mm"],
            self.normalizer.target_short_side_mm + 0.5,  # 浮動小数点誤差を考慮
        )

    def test_long_side_within_max(self):
        """ミニチュアモードで長辺が max_long_side_mm 以下になること"""
        src = _make_glb(self.tmpdir, size_m=1.0)
        result = self.normalizer.normalize(src, miniature=True)
        self.assertLessEqual(
            result["long_side_mm"],
            self.normalizer.max_long_side_mm + 0.5,
        )

    def test_scale_factor_positive(self):
        """scale_factor が正の値であること"""
        src = _make_glb(self.tmpdir, size_m=1.0)
        result = self.normalizer.normalize(src, miniature=True)
        self.assertGreater(result["scale_factor"], 0.0)

    def test_no_miniature_returns_scale_1(self):
        """miniature=False のとき scale_factor=1.0 であること"""
        src = _make_glb(self.tmpdir, size_m=1.0)
        result = self.normalizer.normalize(src, miniature=False)
        self.assertAlmostEqual(result["scale_factor"], 1.0, places=6)

    def test_custom_output_path(self):
        """output_path を指定した場合そのパスに保存されること"""
        src = _make_glb(self.tmpdir, size_m=1.0)
        dst = str(Path(self.tmpdir) / "custom_output.glb")
        result = self.normalizer.normalize(src, output_path=dst)
        self.assertEqual(result["output_path"], dst)
        self.assertTrue(Path(dst).exists())

    def test_aspect_ratio_preserved(self):
        """スケール後もアスペクト比が元と同じであること（±1%）"""
        src = _make_glb(self.tmpdir, size_m=1.0)
        result = self.normalizer.normalize(src, miniature=True)
        orig = result["original_extents_m"]
        scaled = result["scaled_extents_mm"]
        # 全次元を同じ倍率でスケールしているはずなので比率が保持される
        sf = result["scale_factor"]
        for orig_dim, scaled_dim in zip(orig, scaled):
            expected = orig_dim * sf * 1000.0
            self.assertAlmostEqual(scaled_dim, expected, delta=0.1)


# ============================================================
# PhysicsProcessor.assign_properties テスト
# ============================================================

class TestAssignProperties(unittest.TestCase):

    def setUp(self):
        self.processor = PhysicsProcessor(_MATERIAL_CFG, seed=42)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mesh_path(self, size_m: float = 0.06) -> str:
        return _make_glb(self.tmpdir, size_m=size_m)

    def test_returns_required_keys(self):
        """assign_properties が必須フィールドをすべて返すこと"""
        path = self._mesh_path()
        result = self.processor.assign_properties(path, "polycarbonate")
        for key in ("material", "density_kg_m3", "static_friction",
                    "dynamic_friction", "restitution", "volume_m3", "mass_kg"):
            self.assertIn(key, result, f"必須キーが欠落: {key}")

    def test_known_material_used(self):
        """指定した材質キーが結果に反映されること"""
        path = self._mesh_path()
        result = self.processor.assign_properties(path, "polycarbonate", randomize=False)
        self.assertEqual(result["material"], "polycarbonate")

    def test_density_matches_config(self):
        """randomize=False のとき密度が YAML 定義値と一致すること"""
        path = self._mesh_path()
        result = self.processor.assign_properties(path, "polycarbonate", randomize=False)
        self.assertAlmostEqual(result["density_kg_m3"], 1200.0, delta=0.01)

    def test_randomize_changes_values(self):
        """randomize=True のとき値が変化すること"""
        path = self._mesh_path()
        base = self.processor.assign_properties(path, "polycarbonate", randomize=False)
        varied = self.processor.assign_properties(path, "polycarbonate", randomize=True)
        # 少なくともいずれかの値が変化しているはず
        changed = any(
            base[k] != varied[k]
            for k in ("density_kg_m3", "static_friction", "dynamic_friction", "restitution")
        )
        self.assertTrue(changed)

    def test_randomize_within_range(self):
        """randomize=True のとき値が ±15% 範囲に収まること"""
        path = self._mesh_path()
        base = self.processor.assign_properties(path, "polycarbonate", randomize=False)
        for _ in range(10):
            varied = self.processor.assign_properties(path, "polycarbonate", randomize=True)
            self.assertGreaterEqual(varied["density_kg_m3"], base["density_kg_m3"] * 0.85 - 0.01)
            self.assertLessEqual(varied["density_kg_m3"], base["density_kg_m3"] * 1.15 + 0.01)

    def test_mass_positive(self):
        """mass_kg が正の値であること"""
        path = self._mesh_path()
        result = self.processor.assign_properties(path, "nylon")
        self.assertGreater(result["mass_kg"], 0.0)

    def test_unknown_material_fallback(self):
        """未定義材質は composite_hard_suitcase にフォールバックすること"""
        path = self._mesh_path()
        result = self.processor.assign_properties(path, "nonexistent_material", randomize=False)
        self.assertEqual(result["material"], "composite_hard_suitcase")

    def test_all_materials_valid(self):
        """material_properties.yaml の全材質で assign_properties が成功すること"""
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(_MATERIAL_CFG)
        path = self._mesh_path()
        for mat_key in cfg.get("materials", {}).keys():
            result = self.processor.assign_properties(path, mat_key, randomize=False)
            self.assertEqual(result["material"], mat_key, f"材質 {mat_key} で失敗")


# ============================================================
# PhysicsProcessor._resolve_material テスト
# ============================================================

class TestResolveMaterial(unittest.TestCase):

    def setUp(self):
        self.processor = PhysicsProcessor(_MATERIAL_CFG, seed=42)

    def test_explicit_material_used(self):
        """material が直接指定された場合それが使われること"""
        result = self.processor._resolve_material("nylon", "hard_suitcase")
        self.assertEqual(result, "nylon")

    def test_luggage_type_fallback(self):
        """material=None の場合 luggage_type から解決されること"""
        result = self.processor._resolve_material(None, "hard_suitcase")
        self.assertEqual(result, "composite_hard_suitcase")

    def test_unknown_both_falls_back(self):
        """material も luggage_type も不明なら composite_hard_suitcase"""
        result = self.processor._resolve_material(None, None)
        self.assertEqual(result, "composite_hard_suitcase")


# ============================================================
# PhysicsProcessor.generate_collision テスト（CoACD モック）
# ============================================================

class TestGenerateCollision(unittest.TestCase):

    def setUp(self):
        self.processor = PhysicsProcessor(_MATERIAL_CFG)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("src.physics_processor.PhysicsProcessor.generate_collision")
    def test_generate_collision_called(self, mock_gen):
        """generate_collision が適切な引数で呼ばれること"""
        mock_gen.return_value = ["col_0.stl", "col_1.stl"]
        paths = self.processor.generate_collision("mesh.glb", self.tmpdir, threshold=0.08)
        mock_gen.assert_called_once_with("mesh.glb", self.tmpdir, threshold=0.08)
        self.assertEqual(len(paths), 2)

    def test_generate_collision_import_error(self):
        """coacd が未インストールの場合 ImportError が送出されること"""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "coacd":
                raise ImportError("No module named 'coacd'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with self.assertRaises(ImportError):
                self.processor.generate_collision("mesh.glb", self.tmpdir)


# ============================================================
# PhysicsProcessor.process_single テスト（CoACD モック）
# ============================================================

class TestProcessSingle(unittest.TestCase):

    def setUp(self):
        self.processor = PhysicsProcessor(_MATERIAL_CFG, seed=42)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch.object(PhysicsProcessor, "generate_collision")
    def test_process_single_success(self, mock_collision):
        """process_single が status=success を返すこと"""
        mock_collision.return_value = ["col_0.stl"]
        src = _make_glb(self.tmpdir, name="asset.glb", size_m=1.0)
        result = self.processor.process_single(
            src, self.tmpdir, material="polycarbonate", luggage_type="hard_suitcase"
        )
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["visual_path"])
        self.assertIn("mass_kg", result["physics"])

    @patch.object(PhysicsProcessor, "generate_collision")
    def test_process_single_creates_physics_json(self, mock_collision):
        """process_single が physics.json を作成すること"""
        mock_collision.return_value = ["col_0.stl"]
        src = _make_glb(self.tmpdir, name="asset.glb", size_m=1.0)
        self.processor.process_single(src, self.tmpdir, material="nylon")
        asset_id = Path(src).stem
        physics_json = Path(self.tmpdir) / asset_id / "physics.json"
        self.assertTrue(physics_json.exists())
        with open(physics_json) as f:
            data = json.load(f)
        self.assertIn("mass_kg", data)

    @patch.object(PhysicsProcessor, "generate_collision")
    def test_process_single_creates_visual_glb(self, mock_collision):
        """process_single が visual.glb を作成すること"""
        mock_collision.return_value = []
        src = _make_glb(self.tmpdir, name="asset.glb", size_m=1.0)
        result = self.processor.process_single(src, self.tmpdir)
        self.assertTrue(Path(result["visual_path"]).exists())

    @patch.object(PhysicsProcessor, "generate_collision")
    def test_process_single_error_continues(self, mock_collision):
        """generate_collision が失敗しても status=failed で返ること（例外を伝播しない）"""
        mock_collision.side_effect = RuntimeError("CoACD crash")
        src = _make_glb(self.tmpdir, name="asset.glb", size_m=1.0)
        result = self.processor.process_single(src, self.tmpdir)
        self.assertEqual(result["status"], "failed")
        self.assertIsNotNone(result["error"])


# ============================================================
# PhysicsProcessor.process_batch テスト
# ============================================================

class TestProcessBatch(unittest.TestCase):

    def setUp(self):
        self.processor = PhysicsProcessor(_MATERIAL_CFG, seed=42)
        self.tmpdir = tempfile.mkdtemp()
        self.mesh_dir = Path(self.tmpdir) / "meshes"
        self.mesh_dir.mkdir()
        self.output_dir = Path(self.tmpdir) / "assets"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_glbs(self, n: int) -> None:
        for i in range(n):
            box = trimesh.creation.box(extents=[1.0, 0.6, 1.4])
            box.export(str(self.mesh_dir / f"mesh_{i:03d}.glb"), file_type="glb")

    @patch.object(PhysicsProcessor, "generate_collision")
    def test_batch_returns_summary_keys(self, mock_collision):
        """process_batch が total/success/failed/results を返すこと"""
        mock_collision.return_value = ["col_0.stl"]
        self._create_glbs(2)
        result = self.processor.process_batch(str(self.mesh_dir), str(self.output_dir))
        for key in ("total", "success", "failed", "results"):
            self.assertIn(key, result)

    @patch.object(PhysicsProcessor, "generate_collision")
    def test_batch_empty_directory(self, mock_collision):
        """空ディレクトリでも total=0 で正常終了すること"""
        result = self.processor.process_batch(str(self.mesh_dir), str(self.output_dir))
        self.assertEqual(result["total"], 0)

    @patch.object(PhysicsProcessor, "generate_collision")
    def test_batch_resume_skips_existing(self, mock_collision):
        """resume=True のとき physics.json 存在アセットをスキップすること"""
        mock_collision.return_value = ["col_0.stl"]
        self._create_glbs(2)
        # 1 回目
        self.processor.process_batch(str(self.mesh_dir), str(self.output_dir))
        first_count = mock_collision.call_count
        # 2 回目: スキップされるはず
        mock_collision.reset_mock()
        self.processor.process_batch(str(self.mesh_dir), str(self.output_dir), resume=True)
        self.assertEqual(mock_collision.call_count, 0)

    @patch.object(PhysicsProcessor, "generate_collision")
    def test_batch_uses_vlm_metadata(self, mock_collision):
        """metadata_json があれば detected_type を material 解決に使用すること"""
        mock_collision.return_value = []
        self._create_glbs(1)
        # VLM QA 結果 JSON を作成
        meta_path = Path(self.tmpdir) / "vlm.json"
        mesh_key = "mesh_000"
        with open(meta_path, "w") as f:
            json.dump({"results": [
                {"mesh_path": str(self.mesh_dir / "mesh_000.glb"),
                 "detected_type": "backpack",
                 "detected_material": "nylon",
                 "pass": True}
            ]}, f)

        result = self.processor.process_batch(
            str(self.mesh_dir), str(self.output_dir),
            metadata_json=str(meta_path),
        )
        # physics.json の中身を確認
        physics_path = self.output_dir / "mesh_000" / "physics.json"
        if physics_path.exists():
            with open(physics_path) as f:
                data = json.load(f)
            # nylon か composite_backpack にマッピングされること
            self.assertIn(data.get("material"), ("nylon", "composite_backpack"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
