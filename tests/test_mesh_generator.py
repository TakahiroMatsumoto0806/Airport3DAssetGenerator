"""
T-3.1: 3D モデル生成エンジン テスト

検証項目（仕様書記載）:
  - 1 枚のテスト画像で 3D 生成し、GLB ファイルが存在する
  - trimesh.load() で GLB を読み込めること（有効なメッシュ）
  - ファイルサイズ > 0

追加検証:
  - generate_single() の基本動作（モック）
  - generate_batch() のスキーマ・中断再開・失敗継続
  - simplify・texture_size パラメータが正しく渡されること
  - seed による _set_seed 呼び出し確認
  - unload() で GPU メモリ解放

trellis2 / o_voxel / torch は DGX Spark 専用環境なのでモック化して実行する。
実動作確認は tests/test_gpu_models.py (T-0.3) を参照。

実行方法:
    pytest tests/test_mesh_generator.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# テスト用ユーティリティ
# ============================================================

def _make_test_image(path: Path) -> Path:
    from PIL import Image
    img = Image.new("RGB", (512, 512), color=(180, 180, 180))
    img.save(path)
    return path


def _make_real_glb(path: Path) -> Path:
    """trimesh で実際の GLB ファイルを生成する"""
    import trimesh
    mesh = trimesh.creation.box()
    mesh.export(str(path))
    return path


def _build_sys_modules_patch(n_vertices: int = 10000) -> tuple:
    """
    sys.modules に注入するモジュールモックと関連オブジェクトを返す。

    Returns:
        (modules_dict, mock_pipeline, mesh_mock, mock_o_voxel, mock_torch)
    """
    import numpy as np

    # --- mesh モック ---
    mesh_mock = MagicMock()
    mesh_mock.vertices = np.zeros((n_vertices, 3), dtype=np.float32)
    mesh_mock.faces = MagicMock()
    mesh_mock.attrs = MagicMock()
    mesh_mock.coords = MagicMock()
    mesh_mock.layout = MagicMock()
    mesh_mock.voxel_size = 0.01

    # --- o_voxel.postprocess.to_glb ---
    glb_mock = MagicMock()
    captured_exports = []

    def fake_export(path, **kwargs):
        import trimesh
        captured_exports.append(path)
        # 実際に読める GLB ファイルを書き出す
        trimesh.creation.box().export(path)

    glb_mock.export = fake_export

    mock_o_voxel = MagicMock()
    mock_o_voxel.postprocess.to_glb.return_value = glb_mock

    # --- trellis2.pipelines ---
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = [mesh_mock]
    mock_pipeline.cuda.return_value = mock_pipeline

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

    mock_trellis2_pipelines = MagicMock()
    mock_trellis2_pipelines.Trellis2ImageTo3DPipeline = mock_pipeline_cls

    # --- torch ---
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    modules = {
        "trellis2": MagicMock(),
        "trellis2.pipelines": mock_trellis2_pipelines,
        "o_voxel": mock_o_voxel,
        "torch": mock_torch,
        "cv2": MagicMock(),
    }

    return modules, mock_pipeline, mesh_mock, mock_o_voxel, mock_torch, captured_exports


class _MeshGenTestBase(unittest.TestCase):
    """
    sys.modules パッチを setUp/tearDown で管理するベースクラス。

    各テストメソッドは self.gen (MeshGenerator) と関連モックを使用できる。
    sys.modules パッチは setUp で開始し tearDown で終了するため、
    generate_single() 内の `import torch` / `import o_voxel` も正しくモック化される。
    """

    def setUp(self, n_vertices: int = 10000):
        (self.modules, self.mock_pipeline, self.mesh_mock,
         self.mock_o_voxel, self.mock_torch, self.exported_paths
         ) = _build_sys_modules_patch(n_vertices)

        # src.mesh_generator をリロードして新しいモックが適用されるようにする
        if "src.mesh_generator" in sys.modules:
            del sys.modules["src.mesh_generator"]

        self._patcher = patch.dict("sys.modules", self.modules)
        self._patcher.start()

        from src.mesh_generator import MeshGenerator
        self.gen = MeshGenerator.__new__(MeshGenerator)
        self.gen._pipeline = self.mock_pipeline
        self.gen._pipeline_cls = MagicMock()

        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        # _pipeline を None にして __del__ が後のテストの mock_torch を汚染しないようにする
        if hasattr(self, "gen") and self.gen is not None:
            self.gen._pipeline = None
        self._patcher.stop()
        if "src.mesh_generator" in sys.modules:
            del sys.modules["src.mesh_generator"]
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ============================================================
# __init__ テスト（独立）
# ============================================================

class TestMeshGeneratorInit(unittest.TestCase):

    def tearDown(self):
        if "src.mesh_generator" in sys.modules:
            del sys.modules["src.mesh_generator"]

    def test_init_loads_pipeline(self):
        """from_pretrained と cuda() が呼ばれること"""
        modules, mock_pipeline, _, _, mock_torch, _ = _build_sys_modules_patch()
        mock_pipeline_cls = modules["trellis2.pipelines"].Trellis2ImageTo3DPipeline

        if "src.mesh_generator" in sys.modules:
            del sys.modules["src.mesh_generator"]

        with patch.dict("sys.modules", modules):
            from src.mesh_generator import MeshGenerator
            MeshGenerator("/fake/trellis")

        mock_pipeline_cls.from_pretrained.assert_called_once_with("/fake/trellis")
        mock_pipeline.cuda.assert_called_once()

    def test_import_error_on_missing_trellis2(self):
        """trellis2 がない場合に ImportError を送出すること"""
        if "src.mesh_generator" in sys.modules:
            del sys.modules["src.mesh_generator"]

        broken_modules = {
            "trellis2": None,
            "trellis2.pipelines": None,
            "o_voxel": MagicMock(),
            "torch": MagicMock(),
            "cv2": MagicMock(),
        }
        with patch.dict("sys.modules", broken_modules):
            from src.mesh_generator import MeshGenerator
            with self.assertRaises((ImportError, TypeError, AttributeError)):
                MeshGenerator("/fake/trellis")


# ============================================================
# generate_single() テスト
# ============================================================

class TestMeshGeneratorSingle(_MeshGenTestBase):

    # ---- 仕様書記載の必須テスト ----

    def test_glb_file_exists_after_generation(self):
        """GLB ファイルが生成されること"""
        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        self.gen.generate_single(str(img_path), output_path=str(glb_path))

        self.assertTrue(glb_path.exists(), "GLB ファイルが存在しない")

    def test_glb_file_size_greater_than_zero(self):
        """GLB ファイルサイズが 0 より大きいこと"""
        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        self.gen.generate_single(str(img_path), output_path=str(glb_path))

        self.assertGreater(glb_path.stat().st_size, 0, "GLB ファイルサイズが 0")

    def test_glb_loadable_by_trimesh(self):
        """trimesh.load() で GLB を読み込めること（有効なメッシュ）"""
        import trimesh

        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        self.gen.generate_single(str(img_path), output_path=str(glb_path))

        loaded = trimesh.load(str(glb_path))
        self.assertIsNotNone(loaded, "trimesh.load() が None を返した")

    # ---- 追加テスト ----

    def test_returns_glb_path_string(self):
        """generate_single() が GLB パスの文字列を返すこと"""
        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        result = self.gen.generate_single(str(img_path), output_path=str(glb_path))

        self.assertIsInstance(result, str)
        self.assertTrue(result.endswith(".glb"))

    def test_file_not_found_raises(self):
        """存在しない画像は FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.gen.generate_single("/nonexistent/image.png")

    def test_seed_calls_set_seed(self):
        """generate_single() が _set_seed(seed) を呼ぶこと"""
        called = []
        self.gen._set_seed = lambda s: called.append(s)

        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        self.gen.generate_single(str(img_path), seed=77, output_path=str(glb_path))

        self.assertIn(77, called, "seed=77 が _set_seed に渡されていない")

    def test_pipeline_run_called_with_pil_image(self):
        """pipeline.run() が PIL.Image を受け取ること"""
        from PIL import Image as PILImage

        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        self.gen.generate_single(str(img_path), output_path=str(glb_path))

        self.mock_pipeline.run.assert_called_once()
        arg = self.mock_pipeline.run.call_args[0][0]
        self.assertIsInstance(arg, PILImage.Image)

    def test_texture_size_passed_to_to_glb(self):
        """texture_size=1024 が to_glb() に渡されること"""
        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        self.gen.generate_single(str(img_path), texture_size=1024, output_path=str(glb_path))

        kwargs = self.mock_o_voxel.postprocess.to_glb.call_args[1]
        self.assertEqual(kwargs["texture_size"], 1024)

    def test_simplify_applied_to_mesh(self):
        """simplify=0.95 のとき mesh.simplify() が頂点数×0.95 で呼ばれること"""
        n = 20000
        self.setUp(n_vertices=n)  # 明示的に頂点数を設定して再初期化

        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        self.gen.generate_single(str(img_path), simplify=0.95, output_path=str(glb_path))

        self.mesh_mock.simplify.assert_called_once()
        target = self.mesh_mock.simplify.call_args[0][0]
        expected = int(n * 0.95)
        self.assertEqual(target, expected,
                         f"simplify の頂点数が不正: {target} (期待: {expected})")

    def test_auto_output_path(self):
        """output_path=None のとき入力と同ディレクトリに .glb が生成されること"""
        img_path = Path(self.tmpdir) / "auto_test.png"
        _make_test_image(img_path)

        result = self.gen.generate_single(str(img_path))

        self.assertEqual(Path(result), Path(self.tmpdir) / "auto_test.glb")


# ============================================================
# generate_batch() テスト
# ============================================================

class TestMeshGeneratorBatch(_MeshGenTestBase):

    def setUp(self):
        super().setUp()
        self.image_dir = Path(self.tmpdir) / "images"
        self.image_dir.mkdir()
        self.output_dir = Path(self.tmpdir) / "meshes"

    def _create_images(self, n: int) -> list[Path]:
        paths = []
        for i in range(n):
            p = self.image_dir / f"{i:06d}_test.png"
            _make_test_image(p)
            paths.append(p)
        return paths

    def test_batch_result_schema(self):
        """generate_batch() の各エントリが正しいスキーマを持つこと"""
        self._create_images(3)
        results = self.gen.generate_batch(str(self.image_dir), str(self.output_dir))

        self.assertEqual(len(results), 3)
        required = {"image_path", "glb_path", "seed", "status", "error", "timestamp"}
        for r in results:
            missing = required - set(r.keys())
            self.assertFalse(missing, f"不足キー: {missing}")

    def test_generated_glb_exists(self):
        """生成された GLB ファイルが存在すること"""
        self._create_images(2)
        results = self.gen.generate_batch(str(self.image_dir), str(self.output_dir))

        for r in results:
            if r["status"] == "generated":
                self.assertIsNotNone(r["glb_path"])
                self.assertTrue(Path(r["glb_path"]).exists())

    def test_metadata_json_saved(self):
        """generation_metadata.json が保存されること"""
        self._create_images(2)
        self.gen.generate_batch(str(self.image_dir), str(self.output_dir))

        meta_path = self.output_dir / "generation_metadata.json"
        self.assertTrue(meta_path.exists())
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["results"]), 2)

    def test_resume_skips_existing_glb(self):
        """既存 GLB があればスキップされること"""
        self._create_images(3)

        # 1 回目
        self.gen.generate_batch(str(self.image_dir), str(self.output_dir))

        # pipeline.run の呼び出しカウントをリセット
        self.mock_pipeline.run.reset_mock()

        # 2 回目（再開）
        results2 = self.gen.generate_batch(str(self.image_dir), str(self.output_dir))

        skipped = [r for r in results2 if r["status"] == "skipped"]
        self.assertEqual(len(skipped), 3, f"スキップ件数が不正: {len(skipped)}")
        # 2 回目は pipeline.run が呼ばれないこと
        self.mock_pipeline.run.assert_not_called()

    def test_failure_continues_batch(self):
        """1 件失敗してもバッチが継続されること"""
        self._create_images(3)

        fail_count = [0]
        orig_single = self.gen.generate_single.__func__  # unbound

        def flaky_single(self_inner, image_path, seed=42, **kwargs):
            if "000001" in str(image_path) and fail_count[0] == 0:
                fail_count[0] += 1
                raise RuntimeError("Simulated CUDA OOM")
            return orig_single(self_inner, image_path, seed=seed, **kwargs)

        import types
        self.gen.generate_single = types.MethodType(flaky_single, self.gen)

        results = self.gen.generate_batch(str(self.image_dir), str(self.output_dir))

        statuses = [r["status"] for r in results]
        self.assertIn("failed", statuses, "failed エントリがない")
        non_failed = [r for r in results if r["status"] != "failed"]
        self.assertGreater(len(non_failed), 0, "全件失敗になっている")

    def test_seeds_are_index_based(self):
        """シードがインデックス + offset で割り当てられること"""
        self._create_images(3)
        captured_seeds = []

        orig_single = self.gen.generate_single
        def capture(image_path, seed=42, **kwargs):
            captured_seeds.append(seed)
            return orig_single(image_path, seed=seed, **kwargs)
        self.gen.generate_single = capture

        self.gen.generate_batch(str(self.image_dir), str(self.output_dir), seed_offset=10)
        self.assertEqual(captured_seeds, [10, 11, 12])

    def test_empty_directory(self):
        """空ディレクトリでもエラーにならないこと"""
        results = self.gen.generate_batch(str(self.image_dir), str(self.output_dir))
        self.assertEqual(results, [])

    def test_partial_resume(self):
        """一部生成済みの場合、残りのみ生成されること"""
        self._create_images(4)

        # 全 4 件生成してメタデータを記録
        self.gen.generate_batch(str(self.image_dir), str(self.output_dir))
        self.mock_pipeline.run.reset_mock()

        # GLB を 2 件削除してシミュレート
        glbs = sorted(self.output_dir.glob("*.glb"))
        for glb in glbs[:2]:
            glb.unlink()

        # 同じ image_dir・output_dir で再実行
        results = self.gen.generate_batch(str(self.image_dir), str(self.output_dir))

        skipped = [r for r in results if r["status"] == "skipped"]
        generated = [r for r in results if r["status"] == "generated"]
        self.assertEqual(len(skipped), 2, f"スキップ件数が不正: {len(skipped)}")
        self.assertEqual(len(generated), 2, f"生成件数が不正: {len(generated)}")


# ============================================================
# unload() テスト
# ============================================================

class TestMeshGeneratorUnload(_MeshGenTestBase):

    def test_unload_clears_pipeline(self):
        """unload() 後に _pipeline が None になること"""
        self.assertIsNotNone(self.gen._pipeline)
        self.gen.unload()
        self.assertIsNone(self.gen._pipeline)

    def test_unload_calls_cuda_empty_cache(self):
        """unload() が torch.cuda.empty_cache() を呼ぶこと"""
        self.mock_torch.cuda.empty_cache.reset_mock()
        self.gen.unload()
        self.mock_torch.cuda.empty_cache.assert_called_once()

    def test_unload_idempotent(self):
        """unload() を複数回呼んでもエラーにならないこと"""
        self.gen.unload()
        self.gen.unload()  # 2 回目もエラーなし


if __name__ == "__main__":
    unittest.main(verbosity=2)
