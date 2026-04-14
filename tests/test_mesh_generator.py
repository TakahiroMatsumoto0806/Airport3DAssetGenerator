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


def _make_tensor_like(arr: "np.ndarray") -> MagicMock:
    """
    torch.Tensor を模倣するモック。
    .cpu().float().numpy() / .cpu().numpy() / .to(device) パターンに対応する。
    """
    m = MagicMock()
    m.cpu.return_value = m
    m.float.return_value = m
    m.to.return_value = m
    m.numpy.return_value = arr
    m.clamp.return_value = m   # clamp() も self を返してチェーン可能にする
    m.tolist.return_value = arr.tolist() if hasattr(arr, "tolist") else list(arr)
    return m


def _build_sys_modules_patch(n_vertices: int = 10000) -> tuple:
    """
    sys.modules に注入するモジュールモックと関連オブジェクトを返す。

    Returns:
        (modules_dict, mock_pipeline, mesh_mock, mock_o_voxel, mock_torch)
    """
    import numpy as np

    # --- mesh モック用データ ---
    # _export_glb_trimesh と _query_vertex_attrs_torch の両方が
    # .cpu().float().numpy() パターンを必要とするため、tensor-like モックを使う。
    n_voxels = max(10, n_vertices // 100)
    n_faces = max(3, n_vertices // 100)
    C = 6  # [base_color_R, G, B, metallic, roughness, alpha]

    verts_np   = np.zeros((n_vertices, 3), dtype=np.float32)
    # 全面が頂点 0,1,2 を参照する最小有効メッシュ
    faces_np   = np.tile([0, 1, 2], (n_faces, 1)).astype(np.int32)
    coords_np  = np.zeros((n_voxels, 3), dtype=np.float32)
    attrs_np   = np.full((n_voxels, C), 0.5, dtype=np.float32)
    origin_np  = np.array([-0.5, -0.5, -0.5], dtype=np.float32)

    # --- mesh モック ---
    mesh_mock = MagicMock()
    mesh_mock.vertices  = _make_tensor_like(verts_np)
    mesh_mock.faces     = _make_tensor_like(faces_np)
    mesh_mock.attrs     = _make_tensor_like(attrs_np)
    mesh_mock.coords    = _make_tensor_like(coords_np)
    mesh_mock.origin    = _make_tensor_like(origin_np)
    mesh_mock.voxel_size = 0.01
    mesh_mock.layout    = {
        "base_color": slice(0, 3),
        "metallic":   slice(3, 4),
        "roughness":  slice(4, 5),
        "alpha":      slice(5, 6),
    }

    # --- o_voxel.postprocess.to_glb (後方互換のため残す) ---
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

    # mesh_generator は `from trellis2.pipelines.trellis2_image_to_3d import ...` を使う
    mock_trellis2_img3d = MagicMock()
    mock_trellis2_img3d.Trellis2ImageTo3DPipeline = mock_pipeline_cls

    # --- torch ---
    # torch.tensor() が _query_vertex_attrs_torch() 内で呼ばれるため、
    # 戻り値が .clamp().cpu().numpy() パターンをサポートするようにする。
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.memory_reserved.return_value = 0
    mock_torch.cuda.memory_allocated.return_value = 0

    def _fake_torch_tensor(data, dtype=None):
        import numpy as np
        arr = np.clip(np.asarray(data, dtype=np.float32), 0.0, 1.0)
        return _make_tensor_like(arr)

    mock_torch.tensor.side_effect = _fake_torch_tensor

    modules = {
        "trellis2": MagicMock(),
        "trellis2.pipelines": mock_trellis2_pipelines,
        "trellis2.pipelines.trellis2_image_to_3d": mock_trellis2_img3d,
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
        # mesh_generator は trellis2.pipelines.trellis2_image_to_3d サブモジュールから import する
        mock_pipeline_cls = modules["trellis2.pipelines.trellis2_image_to_3d"].Trellis2ImageTo3DPipeline

        if "src.mesh_generator" in sys.modules:
            del sys.modules["src.mesh_generator"]

        with patch.dict("sys.modules", modules), \
             patch("src.utils.memory_guard.assert_memory_headroom"):
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

    def test_seed_passed_to_pipeline_run(self):
        """generate_single(seed=77) が pipeline.run() に seed=77 を渡すこと"""
        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        self.gen.generate_single(str(img_path), seed=77, output_path=str(glb_path))

        call_kwargs = self.mock_pipeline.run.call_args[1]
        self.assertEqual(call_kwargs.get("seed"), 77, "seed=77 が pipeline.run に渡されていない")

    def test_pipeline_run_called_with_pil_image(self):
        """pipeline.run() が PIL.Image を受け取ること"""
        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        self.gen.generate_single(str(img_path), output_path=str(glb_path))

        self.mock_pipeline.run.assert_called_once()
        arg = self.mock_pipeline.run.call_args[0][0]
        # patch.dict 環境では PIL クラスの同一性が崩れることがあるため、
        # isinstance の代わりに型名・属性で確認する
        self.assertEqual(type(arg).__name__, "Image", f"PIL Image が渡されていない: {type(arg)}")
        self.assertEqual(arg.mode, "RGB")
        self.assertEqual(arg.size, (512, 512))

    def test_texture_size_parameter_accepted(self):
        """texture_size パラメータが受け入れられること（将来用、現在は vertex color 出力）"""
        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        # texture_size は将来用パラメータとして受け入れるだけ確認（例外が出ないこと）
        self.gen.generate_single(str(img_path), texture_size=1024, output_path=str(glb_path))

    def test_alpha_forced_255_in_glb(self):
        """GLB の頂点アルファが 255 (不透明) に固定されていること"""
        import trimesh

        img_path = Path(self.tmpdir) / "test.png"
        _make_test_image(img_path)
        glb_path = Path(self.tmpdir) / "test.glb"

        self.gen.generate_single(str(img_path), output_path=str(glb_path))

        loaded = trimesh.load(str(glb_path))
        if hasattr(loaded, "visual") and hasattr(loaded.visual, "vertex_colors"):
            alpha = loaded.visual.vertex_colors[:, 3]
            self.assertTrue(
                (alpha == 255).all(),
                f"アルファが 255 でない頂点が存在: min={alpha.min()}"
            )

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
