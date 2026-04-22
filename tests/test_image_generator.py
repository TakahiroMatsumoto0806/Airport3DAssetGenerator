"""
T-2.1: 画像生成エンジン テスト

検証項目（仕様書記載）:
  - 3 つのプロンプトで画像生成し、ファイルサイズ > 0、解像度 1024×1024 を検証

追加検証:
  - generate_single() が PIL.Image を返す
  - generate_batch() のメタデータ JSON スキーマ
  - 中断再開（既存ファイルスキップ）
  - 失敗時の継続動作
  - unload() で GPU メモリが解放される

GPU なし環境でも動作するよう、diffusers.FluxPipeline をモック化して実行する。
DGX Spark 上での実際のモデルテストは tests/test_gpu_models.py (T-0.3) を参照。

実行方法:
    pytest tests/test_image_generator.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# プロジェクトルートを sys.path に追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---- FLUX モックファクトリ ----

def _make_flux_mock(width: int = 1024, height: int = 1024) -> MagicMock:
    """FluxPipeline の from_pretrained + __call__ をモック化する"""
    from PIL import Image as PILImage

    dummy_image = PILImage.new("RGB", (width, height), color=(200, 200, 200))
    mock_output = MagicMock()
    mock_output.images = [dummy_image]

    mock_pipe = MagicMock()
    mock_pipe.return_value = mock_output
    mock_pipe.to.return_value = mock_pipe  # .to("cuda") が自分を返す

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipe
    return mock_pipeline_cls, mock_pipe


# ---- テストヘルパー ----

def _make_prompt_list(n: int) -> list[dict]:
    """テスト用プロンプトリストを生成"""
    return [
        {
            "prompt": f"A black hard-shell suitcase, prompt {i}, on pure white background, studio lighting, photorealistic",
            "metadata": {
                "luggage_type": "hard_suitcase",
                "subcategory": "carry_on",
                "color": "black",
                "material": "polycarbonate",
                "texture": "smooth surface",
                "style": "modern minimalist design",
                "condition": "brand new",
                "prompt_id": f"test{i:06d}",
            },
        }
        for i in range(n)
    ]


# ============================================================
# テストクラス
# ============================================================

class TestImageGeneratorInit(unittest.TestCase):
    """__init__ テスト"""

    @patch("src.utils.memory_guard.assert_memory_headroom")
    def test_init_loads_pipeline(self, _mock_headroom):
        """from_pretrained が呼ばれ、パイプラインが device に移動すること"""
        mock_cls, mock_pipe = _make_flux_mock()

        with patch.dict("sys.modules", {"torch": MagicMock(), "diffusers": MagicMock()}):
            import importlib
            import sys as _sys

            # torch と diffusers をモック差し替え
            mock_torch = MagicMock()
            mock_torch.bfloat16 = "bfloat16"
            mock_diffusers = MagicMock()
            mock_diffusers.FluxPipeline = mock_cls

            _sys.modules["torch"] = mock_torch
            _sys.modules["diffusers"] = mock_diffusers

            # src.image_generator をリロードしてモックを反映
            if "src.image_generator" in _sys.modules:
                del _sys.modules["src.image_generator"]
            from src.image_generator import ImageGenerator

            gen = ImageGenerator(model_path="/fake/flux", device="cuda")

            mock_cls.from_pretrained.assert_called_once_with(
                "/fake/flux", torch_dtype="bfloat16"
            )
            mock_pipe.to.assert_called_once_with("cuda")

            # クリーンアップ
            if "src.image_generator" in _sys.modules:
                del _sys.modules["src.image_generator"]


class TestImageGeneratorSingle(unittest.TestCase):
    """generate_single() テスト"""

    def _get_generator(self, mock_cls, mock_pipe):
        """モック化された ImageGenerator を返す"""
        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()
        mock_diffusers = MagicMock()
        mock_diffusers.FluxPipeline = mock_cls

        sys.modules["torch"] = mock_torch
        sys.modules["diffusers"] = mock_diffusers
        if "src.image_generator" in sys.modules:
            del sys.modules["src.image_generator"]
        from src.image_generator import ImageGenerator

        with patch("src.utils.memory_guard.assert_memory_headroom"):
            gen = ImageGenerator(model_path="/fake/flux", device="cuda")
        return gen, mock_torch

    def setUp(self):
        self.mock_cls, self.mock_pipe = _make_flux_mock(1024, 1024)
        self._orig_torch = sys.modules.get("torch")
        self._orig_diffusers = sys.modules.get("diffusers")

    def tearDown(self):
        for mod in ["src.image_generator"]:
            if mod in sys.modules:
                del sys.modules[mod]
        # restore real torch/diffusers so later tests are not polluted
        if self._orig_torch is not None:
            sys.modules["torch"] = self._orig_torch
        else:
            sys.modules.pop("torch", None)
        if self._orig_diffusers is not None:
            sys.modules["diffusers"] = self._orig_diffusers
        else:
            sys.modules.pop("diffusers", None)

    def test_returns_pil_image(self):
        """generate_single() が PIL.Image を返すこと"""
        from PIL import Image as PILImage

        gen, _ = self._get_generator(self.mock_cls, self.mock_pipe)
        result = gen.generate_single("test prompt")
        self.assertIsInstance(result, PILImage.Image)

    def test_resolution_1024x1024(self):
        """仕様書記載: 解像度が 1024×1024 であること"""
        gen, _ = self._get_generator(self.mock_cls, self.mock_pipe)
        image = gen.generate_single("test prompt")
        self.assertEqual(image.size, (1024, 1024))

    def test_flux_called_with_correct_params(self):
        """num_inference_steps=4, guidance_scale=0.0 で呼ばれること"""
        gen, _ = self._get_generator(self.mock_cls, self.mock_pipe)
        gen.generate_single("test prompt", seed=None)

        call_kwargs = self.mock_pipe.call_args[1]
        self.assertEqual(call_kwargs["num_inference_steps"], 4)
        self.assertEqual(call_kwargs["guidance_scale"], 0.0)
        self.assertEqual(call_kwargs["height"], 1024)
        self.assertEqual(call_kwargs["width"], 1024)

    def test_seed_creates_generator(self):
        """seed 指定時に torch.Generator が作成されること"""
        gen, mock_torch = self._get_generator(self.mock_cls, self.mock_pipe)
        gen.generate_single("test prompt", seed=42)
        mock_torch.Generator.assert_called_once_with(device="cuda")
        mock_torch.Generator.return_value.manual_seed.assert_called_once_with(42)


class TestImageGeneratorBatch(unittest.TestCase):
    """generate_batch() テスト — 仕様書記載の必須テスト含む"""

    def setUp(self):
        """モック化した ImageGenerator をセットアップ"""
        from PIL import Image as PILImage

        self._orig_torch = sys.modules.get("torch")
        self._orig_diffusers = sys.modules.get("diffusers")

        self.dummy_image = PILImage.new("RGB", (1024, 1024), color=(200, 200, 200))
        mock_output = MagicMock()
        mock_output.images = [self.dummy_image]
        mock_pipe = MagicMock()
        mock_pipe.return_value = mock_output
        mock_pipe.to.return_value = mock_pipe
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_pipe

        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()
        mock_diffusers = MagicMock()
        mock_diffusers.FluxPipeline = mock_cls

        sys.modules["torch"] = mock_torch
        sys.modules["diffusers"] = mock_diffusers
        if "src.image_generator" in sys.modules:
            del sys.modules["src.image_generator"]
        from src.image_generator import ImageGenerator

        with patch("src.utils.memory_guard.assert_memory_headroom"):
            self.gen = ImageGenerator(model_path="/fake/flux", device="cuda")
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        for mod in ["src.image_generator"]:
            if mod in sys.modules:
                del sys.modules[mod]
        if self._orig_torch is not None:
            sys.modules["torch"] = self._orig_torch
        else:
            sys.modules.pop("torch", None)
        if self._orig_diffusers is not None:
            sys.modules["diffusers"] = self._orig_diffusers
        else:
            sys.modules.pop("diffusers", None)

    # ---- 仕様書記載の必須テスト ----

    def test_three_prompts_file_size_and_resolution(self):
        """
        仕様書記載:
          3 つのプロンプトで画像生成し、ファイルサイズ > 0、解像度 1024×1024 を検証
        """
        from PIL import Image as PILImage

        prompts = _make_prompt_list(3)
        results = self.gen.generate_batch(prompts, output_dir=self.tmpdir)

        self.assertEqual(len(results), 3, "結果件数が 3 でない")

        for i, r in enumerate(results):
            self.assertEqual(r["status"], "generated", f"[{i}] status が generated でない")
            self.assertIsNotNone(r["image_path"], f"[{i}] image_path が None")

            path = Path(r["image_path"])
            # ファイルサイズ > 0
            self.assertTrue(path.exists(), f"[{i}] ファイルが存在しない: {path}")
            self.assertGreater(path.stat().st_size, 0, f"[{i}] ファイルサイズが 0")

            # 解像度 1024×1024
            img = PILImage.open(path)
            self.assertEqual(img.size, (1024, 1024), f"[{i}] 解像度が不正: {img.size}")

    # ---- 追加テスト ----

    def test_metadata_json_created(self):
        """generation_metadata.json が作成されること"""
        prompts = _make_prompt_list(3)
        self.gen.generate_batch(prompts, output_dir=self.tmpdir)

        meta_path = Path(self.tmpdir) / "generation_metadata.json"
        self.assertTrue(meta_path.exists(), "generation_metadata.json が存在しない")

        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("total", data)
        self.assertIn("generated", data)
        self.assertIn("skipped", data)
        self.assertIn("failed", data)
        self.assertIn("results", data)
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["generated"], 3)

    def test_metadata_result_schema(self):
        """各結果エントリに必要なフィールドが揃っていること"""
        prompts = _make_prompt_list(2)
        results = self.gen.generate_batch(prompts, output_dir=self.tmpdir)

        required_keys = {"prompt", "metadata", "image_path", "seed", "status", "error"}
        for i, r in enumerate(results):
            missing = required_keys - set(r.keys())
            self.assertFalse(missing, f"[{i}] 不足フィールド: {missing}")

    def test_resume_skips_existing(self):
        """中断再開: 既存ファイルがスキップされること"""
        prompts = _make_prompt_list(3)

        # 1 回目の実行
        self.gen.generate_batch(prompts, output_dir=self.tmpdir)

        # 2 回目の実行 — 全件スキップになるはず
        results2 = self.gen.generate_batch(prompts, output_dir=self.tmpdir)

        skipped = [r for r in results2 if r["status"] == "skipped"]
        self.assertEqual(len(skipped), 3, f"スキップ件数が不正: {len(skipped)}")

    def test_partial_resume(self):
        """一部のみ生成済みの場合、未生成分だけ生成されること"""
        prompts = _make_prompt_list(4)

        # 最初の 2 件だけ生成
        self.gen.generate_batch(prompts[:2], output_dir=self.tmpdir)

        # 4 件全体で再実行
        results = self.gen.generate_batch(prompts, output_dir=self.tmpdir)

        skipped = [r for r in results if r["status"] == "skipped"]
        generated = [r for r in results if r["status"] == "generated"]
        self.assertEqual(len(skipped), 2, f"スキップ件数が不正: {len(skipped)}")
        self.assertEqual(len(generated), 2, f"生成件数が不正: {len(generated)}")

    def test_failure_continues_batch(self):
        """1 件失敗しても残りのバッチ処理が継続されること"""
        from PIL import Image as PILImage

        fail_count = 0

        def flaky_single(prompt, **kwargs):
            nonlocal fail_count
            if "prompt 1" in prompt and fail_count == 0:
                fail_count += 1
                raise RuntimeError("Simulated CUDA OOM")
            return PILImage.new("RGB", (1024, 1024), color=(200, 200, 200))

        self.gen.generate_single = flaky_single

        prompts = _make_prompt_list(3)
        results = self.gen.generate_batch(prompts, output_dir=self.tmpdir)

        statuses = [r["status"] for r in results]
        self.assertIn("failed", statuses, "失敗エントリがない")
        # 失敗以外は generated であること
        non_failed = [r for r in results if r["status"] != "failed"]
        self.assertGreater(len(non_failed), 0, "全件失敗になっている")

    def test_output_dir_created_automatically(self):
        """出力ディレクトリが自動作成されること"""
        nested_dir = Path(self.tmpdir) / "a" / "b" / "c"
        prompts = _make_prompt_list(1)
        self.gen.generate_batch(prompts, output_dir=str(nested_dir))
        self.assertTrue(nested_dir.exists())

    def test_seeds_applied(self):
        """seeds リストが generate_single に渡されること"""
        call_seeds = []
        original_single = self.gen.generate_single.__func__ if hasattr(
            self.gen.generate_single, "__func__"
        ) else None

        from PIL import Image as PILImage

        def capture_single(prompt, seed=None, **kwargs):
            call_seeds.append(seed)
            return PILImage.new("RGB", (1024, 1024))

        self.gen.generate_single = capture_single

        prompts = _make_prompt_list(3)
        custom_seeds = [100, 200, 300]
        self.gen.generate_batch(prompts, output_dir=self.tmpdir, seeds=custom_seeds)

        self.assertEqual(call_seeds, custom_seeds)

    def test_default_seeds_are_random(self):
        """seeds が None の場合、ランダムシードが生成されること。

        以前はインデックス値 [0, 1, 2, ...] を seed に使っていたため、実行のたびに
        同じ画像が生成される不具合があった。generate_batch は seeds=None のとき
        random.randint(0, 2**32-1) でシードを生成する仕様に修正されている。
        """
        call_seeds = []

        from PIL import Image as PILImage

        def capture_single(prompt, seed=None, **kwargs):
            call_seeds.append(seed)
            return PILImage.new("RGB", (1024, 1024))

        self.gen.generate_single = capture_single

        prompts = _make_prompt_list(3)
        self.gen.generate_batch(prompts, output_dir=self.tmpdir, seeds=None)

        # 3 件のシードが生成され、インデックス値 [0, 1, 2] ではないこと
        self.assertEqual(len(call_seeds), 3)
        self.assertNotEqual(call_seeds, [0, 1, 2],
                            "seeds=None でインデックス値が使われている（ランダム化されていない）")
        # 全てのシードが 32-bit 整数範囲内であること
        for s in call_seeds:
            self.assertIsInstance(s, int)
            self.assertGreaterEqual(s, 0)
            self.assertLess(s, 2**32)


class TestImageGeneratorUnload(unittest.TestCase):
    """unload() テスト"""

    def setUp(self):
        self._orig_torch = sys.modules.get("torch")
        self._orig_diffusers = sys.modules.get("diffusers")

        mock_cls, mock_pipe = _make_flux_mock()
        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.cuda.is_available.return_value = True
        mock_diffusers = MagicMock()
        mock_diffusers.FluxPipeline = mock_cls

        # flush_cuda_memory() が memory_reserved()/memory_allocated() を呼ぶため
        # f-string フォーマット (.2f) が通るよう数値を返すよう設定
        mock_torch.cuda.memory_reserved.return_value = 0
        mock_torch.cuda.memory_allocated.return_value = 0

        sys.modules["torch"] = mock_torch
        sys.modules["diffusers"] = mock_diffusers
        if "src.image_generator" in sys.modules:
            del sys.modules["src.image_generator"]
        from src.image_generator import ImageGenerator

        with patch("src.utils.memory_guard.assert_memory_headroom"):
            self.gen = ImageGenerator(model_path="/fake/flux", device="cuda")
        self.mock_torch = mock_torch

    def tearDown(self):
        if "src.image_generator" in sys.modules:
            del sys.modules["src.image_generator"]
        if self._orig_torch is not None:
            sys.modules["torch"] = self._orig_torch
        else:
            sys.modules.pop("torch", None)
        if self._orig_diffusers is not None:
            sys.modules["diffusers"] = self._orig_diffusers
        else:
            sys.modules.pop("diffusers", None)

    def test_unload_clears_pipe(self):
        """unload() 後に _pipe が None になること"""
        self.assertIsNotNone(self.gen._pipe)
        self.gen.unload()
        self.assertIsNone(self.gen._pipe)

    def test_unload_calls_cuda_empty_cache(self):
        """unload() が torch.cuda.empty_cache() を呼ぶこと"""
        self.gen.unload()
        self.mock_torch.cuda.empty_cache.assert_called_once()

    def test_unload_idempotent(self):
        """unload() を複数回呼んでもエラーにならないこと"""
        self.gen.unload()
        self.gen.unload()  # 2 回目


if __name__ == "__main__":
    unittest.main(verbosity=2)
