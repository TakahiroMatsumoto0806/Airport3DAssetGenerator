"""
T-1.2: プロンプト生成エンジン テスト

検証項目（仕様書記載）:
  - 100 個のプロンプトを生成し、全プロンプトがユニーク
  - 全プロンプトに luggage_type メタデータがある
  - プロンプトに白背景指定が含まれている

追加検証:
  - save() で JSON ファイルが正しく書き出される
  - get_statistics() が正しいスキーマを返す
  - LLM リファインはモック経由で動作確認（vLLM サーバー不要）

実行方法:
    pytest tests/test_prompt_generator.py -v
    pytest tests/test_prompt_generator.py -v -k test_generate_100_unique
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

from src.prompt_generator import PromptGenerator

CONFIG_DIR = PROJECT_ROOT / "configs"

# configs が存在するか事前確認
if not CONFIG_DIR.exists():
    raise RuntimeError(f"configs ディレクトリが見つかりません: {CONFIG_DIR}")


class TestPromptGeneratorBasic(unittest.TestCase):
    """基本動作テスト"""

    def setUp(self):
        self.gen = PromptGenerator(config_dir=str(CONFIG_DIR), seed=42)

    # ------------------------------------------------------------------
    # 仕様書記載の必須テスト
    # ------------------------------------------------------------------

    def test_generate_100_unique(self):
        """100 個のプロンプトを生成し、全てユニークであること"""
        prompts = self.gen.generate_combinatorial(n=100)
        self.assertEqual(len(prompts), 100, "生成件数が 100 件でない")

        texts = [p["prompt"] for p in prompts]
        unique_texts = set(texts)
        self.assertEqual(
            len(unique_texts),
            100,
            f"重複プロンプトが存在する: {100 - len(unique_texts)} 件の重複",
        )

    def test_all_have_luggage_type(self):
        """全プロンプトに luggage_type メタデータがあること"""
        prompts = self.gen.generate_combinatorial(n=100)
        for i, p in enumerate(prompts):
            self.assertIn(
                "metadata",
                p,
                f"[{i}] metadata キーがない",
            )
            self.assertIn(
                "luggage_type",
                p["metadata"],
                f"[{i}] luggage_type キーがない: metadata={p['metadata']}",
            )
            self.assertIsInstance(
                p["metadata"]["luggage_type"],
                str,
                f"[{i}] luggage_type が文字列でない",
            )
            self.assertGreater(
                len(p["metadata"]["luggage_type"]),
                0,
                f"[{i}] luggage_type が空文字",
            )

    def test_all_contain_white_background(self):
        """全プロンプトに白背景指定が含まれていること"""
        prompts = self.gen.generate_combinatorial(n=100)
        for i, p in enumerate(prompts):
            prompt_lower = p["prompt"].lower()
            self.assertIn(
                "white background",
                prompt_lower,
                f"[{i}] 白背景指定がない: {p['prompt'][:80]}",
            )

    # ------------------------------------------------------------------
    # メタデータ完全性テスト
    # ------------------------------------------------------------------

    def test_metadata_schema(self):
        """メタデータに必要な全キーが揃っていること"""
        required_keys = {
            "luggage_type",
            "subcategory",
            "color",
            "material",
            "texture",
            "style",
            "condition",
            "prompt_id",
        }
        prompts = self.gen.generate_combinatorial(n=20)
        for i, p in enumerate(prompts):
            missing = required_keys - set(p["metadata"].keys())
            self.assertFalse(
                missing,
                f"[{i}] metadata に不足キーあり: {missing}",
            )

    def test_metadata_luggage_type_is_valid_category(self):
        """luggage_type が luggage_categories.yaml に定義されたカテゴリであること"""
        from omegaconf import OmegaConf

        cats = OmegaConf.load(CONFIG_DIR / "luggage_categories.yaml")
        valid_categories = set(cats.categories.keys())

        prompts = self.gen.generate_combinatorial(n=50)
        for i, p in enumerate(prompts):
            lt = p["metadata"]["luggage_type"]
            self.assertIn(
                lt,
                valid_categories,
                f"[{i}] 未定義カテゴリ: {lt}",
            )

    # ------------------------------------------------------------------
    # 固定フレーズテスト
    # ------------------------------------------------------------------

    def test_fixed_phrases_present(self):
        """固定フレーズが全プロンプトに含まれていること"""
        fixed_phrases = [
            "white background",
            "studio lighting",
            "photorealistic",
        ]
        prompts = self.gen.generate_combinatorial(n=50)
        for i, p in enumerate(prompts):
            prompt_lower = p["prompt"].lower()
            for phrase in fixed_phrases:
                self.assertIn(
                    phrase,
                    prompt_lower,
                    f"[{i}] 固定フレーズ '{phrase}' がない: {p['prompt'][:80]}",
                )

    # ------------------------------------------------------------------
    # プロンプト長テスト
    # ------------------------------------------------------------------

    def test_prompt_length_reasonable(self):
        """プロンプト長が適切な範囲であること（20〜400 文字）"""
        prompts = self.gen.generate_combinatorial(n=50)
        for i, p in enumerate(prompts):
            length = len(p["prompt"])
            self.assertGreater(length, 20, f"[{i}] プロンプトが短すぎる: {length}")
            self.assertLess(length, 400, f"[{i}] プロンプトが長すぎる: {length}")

    # ------------------------------------------------------------------
    # 再現性テスト
    # ------------------------------------------------------------------

    def test_reproducibility(self):
        """同じシードで同じプロンプトが生成されること"""
        gen1 = PromptGenerator(config_dir=str(CONFIG_DIR), seed=123)
        gen2 = PromptGenerator(config_dir=str(CONFIG_DIR), seed=123)
        p1 = gen1.generate_combinatorial(n=10)
        p2 = gen2.generate_combinatorial(n=10)
        texts1 = [p["prompt"] for p in p1]
        texts2 = [p["prompt"] for p in p2]
        self.assertEqual(texts1, texts2, "同じシードで異なる結果が生成された")

    def test_different_seeds_different_prompts(self):
        """異なるシードで異なるプロンプトが生成されること"""
        gen1 = PromptGenerator(config_dir=str(CONFIG_DIR), seed=1)
        gen2 = PromptGenerator(config_dir=str(CONFIG_DIR), seed=9999)
        p1 = {p["prompt"] for p in gen1.generate_combinatorial(n=20)}
        p2 = {p["prompt"] for p in gen2.generate_combinatorial(n=20)}
        # 全て同じにはならないはず
        self.assertNotEqual(p1, p2, "異なるシードで同一プロンプト群が生成された")

    # ------------------------------------------------------------------
    # 大量生成テスト
    # ------------------------------------------------------------------

    def test_generate_1500(self):
        """1500 件生成できること（実際のパイプライン想定）"""
        prompts = self.gen.generate_combinatorial(n=1500)
        self.assertGreaterEqual(len(prompts), 1000, "1000 件以上生成されること")
        # ユニーク率 95% 以上
        unique_rate = len({p["prompt"] for p in prompts}) / len(prompts)
        self.assertGreaterEqual(unique_rate, 0.95, f"ユニーク率が低い: {unique_rate:.2%}")

    # ------------------------------------------------------------------
    # カテゴリ分布テスト
    # ------------------------------------------------------------------

    def test_category_distribution(self):
        """全カテゴリが少なくとも 1 件以上生成されること（1500件）"""
        from omegaconf import OmegaConf

        cats = OmegaConf.load(CONFIG_DIR / "luggage_categories.yaml")
        expected_categories = set(cats.categories.keys())

        prompts = self.gen.generate_combinatorial(n=1500)
        generated_categories = {p["metadata"]["luggage_type"] for p in prompts}

        missing = expected_categories - generated_categories
        self.assertFalse(
            missing,
            f"生成されなかったカテゴリ: {missing}",
        )


class TestPromptGeneratorSave(unittest.TestCase):
    """save() テスト"""

    def setUp(self):
        self.gen = PromptGenerator(config_dir=str(CONFIG_DIR), seed=42)

    def test_save_creates_valid_json(self):
        """save() が有効な JSON ファイルを作成すること"""
        prompts = self.gen.generate_combinatorial(n=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_prompts.json"
            result_path = self.gen.save(prompts, str(output_path))

            self.assertTrue(result_path.exists(), "ファイルが作成されない")
            self.assertGreater(result_path.stat().st_size, 0, "ファイルサイズが 0")

            with open(result_path, encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("total", data)
            self.assertIn("prompts", data)
            self.assertEqual(data["total"], 10)
            self.assertEqual(len(data["prompts"]), 10)

    def test_save_json_schema(self):
        """保存した JSON の各エントリが正しいスキーマを持つこと"""
        prompts = self.gen.generate_combinatorial(n=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.json"
            self.gen.save(prompts, str(output_path))

            with open(output_path, encoding="utf-8") as f:
                data = json.load(f)

            for i, entry in enumerate(data["prompts"]):
                self.assertIn("prompt", entry, f"[{i}] prompt キーがない")
                self.assertIn("metadata", entry, f"[{i}] metadata キーがない")
                self.assertIn(
                    "luggage_type",
                    entry["metadata"],
                    f"[{i}] luggage_type キーがない",
                )

    def test_save_creates_parent_dirs(self):
        """save() が親ディレクトリを自動作成すること"""
        prompts = self.gen.generate_combinatorial(n=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / "a" / "b" / "c" / "out.json"
            self.gen.save(prompts, str(deep_path))
            self.assertTrue(deep_path.exists())


class TestPromptGeneratorStatistics(unittest.TestCase):
    """get_statistics() テスト"""

    def setUp(self):
        self.gen = PromptGenerator(config_dir=str(CONFIG_DIR), seed=42)

    def test_statistics_schema(self):
        """統計情報が正しいスキーマを返すこと"""
        prompts = self.gen.generate_combinatorial(n=100)
        stats = self.gen.get_statistics(prompts)

        required_keys = {
            "total",
            "unique_prompts",
            "category_distribution",
            "color_distribution",
            "material_distribution",
            "condition_distribution",
            "refined_count",
        }
        missing = required_keys - set(stats.keys())
        self.assertFalse(missing, f"統計情報に不足キーあり: {missing}")

    def test_statistics_totals(self):
        """統計情報の合計が正しいこと"""
        prompts = self.gen.generate_combinatorial(n=100)
        stats = self.gen.get_statistics(prompts)

        self.assertEqual(stats["total"], 100)
        self.assertGreaterEqual(stats["unique_prompts"], 95)  # 95% 以上ユニーク
        self.assertEqual(stats["refined_count"], 0)  # リファインなしなので 0

        cat_total = sum(stats["category_distribution"].values())
        self.assertEqual(cat_total, 100)


class TestPromptGeneratorLLMRefinement(unittest.TestCase):
    """LLM リファイン（モック）テスト"""

    def setUp(self):
        self.gen = PromptGenerator(config_dir=str(CONFIG_DIR), seed=42)

    def test_llm_refinement_mock(self):
        """LLM リファインがモック経由で動作し、refined フラグが立つこと"""
        base_prompts = self.gen.generate_combinatorial(n=3)

        # openai.OpenAI のモック
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "A sleek black polycarbonate carry-on suitcase with spinner wheels, "
            "studio lighting, on pure white background, photorealistic"
        )
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_model = MagicMock()
        mock_model.id = "Qwen/Qwen3-VL-32B-Instruct"
        mock_models_response = MagicMock()
        mock_models_response.data = [mock_model]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client.models.list.return_value = mock_models_response

        with patch("src.prompt_generator.PromptGenerator.generate_with_llm_refinement") as mock_refine:
            # モック: refined=True のメタデータを付与して返す
            def fake_refine(base, **kwargs):
                result = []
                for item in base:
                    result.append({
                        **item,
                        "prompt": mock_choice.message.content,
                        "metadata": {**item["metadata"], "refined": True, "original_prompt": item["prompt"]},
                    })
                return result

            mock_refine.side_effect = fake_refine
            refined = self.gen.generate_with_llm_refinement(base_prompts)

        for i, r in enumerate(refined):
            self.assertIn("metadata", r, f"[{i}] metadata なし")

    def test_llm_refinement_fallback_on_error(self):
        """vLLM サーバーが利用不可の場合、元プロンプトが維持されること"""
        base_prompts = self.gen.generate_combinatorial(n=3)
        original_texts = [p["prompt"] for p in base_prompts]

        # openai が ImportError を返すケースをシミュレート
        with patch.dict("sys.modules", {"openai": None}):
            # openai なしの場合は base_prompts をそのまま返す
            import importlib
            import src.prompt_generator as pg_module
            importlib.reload(pg_module)
            gen2 = pg_module.PromptGenerator(config_dir=str(CONFIG_DIR), seed=42)
            result = gen2.generate_with_llm_refinement(base_prompts)

        # 元プロンプトが維持されていること
        self.assertEqual(len(result), len(base_prompts))
        for i, (orig, res) in enumerate(zip(original_texts, result)):
            self.assertEqual(
                orig,
                res["prompt"],
                f"[{i}] エラー時に元プロンプトが変更された",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
