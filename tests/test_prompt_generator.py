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
        """プロンプト長が適切な範囲であること（20〜900 文字）。
        A-2/A-3 で受託手荷物状態フレーズ・カテゴリ別状態フレーズが追加されたため、
        上限を 900 文字に更新（LLM リファイン後は 60-75 語以内に圧縮される）。
        """
        prompts = self.gen.generate_combinatorial(n=50)
        for i, p in enumerate(prompts):
            length = len(p["prompt"])
            self.assertGreater(length, 20, f"[{i}] プロンプトが短すぎる: {length}")
            self.assertLess(length, 900, f"[{i}] プロンプトが長すぎる: {length}")

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

    # リファイン後の期待プロンプト（冗長タグを除去した 60-75 words の例）
    _REFINED_PROMPT = (
        "single product photo, object centered, medium travel duffel with handles, "
        "glossy black, heavy-duty canvas, modern minimalist design, brand new, "
        "solid white background, flat studio lighting, no background shadows, frontal view, "
        "fully visible, photorealistic"
    )

    def setUp(self):
        self.gen = PromptGenerator(config_dir=str(CONFIG_DIR), seed=42)

    def _make_mock_client(self, refined_text: str) -> MagicMock:
        """OpenAI クライアントをモック化する"""
        mock_choice = MagicMock()
        mock_choice.message.content = refined_text
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_llm_refinement_mock(self):
        """LLM リファインが OpenAI クライアント経由で動作し、refined メタデータが付与されること"""
        base_prompts = self.gen.generate_combinatorial(n=3)

        with patch.object(self.gen.__class__, "_wait_for_vllm", return_value=None), \
             patch("openai.OpenAI", return_value=self._make_mock_client(self._REFINED_PROMPT)):
            refined = self.gen.generate_with_llm_refinement(base_prompts)

        self.assertEqual(len(refined), 3)
        for i, r in enumerate(refined):
            self.assertIn("metadata", r, f"[{i}] metadata なし")
            self.assertTrue(r["metadata"].get("refined"), f"[{i}] refined フラグが立っていない")
            self.assertIn("original_prompt", r["metadata"], f"[{i}] original_prompt が保存されていない")
            # リファイン後のプロンプトが元プロンプトと異なること
            self.assertNotEqual(r["prompt"], r["metadata"]["original_prompt"],
                                f"[{i}] リファイン結果が元プロンプトと同じ")

    def test_llm_refinement_output_fits_clip_budget(self):
        """リファイン後プロンプトが CLIP 77 token 目標（≈ 350 字 / 75 words）以内であること"""
        base_prompts = self.gen.generate_combinatorial(n=5)

        with patch.object(self.gen.__class__, "_wait_for_vllm", return_value=None), \
             patch("openai.OpenAI", return_value=self._make_mock_client(self._REFINED_PROMPT)):
            refined = self.gen.generate_with_llm_refinement(base_prompts)

        for i, r in enumerate(refined):
            length = len(r["prompt"])
            words = len(r["prompt"].split())
            self.assertLessEqual(
                words, 80,
                f"[{i}] リファイン後プロンプトが 80 words を超えている: {words} words"
            )
            self.assertLessEqual(
                length, 400,
                f"[{i}] リファイン後プロンプトが 400 字を超えている: {length} 字"
            )

    def test_llm_refinement_vllm_unavailable_raises(self):
        """vLLM サーバーが起動していない場合 RuntimeError で異常終了すること"""
        base_prompts = self.gen.generate_combinatorial(n=2)

        # _wait_for_vllm が即タイムアウトするようモック
        with patch.object(
            self.gen.__class__, "_wait_for_vllm",
            side_effect=RuntimeError(
                "[PromptGenerator] vLLM サーバーに接続できませんでした。\n"
                "  URL      : http://localhost:8001/health\n"
                "  待機時間 : 300 秒"
            )
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.gen.generate_with_llm_refinement(base_prompts)

        self.assertIn("vLLM", str(ctx.exception))
        self.assertIn("接続できませんでした", str(ctx.exception))

    def test_llm_refinement_vllm_unavailable_error_contains_instructions(self):
        """RuntimeError メッセージに対処方法が含まれること"""
        base_prompts = self.gen.generate_combinatorial(n=1)

        with patch.object(
            self.gen.__class__, "_wait_for_vllm",
            side_effect=RuntimeError(
                "[PromptGenerator] vLLM サーバーに接続できませんでした。\n"
                "  対処方法 : vLLM サーバーを起動してから再実行してください。"
            )
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.gen.generate_with_llm_refinement(base_prompts)

        self.assertIn("対処方法", str(ctx.exception))

    def test_wait_for_vllm_returns_immediately_when_available(self):
        """ヘルスチェックが 200 を返す場合 RuntimeError を送出しないこと"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("src.prompt_generator.PromptGenerator._wait_for_vllm") as mock_wait:
            mock_wait.return_value = None  # 正常終了
            # _wait_for_vllm が呼ばれてもエラーにならないことを確認
            with patch("openai.OpenAI", return_value=self._make_mock_client(self._REFINED_PROMPT)):
                base_prompts = self.gen.generate_combinatorial(n=1)
                result = self.gen.generate_with_llm_refinement(base_prompts)
        self.assertEqual(len(result), 1)

    def test_wait_for_vllm_raises_after_timeout(self):
        """_wait_for_vllm がタイムアウト後に RuntimeError を送出すること"""
        # 存在しないポートに極短タイムアウトで接続 → 即タイムアウト
        with self.assertRaises(RuntimeError) as ctx:
            self.gen._wait_for_vllm(
                "http://localhost:19999/v1",
                timeout=0.1,         # 実質即タイムアウト
                poll_interval=0.05,
            )
        self.assertIn("接続できませんでした", str(ctx.exception))


class TestCheckedBaggagePromptQuality(unittest.TestCase):
    """A-1/A-2/A-3: 受託手荷物プロンプト品質テスト"""

    def setUp(self):
        self.gen = PromptGenerator(config_dir=str(CONFIG_DIR), seed=42)

    # ---- A-1: 機内持ち込みサブカテゴリが生成されないこと ----

    def test_a1_hard_suitcase_no_carry_on(self):
        """hard_suitcase の carry_on サブカテゴリが生成されないこと"""
        from omegaconf import OmegaConf
        type_cfg = self.gen._templates.attributes.type
        sub_keys = list(OmegaConf.to_container(type_cfg.hard_suitcase).keys())
        self.assertNotIn("carry_on", sub_keys, "hard_suitcase に carry_on サブカテゴリが残っている")

    def test_a1_soft_suitcase_no_carry_on(self):
        """soft_suitcase の carry_on サブカテゴリが生成されないこと"""
        from omegaconf import OmegaConf
        type_cfg = self.gen._templates.attributes.type
        sub_keys = list(OmegaConf.to_container(type_cfg.soft_suitcase).keys())
        self.assertNotIn("carry_on", sub_keys, "soft_suitcase に carry_on サブカテゴリが残っている")

    def test_a1_backpack_no_daypack(self):
        """backpack の daypack サブカテゴリが生成されないこと"""
        from omegaconf import OmegaConf
        type_cfg = self.gen._templates.attributes.type
        sub_keys = list(OmegaConf.to_container(type_cfg.backpack).keys())
        self.assertNotIn("daypack", sub_keys, "backpack に daypack サブカテゴリが残っている")

    def test_a1_large_scale_no_carry_on_subcategory(self):
        """1500件生成して carry_on / daypack がメタデータに現れないこと"""
        prompts = self.gen.generate_combinatorial(n=1500)
        for p in prompts:
            sub = p["metadata"]["subcategory"]
            cat = p["metadata"]["luggage_type"]
            if cat in ("hard_suitcase", "soft_suitcase"):
                self.assertNotEqual(sub, "carry_on", f"{cat} に carry_on が生成された")
            if cat == "backpack":
                self.assertNotEqual(sub, "daypack", f"backpack に daypack が生成された")



if __name__ == "__main__":
    unittest.main(verbosity=2)
