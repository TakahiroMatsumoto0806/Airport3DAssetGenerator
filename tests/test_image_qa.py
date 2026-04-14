"""
T-2.2: 画像検品エンジン テスト

検証項目（仕様書記載）:
  - ダミー画像（白画像 + 実画像）で評価 JSON が正しいスキーマで返ること

追加検証:
  - 合格基準ルール（realism>=7, integrity>=7, !has_artifacts）の適用
  - verdict の正確な判定（pass / review / reject）
  - VLM 応答の JSON パース（正常系・壊れた JSON・thinking タグ付き）
  - evaluate_batch() のスキーマ・approved コピー・中断再開
  - get_statistics() のスキーマと集計
  - VLM 呼び出し失敗時のフォールバック

実行方法:
    pytest tests/test_image_qa.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# プロジェクトルートを sys.path に追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# テスト用ユーティリティ
# ============================================================

def _make_dummy_image(path: Path, color: tuple = (255, 255, 255), size=(64, 64)) -> Path:
    """テスト用ダミー画像を生成して保存する"""
    from PIL import Image
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


def _make_vllm_response(content: str) -> MagicMock:
    """vLLM OpenAI 互換レスポンスのモックを返す"""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def _good_qa_json() -> str:
    """合格スコアの正常 JSON 応答"""
    return json.dumps({
        "realism_score": 8,
        "object_integrity": 8,
        "background_clean": True,
        "luggage_type": "hard_suitcase",
        "has_artifacts": False,
        "handle_retracted": True,
        "is_bag_closed": True,
        "is_checked_baggage_appropriate": True,
        "checked_in_ready": True,
        "material_estimate": "polycarbonate",
        "pass": True,
        "verdict": "pass",
        "reason": "",
    })


def _borderline_qa_json() -> str:
    """ボーダーライン（review）の JSON 応答"""
    return json.dumps({
        "realism_score": 6,
        "object_integrity": 6,
        "background_clean": True,
        "luggage_type": "backpack",
        "has_artifacts": False,
        "handle_retracted": None,
        "is_bag_closed": True,
        "is_checked_baggage_appropriate": True,
        "checked_in_ready": True,
        "material_estimate": "nylon",
        "pass": False,
        "verdict": "review",
        "reason": "slightly low realism",
    })


def _reject_qa_json() -> str:
    """不合格の JSON 応答"""
    return json.dumps({
        "realism_score": 3,
        "object_integrity": 4,
        "background_clean": False,
        "luggage_type": "unknown",
        "has_artifacts": True,
        "handle_retracted": False,
        "is_bag_closed": False,
        "is_checked_baggage_appropriate": False,
        "checked_in_ready": False,
        "material_estimate": "unknown",
        "pass": False,
        "verdict": "reject",
        "reason": "heavy artifacts and low realism",
    })


def _build_image_qa(mock_response_content: str = None) -> "ImageQA":
    """モック化した ImageQA インスタンスを返す"""
    mock_model = MagicMock()
    mock_model.id = "Qwen/Qwen3-VL-32B-Instruct"
    mock_models_resp = MagicMock()
    mock_models_resp.data = [mock_model]

    mock_client = MagicMock()
    mock_client.models.list.return_value = mock_models_resp

    if mock_response_content is not None:
        mock_client.chat.completions.create.return_value = _make_vllm_response(
            mock_response_content
        )

    from src.image_qa import (
        ImageQA,
        MIN_REALISM, MIN_INTEGRITY,
        _SYSTEM_PROMPT, _USER_PROMPT_TEMPLATE,
    )
    qa = ImageQA.__new__(ImageQA)
    qa._client = mock_client
    qa._model_name = "Qwen/Qwen3-VL-32B-Instruct"
    qa._max_tokens = 512
    qa._temperature = 0.0
    qa._system_prompt = _SYSTEM_PROMPT
    qa._user_prompt_template = _USER_PROMPT_TEMPLATE
    qa._min_realism = MIN_REALISM
    qa._min_integrity = MIN_INTEGRITY
    qa._min_coverage_pct = 50
    qa._require_fully_visible = True
    qa._require_contrast_sufficient = True
    qa._require_no_background_shadow = True
    qa._require_sharp_focus = True
    qa._require_camera_angle_ok = True
    qa._require_no_artifacts = True
    qa._require_handle_retracted = True
    qa._require_bag_closed = True
    qa._require_checked_baggage_appropriate = True
    qa._require_checked_in_ready = True
    return qa


# ============================================================
# _parse_json_response テスト
# ============================================================

class TestParseJsonResponse(unittest.TestCase):
    """内部 JSON パーサーのテスト"""

    def setUp(self):
        from src.image_qa import _parse_json_response
        self._parse = _parse_json_response

    def test_plain_json(self):
        """純粋な JSON テキストをパースできること"""
        result = self._parse('{"realism_score": 8, "pass": true}')
        self.assertEqual(result["realism_score"], 8)
        self.assertTrue(result["pass"])

    def test_json_with_code_block(self):
        """コードブロックで囲まれた JSON をパースできること"""
        result = self._parse('```json\n{"realism_score": 7}\n```')
        self.assertEqual(result["realism_score"], 7)

    def test_json_with_thinking_tag(self):
        """<think>タグを除去して JSON をパースできること"""
        text = "<think>Let me evaluate...</think>\n{\"realism_score\": 9}"
        result = self._parse(text)
        self.assertEqual(result["realism_score"], 9)

    def test_json_with_surrounding_text(self):
        """前後にテキストがある場合も JSON を抽出できること"""
        text = 'Here is my evaluation: {"realism_score": 8} End.'
        result = self._parse(text)
        self.assertEqual(result["realism_score"], 8)

    def test_invalid_json_raises(self):
        """JSON が全くない場合は ValueError を送出すること"""
        with self.assertRaises(ValueError):
            self._parse("No JSON here at all")


# ============================================================
# _validate_and_normalize テスト
# ============================================================

class TestValidateAndNormalize(unittest.TestCase):
    """合格基準ルールの適用テスト"""

    def setUp(self):
        from src.image_qa import _validate_and_normalize
        self._validate = _validate_and_normalize

    def test_pass_when_all_criteria_met(self):
        """realism>=7, integrity>=7, !artifacts, handle_retracted, bag_closed, checked_baggage_ok → pass"""
        raw = {
            "realism_score": 8, "object_integrity": 8,
            "background_clean": True, "luggage_type": "hard_suitcase",
            "has_artifacts": False, "handle_retracted": True,
            "is_bag_closed": True, "is_checked_baggage_appropriate": True,
            "material_estimate": "polycarbonate", "pass": True,
            "verdict": "pass", "reason": "",
        }
        result = self._validate(raw)
        self.assertTrue(result["pass"])
        self.assertEqual(result["verdict"], "pass")

    def test_reject_when_realism_low(self):
        """realism < 7 → 不合格"""
        raw = {
            "realism_score": 5, "object_integrity": 8,
            "background_clean": True, "luggage_type": "backpack",
            "has_artifacts": False, "handle_retracted": None,
            "material_estimate": "nylon", "pass": True,  # VLM は pass と言っているが
            "verdict": "pass", "reason": "",
        }
        result = self._validate(raw)
        self.assertFalse(result["pass"])  # ルールで上書きされる
        self.assertNotEqual(result["verdict"], "pass")

    def test_reject_when_integrity_low(self):
        """integrity < 7 → 不合格"""
        raw = {
            "realism_score": 8, "object_integrity": 5,
            "background_clean": True, "luggage_type": "duffel_bag",
            "has_artifacts": False, "handle_retracted": None,
            "material_estimate": "nylon", "pass": True,
            "verdict": "pass", "reason": "",
        }
        result = self._validate(raw)
        self.assertFalse(result["pass"])

    def test_reject_when_has_artifacts(self):
        """has_artifacts == True → 不合格"""
        raw = {
            "realism_score": 8, "object_integrity": 8,
            "background_clean": True, "luggage_type": "hard_suitcase",
            "has_artifacts": True, "handle_retracted": True,
            "material_estimate": "polycarbonate", "pass": True,
            "verdict": "pass", "reason": "",
        }
        result = self._validate(raw)
        self.assertFalse(result["pass"])
        self.assertEqual(result["verdict"], "reject")

    def test_review_verdict_for_borderline(self):
        """スコア 6 はボーダーライン → review"""
        raw = {
            "realism_score": 6, "object_integrity": 6,
            "background_clean": True, "luggage_type": "backpack",
            "has_artifacts": False, "handle_retracted": None,
            "material_estimate": "nylon", "pass": False,
            "verdict": "review", "reason": "slightly low",
        }
        result = self._validate(raw)
        self.assertFalse(result["pass"])
        self.assertEqual(result["verdict"], "review")

    def test_score_clamped_to_1_10(self):
        """スコアは 1–10 にクランプされること"""
        raw = {
            "realism_score": 15, "object_integrity": -3,
            "background_clean": True, "luggage_type": "hard_suitcase",
            "has_artifacts": False, "handle_retracted": None,
            "material_estimate": "canvas", "pass": False,
            "verdict": "reject", "reason": "",
        }
        result = self._validate(raw)
        self.assertEqual(result["realism_score"], 10)
        self.assertEqual(result["object_integrity"], 1)

    def test_handle_retracted_none_stays_none(self):
        """handle_retracted が null → None のまま保持（ハンドルなしアイテムは合格扱い）"""
        raw = {
            "realism_score": 7, "object_integrity": 7,
            "background_clean": True, "luggage_type": "backpack",
            "has_artifacts": False, "handle_retracted": None,
            "is_bag_closed": True, "is_checked_baggage_appropriate": True,
            "material_estimate": "nylon", "pass": True,
            "verdict": "pass", "reason": "",
        }
        result = self._validate(raw)
        self.assertIsNone(result["handle_retracted"])
        # ハンドルなし（null）は合格条件を満たす
        self.assertTrue(result["pass"])


# ============================================================
# evaluate_single テスト
# ============================================================

class TestImageQAEvaluateSingle(unittest.TestCase):
    """evaluate_single() テスト — 仕様書記載の必須テスト含む"""

    def _get_qa(self, response_content: str):
        return _build_image_qa(response_content)

    # ---- 仕様書記載の必須テスト ----

    def test_dummy_white_image_returns_correct_schema(self):
        """
        仕様書記載:
          ダミー画像（白画像）で評価 JSON が正しいスキーマで返ること
        """
        qa = self._get_qa(_good_qa_json())
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "white.png"
            _make_dummy_image(img_path, color=(255, 255, 255))

            result = qa.evaluate_single(str(img_path))

        self._assert_schema(result)

    def test_dummy_gray_image_returns_correct_schema(self):
        """
        仕様書記載:
          ダミー画像（実画像相当のグレー画像）で評価 JSON が正しいスキーマで返ること
        """
        qa = self._get_qa(_good_qa_json())
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "gray.png"
            _make_dummy_image(img_path, color=(150, 150, 150), size=(128, 128))

            result = qa.evaluate_single(str(img_path))

        self._assert_schema(result)

    def _assert_schema(self, result: dict):
        """QA_RESULT_SCHEMA のキーが全て存在し型が正しいこと"""
        required_keys = {
            "realism_score", "object_integrity", "background_clean",
            "luggage_type", "has_artifacts", "handle_retracted",
            "material_estimate", "pass", "verdict", "reason",
        }
        missing = required_keys - set(result.keys())
        self.assertFalse(missing, f"不足キー: {missing}")

        self.assertIsInstance(result["realism_score"], int)
        self.assertIsInstance(result["object_integrity"], int)
        self.assertIsInstance(result["background_clean"], bool)
        self.assertIsInstance(result["luggage_type"], str)
        self.assertIsInstance(result["has_artifacts"], bool)
        self.assertIsInstance(result["handle_retracted"], bool)
        self.assertIsInstance(result["material_estimate"], str)
        self.assertIsInstance(result["pass"], bool)
        self.assertIn(result["verdict"], ("pass", "review", "reject"))
        self.assertIsInstance(result["reason"], str)

    # ---- 追加テスト ----

    def test_pass_result_for_good_image(self):
        """合格 JSON → result["pass"] == True"""
        qa = self._get_qa(_good_qa_json())
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "good.png"
            _make_dummy_image(img_path)
            result = qa.evaluate_single(str(img_path))
        self.assertTrue(result["pass"])
        self.assertEqual(result["verdict"], "pass")

    def test_reject_result_for_bad_image(self):
        """不合格 JSON → result["pass"] == False, verdict == "reject" """
        qa = self._get_qa(_reject_qa_json())
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "bad.png"
            _make_dummy_image(img_path)
            result = qa.evaluate_single(str(img_path))
        self.assertFalse(result["pass"])
        self.assertEqual(result["verdict"], "reject")

    def test_review_result_for_borderline_image(self):
        """ボーダーライン JSON → verdict == "review" """
        qa = self._get_qa(_borderline_qa_json())
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "border.png"
            _make_dummy_image(img_path)
            result = qa.evaluate_single(str(img_path))
        self.assertEqual(result["verdict"], "review")

    def test_file_not_found_raises(self):
        """存在しないファイルは FileNotFoundError"""
        qa = self._get_qa(_good_qa_json())
        with self.assertRaises(FileNotFoundError):
            qa.evaluate_single("/nonexistent/path/image.png")

    def test_broken_json_response_raises_runtime_error(self):
        """VLM が壊れた JSON を返した場合 RuntimeError"""
        qa = self._get_qa("This is not JSON at all")
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "img.png"
            _make_dummy_image(img_path)
            with self.assertRaises(RuntimeError):
                qa.evaluate_single(str(img_path))

    def test_thinking_tag_in_response_handled(self):
        """<think>タグ付き応答でも正しくパースできること"""
        response = f"<think>Let me think...</think>\n{_good_qa_json()}"
        qa = self._get_qa(response)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "img.png"
            _make_dummy_image(img_path)
            result = qa.evaluate_single(str(img_path))
        self.assertIn("pass", result)

    def test_vllm_called_with_think(self):
        """VLM へのリクエストに /think が含まれていること（B-2: thinking モード常時 ON）"""
        qa = self._get_qa(_good_qa_json())
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "img.png"
            _make_dummy_image(img_path)
            qa.evaluate_single(str(img_path))

        call_args = qa._client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        # user メッセージのテキスト部分に /think が含まれること（B-2: thinking モード常時 ON）
        user_content = messages[-1]["content"]
        text_parts = [c["text"] for c in user_content if c.get("type") == "text"]
        self.assertTrue(
            any("/think" in t for t in text_parts),
            "/think が VLM リクエストに含まれていない",
        )


# ============================================================
# evaluate_batch テスト
# ============================================================

class TestImageQAEvaluateBatch(unittest.TestCase):
    """evaluate_batch() テスト"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.image_dir = Path(self.tmpdir) / "images"
        self.image_dir.mkdir()
        self.approved_dir = Path(self.tmpdir) / "approved"
        self.output_json = Path(self.tmpdir) / "qa_results.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_images(self, n: int) -> list[Path]:
        paths = []
        for i in range(n):
            p = self.image_dir / f"{i:06d}_test.png"
            _make_dummy_image(p)
            paths.append(p)
        return paths

    def _qa_with_responses(self, responses: list[str]):
        """複数レスポンスを順番に返す QA インスタンス"""
        qa = _build_image_qa()
        side_effects = [_make_vllm_response(r) for r in responses]
        qa._client.chat.completions.create.side_effect = side_effects
        return qa

    def test_batch_result_schema(self):
        """evaluate_batch() が正しいサマリスキーマを返すこと"""
        self._create_images(3)
        qa = self._qa_with_responses([_good_qa_json()] * 3)

        summary = qa.evaluate_batch(
            str(self.image_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
        )

        required_keys = {"total", "evaluated", "passed", "reviewed", "rejected", "failed_eval", "pass_rate", "results"}
        missing = required_keys - set(summary.keys())
        self.assertFalse(missing, f"サマリに不足キー: {missing}")
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["evaluated"], 3)

    def test_pass_images_copied_to_approved(self):
        """合格画像が approved_dir にコピーされること"""
        paths = self._create_images(3)
        qa = self._qa_with_responses([_good_qa_json()] * 3)

        qa.evaluate_batch(
            str(self.image_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
        )

        for p in paths:
            dst = self.approved_dir / p.name
            self.assertTrue(dst.exists(), f"合格画像がコピーされていない: {dst}")

    def test_reject_images_not_copied(self):
        """不合格画像は approved_dir にコピーされないこと"""
        paths = self._create_images(2)
        qa = self._qa_with_responses([_reject_qa_json()] * 2)

        qa.evaluate_batch(
            str(self.image_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
        )

        for p in paths:
            dst = self.approved_dir / p.name
            self.assertFalse(dst.exists(), f"不合格画像がコピーされた: {dst}")

    def test_review_images_copied_to_approved(self):
        """review 画像も approved_dir にコピーされること"""
        paths = self._create_images(2)
        qa = self._qa_with_responses([_borderline_qa_json()] * 2)

        qa.evaluate_batch(
            str(self.image_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
        )

        for p in paths:
            dst = self.approved_dir / p.name
            self.assertTrue(dst.exists(), f"review 画像がコピーされていない: {dst}")

    def test_output_json_created(self):
        """結果 JSON ファイルが作成されること"""
        self._create_images(2)
        qa = self._qa_with_responses([_good_qa_json()] * 2)

        qa.evaluate_batch(
            str(self.image_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
        )

        self.assertTrue(self.output_json.exists())
        with open(self.output_json, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 2)

    def test_result_entry_schema(self):
        """各結果エントリに必要なフィールドがあること"""
        self._create_images(1)
        qa = self._qa_with_responses([_good_qa_json()])

        summary = qa.evaluate_batch(
            str(self.image_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
        )

        entry = summary["results"][0]
        self.assertIn("image_path", entry)
        self.assertIn("filename", entry)
        self.assertIn("verdict", entry)
        self.assertIn("pass", entry)
        self.assertIn("approved_path", entry)

    def test_resume_skips_existing(self):
        """2 回目実行時に既存評価済みエントリがスキップされること"""
        self._create_images(3)
        qa = self._qa_with_responses([_good_qa_json()] * 6)

        # 1 回目
        qa.evaluate_batch(
            str(self.image_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
        )
        first_call_count = qa._client.chat.completions.create.call_count

        # 2 回目 (resume=True)
        qa.evaluate_batch(
            str(self.image_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
            resume=True,
        )
        second_call_count = qa._client.chat.completions.create.call_count

        # 2 回目は VLM 呼び出しが増えていないこと（全スキップ）
        self.assertEqual(
            first_call_count,
            second_call_count,
            "再開時に既評価済み画像が再評価されている",
        )

    def test_eval_failure_continues_batch(self):
        """1 件の VLM 呼び出し失敗でバッチが継続されること"""
        self._create_images(3)
        responses = [
            _good_qa_json(),
            "INVALID_JSON",  # 2 件目失敗
            _good_qa_json(),
        ]
        qa = self._qa_with_responses(responses)

        summary = qa.evaluate_batch(
            str(self.image_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
        )

        self.assertEqual(summary["evaluated"], 3)
        self.assertGreaterEqual(summary["failed_eval"], 1)

    def test_pass_rate_calculation(self):
        """合格率が正しく計算されること"""
        self._create_images(4)
        responses = [
            _good_qa_json(),    # pass
            _good_qa_json(),    # pass
            _reject_qa_json(),  # reject
            _reject_qa_json(),  # reject
        ]
        qa = self._qa_with_responses(responses)

        summary = qa.evaluate_batch(
            str(self.image_dir),
            str(self.output_json),
            approved_dir=str(self.approved_dir),
        )

        self.assertEqual(summary["passed"], 2)
        self.assertAlmostEqual(summary["pass_rate"], 0.5, places=2)


# ============================================================
# get_statistics テスト
# ============================================================

class TestImageQAGetStatistics(unittest.TestCase):
    """get_statistics() テスト"""

    def setUp(self):
        self.qa = _build_image_qa()

    def test_statistics_schema(self):
        """統計情報が正しいスキーマを返すこと"""
        results = [
            {"verdict": "pass", "realism_score": 8, "object_integrity": 8,
             "luggage_type": "hard_suitcase", "material_estimate": "polycarbonate"},
            {"verdict": "reject", "realism_score": 3, "object_integrity": 3,
             "luggage_type": "unknown", "material_estimate": "unknown",
             "reason": "low quality"},
        ]
        stats = self.qa.get_statistics(results)

        required = {
            "total", "passed", "reviewed", "rejected", "pass_rate",
            "avg_realism_score", "avg_object_integrity",
            "luggage_type_distribution", "material_distribution",
            "rejection_reasons",
        }
        missing = required - set(stats.keys())
        self.assertFalse(missing, f"統計に不足キー: {missing}")

    def test_statistics_counts(self):
        """集計値が正しいこと"""
        results = [
            {"verdict": "pass", "realism_score": 8, "object_integrity": 8,
             "luggage_type": "hard_suitcase", "material_estimate": "polycarbonate"},
            {"verdict": "review", "realism_score": 6, "object_integrity": 6,
             "luggage_type": "backpack", "material_estimate": "nylon"},
            {"verdict": "reject", "realism_score": 3, "object_integrity": 3,
             "luggage_type": "unknown", "material_estimate": "unknown",
             "reason": "bad"},
        ]
        stats = self.qa.get_statistics(results)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(stats["reviewed"], 1)
        self.assertEqual(stats["rejected"], 1)
        self.assertAlmostEqual(stats["pass_rate"], 1 / 3, places=4)

    def test_avg_scores(self):
        """平均スコアが正しく計算されること"""
        results = [
            {"verdict": "pass", "realism_score": 8, "object_integrity": 9,
             "luggage_type": "hard_suitcase", "material_estimate": "polycarbonate"},
            {"verdict": "pass", "realism_score": 7, "object_integrity": 7,
             "luggage_type": "backpack", "material_estimate": "nylon"},
        ]
        stats = self.qa.get_statistics(results)
        self.assertAlmostEqual(stats["avg_realism_score"], 7.5)
        self.assertAlmostEqual(stats["avg_object_integrity"], 8.0)

    def test_empty_results(self):
        """空リストで統計を呼んでもエラーにならないこと"""
        stats = self.qa.get_statistics([])
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["pass_rate"], 0.0)


# ============================================================
# 受託手荷物条件（Checked Baggage）テスト
# ============================================================

class TestCheckedBaggageConditions(unittest.TestCase):
    """受託手荷物状態チェックの新規QAフィールドのテスト"""

    def setUp(self):
        from src.image_qa import _validate_and_normalize
        self._validate = _validate_and_normalize

    def _good_raw(self, **overrides) -> dict:
        """合格基準を満たす最小限の raw データ"""
        base = {
            "realism_score": 8, "object_integrity": 8,
            "background_clean": True, "luggage_type": "hard_suitcase",
            "has_artifacts": False, "handle_retracted": True,
            "is_bag_closed": True, "is_checked_baggage_appropriate": True,
            "checked_in_ready": True,
            "material_estimate": "polycarbonate",
            "is_fully_visible": True, "contrast_sufficient": True,
            "object_coverage_pct": 65, "has_background_shadow": False,
            "is_sharp_focus": True, "camera_angle_ok": True,
            "pass": True, "verdict": "pass", "reason": "",
        }
        base.update(overrides)
        return base

    def test_handle_extended_causes_reject(self):
        """キャリーハンドルが伸びた状態（False）→ 不合格"""
        result = self._validate(self._good_raw(handle_retracted=False))
        self.assertFalse(result["pass"])
        self.assertEqual(result["verdict"], "reject")

    def test_handle_retracted_passes(self):
        """キャリーハンドルが収納（True）→ 合格"""
        result = self._validate(self._good_raw(handle_retracted=True))
        self.assertTrue(result["pass"])

    def test_handle_none_passes(self):
        """ハンドルなしアイテム（null）→ 合格（バックパック等）"""
        result = self._validate(self._good_raw(handle_retracted=None))
        self.assertIsNone(result["handle_retracted"])
        self.assertTrue(result["pass"])

    def test_bag_open_causes_reject(self):
        """バッグが開いた状態（is_bag_closed=False）→ 不合格"""
        result = self._validate(self._good_raw(is_bag_closed=False))
        self.assertFalse(result["pass"])
        self.assertEqual(result["verdict"], "reject")

    def test_bag_closed_passes(self):
        """バッグが閉じた状態（is_bag_closed=True）→ 合格条件を満たす"""
        result = self._validate(self._good_raw(is_bag_closed=True))
        self.assertTrue(result["pass"])

    def test_small_handbag_causes_reject(self):
        """受託手荷物不適切（小さなハンドバッグ等）→ 即 reject"""
        result = self._validate(self._good_raw(is_checked_baggage_appropriate=False))
        self.assertFalse(result["pass"])
        self.assertEqual(result["verdict"], "reject")

    def test_checked_baggage_appropriate_passes(self):
        """受託手荷物として適切なサイズ・種別 → 合格条件を満たす"""
        result = self._validate(self._good_raw(is_checked_baggage_appropriate=True))
        self.assertTrue(result["pass"])

    def test_new_fields_present_in_result(self):
        """新規フィールドが結果 dict に含まれること（B-1: checked_in_ready を含む）"""
        result = self._validate(self._good_raw())
        self.assertIn("handle_retracted", result)
        self.assertIn("is_bag_closed", result)
        self.assertIn("is_checked_baggage_appropriate", result)
        self.assertIn("checked_in_ready", result)

    def test_missing_new_fields_default_to_safe(self):
        """新規フィールドが VLM 応答に含まれない場合はデフォルト（合格扱い）"""
        raw = {
            "realism_score": 8, "object_integrity": 8,
            "background_clean": True, "luggage_type": "hard_suitcase",
            "has_artifacts": False,
            # handle_retracted / is_bag_closed / is_checked_baggage_appropriate /
            # checked_in_ready を省略
            "material_estimate": "polycarbonate",
        }
        result = self._validate(raw)
        # デフォルト値（安全側）で合格基準を満たすこと
        self.assertTrue(result["is_bag_closed"])
        self.assertTrue(result["is_checked_baggage_appropriate"])
        self.assertTrue(result["checked_in_ready"])

    def test_all_three_checked_baggage_failures_reject(self):
        """3つの受託手荷物条件がすべて NG → reject"""
        result = self._validate(self._good_raw(
            handle_retracted=False,
            is_bag_closed=False,
            is_checked_baggage_appropriate=False,
        ))
        self.assertFalse(result["pass"])
        self.assertEqual(result["verdict"], "reject")

    def test_checked_in_ready_false_causes_reject(self):
        """チェックイン不可状態（checked_in_ready=False）→ 即 reject"""
        result = self._validate(self._good_raw(checked_in_ready=False))
        self.assertFalse(result["pass"])
        self.assertEqual(result["verdict"], "reject")

    def test_checked_in_ready_true_passes(self):
        """チェックイン可能状態（checked_in_ready=True）→ 合格条件を満たす"""
        result = self._validate(self._good_raw(checked_in_ready=True))
        self.assertTrue(result["pass"])
        self.assertEqual(result["verdict"], "pass")

    def test_checked_in_ready_in_result_dict(self):
        """checked_in_ready フィールドが結果 dict に含まれること"""
        result = self._validate(self._good_raw(checked_in_ready=False))
        self.assertIn("checked_in_ready", result)
        self.assertFalse(result["checked_in_ready"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
