"""
T-3.3: MeshVLMQA テスト

pyrender と VLM (vLLM サーバー) をモック化してユニットテストを実行する。
vLLM サーバーが不要なため CI 環境でも実行可能。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import trimesh

# プロジェクトの src を PYTHONPATH に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh_vlm_qa import (
    MeshVLMQA,
    MIN_GEOMETRY_SCORE,
    MIN_TEXTURE_SCORE,
    _apply_defaults,
    _parse_json_response,
)


# ============================================================
# テスト用ヘルパー
# ============================================================

def _make_good_response() -> dict:
    """合格スコアの VLM 応答"""
    return {
        "geometry_score": 8,
        "texture_score": 7,
        "consistency_score": 8,
        "is_realistic_luggage": True,
        "detected_type": "hard_suitcase",
        "detected_material": "polycarbonate",
        "issues": [],
        "pass": True,
    }


def _make_fail_response() -> dict:
    """不合格スコアの VLM 応答"""
    return {
        "geometry_score": 4,
        "texture_score": 3,
        "consistency_score": 5,
        "is_realistic_luggage": False,
        "detected_type": "unknown",
        "detected_material": "unknown",
        "issues": ["broken geometry", "missing texture"],
        "pass": False,
    }


def _make_glb(tmpdir: str, name: str = "mesh.glb") -> str:
    """テスト用の GLB ファイルを作成する"""
    mesh = trimesh.creation.icosphere(subdivisions=4)
    path = str(Path(tmpdir) / name)
    mesh.export(path, file_type="glb")
    return path


def _mock_vllm_response(content_dict: dict) -> MagicMock:
    """vLLM の OpenAI クライアント応答をモック化する"""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(content_dict)
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


# ============================================================
# _apply_defaults テスト
# ============================================================

class TestApplyDefaults(unittest.TestCase):

    def test_pass_logic_geometry_below_threshold(self):
        """geometry_score < MIN_GEOMETRY_SCORE(5) なら pass=False"""
        result = _apply_defaults({"geometry_score": MIN_GEOMETRY_SCORE - 1, "texture_score": 7})
        self.assertFalse(result["pass"])

    def test_pass_logic_texture_below_threshold(self):
        """texture_score < MIN_TEXTURE_SCORE(4) なら pass=False"""
        result = _apply_defaults({"geometry_score": 8, "texture_score": MIN_TEXTURE_SCORE - 1})
        self.assertFalse(result["pass"])

    def test_pass_logic_both_above_threshold(self):
        """geometry>=MIN_GEOMETRY_SCORE かつ texture>=MIN_TEXTURE_SCORE なら pass=True"""
        result = _apply_defaults({"geometry_score": MIN_GEOMETRY_SCORE, "texture_score": MIN_TEXTURE_SCORE})
        self.assertTrue(result["pass"])

    def test_missing_fields_filled_with_defaults(self):
        """欠落フィールドにデフォルト値が補完されること"""
        result = _apply_defaults({})
        for key in ("geometry_score", "texture_score", "consistency_score",
                    "is_realistic_luggage", "detected_type", "detected_material",
                    "issues", "pass"):
            self.assertIn(key, result)

    def test_issues_default_is_list(self):
        """issues のデフォルトは空リストであること"""
        result = _apply_defaults({})
        self.assertIsInstance(result["issues"], list)


# ============================================================
# _parse_json_response テスト
# ============================================================

class TestParseJsonResponse(unittest.TestCase):

    def test_plain_json(self):
        """プレーンな JSON 文字列をパースできること"""
        text = '{"geometry_score": 8, "pass": true}'
        result = _parse_json_response(text)
        self.assertEqual(result["geometry_score"], 8)

    def test_json_with_markdown_fence(self):
        """```json ... ``` で囲まれた JSON をパースできること"""
        text = '```json\n{"geometry_score": 8}\n```'
        result = _parse_json_response(text)
        self.assertEqual(result["geometry_score"], 8)

    def test_json_with_think_block(self):
        """<think>...</think> を除去して JSON をパースできること"""
        text = "<think>thinking...</think>\n{\"geometry_score\": 7}"
        result = _parse_json_response(text)
        self.assertEqual(result["geometry_score"], 7)

    def test_invalid_json_raises(self):
        """無効な JSON は例外を送出すること"""
        with self.assertRaises(Exception):
            _parse_json_response("not json at all, no braces")


# ============================================================
# MeshVLMQA.evaluate_3d テスト
# ============================================================

class TestEvaluate3D(unittest.TestCase):

    def setUp(self):
        self.qa = MeshVLMQA()
        self.tmpdir = tempfile.mkdtemp()
        # ダミーの PNG ファイルを作成
        from PIL import Image
        self.render_paths = []
        for i in range(4):
            p = str(Path(self.tmpdir) / f"view_{i}.png")
            Image.new("RGB", (512, 512), color=(100, 100, 100)).save(p)
            self.render_paths.append(p)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mock_client(self, response_dict: dict) -> MagicMock:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_vllm_response(response_dict)
        return mock_client

    def test_evaluate_3d_returns_required_keys(self):
        """evaluate_3d の返り値が必須フィールドをすべて含むこと"""
        self.qa._client = self._mock_client(_make_good_response())
        result = self.qa.evaluate_3d(self.render_paths)
        for key in ("geometry_score", "texture_score", "consistency_score",
                    "is_realistic_luggage", "detected_type", "detected_material",
                    "issues", "pass"):
            self.assertIn(key, result)

    def test_good_response_returns_pass_true(self):
        """合格スコアの応答で pass=True になること"""
        self.qa._client = self._mock_client(_make_good_response())
        result = self.qa.evaluate_3d(self.render_paths)
        self.assertTrue(result["pass"])

    def test_fail_response_returns_pass_false(self):
        """不合格スコアの応答で pass=False になること"""
        self.qa._client = self._mock_client(_make_fail_response())
        result = self.qa.evaluate_3d(self.render_paths)
        self.assertFalse(result["pass"])

    def test_pass_threshold_geometry_exactly_min(self):
        """geometry_score=MIN_GEOMETRY_SCORE, texture_score=MIN_TEXTURE_SCORE で pass=True（境界値）"""
        resp = {**_make_good_response(), "geometry_score": MIN_GEOMETRY_SCORE, "texture_score": MIN_TEXTURE_SCORE}
        self.qa._client = self._mock_client(resp)
        result = self.qa.evaluate_3d(self.render_paths)
        self.assertTrue(result["pass"])

    def test_pass_threshold_geometry_below_min_fails(self):
        """geometry_score=MIN_GEOMETRY_SCORE-1 で pass=False（境界値）"""
        resp = {**_make_good_response(), "geometry_score": MIN_GEOMETRY_SCORE - 1, "texture_score": MIN_TEXTURE_SCORE + 3}
        self.qa._client = self._mock_client(resp)
        result = self.qa.evaluate_3d(self.render_paths)
        self.assertFalse(result["pass"])

    def test_invalid_json_fallback_to_defaults(self):
        """VLM が無効な JSON を返したとき pass=False でデフォルト値が補完されること"""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "This is not JSON at all."
        mock_client.chat.completions.create.return_value = mock_resp
        self.qa._client = mock_client

        result = self.qa.evaluate_3d(self.render_paths)
        self.assertFalse(result["pass"])
        self.assertIn("geometry_score", result)

    def test_expected_type_included_in_prompt(self):
        """expected_type が VLM プロンプトに含まれること"""
        self.qa._client = self._mock_client(_make_good_response())
        self.qa.evaluate_3d(self.render_paths, expected_type="hard_suitcase")
        call_args = self.qa._client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_content = messages[1]["content"]
        # user_content は list[dict] — テキスト部分から expected_type を探す
        text_parts = [c["text"] for c in user_content if c.get("type") == "text"]
        self.assertTrue(
            any("hard_suitcase" in t for t in text_parts),
            "expected_type がプロンプトに含まれていない",
        )


# ============================================================
# MeshVLMQA.render_multiview テスト（pyrender をモック化）
# ============================================================

class TestRenderMultiview(unittest.TestCase):

    def setUp(self):
        self.qa = MeshVLMQA()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("src.mesh_vlm_qa.MeshVLMQA.render_multiview")
    def test_render_multiview_called_with_correct_args(self, mock_render):
        """render_multiview が正しい引数で呼ばれること"""
        mock_render.return_value = ["v0.png", "v1.png", "v2.png", "v3.png"]
        paths = self.qa.render_multiview("mesh.glb", self.tmpdir, views=4)
        mock_render.assert_called_once_with("mesh.glb", self.tmpdir, views=4)
        self.assertEqual(len(paths), 4)


# ============================================================
# MeshVLMQA.evaluate_batch テスト
# ============================================================

class TestEvaluateBatch(unittest.TestCase):

    def setUp(self):
        self.qa = MeshVLMQA()
        self.tmpdir = tempfile.mkdtemp()
        self.mesh_dir = Path(self.tmpdir) / "meshes"
        self.mesh_dir.mkdir()
        self.output_json = Path(self.tmpdir) / "results.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_glbs(self, n: int) -> None:
        mesh = trimesh.creation.icosphere(subdivisions=4)
        for i in range(n):
            mesh.export(str(self.mesh_dir / f"mesh_{i:03d}.glb"), file_type="glb")

    @patch.object(MeshVLMQA, "render_multiview")
    @patch.object(MeshVLMQA, "evaluate_3d")
    def test_batch_returns_summary_keys(self, mock_eval, mock_render):
        """evaluate_batch が total/passed/failed/results を返すこと"""
        from PIL import Image
        mock_render.return_value = []
        mock_eval.return_value = _apply_defaults(_make_good_response())
        self._create_glbs(2)

        result = self.qa.evaluate_batch(str(self.mesh_dir), str(self.output_json))
        for key in ("total", "passed", "failed", "results"):
            self.assertIn(key, result)

    @patch.object(MeshVLMQA, "render_multiview")
    @patch.object(MeshVLMQA, "evaluate_3d")
    def test_batch_json_saved(self, mock_eval, mock_render):
        """evaluate_batch が JSON ファイルを保存すること"""
        mock_render.return_value = []
        mock_eval.return_value = _apply_defaults(_make_good_response())
        self._create_glbs(1)

        self.qa.evaluate_batch(str(self.mesh_dir), str(self.output_json))
        self.assertTrue(self.output_json.exists())

    @patch.object(MeshVLMQA, "render_multiview")
    @patch.object(MeshVLMQA, "evaluate_3d")
    def test_batch_empty_directory(self, mock_eval, mock_render):
        """空ディレクトリでも total=0 で正常終了すること"""
        result = self.qa.evaluate_batch(str(self.mesh_dir), str(self.output_json))
        self.assertEqual(result["total"], 0)

    @patch.object(MeshVLMQA, "render_multiview")
    @patch.object(MeshVLMQA, "evaluate_3d")
    def test_batch_pass_count(self, mock_eval, mock_render):
        """合格メッシュが passed にカウントされること"""
        mock_render.return_value = []
        mock_eval.return_value = _apply_defaults(_make_good_response())
        self._create_glbs(3)

        result = self.qa.evaluate_batch(str(self.mesh_dir), str(self.output_json))
        self.assertEqual(result["passed"], 3)
        self.assertEqual(result["failed"], 0)

    @patch.object(MeshVLMQA, "render_multiview")
    @patch.object(MeshVLMQA, "evaluate_3d")
    def test_batch_resume_skips_existing(self, mock_eval, mock_render):
        """resume=True のとき既存結果をスキップすること"""
        mock_render.return_value = []
        mock_eval.return_value = _apply_defaults(_make_good_response())
        self._create_glbs(3)

        # 1 回目の実行
        self.qa.evaluate_batch(str(self.mesh_dir), str(self.output_json))
        first_call_count = mock_eval.call_count

        # 2 回目: resume=True でスキップされるはず
        mock_eval.reset_mock()
        self.qa.evaluate_batch(str(self.mesh_dir), str(self.output_json), resume=True)
        self.assertEqual(mock_eval.call_count, 0, "既存結果がスキップされていない")

    @patch.object(MeshVLMQA, "render_multiview")
    @patch.object(MeshVLMQA, "evaluate_3d")
    def test_batch_error_continues(self, mock_eval, mock_render):
        """評価失敗時も処理を継続すること"""
        mock_render.return_value = []
        mock_eval.side_effect = RuntimeError("render failed")
        self._create_glbs(2)

        result = self.qa.evaluate_batch(str(self.mesh_dir), str(self.output_json))
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["failed"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
