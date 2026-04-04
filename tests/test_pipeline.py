"""
T-6.1: AL3DGPipeline テスト

全モデルはモック化してユニットテストを実行する。
設定読み込み・ステップ解決・サマリー保存のロジックを検証。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import AL3DGPipeline


def _make_cfg(tmpdir: str) -> object:
    """テスト用の最小 DictConfig を作成する"""
    raw = {
        "project": {"name": "al3dg_test", "output_dir": tmpdir, "seed": 42},
        "generation": {"target_count": 10, "image_batch_size": 4,
                       "prompt_count_per_category": 2},
        "steps": {
            "t1_prompt_generation": True,
            "t2_image_generation": True,
            "t2_image_qa": True,
            "t3_mesh_generation": True,
            "t3_mesh_qa": True,
            "t3_mesh_vlm_qa": True,
            "t4_physics": True,
            "t4_sim_export": True,
            "t5_diversity_report": True,
        },
        "paths": {
            "prompts_dir": f"{tmpdir}/prompts",
            "images_dir": f"{tmpdir}/images",
            "images_approved_dir": f"{tmpdir}/images_approved",
            "meshes_raw_dir": f"{tmpdir}/meshes_raw",
            "meshes_approved_dir": f"{tmpdir}/meshes_approved",
            "assets_final_dir": f"{tmpdir}/assets_final",
            "reports_dir": f"{tmpdir}/reports",
        },
        "models": {
            "flux": {"model_id": "test/flux", "model_dir": "/tmp/flux",
                     "device": "cpu", "dtype": "float32", "num_inference_steps": 1},
            "trellis": {"model_dir": "/tmp/trellis"},
            "vlm": {"base_url": "http://localhost:8000/v1",
                    "model_name": "test/vlm", "temperature": 0.1, "max_tokens": 256},
            "clip": {"model_name": "ViT-L-14",
                     "pretrained": "datacomp_xl_s13b_b90k", "device": "cpu"},
        },
        "prompt_generation": {
            "configs_dir": "configs",
            "output_file": f"{tmpdir}/prompts/prompts.json",
            "seed": 42,
        },
        "image_generation": {
            "output_dir": f"{tmpdir}/images",
            "width": 256, "height": 256,
            "num_inference_steps": 1, "guidance_scale": 0.0, "resume": True,
        },
        "image_qa": {
            "output_dir": f"{tmpdir}/images_approved",
            "thresholds": {"realism": 7, "integrity": 7},
            "thinking_mode": False, "batch_size": 4, "resume": True,
        },
        "mesh_generation": {
            "output_dir": f"{tmpdir}/meshes_raw", "seed": 42, "resume": True,
        },
        "mesh_qa": {
            "thresholds": {"min_faces": 5000, "max_faces": 100000,
                           "max_aspect_ratio": 20.0},
            "repair": True,
            "output_dir": f"{tmpdir}/meshes_approved",
            "resume": True,
        },
        "mesh_vlm_qa": {
            "output_dir": f"{tmpdir}/meshes_approved",
            "render_dir": f"{tmpdir}/renders",
            "thresholds": {"geometry": 7, "texture": 6},
            "thinking_mode": True,
            "azimuths": [0, 90, 180, 270],
            "render_size": [512, 512],
            "resume": True,
        },
        "physics": {
            "output_dir": f"{tmpdir}/assets_final",
            "coacd_threshold": 0.08, "max_convex_hulls": 16,
            "miniature": True, "resume": True,
        },
        "sim_export": {
            "output_dir": f"{tmpdir}/assets_final",
            "format": "both", "resume": True,
        },
        "diversity": {
            "output_dir": f"{tmpdir}/reports",
            "near_dup_threshold": 0.95, "embed_batch_size": 32,
            "size_realism_refs": {},
        },
    }
    return OmegaConf.create(raw)


# ============================================================
# 設定ロード・ステップ解決テスト
# ============================================================

class TestPipelineInit(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _make_cfg(self.tmpdir)

    def test_init_with_cfg(self):
        """DictConfig で初期化できること"""
        pipeline = AL3DGPipeline(self.cfg)
        self.assertIsNotNone(pipeline)

    def test_resolve_steps_all(self):
        """steps=None のとき全ステップが有効になること"""
        pipeline = AL3DGPipeline(self.cfg)
        steps = pipeline._resolve_steps(None)
        self.assertIn("prompt", steps)
        self.assertIn("image", steps)
        self.assertIn("diversity", steps)

    def test_resolve_steps_explicit(self):
        """明示的にリストを渡したときそのまま返ること"""
        pipeline = AL3DGPipeline(self.cfg)
        steps = pipeline._resolve_steps(["prompt", "diversity"])
        self.assertEqual(steps, ["prompt", "diversity"])

    def test_resolve_steps_disabled(self):
        """cfg.steps で False にしたステップは除外されること"""
        cfg = _make_cfg(self.tmpdir)
        OmegaConf.update(cfg, "steps.t1_prompt_generation", False)
        pipeline = AL3DGPipeline(cfg)
        steps = pipeline._resolve_steps(None)
        self.assertNotIn("prompt", steps)

    def test_save_run_summary_creates_json(self):
        """_save_run_summary が JSON ファイルを作成すること"""
        pipeline = AL3DGPipeline(self.cfg)
        pipeline._save_run_summary({"step1": {"count": 5}})
        summary_path = Path(self.tmpdir) / "reports" / "pipeline_run_summary.json"
        self.assertTrue(summary_path.exists())

    def test_save_run_summary_valid_json(self):
        """保存された JSON が有効であること"""
        pipeline = AL3DGPipeline(self.cfg)
        pipeline._save_run_summary({"step1": {"count": 5, "path": Path("/tmp/x")}})
        summary_path = Path(self.tmpdir) / "reports" / "pipeline_run_summary.json"
        with open(summary_path) as f:
            data = json.load(f)
        self.assertIn("step1", data)


# ============================================================
# run() インテグレーションテスト（全モジュールをモック）
# ============================================================

class TestPipelineRun(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = _make_cfg(self.tmpdir)
        # prompts.json を事前に作成
        Path(self.tmpdir, "prompts").mkdir(parents=True, exist_ok=True)
        prompts_path = Path(self.tmpdir) / "prompts" / "prompts.json"
        with open(prompts_path, "w") as f:
            json.dump([{"prompt": "test", "luggage_type": "backpack"}], f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("src.pipeline.AL3DGPipeline.run_prompt_generation", return_value={"count": 2})
    @patch("src.pipeline.AL3DGPipeline.run_image_generation", return_value={"success": 2, "failed": 0})
    @patch("src.pipeline.AL3DGPipeline.run_image_qa", return_value={"approved": 2, "rejected": 0})
    @patch("src.pipeline.AL3DGPipeline.run_mesh_generation", return_value={"success": 2, "failed": 0})
    @patch("src.pipeline.AL3DGPipeline.run_mesh_qa", return_value={"approved": 2, "rejected": 0})
    @patch("src.pipeline.AL3DGPipeline.run_mesh_vlm_qa", return_value={"approved": 2, "rejected": 0})
    @patch("src.pipeline.AL3DGPipeline.run_physics", return_value={"success": 2, "failed": 0})
    @patch("src.pipeline.AL3DGPipeline.run_sim_export", return_value={"success": 2, "failed": 0})
    @patch("src.pipeline.AL3DGPipeline.run_diversity_report", return_value={"html_path": "/tmp/r.html"})
    def test_run_calls_all_steps(self, mock_div, mock_exp, mock_phys, mock_vlm,
                                  mock_mqa, mock_mesh, mock_iqa, mock_img, mock_prm):
        """run() が全ステップを呼び出すこと"""
        pipeline = AL3DGPipeline(self.cfg)
        results = pipeline.run()
        mock_prm.assert_called_once()
        mock_img.assert_called_once()
        mock_iqa.assert_called_once()
        mock_mesh.assert_called_once()
        mock_mqa.assert_called_once()
        mock_vlm.assert_called_once()
        mock_phys.assert_called_once()
        mock_exp.assert_called_once()
        mock_div.assert_called_once()
        self.assertIn("prompt", results)
        self.assertIn("diversity", results)

    @patch("src.pipeline.AL3DGPipeline.run_prompt_generation", return_value={"count": 2})
    @patch("src.pipeline.AL3DGPipeline.run_diversity_report", return_value={"html_path": "/tmp/r.html"})
    def test_run_selective_steps(self, mock_div, mock_prm):
        """steps= で指定したステップだけ実行されること"""
        pipeline = AL3DGPipeline(self.cfg)
        results = pipeline.run(steps=["prompt", "diversity"])
        mock_prm.assert_called_once()
        mock_div.assert_called_once()
        self.assertNotIn("image", results)

    @patch("src.pipeline.AL3DGPipeline.run_prompt_generation", return_value={"count": 2})
    def test_run_returns_dict(self, mock_prm):
        """run() が dict を返すこと"""
        pipeline = AL3DGPipeline(self.cfg)
        results = pipeline.run(steps=["prompt"])
        self.assertIsInstance(results, dict)

    @patch("src.pipeline.AL3DGPipeline.run_prompt_generation", return_value={"count": 2})
    def test_run_saves_summary(self, mock_prm):
        """run() がサマリー JSON を保存すること"""
        pipeline = AL3DGPipeline(self.cfg)
        pipeline.run(steps=["prompt"])
        summary_path = Path(self.tmpdir) / "reports" / "pipeline_run_summary.json"
        self.assertTrue(summary_path.exists())


# ============================================================
# pipeline_config.yaml 読み込みテスト
# ============================================================

class TestPipelineConfigLoad(unittest.TestCase):

    def test_load_pipeline_config(self):
        """pipeline_config.yaml が読み込めること"""
        cfg_path = Path(__file__).parent.parent / "configs" / "pipeline_config.yaml"
        self.assertTrue(cfg_path.exists(), "pipeline_config.yaml が見つかりません")
        cfg = OmegaConf.load(str(cfg_path))
        self.assertIn("project", cfg)
        self.assertIn("steps", cfg)
        self.assertIn("models", cfg)
        self.assertIn("paths", cfg)

    def test_config_has_required_step_keys(self):
        """pipeline_config.yaml に必須ステップキーが含まれること"""
        cfg_path = Path(__file__).parent.parent / "configs" / "pipeline_config.yaml"
        cfg = OmegaConf.load(str(cfg_path))
        for key in ("t1_prompt_generation", "t2_image_generation", "t2_image_qa",
                    "t3_mesh_generation", "t3_mesh_qa", "t3_mesh_vlm_qa",
                    "t4_physics", "t4_sim_export", "t5_diversity_report"):
            self.assertIn(key, cfg.steps, f"steps.{key} が見つかりません")

    def test_config_has_model_sections(self):
        """pipeline_config.yaml にモデルセクションが含まれること"""
        cfg_path = Path(__file__).parent.parent / "configs" / "pipeline_config.yaml"
        cfg = OmegaConf.load(str(cfg_path))
        for model in ("flux", "trellis", "vlm", "clip"):
            self.assertIn(model, cfg.models, f"models.{model} が見つかりません")


if __name__ == "__main__":
    unittest.main(verbosity=2)
