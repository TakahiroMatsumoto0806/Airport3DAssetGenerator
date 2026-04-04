"""
T-6.2: CLI スクリプト テスト

run_pipeline.py / run_step.py / generate_report.py の
引数パース・ロジックを unittest で検証。
実際のモデルロードは行わない。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# run_pipeline.py テスト
# ============================================================

class TestRunPipelineCLI(unittest.TestCase):

    def _run(self, argv: list[str]) -> int:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_pipeline",
            Path(__file__).parent.parent / "scripts" / "run_pipeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        with patch.object(sys, "argv", ["run_pipeline.py"] + argv):
            spec.loader.exec_module(mod)
        return mod

    def test_parse_args_defaults(self):
        from scripts.run_pipeline import parse_args
        with patch.object(sys, "argv", ["run_pipeline.py"]):
            args = parse_args()
        self.assertEqual(args.config, "configs/pipeline_config.yaml")
        self.assertIsNone(args.steps)
        self.assertFalse(args.no_resume)

    def test_parse_args_steps(self):
        from scripts.run_pipeline import parse_args
        with patch.object(sys, "argv",
                          ["run_pipeline.py", "--steps", "prompt", "diversity"]):
            args = parse_args()
        self.assertEqual(args.steps, ["prompt", "diversity"])

    def test_parse_args_no_resume(self):
        from scripts.run_pipeline import parse_args
        with patch.object(sys, "argv", ["run_pipeline.py", "--no-resume"]):
            args = parse_args()
        self.assertTrue(args.no_resume)

    def test_missing_config_returns_1(self):
        from scripts.run_pipeline import main
        with patch.object(sys, "argv",
                          ["run_pipeline.py", "--config", "/nonexistent/config.yaml"]):
            ret = main()
        self.assertEqual(ret, 1)

    @patch("scripts.run_pipeline.AL3DGPipeline")
    def test_main_calls_pipeline_run(self, MockPipeline):
        """main() が pipeline.run() を呼び出すこと"""
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"prompt": {"count": 2}}
        MockPipeline.return_value = mock_instance

        cfg_path = Path(__file__).parent.parent / "configs" / "pipeline_config.yaml"
        from scripts.run_pipeline import main
        with patch.object(sys, "argv",
                          ["run_pipeline.py", "--config", str(cfg_path),
                           "--steps", "prompt"]):
            ret = main()

        self.assertEqual(ret, 0)
        mock_instance.run.assert_called_once()


# ============================================================
# run_step.py テスト
# ============================================================

class TestRunStepCLI(unittest.TestCase):

    def test_parse_args_step_required(self):
        from scripts.run_step import parse_args
        with self.assertRaises(SystemExit):
            with patch.object(sys, "argv", ["run_step.py"]):
                parse_args()

    def test_parse_args_step_valid(self):
        from scripts.run_step import parse_args
        with patch.object(sys, "argv", ["run_step.py", "--step", "prompt"]):
            args = parse_args()
        self.assertEqual(args.step, "prompt")

    def test_parse_args_invalid_step(self):
        from scripts.run_step import parse_args
        with self.assertRaises(SystemExit):
            with patch.object(sys, "argv", ["run_step.py", "--step", "invalid"]):
                parse_args()

    def test_missing_config_returns_1(self):
        from scripts.run_step import main
        with patch.object(sys, "argv",
                          ["run_step.py", "--step", "prompt",
                           "--config", "/nonexistent.yaml"]):
            ret = main()
        self.assertEqual(ret, 1)

    @patch("scripts.run_step.AL3DGPipeline")
    def test_main_calls_correct_step(self, MockPipeline):
        """--step prompt で run_prompt_generation が呼ばれること"""
        mock_instance = MagicMock()
        mock_instance.run_prompt_generation.return_value = {"count": 3}
        MockPipeline.return_value = mock_instance

        cfg_path = Path(__file__).parent.parent / "configs" / "pipeline_config.yaml"
        from scripts.run_step import main
        with patch.object(sys, "argv",
                          ["run_step.py", "--step", "prompt",
                           "--config", str(cfg_path)]):
            ret = main()

        self.assertEqual(ret, 0)
        mock_instance.run_prompt_generation.assert_called_once()

    @patch("scripts.run_step.AL3DGPipeline")
    def test_sim_export_format_override(self, MockPipeline):
        """--format mjcf が設定に反映されること"""
        mock_instance = MagicMock()
        mock_instance.run_sim_export.return_value = {"success": 1}
        MockPipeline.return_value = mock_instance

        cfg_path = Path(__file__).parent.parent / "configs" / "pipeline_config.yaml"
        from scripts.run_step import main
        with patch.object(sys, "argv",
                          ["run_step.py", "--step", "sim_export",
                           "--config", str(cfg_path), "--format", "mjcf"]):
            ret = main()

        self.assertEqual(ret, 0)
        # format が "mjcf" にオーバーライドされていること
        call_cfg = MockPipeline.call_args[0][0]
        self.assertEqual(call_cfg.sim_export.format, "mjcf")


# ============================================================
# generate_report.py テスト
# ============================================================

class TestGenerateReportCLI(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # assets_final 作成
        assets_dir = Path(self.tmpdir) / "assets_final"
        assets_dir.mkdir(parents=True)
        # ダミー physics.json
        for i in range(3):
            asset_dir = assets_dir / f"asset_{i:03d}"
            asset_dir.mkdir()
            phys = {
                "asset_id": f"asset_{i:03d}",
                "luggage_type": "hard_suitcase",
                "mass_kg": 0.5,
                "scale": {"scaled_extents_mm": [50.0, 35.0, 70.0]},
            }
            with open(asset_dir / "physics.json", "w") as f:
                json.dump(phys, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parse_args_defaults(self):
        from scripts.generate_report import parse_args
        with patch.object(sys, "argv", ["generate_report.py"]):
            args = parse_args()
        self.assertFalse(args.no_clip)
        self.assertEqual(args.threshold, 0.95)

    def test_collect_metadata(self):
        """collect_metadata が physics.json を正しく収集すること"""
        from scripts.generate_report import collect_metadata
        meta, mesh_info = collect_metadata(Path(self.tmpdir) / "assets_final")
        self.assertEqual(len(meta), 3)
        self.assertEqual(len(mesh_info), 3)
        self.assertEqual(meta[0]["luggage_type"], "hard_suitcase")

    @patch("scripts.generate_report.DiversityEvaluator")
    def test_main_no_clip(self, MockEval):
        """--no-clip でレポートが生成されること"""
        mock_ev = MagicMock()
        mock_ev.generate_report.return_value = str(
            Path(self.tmpdir) / "reports" / "diversity_report.html"
        )
        mock_ev.check_size_realism.return_value = {
            "realistic": 3, "unrealistic": 0, "unknown": 0, "total": 3,
        }
        MockEval.return_value = mock_ev

        cfg_path = Path(__file__).parent.parent / "configs" / "pipeline_config.yaml"
        from scripts.generate_report import main
        with patch.object(sys, "argv", [
            "generate_report.py",
            "--no-clip",
            "--assets-dir", str(Path(self.tmpdir) / "assets_final"),
            "--output-dir", str(Path(self.tmpdir) / "reports"),
            "--config", str(cfg_path),
        ]):
            ret = main()

        self.assertEqual(ret, 0)
        mock_ev.generate_report.assert_called_once()
        # CLIP 関連メソッドは呼ばれないこと
        mock_ev.load_model.assert_not_called()
        mock_ev.compute_clip_embeddings.assert_not_called()

    def test_missing_assets_dir_returns_1(self):
        from scripts.generate_report import main
        with patch.object(sys, "argv", [
            "generate_report.py",
            "--no-clip",
            "--assets-dir", "/nonexistent/assets",
        ]):
            ret = main()
        self.assertEqual(ret, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
