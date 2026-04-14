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
                          ["run_pipeline.py", "--steps", "prompt", "sim_export"]):
            args = parse_args()
        self.assertEqual(args.steps, ["prompt", "sim_export"])

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



# ============================================================
# generate_report.py テスト
# ============================================================

class TestGenerateReportCLI(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parse_args_defaults(self):
        from scripts.generate_report import parse_args
        with patch.object(sys, "argv", ["generate_report.py"]):
            args = parse_args()
        self.assertEqual(args.step, "image_qa")
        self.assertEqual(args.config, "configs/pipeline_config.yaml")
        self.assertIsNone(args.output_dir)

    def test_parse_args_step_pass_rate(self):
        from scripts.generate_report import parse_args
        with patch.object(sys, "argv", ["generate_report.py", "--step", "pass_rate"]):
            args = parse_args()
        self.assertEqual(args.step, "pass_rate")

    def test_parse_args_invalid_step_exits(self):
        from scripts.generate_report import parse_args
        with self.assertRaises(SystemExit):
            with patch.object(sys, "argv", ["generate_report.py", "--step", "diversity"]):
                parse_args()

    def test_missing_qa_results_returns_1(self):
        """QA 結果ファイルが存在しない場合に main() が 1 を返すこと"""
        from scripts.generate_report import main
        with patch.object(sys, "argv", [
            "generate_report.py",
            "--step", "image_qa",
            "--qa-results", "/nonexistent/qa_results.json",
        ]):
            ret = main()
        self.assertEqual(ret, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
