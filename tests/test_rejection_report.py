"""
D-1: 画像 QA リジェクト分析レポート テスト

検証項目:
  - サンプル QA 結果 JSON から条件別カウントが正しく集計される
  - 出力 JSON のスキーマが正しい
  - カテゴリ別合格率が正しく計算される
  - Top 10 リジェクト理由が正しく集計される
  - HTML ファイルが生成される

実行方法:
    pytest tests/test_rejection_report.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_report import analyze_image_qa_rejections


def _make_qa_result(
    verdict: str = "pass",
    luggage_type: str = "hard_suitcase",
    realism_score: int = 8,
    object_integrity: int = 8,
    has_artifacts: bool = False,
    handle_retracted=True,
    is_bag_closed: bool = True,
    is_checked_baggage_appropriate: bool = True,
    checked_in_ready: bool = True,
    is_fully_visible: bool = True,
    contrast_sufficient: bool = True,
    object_coverage_pct: int = 65,
    has_background_shadow: bool = False,
    is_sharp_focus: bool = True,
    camera_angle_ok: bool = True,
    reason: str = "",
) -> dict:
    """テスト用の QA 結果エントリを生成する"""
    return {
        "verdict": verdict,
        "luggage_type": luggage_type,
        "realism_score": realism_score,
        "object_integrity": object_integrity,
        "has_artifacts": has_artifacts,
        "handle_retracted": handle_retracted,
        "is_bag_closed": is_bag_closed,
        "is_checked_baggage_appropriate": is_checked_baggage_appropriate,
        "checked_in_ready": checked_in_ready,
        "is_fully_visible": is_fully_visible,
        "contrast_sufficient": contrast_sufficient,
        "object_coverage_pct": object_coverage_pct,
        "has_background_shadow": has_background_shadow,
        "is_sharp_focus": is_sharp_focus,
        "camera_angle_ok": camera_angle_ok,
        "reason": reason,
    }


class TestRejectionReportSummary(unittest.TestCase):
    """全体サマリーの集計テスト"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir) / "reports"
        self.qa_results_path = Path(self.tmpdir) / "qa_results.json"

    def _write_results(self, results: list[dict]) -> None:
        with open(self.qa_results_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f)

    def test_summary_counts(self):
        """pass/review/reject カウントが正しく集計されること"""
        results = [
            _make_qa_result(verdict="pass"),
            _make_qa_result(verdict="pass"),
            _make_qa_result(verdict="review", reason="borderline quality"),
            _make_qa_result(verdict="reject", reason="heavy artifacts"),
            _make_qa_result(verdict="reject", reason="handle extended"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)

        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["pass"], 2)
        self.assertEqual(summary["review"], 1)
        self.assertEqual(summary["reject"], 2)

    def test_pass_rate_calculation(self):
        """合格率が正しく計算されること"""
        results = [
            _make_qa_result(verdict="pass"),
            _make_qa_result(verdict="pass"),
            _make_qa_result(verdict="reject", reason="artifact"),
            _make_qa_result(verdict="reject", reason="shadow"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)

        self.assertAlmostEqual(summary["pass_rate"], 0.5, places=4)

    def test_zero_results(self):
        """結果が0件の場合に pass_rate が 0 であること"""
        self._write_results([])
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["pass_rate"], 0.0)

    def test_all_pass(self):
        """全件合格の場合に pass_rate が 1.0 であること"""
        results = [_make_qa_result(verdict="pass") for _ in range(5)]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertEqual(summary["reject"], 0)


class TestRejectionReportConditionCounts(unittest.TestCase):
    """条件別失敗カウントのテスト"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir) / "reports"
        self.qa_results_path = Path(self.tmpdir) / "qa_results.json"

    def _write_results(self, results: list[dict]) -> None:
        with open(self.qa_results_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f)

    def test_realism_low_counted(self):
        """realism_score < 7 が realism_low にカウントされること"""
        results = [
            _make_qa_result(verdict="reject", realism_score=5, reason="low realism"),
            _make_qa_result(verdict="pass"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertEqual(summary["condition_counts"]["realism_low"], 1)

    def test_handle_extended_counted(self):
        """handle_retracted=False が handle_extended にカウントされること"""
        results = [
            _make_qa_result(verdict="reject", handle_retracted=False, reason="handle extended"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertEqual(summary["condition_counts"]["handle_extended"], 1)

    def test_handle_none_not_counted_as_extended(self):
        """handle_retracted=None (ハンドルなし) は handle_extended にカウントされないこと"""
        results = [
            _make_qa_result(verdict="reject", handle_retracted=None, reason="other reason"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertEqual(summary["condition_counts"]["handle_extended"], 0)

    def test_bag_open_counted(self):
        """is_bag_closed=False が bag_open にカウントされること"""
        results = [
            _make_qa_result(verdict="reject", is_bag_closed=False, reason="bag open"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertEqual(summary["condition_counts"]["bag_open"], 1)

    def test_not_checked_in_ready_counted(self):
        """checked_in_ready=False が not_checked_in_ready にカウントされること（B-1）"""
        results = [
            _make_qa_result(verdict="reject", checked_in_ready=False, reason="not ready"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertEqual(summary["condition_counts"]["not_checked_in_ready"], 1)

    def test_pass_records_not_counted(self):
        """pass した記録は条件別カウントに含まれないこと"""
        results = [
            _make_qa_result(verdict="pass", realism_score=3),  # pass だがスコア低い
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        # pass は non_pass リストに入らないのでカウント 0
        self.assertEqual(summary["condition_counts"]["realism_low"], 0)

    def test_shadow_counted(self):
        """has_background_shadow=True が shadow にカウントされること"""
        results = [
            _make_qa_result(verdict="reject", has_background_shadow=True, reason="shadow"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertEqual(summary["condition_counts"]["shadow"], 1)

    def test_multiple_conditions_on_same_record(self):
        """1件のリジェクトが複数条件に同時にカウントされること"""
        results = [
            _make_qa_result(
                verdict="reject",
                realism_score=4,
                has_background_shadow=True,
                is_bag_closed=False,
                reason="multiple issues",
            )
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        cond = summary["condition_counts"]
        self.assertEqual(cond["realism_low"], 1)
        self.assertEqual(cond["shadow"], 1)
        self.assertEqual(cond["bag_open"], 1)


class TestRejectionReportCategoryPassRates(unittest.TestCase):
    """カテゴリ別合格率のテスト"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir) / "reports"
        self.qa_results_path = Path(self.tmpdir) / "qa_results.json"

    def _write_results(self, results: list[dict]) -> None:
        with open(self.qa_results_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f)

    def test_category_pass_rate_correct(self):
        """カテゴリ別合格率が正しく計算されること"""
        results = [
            _make_qa_result(verdict="pass", luggage_type="hard_suitcase"),
            _make_qa_result(verdict="pass", luggage_type="hard_suitcase"),
            _make_qa_result(verdict="reject", luggage_type="hard_suitcase", reason="shadow"),
            _make_qa_result(verdict="pass", luggage_type="backpack"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)

        hs = summary["category_pass_rates"]["hard_suitcase"]
        self.assertEqual(hs["total"], 3)
        self.assertEqual(hs["pass"], 2)
        self.assertAlmostEqual(hs["pass_rate"], 2 / 3, places=4)

        bp = summary["category_pass_rates"]["backpack"]
        self.assertEqual(bp["total"], 1)
        self.assertEqual(bp["pass"], 1)
        self.assertAlmostEqual(bp["pass_rate"], 1.0, places=4)

    def test_category_with_all_reject(self):
        """全件不合格カテゴリの合格率が 0.0 であること"""
        results = [
            _make_qa_result(verdict="reject", luggage_type="stroller", reason="not folded"),
            _make_qa_result(verdict="reject", luggage_type="stroller", reason="not folded"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertEqual(summary["category_pass_rates"]["stroller"]["pass_rate"], 0.0)


class TestRejectionReportTopReasons(unittest.TestCase):
    """Top 10 リジェクト理由のテスト"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir) / "reports"
        self.qa_results_path = Path(self.tmpdir) / "qa_results.json"

    def _write_results(self, results: list[dict]) -> None:
        with open(self.qa_results_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f)

    def test_top_reasons_sorted_by_count(self):
        """リジェクト理由が頻出順に並んでいること"""
        results = [
            _make_qa_result(verdict="reject", reason="shadow on background"),
            _make_qa_result(verdict="reject", reason="shadow on background"),
            _make_qa_result(verdict="reject", reason="shadow on background"),
            _make_qa_result(verdict="reject", reason="handle extended"),
            _make_qa_result(verdict="reject", reason="handle extended"),
            _make_qa_result(verdict="reject", reason="bag open"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        reasons = summary["top_reject_reasons"]
        self.assertEqual(reasons[0]["reason"], "shadow on background")
        self.assertEqual(reasons[0]["count"], 3)
        self.assertEqual(reasons[1]["count"], 2)

    def test_top_reasons_max_10(self):
        """Top reasons が最大 10 件であること"""
        results = [
            _make_qa_result(verdict="reject", reason=f"reason_{i}")
            for i in range(20)
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertLessEqual(len(summary["top_reject_reasons"]), 10)

    def test_pass_records_excluded_from_reasons(self):
        """pass した記録のリジェクト理由が top_reasons に含まれないこと"""
        results = [
            _make_qa_result(verdict="pass", reason=""),
            _make_qa_result(verdict="reject", reason="shadow"),
        ]
        self._write_results(results)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        reasons = [r["reason"] for r in summary["top_reject_reasons"]]
        self.assertNotIn("", reasons)


class TestRejectionReportOutputFiles(unittest.TestCase):
    """出力ファイル生成のテスト"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir) / "reports"
        self.qa_results_path = Path(self.tmpdir) / "qa_results.json"

    def _write_results(self, results: list[dict]) -> None:
        with open(self.qa_results_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f)

    def test_json_file_created(self):
        """JSON ファイルが生成されること"""
        self._write_results([_make_qa_result()])
        analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        json_path = self.output_dir / "image_qa_rejection_analysis.json"
        self.assertTrue(json_path.exists(), "JSON ファイルが生成されていない")

    def test_html_file_created(self):
        """HTML ファイルが生成されること"""
        self._write_results([_make_qa_result()])
        analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        html_path = self.output_dir / "image_qa_rejection_analysis.html"
        self.assertTrue(html_path.exists(), "HTML ファイルが生成されていない")

    def test_json_schema_correct(self):
        """出力 JSON が正しいスキーマを持つこと"""
        results = [
            _make_qa_result(verdict="pass"),
            _make_qa_result(verdict="reject", reason="shadow"),
        ]
        self._write_results(results)
        analyze_image_qa_rejections(self.qa_results_path, self.output_dir)

        json_path = self.output_dir / "image_qa_rejection_analysis.json"
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        required_keys = {
            "total", "pass", "review", "reject", "pass_rate",
            "condition_counts", "category_pass_rates", "top_reject_reasons",
        }
        missing = required_keys - set(data.keys())
        self.assertFalse(missing, f"JSON に不足キーあり: {missing}")

    def test_condition_counts_schema(self):
        """condition_counts が全必須条件キーを持つこと"""
        self._write_results([_make_qa_result(verdict="reject", reason="x")])
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)

        required_conditions = {
            "realism_low", "integrity_low", "has_artifacts",
            "handle_extended", "bag_open",
            "not_checked_baggage_appropriate", "not_checked_in_ready",
            "shadow", "blur", "coverage_low", "bad_angle",
        }
        missing = required_conditions - set(summary["condition_counts"].keys())
        self.assertFalse(missing, f"condition_counts に不足キーあり: {missing}")

    def test_missing_qa_file_returns_empty(self):
        """QA 結果ファイルが存在しない場合に空 dict を返すこと"""
        summary = analyze_image_qa_rejections(
            Path(self.tmpdir) / "nonexistent.json",
            self.output_dir,
        )
        self.assertEqual(summary, {})

    def test_flat_results_list_format(self):
        """results キーなしのフラットリスト形式も受け付けること"""
        flat_results = [
            _make_qa_result(verdict="pass"),
            _make_qa_result(verdict="reject", reason="shadow"),
        ]
        with open(self.qa_results_path, "w", encoding="utf-8") as f:
            json.dump(flat_results, f)
        summary = analyze_image_qa_rejections(self.qa_results_path, self.output_dir)
        self.assertEqual(summary["total"], 2)

    def test_html_contains_key_sections(self):
        """HTML ファイルに主要セクションが含まれること"""
        results = [
            _make_qa_result(verdict="pass"),
            _make_qa_result(verdict="reject", reason="shadow on background"),
        ]
        self._write_results(results)
        analyze_image_qa_rejections(self.qa_results_path, self.output_dir)

        html_path = self.output_dir / "image_qa_rejection_analysis.html"
        content = html_path.read_text(encoding="utf-8")
        self.assertIn("条件別失敗カウント", content)
        self.assertIn("カテゴリ別合格率", content)
        self.assertIn("Top 10 リジェクト理由", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
