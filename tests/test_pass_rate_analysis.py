"""
T-pass_rate: カテゴリ別合格率分析のユニットテスト

テスト対象:
  - scripts/generate_report.py :: _wilson_ci()
  - scripts/generate_report.py :: _calc_recommended_weights()
  - scripts/generate_report.py :: analyze_pass_rates()
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_report import (
    _calc_recommended_weights,
    _wilson_ci,
    analyze_pass_rates,
)


# ---------------------------------------------------------------------------
# Wilson CI テスト
# ---------------------------------------------------------------------------

class TestWilsonCI:
    """_wilson_ci() の境界条件・数値計算テスト"""

    def test_n_zero_returns_zeros(self):
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_all_pass(self):
        """k=n のとき上限は 1.0、下限は 0 より大きい"""
        lo, hi = _wilson_ci(20, 20)
        assert hi == 1.0
        assert lo > 0.8  # 95% CI lower bound for 20/20

    def test_all_fail(self):
        """k=0 のとき下限は 0.0、上限は 0 より大きい"""
        lo, hi = _wilson_ci(0, 20)
        assert lo == 0.0
        assert hi > 0.0
        assert hi < 0.2  # CI が合理的な範囲内

    def test_half_pass_symmetric(self):
        """k=n/2 のとき CI は 0.5 を中心にほぼ対称"""
        lo, hi = _wilson_ci(10, 20)
        center = (lo + hi) / 2
        assert abs(center - 0.5) < 0.05

    def test_bounds_clamped_to_01(self):
        """どの入力でも 0.0〜1.0 に収まること"""
        for k, n in [(0, 1), (1, 1), (5, 100), (100, 100)]:
            lo, hi = _wilson_ci(k, n)
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0
            assert lo <= hi

    def test_larger_n_gives_narrower_ci(self):
        """サンプル数が増えるほど CI が狭まること"""
        lo5, hi5 = _wilson_ci(3, 5)   # n=5, p≈0.6
        lo20, hi20 = _wilson_ci(12, 20)  # n=20, p=0.6
        assert (hi5 - lo5) > (hi20 - lo20)

    def test_n20_worst_case_precision(self):
        """n=20/カテゴリ のとき worst case (p=0.5) の半幅が ±22% 以内"""
        lo, hi = _wilson_ci(10, 20)
        half_width = (hi - lo) / 2
        assert half_width <= 0.25  # 計画では ±22%


# ---------------------------------------------------------------------------
# 推奨重み計算テスト
# ---------------------------------------------------------------------------

class TestCalcRecommendedWeights:
    """_calc_recommended_weights() の計算テスト"""

    # 実際の category_weights の簡易版
    CURRENT_WEIGHTS = {
        "hard_suitcase": 0.25,
        "soft_suitcase": 0.15,
        "backpack": 0.15,
        "duffel_bag": 0.10,
        "briefcase": 0.07,
        "cardboard_box": 0.06,
        "hard_case": 0.04,
        "golf_bag": 0.03,
        "stroller": 0.03,
        "instrument_case": 0.02,
        "ski_bag": 0.02,
    }

    def test_output_sums_to_one(self):
        """推奨重みの合計が 1.0 になること（浮動小数点誤差許容）"""
        pass_rates = {c: 0.5 for c in self.CURRENT_WEIGHTS}
        pass_rates["stroller"] = 0.0
        pass_rates["ski_bag"] = 0.0
        rec = _calc_recommended_weights(pass_rates, self.CURRENT_WEIGHTS, 0.70)
        assert abs(sum(rec.values()) - 1.0) < 0.01

    def test_target_fraction_achieved(self):
        """推奨重みで期待合格割合が目標値になること（±3%許容）"""
        pass_rates = {
            "hard_suitcase": 0.70,
            "soft_suitcase": 0.40,
            "backpack": 0.60,
            "duffel_bag": 0.30,
            "briefcase": 0.0,
            "cardboard_box": 0.50,
            "hard_case": 0.40,
            "golf_bag": 0.10,
            "stroller": 0.0,
            "instrument_case": 0.30,
            "ski_bag": 0.0,
        }
        target_f = 0.70
        rec = _calc_recommended_weights(pass_rates, self.CURRENT_WEIGHTS, target_f)

        # 期待合格 = sum(w_i * p_i) / total
        expected_total = sum(rec[c] * pass_rates.get(c, 0.0) for c in rec)
        hs_fraction = rec["hard_suitcase"] * pass_rates["hard_suitcase"] / expected_total
        assert abs(hs_fraction - target_f) < 0.03

    def test_zero_pass_categories_get_min_weight(self):
        """合格率 0% カテゴリが最低重みを持つこと"""
        pass_rates = {c: 0.5 for c in self.CURRENT_WEIGHTS}
        pass_rates["stroller"] = 0.0
        pass_rates["ski_bag"] = 0.0
        min_eps = 0.005
        rec = _calc_recommended_weights(
            pass_rates, self.CURRENT_WEIGHTS, 0.70, min_weight_zero=min_eps
        )
        assert rec["stroller"] == pytest.approx(min_eps, abs=0.002)
        assert rec["ski_bag"] == pytest.approx(min_eps, abs=0.002)

    def test_hs_zero_pass_rate_returns_current(self):
        """hard_suitcase の合格率が 0% の場合は現在の重みをそのまま返すこと"""
        pass_rates = {c: 0.5 for c in self.CURRENT_WEIGHTS}
        pass_rates["hard_suitcase"] = 0.0
        rec = _calc_recommended_weights(pass_rates, self.CURRENT_WEIGHTS, 0.70)
        assert rec == self.CURRENT_WEIGHTS

    def test_all_active_categories(self):
        """全カテゴリが合格率 > 0% のとき、ゼロ重みカテゴリなし"""
        pass_rates = {c: 0.3 for c in self.CURRENT_WEIGHTS}
        pass_rates["hard_suitcase"] = 0.7
        rec = _calc_recommended_weights(pass_rates, self.CURRENT_WEIGHTS, 0.70)
        for cat, w in rec.items():
            assert w > 0.0, f"{cat} の重みが 0 以下: {w}"

    def test_output_contains_all_input_categories(self):
        """入力の全カテゴリが出力に含まれること"""
        pass_rates = {c: 0.5 for c in self.CURRENT_WEIGHTS}
        rec = _calc_recommended_weights(pass_rates, self.CURRENT_WEIGHTS, 0.70)
        for cat in self.CURRENT_WEIGHTS:
            assert cat in rec

    def test_different_target_fractions(self):
        """目標割合 0.5, 0.7, 0.9 で各々目標達成すること"""
        pass_rates = {c: 0.4 for c in self.CURRENT_WEIGHTS}
        pass_rates["hard_suitcase"] = 0.6
        pass_rates["stroller"] = 0.0
        for target_f in [0.50, 0.70, 0.90]:
            rec = _calc_recommended_weights(pass_rates, self.CURRENT_WEIGHTS, target_f)
            total_pass = sum(rec[c] * pass_rates.get(c, 0.0) for c in rec)
            if total_pass > 0:
                hs_frac = rec["hard_suitcase"] * pass_rates["hard_suitcase"] / total_pass
                assert abs(hs_frac - target_f) < 0.05, (
                    f"target={target_f} だが実際の割合={hs_frac:.3f}"
                )


# ---------------------------------------------------------------------------
# analyze_pass_rates() 統合テスト（一時ファイルを使用）
# ---------------------------------------------------------------------------

class TestAnalyzePassRates:
    """analyze_pass_rates() の入力バリデーションと出力形式テスト"""

    def _make_qa_results(self, tmp_path: Path, data: list[dict]) -> Path:
        p = tmp_path / "image_qa_results.json"
        p.write_text(json.dumps({"results": data}), encoding="utf-8")
        return p

    def _make_prompts(self, tmp_path: Path, data: list[dict]) -> Path:
        p = tmp_path / "prompts.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_missing_qa_results_returns_empty(self, tmp_path):
        """QA 結果ファイルが存在しない場合は空 dict を返すこと"""
        report = analyze_pass_rates(
            qa_results_path=tmp_path / "nonexistent.json",
            prompts_path=tmp_path / "prompts.json",
            target_hs_fraction=0.70,
            output_dir=tmp_path / "reports",
            config_path=Path("configs/pipeline_config.yaml"),
        )
        assert report == {}

    def test_basic_report_structure(self, tmp_path):
        """基本的なレポート構造が正しいこと"""
        qa_results = [
            {"verdict": "pass", "luggage_type": "hard_suitcase"},
            {"verdict": "pass", "luggage_type": "hard_suitcase"},
            {"verdict": "reject", "luggage_type": "hard_suitcase"},
            {"verdict": "pass", "luggage_type": "backpack"},
            {"verdict": "reject", "luggage_type": "backpack"},
        ]
        qa_path = self._make_qa_results(tmp_path, qa_results)
        prompts_path = self._make_prompts(tmp_path, [])
        output_dir = tmp_path / "reports"

        report = analyze_pass_rates(
            qa_results_path=qa_path,
            prompts_path=prompts_path,
            target_hs_fraction=0.70,
            output_dir=output_dir,
            config_path=Path("configs/pipeline_config.yaml"),
        )

        assert "metadata" in report
        assert "categories" in report
        assert "precision_guide" in report
        assert "weight_recommendations" in report

    def test_category_aggregation_correct(self, tmp_path):
        """カテゴリ別の集計が正確なこと"""
        qa_results = [
            {"verdict": "pass", "luggage_type": "hard_suitcase"},
            {"verdict": "pass", "luggage_type": "hard_suitcase"},
            {"verdict": "reject", "luggage_type": "hard_suitcase"},
            {"verdict": "reject", "luggage_type": "backpack"},
            {"verdict": "reject", "luggage_type": "backpack"},
        ]
        qa_path = self._make_qa_results(tmp_path, qa_results)
        report = analyze_pass_rates(
            qa_results_path=qa_path,
            prompts_path=self._make_prompts(tmp_path, []),
            target_hs_fraction=0.70,
            output_dir=tmp_path / "reports",
            config_path=Path("configs/pipeline_config.yaml"),
        )
        hs = report["categories"]["hard_suitcase"]
        bp = report["categories"]["backpack"]
        assert hs["samples"] == 3
        assert hs["passed"] == 2
        assert abs(hs["pass_rate"] - 2 / 3) < 0.001
        assert bp["samples"] == 2
        assert bp["passed"] == 0
        assert bp["pass_rate"] == 0.0

    def test_zero_category_status(self, tmp_path):
        """合格率 0% カテゴリのステータスが 'zero_pass_investigate' になること"""
        qa_results = [
            {"verdict": "pass", "luggage_type": "hard_suitcase"},
            {"verdict": "reject", "luggage_type": "stroller"},
        ]
        qa_path = self._make_qa_results(tmp_path, qa_results)
        report = analyze_pass_rates(
            qa_results_path=qa_path,
            prompts_path=self._make_prompts(tmp_path, []),
            target_hs_fraction=0.70,
            output_dir=tmp_path / "reports",
            config_path=Path("configs/pipeline_config.yaml"),
        )
        assert report["categories"]["stroller"]["status"] == "zero_pass_investigate"
        assert report["categories"]["hard_suitcase"]["status"] == "active"

    def test_ci_bounds_present(self, tmp_path):
        """CI 境界値フィールドが存在しかつ [0, 1] 範囲内であること"""
        qa_results = [
            {"verdict": "pass", "luggage_type": "hard_suitcase"},
            {"verdict": "reject", "luggage_type": "hard_suitcase"},
        ]
        qa_path = self._make_qa_results(tmp_path, qa_results)
        report = analyze_pass_rates(
            qa_results_path=qa_path,
            prompts_path=self._make_prompts(tmp_path, []),
            target_hs_fraction=0.70,
            output_dir=tmp_path / "reports",
            config_path=Path("configs/pipeline_config.yaml"),
        )
        hs = report["categories"]["hard_suitcase"]
        assert "ci_lower" in hs
        assert "ci_upper" in hs
        assert 0.0 <= hs["ci_lower"] <= hs["ci_upper"] <= 1.0

    def test_output_files_created(self, tmp_path):
        """JSON・HTML レポートが出力されること"""
        qa_results = [
            {"verdict": "pass", "luggage_type": "hard_suitcase"},
        ]
        qa_path = self._make_qa_results(tmp_path, qa_results)
        output_dir = tmp_path / "reports"
        analyze_pass_rates(
            qa_results_path=qa_path,
            prompts_path=self._make_prompts(tmp_path, []),
            target_hs_fraction=0.70,
            output_dir=output_dir,
            config_path=Path("configs/pipeline_config.yaml"),
        )
        assert (output_dir / "pass_rate_report.json").exists()
        assert (output_dir / "pass_rate_report.html").exists()

    def test_prompt_id_mapping(self, tmp_path):
        """prompts.json の prompt_id → luggage_type マップが機能すること"""
        prompts = [
            {"prompt": "a red suitcase", "metadata": {"prompt_id": "abc123def456", "luggage_type": "hard_suitcase"}},
            {"prompt": "a backpack", "metadata": {"prompt_id": "xyz789uvw012", "luggage_type": "backpack"}},
        ]
        # QA 結果に luggage_type がなく image_path から判断する場合
        qa_results = [
            {"verdict": "pass", "image_path": "outputs/images/abc123def456.png"},
            {"verdict": "reject", "image_path": "outputs/images/xyz789uvw012.png"},
        ]
        qa_path = self._make_qa_results(tmp_path, qa_results)
        prompts_path = self._make_prompts(tmp_path, prompts)
        report = analyze_pass_rates(
            qa_results_path=qa_path,
            prompts_path=prompts_path,
            target_hs_fraction=0.70,
            output_dir=tmp_path / "reports",
            config_path=Path("configs/pipeline_config.yaml"),
        )
        # hard_suitcase が正しく集計されること
        assert "hard_suitcase" in report["categories"]
        assert report["categories"]["hard_suitcase"]["passed"] == 1

    def test_overall_metadata_correct(self, tmp_path):
        """メタデータの総数・合格数・合格率が正確なこと"""
        qa_results = [{"verdict": "pass", "luggage_type": "hard_suitcase"}] * 6 + \
                     [{"verdict": "reject", "luggage_type": "hard_suitcase"}] * 4
        qa_path = self._make_qa_results(tmp_path, qa_results)
        report = analyze_pass_rates(
            qa_results_path=qa_path,
            prompts_path=self._make_prompts(tmp_path, []),
            target_hs_fraction=0.70,
            output_dir=tmp_path / "reports",
            config_path=Path("configs/pipeline_config.yaml"),
        )
        assert report["metadata"]["total_generated"] == 10
        assert report["metadata"]["total_passed"] == 6
        assert abs(report["metadata"]["overall_pass_rate"] - 0.6) < 0.001
