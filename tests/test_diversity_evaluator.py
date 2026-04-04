"""
T-5.1: DiversityEvaluator テスト

OpenCLIP はモック化してユニットテストを実行する。
Vendi Score・重複検出・統計計算のロジックを検証。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diversity_evaluator import DiversityEvaluator


# ============================================================
# Vendi Score テスト
# ============================================================

class TestVendiScore(unittest.TestCase):

    def setUp(self):
        self.ev = DiversityEvaluator()

    def test_empty_returns_zero(self):
        self.assertEqual(self.ev.compute_vendi_score(np.zeros((0, 8))), 0.0)

    def test_single_returns_one(self):
        e = np.array([[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(self.ev.compute_vendi_score(e), 1.0, places=3)

    def test_identical_embeddings_score_near_one(self):
        """完全に同一の埋め込みなら Vendi Score ≈ 1"""
        e = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        score = self.ev.compute_vendi_score(e)
        self.assertLess(score, 2.0)

    def test_orthogonal_embeddings_score_near_n(self):
        """直交ベクトルなら Vendi Score ≈ N"""
        n = 4
        e = np.eye(n)
        score = self.ev.compute_vendi_score(e)
        self.assertGreater(score, n * 0.8)
        self.assertLessEqual(score, n + 0.5)

    def test_diverse_greater_than_uniform(self):
        """多様な埋め込みの方が一様より高いスコア"""
        uniform = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
        diverse = np.random.default_rng(42).standard_normal((10, 4))
        diverse /= np.linalg.norm(diverse, axis=1, keepdims=True)
        self.assertGreater(
            self.ev.compute_vendi_score(diverse),
            self.ev.compute_vendi_score(uniform),
        )

    def test_score_is_float(self):
        e = np.eye(5)
        self.assertIsInstance(self.ev.compute_vendi_score(e), float)


# ============================================================
# 近似重複検出テスト
# ============================================================

class TestFindNearDuplicates(unittest.TestCase):

    def setUp(self):
        self.ev = DiversityEvaluator()

    def test_identical_pair_detected(self):
        """完全に同一のベクトルペアが検出されること"""
        e = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        dups = self.ev.find_near_duplicates(e, threshold=0.99)
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0][0], 0)
        self.assertEqual(dups[0][1], 1)

    def test_orthogonal_no_duplicates(self):
        """直交ベクトルはゼロ件"""
        e = np.eye(5)
        dups = self.ev.find_near_duplicates(e, threshold=0.95)
        self.assertEqual(len(dups), 0)

    def test_similarity_value_correct(self):
        """類似度の値が正しく計算されること"""
        e = np.array([[1.0, 0.0], [1.0, 0.0]])
        dups = self.ev.find_near_duplicates(e, threshold=0.9)
        self.assertEqual(len(dups), 1)
        self.assertAlmostEqual(dups[0][2], 1.0, places=5)

    def test_with_image_paths(self):
        """image_paths を渡した場合にパスが含まれること"""
        e = np.array([[1.0, 0.0], [1.0, 0.0]])
        paths = ["a.png", "b.png"]
        dups = self.ev.find_near_duplicates(e, threshold=0.9, image_paths=paths)
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0][0], "a.png")
        self.assertEqual(dups[0][1], "b.png")

    def test_returns_i_less_than_j(self):
        """返り値の i < j であること"""
        e = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]])
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        dups = self.ev.find_near_duplicates(e, threshold=0.9)
        for d in dups:
            self.assertLess(d[0], d[1])


# ============================================================
# サイズ・カテゴリ統計テスト
# ============================================================

class TestComputeSizeDiversity(unittest.TestCase):

    def setUp(self):
        self.ev = DiversityEvaluator()

    def test_returns_required_keys(self):
        info = [{"scale": {"scaled_extents_mm": [50.0, 35.0, 70.0]}}]
        result = self.ev.compute_size_diversity(info)
        for key in ("count", "short_side_mm", "long_side_mm", "volume_mm3"):
            self.assertIn(key, result)

    def test_count_correct(self):
        info = [{"scale": {"scaled_extents_mm": [50.0, 35.0, 70.0]}} for _ in range(5)]
        result = self.ev.compute_size_diversity(info)
        self.assertEqual(result["count"], 5)

    def test_short_side_correct(self):
        info = [{"scale": {"scaled_extents_mm": [50.0, 35.0, 70.0]}}]
        result = self.ev.compute_size_diversity(info)
        self.assertAlmostEqual(result["short_side_mm"]["min"], 35.0, places=2)

    def test_long_side_correct(self):
        info = [{"scale": {"scaled_extents_mm": [50.0, 35.0, 70.0]}}]
        result = self.ev.compute_size_diversity(info)
        self.assertAlmostEqual(result["long_side_mm"]["max"], 70.0, places=2)

    def test_empty_list(self):
        result = self.ev.compute_size_diversity([])
        self.assertEqual(result["count"], 0)

    def test_missing_scale_key_skipped(self):
        info = [{"no_scale_key": True}, {"scale": {"scaled_extents_mm": [50.0, 35.0, 70.0]}}]
        result = self.ev.compute_size_diversity(info)
        self.assertEqual(result["count"], 1)


class TestComputeCategoryDistribution(unittest.TestCase):

    def setUp(self):
        self.ev = DiversityEvaluator()

    def test_returns_required_keys(self):
        meta = [{"luggage_type": "hard_suitcase", "material": "polycarbonate"}]
        result = self.ev.compute_category_distribution(meta)
        for key in ("luggage_type", "material", "detected_type"):
            self.assertIn(key, result)

    def test_counts_correct(self):
        meta = [
            {"luggage_type": "hard_suitcase", "material": "nylon"},
            {"luggage_type": "hard_suitcase", "material": "polycarbonate"},
            {"luggage_type": "backpack", "material": "nylon"},
        ]
        result = self.ev.compute_category_distribution(meta)
        self.assertEqual(result["luggage_type"]["hard_suitcase"], 2)
        self.assertEqual(result["luggage_type"]["backpack"], 1)
        self.assertEqual(result["material"]["nylon"], 2)

    def test_empty_list(self):
        result = self.ev.compute_category_distribution([])
        self.assertEqual(sum(result["luggage_type"].values()), 0)


class TestCheckSizeRealism(unittest.TestCase):

    def setUp(self):
        self.ev = DiversityEvaluator()
        self.refs = {
            "hard_suitcase": {"min_mm": 30.0, "max_mm": 120.0},
        }

    def test_realistic_counted(self):
        info = [{"luggage_type": "hard_suitcase",
                 "scale": {"scaled_extents_mm": [50.0, 35.0, 70.0]}}]
        result = self.ev.check_size_realism(info, self.refs)
        self.assertEqual(result["realistic"], 1)
        self.assertEqual(result["unrealistic"], 0)

    def test_unrealistic_counted(self):
        info = [{"luggage_type": "hard_suitcase",
                 "scale": {"scaled_extents_mm": [5.0, 3.5, 200.0]}}]
        result = self.ev.check_size_realism(info, self.refs)
        self.assertEqual(result["unrealistic"], 1)

    def test_total_count(self):
        info = [
            {"luggage_type": "hard_suitcase", "scale": {"scaled_extents_mm": [50.0, 35.0, 70.0]}},
            {"luggage_type": "hard_suitcase", "scale": {"scaled_extents_mm": [5.0, 3.5, 200.0]}},
        ]
        result = self.ev.check_size_realism(info, self.refs)
        self.assertEqual(result["total"], 2)


# ============================================================
# generate_report テスト
# ============================================================

class TestGenerateReport(unittest.TestCase):

    def setUp(self):
        self.ev = DiversityEvaluator()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_html_file_created(self):
        embeddings = np.eye(5)
        path = self.ev.generate_report(self.tmpdir, embeddings=embeddings)
        self.assertTrue(Path(path).exists())
        self.assertTrue(path.endswith(".html"))

    def test_json_file_created(self):
        embeddings = np.eye(5)
        self.ev.generate_report(self.tmpdir, embeddings=embeddings)
        json_path = Path(self.tmpdir) / "diversity_report.json"
        self.assertTrue(json_path.exists())

    def test_json_has_vendi_score(self):
        embeddings = np.eye(4)
        self.ev.generate_report(self.tmpdir, embeddings=embeddings)
        with open(Path(self.tmpdir) / "diversity_report.json") as f:
            data = json.load(f)
        self.assertIn("vendi_score", data)
        self.assertIsNotNone(data["vendi_score"])

    def test_no_embeddings_still_generates(self):
        """embeddings=None でもレポートが生成されること"""
        path = self.ev.generate_report(self.tmpdir)
        self.assertTrue(Path(path).exists())

    def test_near_duplicates_in_report(self):
        """完全一致の埋め込みがある場合、near_duplicates_count > 0"""
        e = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.ev.generate_report(self.tmpdir, embeddings=e)
        with open(Path(self.tmpdir) / "diversity_report.json") as f:
            data = json.load(f)
        self.assertGreater(data.get("near_duplicates_count", 0), 0)


# ============================================================
# compute_clip_embeddings テスト（OpenCLIP モック）
# ============================================================

class TestComputeClipEmbeddings(unittest.TestCase):

    def setUp(self):
        self.ev = DiversityEvaluator()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_images(self, n: int) -> list[str]:
        from PIL import Image
        paths = []
        for i in range(n):
            p = str(Path(self.tmpdir) / f"img_{i}.png")
            Image.new("RGB", (224, 224), color=(i * 10, 100, 200)).save(p)
            paths.append(p)
        return paths

    def test_not_loaded_raises(self):
        """モデル未ロードで RuntimeError"""
        with self.assertRaises(RuntimeError):
            self.ev.compute_clip_embeddings(["img.png"])

    @patch("src.diversity_evaluator.DiversityEvaluator.compute_clip_embeddings")
    def test_embeddings_shape(self, mock_embed):
        """埋め込みが (N, D) 形状であること（モック）"""
        mock_embed.return_value = np.random.randn(5, 768).astype(np.float32)
        mock_embed.return_value /= np.linalg.norm(mock_embed.return_value, axis=1, keepdims=True)
        result = self.ev.compute_clip_embeddings(["img.png"] * 5)
        self.assertEqual(result.shape[0], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
