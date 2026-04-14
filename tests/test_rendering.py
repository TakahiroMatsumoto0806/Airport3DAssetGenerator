"""
rendering.py テスト

pyrender + OSMesa をモック化してユニットテストを実行する。
headless 環境・GPU なし CI でも動作する。

テスト対象:
  - _look_at_pose()  : カメラポーズ行列の数学的正しさ（純粋計算、モック不要）
  - render_multiview(): 出力ファイル数・命名・ディレクトリ作成（pyrender モック）
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent.parent))

# rendering.py は import 時に PYOPENGL_PLATFORM 警告を出すが CI では無視する
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from src.utils.rendering import _look_at_pose, render_multiview


# ============================================================
# テスト用ヘルパー
# ============================================================

def _make_glb(tmpdir: str, name: str = "mesh.glb") -> str:
    """テスト用 GLB を作成して返す"""
    mesh = trimesh.creation.icosphere(subdivisions=2)
    path = str(Path(tmpdir) / name)
    mesh.export(path, file_type="glb")
    return path


def _make_pyrender_mock() -> MagicMock:
    """pyrender モジュール全体のモック"""
    mock_pr = MagicMock()

    # OffscreenRenderer.render() が (H,W,3) の uint8 配列と depth を返す
    dummy_color = np.full((512, 512, 3), 128, dtype=np.uint8)
    dummy_depth = np.zeros((512, 512), dtype=np.float32)
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = (dummy_color, dummy_depth)
    mock_pr.OffscreenRenderer.return_value = mock_renderer

    # Scene, Camera, Light は MagicMock のまま
    mock_pr.Scene.from_trimesh_scene.return_value = MagicMock()
    return mock_pr


# ============================================================
# _look_at_pose テスト（モック不要の純粋計算）
# ============================================================

class TestLookAtPose(unittest.TestCase):

    def test_returns_4x4_matrix(self):
        """4×4 の numpy 行列を返すこと"""
        pose = _look_at_pose(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0]))
        self.assertEqual(pose.shape, (4, 4))

    def test_eye_stored_in_translation(self):
        """pose[:3, 3] に eye 座標が格納されること"""
        eye = np.array([1.0, 2.0, 3.0])
        pose = _look_at_pose(eye, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(pose[:3, 3], eye, atol=1e-10)

    def test_rotation_columns_orthonormal(self):
        """回転部分 R が正規直交行列であること（R @ R.T ≈ I）"""
        pose = _look_at_pose(np.array([1.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]))
        R = pose[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_homogeneous_row_is_0001(self):
        """最終行が [0, 0, 0, 1] であること"""
        pose = _look_at_pose(np.array([2.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(pose[3], [0, 0, 0, 1])

    def test_degenerate_eye_target_raises(self):
        """eye と target が同一点なら ValueError を送出すること"""
        with self.assertRaises(ValueError):
            _look_at_pose(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))

    def test_overhead_view_uses_fallback_up(self):
        """真上から見下ろす場合（eye=Y軸方向）も正常に計算できること"""
        # eye が up=(0,1,0) と平行な場合、内部でフォールバックを使う
        pose = _look_at_pose(np.array([0.0, 3.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        self.assertEqual(pose.shape, (4, 4))
        R = pose[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_multiple_azimuths_produce_distinct_poses(self):
        """異なるアジマス角から見たポーズは互いに異なること"""
        target = np.array([0.0, 0.0, 0.0])
        angles = [0, 90, 180, 270]
        poses = []
        for deg in angles:
            az = np.radians(deg)
            eye = np.array([2.0 * np.sin(az), 0.0, 2.0 * np.cos(az)])
            poses.append(_look_at_pose(eye, target))
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                self.assertFalse(
                    np.allclose(poses[i], poses[j]),
                    f"az={angles[i]}° と az={angles[j]}° のポーズが同一"
                )


# ============================================================
# render_multiview テスト（pyrender モック）
# ============================================================

class TestRenderMultiview(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.glb_path = _make_glb(self.tmpdir)
        self.output_dir = str(Path(self.tmpdir) / "renders")
        self.mock_pr = _make_pyrender_mock()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, views: int = 4, prefix: str = "view", **kwargs) -> list[str]:
        """pyrender をモックして render_multiview を実行する"""
        with patch.dict("sys.modules", {"pyrender": self.mock_pr}):
            return render_multiview(
                self.glb_path, self.output_dir,
                views=views, prefix=prefix, **kwargs
            )

    def test_returns_n_paths(self):
        """views=4 のとき 4 件のパスを返すこと"""
        result = self._run(views=4)
        self.assertEqual(len(result), 4)

    def test_returns_correct_count_for_different_views(self):
        """views=2 でも 2 件を返すこと"""
        result = self._run(views=2)
        self.assertEqual(len(result), 2)

    def test_output_dir_created(self):
        """出力ディレクトリが自動作成されること"""
        self._run()
        self.assertTrue(Path(self.output_dir).exists())

    def test_output_files_are_png(self):
        """出力ファイルの拡張子がすべて .png であること"""
        result = self._run()
        for path in result:
            self.assertEqual(Path(path).suffix, ".png")

    def test_default_prefix_in_filenames(self):
        """デフォルト prefix 'view' がファイル名に含まれること"""
        result = self._run(prefix="view")
        for path in result:
            self.assertIn("view_", Path(path).name)

    def test_custom_prefix_applied(self):
        """カスタム prefix がファイル名に反映されること"""
        result = self._run(prefix="render")
        for path in result:
            self.assertIn("render_", Path(path).name)

    def test_filenames_are_sequential(self):
        """出力ファイルが view_0.png, view_1.png... の順番であること"""
        result = self._run(views=4)
        for i, path in enumerate(result):
            self.assertEqual(Path(path).name, f"view_{i}.png")

    def test_renderer_deleted_on_completion(self):
        """OffscreenRenderer.delete() が呼ばれること（リソース解放）"""
        self._run()
        self.mock_pr.OffscreenRenderer.return_value.delete.assert_called_once()

    def test_render_called_once_per_view(self):
        """render() が views 回呼ばれること"""
        views = 3
        self._run(views=views)
        self.assertEqual(
            self.mock_pr.OffscreenRenderer.return_value.render.call_count, views
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
