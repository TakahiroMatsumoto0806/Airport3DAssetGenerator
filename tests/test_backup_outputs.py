"""tests/test_backup_outputs.py — backup_outputs.py のテスト

すべて tempfile.TemporaryDirectory を使い、本物の outputs/ には一切触れない。
"""

import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# scripts/ をインポートパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import backup_outputs as bo


def _make_fake_outputs(base: Path) -> Path:
    """テスト用の疑似 outputs/ ディレクトリを作成して返す。"""
    outputs = base / "outputs"
    # ファイルを含む典型的な構造を作成
    (outputs / "assets_final" / "000001").mkdir(parents=True)
    (outputs / "assets_final" / "000001" / "visual.glb").write_text("glb_data")
    (outputs / "assets_final" / "000001" / "physics.json").write_text("{}")
    (outputs / "assets_final" / "collisions").mkdir(parents=True)
    (outputs / "assets_final" / "collisions" / ".gitkeep").touch()
    (outputs / "assets_final" / "isaac").mkdir(parents=True)
    (outputs / "assets_final" / "isaac" / ".gitkeep").touch()
    (outputs / "assets_final" / "metadata").mkdir(parents=True)
    (outputs / "assets_final" / "metadata" / ".gitkeep").touch()
    (outputs / "assets_final" / "mjcf").mkdir(parents=True)
    (outputs / "assets_final" / "mjcf" / ".gitkeep").touch()
    (outputs / "images").mkdir(parents=True)
    (outputs / "images" / "img_000001.png").write_bytes(b"\x89PNG")
    (outputs / "images_approved").mkdir(parents=True)
    (outputs / "images_approved" / ".gitkeep").touch()
    (outputs / "logs").mkdir(parents=True)
    (outputs / "logs" / "pipeline.log").write_text("log data")
    (outputs / "meshes_approved").mkdir(parents=True)
    (outputs / "meshes_approved" / ".gitkeep").touch()
    (outputs / "meshes_raw").mkdir(parents=True)
    (outputs / "meshes_raw" / ".gitkeep").touch()
    (outputs / "prompts").mkdir(parents=True)
    (outputs / "prompts" / "prompts.json").write_text("[]")
    (outputs / "renders").mkdir(parents=True)
    (outputs / "renders" / ".gitkeep").touch()
    (outputs / "reports").mkdir(parents=True)
    (outputs / "reports" / ".gitkeep").touch()
    return outputs


TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")

EXPECTED_LEAF_DIRS = [
    "assets_final/collisions",
    "assets_final/isaac",
    "assets_final/metadata",
    "assets_final/mjcf",
    "images",
    "images_approved",
    "logs",
    "meshes_approved",
    "meshes_raw",
    "prompts",
    "renders",
    "reports",
]


class TestBackupCreatesTimestampedDir(unittest.TestCase):
    def test_backup_creates_timestamped_dir(self):
        """バックアップフォルダが YYYY-MM-DD_HHMMSS 形式で生成される。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = _make_fake_outputs(tmp)
            backup_parent = tmp / "outputs_backup"

            result = bo.backup_outputs(outputs, backup_parent, dry_run=False)

            self.assertTrue(backup_parent.exists())
            entries = list(backup_parent.iterdir())
            self.assertEqual(len(entries), 1)
            dirname = entries[0].name
            self.assertRegex(dirname, TIMESTAMP_PATTERN)
            self.assertEqual(result, backup_parent / dirname)


class TestBackupCopiesAllFiles(unittest.TestCase):
    def test_backup_copies_all_files(self):
        """outputs/ 内の全ファイルがバックアップに存在する。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = _make_fake_outputs(tmp)
            backup_parent = tmp / "outputs_backup"

            original_files = {f.relative_to(outputs) for f in outputs.rglob("*") if f.is_file()}
            dest = bo.backup_outputs(outputs, backup_parent, dry_run=False)

            backed_up_files = {f.relative_to(dest) for f in dest.rglob("*") if f.is_file()}
            self.assertEqual(original_files, backed_up_files)

    def test_backup_preserves_file_contents(self):
        """バックアップ先のファイル内容が元と一致する。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = _make_fake_outputs(tmp)
            backup_parent = tmp / "outputs_backup"

            dest = bo.backup_outputs(outputs, backup_parent, dry_run=False)

            original = (outputs / "prompts" / "prompts.json").read_text()
            backed = (dest / "prompts" / "prompts.json").read_text()
            self.assertEqual(original, backed)


class TestResetCreatesRequiredDirs(unittest.TestCase):
    def test_reset_creates_required_dirs(self):
        """リセット後に期待するサブディレクトリが全て存在する。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = _make_fake_outputs(tmp)

            bo.reset_outputs(outputs, dry_run=False)

            for rel in EXPECTED_LEAF_DIRS:
                self.assertTrue(
                    (outputs / rel).is_dir(),
                    f"リセット後に {rel} が存在しない",
                )

    def test_reset_creates_gitkeep(self):
        """.gitkeep が各リーフディレクトリに存在する。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = _make_fake_outputs(tmp)

            bo.reset_outputs(outputs, dry_run=False)

            for rel in EXPECTED_LEAF_DIRS:
                gitkeep = outputs / rel / ".gitkeep"
                self.assertTrue(
                    gitkeep.exists(),
                    f"{rel}/.gitkeep が存在しない",
                )

    def test_reset_removes_old_content(self):
        """リセット後に旧コンテンツ（アセットファイル等）が残らない。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = _make_fake_outputs(tmp)

            bo.reset_outputs(outputs, dry_run=False)

            # アセットフォルダは消えているはず
            self.assertFalse((outputs / "assets_final" / "000001").exists())
            # 画像ファイルも消えているはず
            self.assertFalse((outputs / "images" / "img_000001.png").exists())
            # ログも消えているはず
            self.assertFalse((outputs / "logs" / "pipeline.log").exists())

    def test_reset_only_gitkeep_files_remain(self):
        """リセット後のファイルはすべて .gitkeep のみ。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = _make_fake_outputs(tmp)

            bo.reset_outputs(outputs, dry_run=False)

            all_files = list(outputs.rglob("*"))
            non_gitkeep = [f for f in all_files if f.is_file() and f.name != ".gitkeep"]
            self.assertEqual(
                non_gitkeep,
                [],
                f".gitkeep 以外のファイルが残っている: {non_gitkeep}",
            )


class TestDryRunNoSideEffects(unittest.TestCase):
    def test_dry_run_backup_no_side_effects(self):
        """--dry-run では backup_outputs がファイルを作成しない。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = _make_fake_outputs(tmp)
            backup_parent = tmp / "outputs_backup"

            bo.backup_outputs(outputs, backup_parent, dry_run=True)

            self.assertFalse(backup_parent.exists(), "dry-run なのにバックアップが作成された")

    def test_dry_run_reset_no_side_effects(self):
        """--dry-run では reset_outputs が outputs/ を変更しない。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = _make_fake_outputs(tmp)

            before = {f.relative_to(outputs) for f in outputs.rglob("*") if f.is_file()}
            bo.reset_outputs(outputs, dry_run=True)
            after = {f.relative_to(outputs) for f in outputs.rglob("*") if f.is_file()}

            self.assertEqual(before, after, "dry-run なのに outputs/ が変更された")


class TestMultipleBackupsPreserved(unittest.TestCase):
    def test_multiple_backups_preserved(self):
        """2回実行しても両方のバックアップが保持される。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = _make_fake_outputs(tmp)
            backup_parent = tmp / "outputs_backup"

            dest1 = bo.backup_outputs(outputs, backup_parent, dry_run=False)
            dest2 = bo.backup_outputs(outputs, backup_parent, dry_run=False)

            entries = list(backup_parent.iterdir())
            self.assertEqual(len(entries), 2)
            self.assertTrue(dest1.exists())
            self.assertTrue(dest2.exists())
            self.assertNotEqual(dest1, dest2)


class TestEmptyOutputsStillWorks(unittest.TestCase):
    def test_empty_outputs_backup(self):
        """outputs/ が空でもバックアップがエラーなく動作する。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = tmp / "outputs"
            outputs.mkdir()
            backup_parent = tmp / "outputs_backup"

            dest = bo.backup_outputs(outputs, backup_parent, dry_run=False)
            self.assertTrue(dest.exists())

    def test_empty_outputs_reset(self):
        """outputs/ が空でもリセットがエラーなく動作し、構造を再作成する。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            outputs = tmp / "outputs"
            outputs.mkdir()

            bo.reset_outputs(outputs, dry_run=False)

            for rel in EXPECTED_LEAF_DIRS:
                self.assertTrue((outputs / rel).is_dir())
                self.assertTrue((outputs / rel / ".gitkeep").exists())


class TestResetDirsMatchScript(unittest.TestCase):
    def test_reset_dirs_constant_matches_expected(self):
        """RESET_DIRS 定数が期待するリーフパスと一致する。"""
        self.assertEqual(sorted(bo.RESET_DIRS), sorted(EXPECTED_LEAF_DIRS))


if __name__ == "__main__":
    unittest.main()
