"""
memory_guard ユニットテスト + モデルアンロード統合テスト

テスト構成:
  Class A: MemoryGuard ユニットテスト (GPU / 外部プロセス不要)
  Class B: flush_cuda_memory 統合テスト (CUDA 環境のみ)
  Class C: モデルクラス unload() 後のメモリ確認 (CUDA + 実モデル不要)
  Class D: stop_vllm_server メモリ待機ロジック (モック)

実行方法:
  # GPU 不要テストのみ (CI 向け)
  pytest tests/test_memory_guard.py -v -m "not gpu"

  # GPU ありテスト (DGX Spark)
  pytest tests/test_memory_guard.py -v
"""

import gc
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ===========================================================================
# Class A: memory_guard ユニットテスト (GPU / 外部プロセス不要)
# ===========================================================================

class TestFlushCudaMemoryNoCuda:
    """flush_cuda_memory — CUDA なし環境での動作確認"""

    def test_returns_zeros_when_cuda_unavailable(self):
        """CUDA が使えない場合は (0.0, 0.0) を返す"""
        from src.utils.memory_guard import flush_cuda_memory

        with patch("torch.cuda.is_available", return_value=False):
            result = flush_cuda_memory()
        assert result == (0.0, 0.0)

    def test_runs_gc_collect_twice(self):
        """gc.collect() が 2 回呼ばれることを確認（参照サイクル対策）"""
        from src.utils.memory_guard import flush_cuda_memory

        with patch("gc.collect") as mock_gc, \
             patch("torch.cuda.is_available", return_value=False):
            flush_cuda_memory()

        assert mock_gc.call_count >= 2, "gc.collect() は最低 2 回必要"

    def test_does_not_raise_without_torch(self):
        """torch が ImportError でも例外を起こさない"""
        import importlib
        import src.utils.memory_guard as mg

        original = sys.modules.get("torch")
        sys.modules["torch"] = None  # import torch → ImportError 相当
        try:
            # 直接呼べばOK (内部で except Exception)
            result = mg.flush_cuda_memory()
            assert isinstance(result, tuple)
        finally:
            if original is not None:
                sys.modules["torch"] = original
            else:
                sys.modules.pop("torch", None)


class TestGetSystemFreeGb:
    """get_system_free_gb — システムメモリ取得"""

    def test_returns_positive_float(self):
        """空きメモリが正の数を返すこと"""
        from src.utils.memory_guard import get_system_free_gb

        free = get_system_free_gb()
        assert isinstance(free, float)
        assert free > 0.0, f"空きメモリが 0 以下: {free}"

    def test_psutil_preferred(self):
        """psutil が使える場合は psutil の値を使う"""
        from src.utils.memory_guard import get_system_free_gb

        mock_vmem = MagicMock()
        mock_vmem.available = 50 * 1024**3  # 50 GiB
        with patch("psutil.virtual_memory", return_value=mock_vmem):
            free = get_system_free_gb()
        assert abs(free - 50.0) < 0.01

    def test_fallback_to_proc_meminfo(self):
        """/proc/meminfo へのフォールバック"""
        from src.utils.memory_guard import get_system_free_gb

        fake_meminfo = "MemTotal: 131072000 kB\nMemAvailable: 62500000 kB\n"
        with patch("builtins.open", unittest.mock.mock_open(read_data=fake_meminfo)), \
             patch.dict(sys.modules, {"psutil": None}):
            free = get_system_free_gb()

        expected = 62_500_000 / (1024**2)  # KiB → GiB
        assert abs(free - expected) < 0.1

    def test_returns_inf_when_both_fail(self):
        """両方失敗した場合 float('inf') を返す"""
        from src.utils.memory_guard import get_system_free_gb

        with patch.dict(sys.modules, {"psutil": None}), \
             patch("builtins.open", side_effect=OSError("no proc")):
            free = get_system_free_gb()
        assert free == float("inf")


class TestWaitUntilFreeGb:
    """wait_until_free_gb — メモリ解放待機ロジック"""

    def test_returns_true_immediately_when_sufficient(self):
        """空きメモリが最初から十分ならすぐ True を返す"""
        from src.utils.memory_guard import wait_until_free_gb

        with patch("src.utils.memory_guard.get_system_free_gb", return_value=80.0):
            ok = wait_until_free_gb(target_gb=60.0, timeout=5.0)
        assert ok is True

    def test_returns_false_on_timeout(self):
        """timeout 以内に条件が満たされない場合 False を返す"""
        from src.utils.memory_guard import wait_until_free_gb

        with patch("src.utils.memory_guard.get_system_free_gb", return_value=10.0), \
             patch("time.sleep"):  # sleep をスキップして高速化
            ok = wait_until_free_gb(target_gb=60.0, timeout=0.05, poll_interval=0.01)
        assert ok is False

    def test_returns_true_after_memory_freed(self):
        """ポーリング中にメモリが解放されたら True を返す"""
        from src.utils.memory_guard import wait_until_free_gb

        call_count = 0

        def mock_free_gb():
            nonlocal call_count
            call_count += 1
            return 10.0 if call_count < 3 else 70.0  # 3 回目に解放

        with patch("src.utils.memory_guard.get_system_free_gb", side_effect=mock_free_gb), \
             patch("time.sleep"):
            ok = wait_until_free_gb(target_gb=60.0, timeout=5.0, poll_interval=0.01)
        assert ok is True
        assert call_count >= 3

    def test_does_not_sleep_after_success(self):
        """条件達成後は余計な sleep をしない（最初から十分な場合）"""
        from src.utils.memory_guard import wait_until_free_gb

        with patch("src.utils.memory_guard.get_system_free_gb", return_value=80.0), \
             patch("time.sleep") as mock_sleep:
            wait_until_free_gb(target_gb=60.0, timeout=5.0)
        mock_sleep.assert_not_called()


class TestAssertMemoryHeadroom:
    """assert_memory_headroom — ロード前アサーション"""

    def test_passes_when_sufficient(self):
        """十分なメモリがある場合は例外なし"""
        from src.utils.memory_guard import assert_memory_headroom

        with patch("src.utils.memory_guard.get_system_free_gb", return_value=80.0):
            assert_memory_headroom(required_gb=20.0, label="FLUX test")

    def test_raises_when_insufficient(self):
        """メモリ不足なら RuntimeError を送出"""
        from src.utils.memory_guard import assert_memory_headroom

        with patch("src.utils.memory_guard.get_system_free_gb", return_value=5.0), \
             pytest.raises(RuntimeError) as exc_info:
            assert_memory_headroom(required_gb=20.0, label="FLUX test")

        assert "FLUX test" in str(exc_info.value)
        assert "5." in str(exc_info.value)   # 空き GiB が含まれる
        assert "20." in str(exc_info.value)  # 必要 GiB が含まれる

    def test_error_message_contains_instructions(self):
        """エラーメッセージに vLLM 停止のヒントが含まれる"""
        from src.utils.memory_guard import assert_memory_headroom

        with patch("src.utils.memory_guard.get_system_free_gb", return_value=5.0), \
             pytest.raises(RuntimeError) as exc_info:
            assert_memory_headroom(required_gb=20.0, label="test")

        assert "vLLM" in str(exc_info.value)


# ===========================================================================
# Class B: flush_cuda_memory 統合テスト (CUDA 環境のみ)
# ===========================================================================

@pytest.mark.gpu
class TestFlushCudaMemoryWithCuda:
    """flush_cuda_memory — 実 CUDA 環境での動作確認"""

    def test_returns_tuple_of_floats(self):
        """CUDA 環境で (float, float) を返す"""
        pytest.importorskip("torch")
        from src.utils.memory_guard import flush_cuda_memory

        result = flush_cuda_memory()
        assert isinstance(result, tuple)
        assert len(result) == 2
        reserved, allocated = result
        assert isinstance(reserved, float)
        assert isinstance(allocated, float)

    def test_allocated_nonnegative(self):
        """allocated >= 0"""
        pytest.importorskip("torch")
        from src.utils.memory_guard import flush_cuda_memory

        _, allocated = flush_cuda_memory()
        assert allocated >= 0.0

    def test_reserved_ge_allocated(self):
        """reserved >= allocated (PyTorch の invariant)"""
        pytest.importorskip("torch")
        from src.utils.memory_guard import flush_cuda_memory

        reserved, allocated = flush_cuda_memory()
        assert reserved >= allocated, (
            f"reserved ({reserved:.3f}) < allocated ({allocated:.3f})"
        )

    def test_small_tensor_freed(self):
        """小さなテンソルをアロケートしてから flush した後に allocated が減る"""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA が利用できません")

        from src.utils.memory_guard import flush_cuda_memory

        flush_cuda_memory()  # ベースライン
        _, base = flush_cuda_memory()

        # 100MB のテンソルを確保してすぐ解放
        try:
            t = torch.zeros(100 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        except Exception:
            pytest.skip("CUDA テンソルアロケートに失敗（OOM または未サポートハードウェア）")
        del t
        _, after = flush_cuda_memory()

        assert after <= base + 1.0, (
            f"テンソル解放後のメモリが増加している: base={base:.3f}, after={after:.3f}"
        )


# ===========================================================================
# Class C: モデルクラス unload() 後のメモリログ確認 (軽量)
# ===========================================================================

class TestModelUnloadCallsFlush:
    """各モデルクラスの unload() が flush_cuda_memory() を呼ぶことを確認"""

    def test_image_generator_unload_calls_flush(self):
        """ImageGenerator.unload() → flush_cuda_memory() が呼ばれる"""
        from src.image_generator import ImageGenerator

        gen = ImageGenerator.__new__(ImageGenerator)
        gen.device = "cpu"

        mock_pipe = MagicMock()
        gen._pipe = mock_pipe

        with patch("src.utils.memory_guard.flush_cuda_memory", return_value=(0.0, 0.0)) as mock_flush:
            gen.unload()

        mock_flush.assert_called_once()
        assert gen._pipe is None

    def test_mesh_generator_unload_calls_flush(self):
        """MeshGenerator.unload() → flush_cuda_memory() が呼ばれる"""
        from src.mesh_generator import MeshGenerator

        gen = MeshGenerator.__new__(MeshGenerator)
        gen._pipeline = MagicMock()

        with patch("src.utils.memory_guard.flush_cuda_memory", return_value=(0.0, 0.0)) as mock_flush:
            gen.unload()

        mock_flush.assert_called_once()
        assert gen._pipeline is None

    def test_unload_noop_when_already_none(self):
        """既に None の場合は flush を呼ばない（二重解放防止）"""
        from src.image_generator import ImageGenerator

        gen = ImageGenerator.__new__(ImageGenerator)
        gen.device = "cpu"
        gen._pipe = None

        with patch("src.utils.memory_guard.flush_cuda_memory") as mock_flush:
            gen.unload()

        mock_flush.assert_not_called()


class TestModelLoadChecksHeadroom:
    """モデルクラスの __init__ / load_model が assert_memory_headroom を呼ぶことを確認"""

    def test_image_generator_checks_headroom_before_load(self):
        """ImageGenerator.__init__ がモデルロード前にメモリ確認する"""
        with patch("src.utils.memory_guard.assert_memory_headroom") as mock_assert, \
             patch("torch.cuda.is_available", return_value=False), \
             patch("diffusers.FluxPipeline.from_pretrained", side_effect=RuntimeError("no model")):
            try:
                from src.image_generator import ImageGenerator
                ImageGenerator(model_path="/tmp/fake_model")
            except RuntimeError:
                pass

        mock_assert.assert_called_once()
        args = mock_assert.call_args[0]
        assert args[1] == "FLUX.1-schnell"


    def test_headroom_check_blocks_load_when_oom(self):
        """メモリ不足時に RuntimeError を送出してモデルロードを阻止する"""
        with patch("src.utils.memory_guard.get_system_free_gb", return_value=2.0), \
             patch("diffusers.FluxPipeline.from_pretrained") as mock_load:
            from src.image_generator import ImageGenerator
            with pytest.raises(RuntimeError, match="MemoryGuard"):
                ImageGenerator(model_path="/tmp/fake_model")

        # モデルロードが呼ばれていないことを確認
        mock_load.assert_not_called()


# ===========================================================================
# Class D: stop_vllm_server メモリ待機ロジック (モック)
# ===========================================================================

class TestStopVllmServerMemoryWait:
    """stop_vllm_server() が wait_until_free_gb を呼ぶことを確認"""

    def _make_pgrep_no_process(self):
        """vLLM プロセスが存在しない pgrep モック"""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        return mock_result

    def _make_pgrep_with_pid(self, pid: str):
        """vLLM プロセスが存在する pgrep モック"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = pid
        return mock_result

    def test_no_vllm_process_returns_true_without_wait(self):
        """vLLM が起動していない場合はメモリ待機せず True を返す"""
        from scripts.run_pipeline import stop_vllm_server

        with patch("subprocess.run", return_value=self._make_pgrep_no_process()), \
             patch("src.utils.memory_guard.wait_until_free_gb") as mock_wait:
            result = stop_vllm_server()

        assert result is True
        mock_wait.assert_not_called()

    def test_vllm_killed_then_waits_for_memory(self):
        """vLLM プロセスを SIGTERM で停止した後に wait_until_free_gb を呼ぶ"""
        from scripts.run_pipeline import stop_vllm_server

        # _VLLM_PATTERNS が 2 パターンあるため、
        # 初期スキャン(2回) + _any_vllm_alive() 内(2回) = 計4回の subprocess.run が必要
        pgrep_responses = [
            self._make_pgrep_with_pid("12345"),  # 初期スキャン パターン1: 存在
            self._make_pgrep_no_process(),        # 初期スキャン パターン2: 存在せず
            self._make_pgrep_no_process(),        # _any_vllm_alive() パターン1: 消滅済み
            self._make_pgrep_no_process(),        # _any_vllm_alive() パターン2: 消滅済み
        ]

        with patch("subprocess.run", side_effect=pgrep_responses), \
             patch("os.kill"), \
             patch("time.sleep"), \
             patch("src.utils.memory_guard.wait_until_free_gb", return_value=True) as mock_wait:
            result = stop_vllm_server()

        assert result is True
        mock_wait.assert_called_once()
        # target_gb が 60 以上であることを確認
        kwargs = mock_wait.call_args
        target_gb = kwargs[1].get("target_gb") or kwargs[0][0]
        assert target_gb >= 60.0, f"target_gb が小さすぎる: {target_gb}"

    def test_wait_timeout_does_not_abort_pipeline(self):
        """wait_until_free_gb がタイムアウトしても停止処理は True を返す（警告のみ）"""
        from scripts.run_pipeline import stop_vllm_server

        # _VLLM_PATTERNS が 2 パターンあるため計4回必要
        pgrep_responses = [
            self._make_pgrep_with_pid("12345"),
            self._make_pgrep_no_process(),
            self._make_pgrep_no_process(),
            self._make_pgrep_no_process(),
        ]

        with patch("subprocess.run", side_effect=pgrep_responses), \
             patch("os.kill"), \
             patch("time.sleep"), \
             patch("src.utils.memory_guard.wait_until_free_gb", return_value=False):
            result = stop_vllm_server()

        # タイムアウトしても True: パイプラインを続行させる（警告は出す）
        assert result is True


# ===========================================================================
# 実行
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
