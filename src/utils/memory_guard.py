"""
GPU / 統合メモリ管理ユーティリティ

DGX Spark (Grace Blackwell GB10, 128GB 統合メモリ) でのモデル逐次ロード制約を
サポートする。

## 背景と解決する問題

GB10 は CPU と GPU が同一の 128GB 物理メモリプールを共有する。
外部プロセス (vLLM) がメモリを保持している状態でモデルをロードしようとすると
OOM になるが、以下の理由でプロセス終了 ≠ メモリ即時解放 となる:

  1. プロセス終了後、OS がページを回収するまでわずかな遅延がある
  2. Python の del + gc.collect() + torch.cuda.empty_cache() だけでは
     参照サイクルを確実に破壊できない場合がある
  3. torch.cuda.memory_reserved() は自プロセスの PyTorch アロケータしか
     反映しないため、別プロセス (vLLM) の占有分は見えない

## 提供する機能

  flush_cuda_memory()
    — Python GC (複数パス) + CUDA キャッシュクリア + 同期を実行する。
    — 全 unload() メソッドから呼び出す統一実装。

  get_system_free_gb()
    — システム全体の空きメモリ (GiB) を返す。
    — GB10 統合メモリ環境では vLLM 等の別プロセスの解放分も反映される。

  wait_until_free_gb(target_gb, timeout, poll_interval)
    — 空きメモリが target_gb 以上になるまでポーリングする。
    — vLLM 停止後にモデルロード前の待機として使用する。

  assert_memory_headroom(required_gb, label)
    — 必要メモリが空いていることをアサートする。
    — モデルロード直前に呼び出してロード中途の OOM を未然に防ぐ。
"""

import gc
import time

from loguru import logger

# 各モデルのロードに必要な最小空きメモリ (GiB)
# 実際のモデルサイズより余裕を持たせた値 (ロード中の一時バッファ込み)
REQUIRED_GB = {
    "FLUX.1-schnell": 20.0,    # BF16 ~12GB + ロード中 CPU ステージング
    "TRELLIS.2-4B":   35.0,    # BF16 ~24GB + ロード中バッファ
    "Qwen3-VL-32B":  110.0,    # 実使用 ~100GB（モデル重量 ~65GB + KV キャッシュ + CUDA グラフ）
    "OpenCLIP":        5.0,    # BF16 ~2GB  + 余裕
}


# ---------------------------------------------------------------------------
# CUDA メモリクリーンアップ
# ---------------------------------------------------------------------------

def flush_cuda_memory() -> tuple[float, float]:
    """
    Python GC (複数パス) + CUDA アロケータキャッシュクリア + 同期を実行する。

    参照サイクルを確実に破壊するために gc.collect() を 2 回実行する。
    CUDA キャッシュクリア前後に synchronize() を挟み、カーネルが完了してから
    キャッシュを解放する。

    Returns:
        (reserved_gb, allocated_gb) — クリア後の PyTorch CUDA メモリ状態。
        CUDA が使えない場合は (0.0, 0.0) を返す。
    """
    # 複数パスで参照サイクルを破壊
    gc.collect()
    gc.collect()

    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0

        # 進行中の CUDA 操作を完了させてからキャッシュを解放
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
        allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
        return reserved_gb, allocated_gb

    except Exception as e:
        logger.debug(f"flush_cuda_memory: CUDA 操作中に例外 (無視): {e}")
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# システムメモリ確認
# ---------------------------------------------------------------------------

def get_system_free_gb() -> float:
    """
    システム全体の空きメモリ (GiB) を返す。

    GB10 統合メモリ環境では CPU / GPU が同一プールを共有するため、
    この値は vLLM 等の別プロセスが保持するメモリも反映する。

    優先順: psutil > /proc/meminfo > float('inf') (判定不能)
    """
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        pass

    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)  # KiB → GiB
    except Exception:
        pass

    logger.warning("空きメモリ取得不能: psutil / /proc/meminfo が利用できません")
    return float("inf")


# ---------------------------------------------------------------------------
# 待機
# ---------------------------------------------------------------------------

def wait_until_free_gb(
    target_gb: float,
    timeout: float = 60.0,
    poll_interval: float = 2.0,
) -> bool:
    """
    システムの空きメモリが target_gb (GiB) 以上になるまで待機する。

    vLLM などの外部プロセス停止後、OS がメモリを回収するまでの遅延を吸収する。

    Args:
        target_gb:     要求する最小空きメモリ (GiB)
        timeout:       最大待機時間 (秒, デフォルト 60)
        poll_interval: ポーリング間隔 (秒, デフォルト 2)

    Returns:
        True  — target_gb 以上の空きメモリが確認できた
        False — timeout 秒以内に条件が満たされなかった
    """
    deadline = time.monotonic() + timeout
    logged_at = -1.0  # 最後にログを出した経過秒

    while True:
        free = get_system_free_gb()
        if free >= target_gb:
            logger.info(
                f"メモリ解放確認: {free:.1f} GiB 空き "
                f"(要求: {target_gb:.1f} GiB)"
            )
            return True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        elapsed = timeout - remaining
        # 10 秒ごとにログを出す (ポーリングのたびに出すとうるさい)
        if elapsed - logged_at >= 10.0:
            logger.info(
                f"メモリ解放待機中: {free:.1f} GiB 空き / "
                f"{target_gb:.1f} GiB 要求 ({elapsed:.0f}s 経過)"
            )
            logged_at = elapsed

        time.sleep(min(poll_interval, remaining))

    free = get_system_free_gb()
    logger.warning(
        f"メモリ解放タイムアウト ({timeout:.0f}s): "
        f"{free:.1f} GiB 空き / {target_gb:.1f} GiB 要求。"
        "続行しますが OOM が発生する可能性があります。"
    )
    return False


# ---------------------------------------------------------------------------
# 事前アサーション
# ---------------------------------------------------------------------------

def assert_memory_headroom(required_gb: float, label: str = "model") -> None:
    """
    モデルロード直前に十分な空きメモリがあることを確認する。

    空きメモリが required_gb を下回る場合は RuntimeError を送出し、
    ロード中途での OOM クラッシュを未然に防ぐ。

    Args:
        required_gb: 必要な最小空きメモリ (GiB)
        label:       エラーメッセージに表示するモデル名
    """
    free = get_system_free_gb()
    if free < required_gb:
        raise RuntimeError(
            f"[MemoryGuard] {label} のロードに必要なメモリが不足しています。\n"
            f"  空きメモリ : {free:.1f} GiB\n"
            f"  必要メモリ : {required_gb:.1f} GiB\n"
            "他のモデルやプロセスがメモリを占有している可能性があります。\n"
            "vLLM が起動中の場合は停止してからロードしてください。"
        )
    logger.info(
        f"[MemoryGuard] {label} ロード前確認 OK: "
        f"{free:.1f} GiB 空き (必要: {required_gb:.1f} GiB)"
    )
