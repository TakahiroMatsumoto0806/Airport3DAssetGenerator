#!/usr/bin/env python3
"""
T-6.2: フルパイプライン実行スクリプト

全ステップを順番に実行する。
DGX Spark での標準的な使用例:

    # 全ステップ実行
    python scripts/run_pipeline.py

    # 設定ファイルを指定
    python scripts/run_pipeline.py --config configs/pipeline_config.yaml

    # 特定ステップのみ実行
    python scripts/run_pipeline.py --steps prompt image image_qa

    # 中断再開なし（全件再実行）
    python scripts/run_pipeline.py --no-resume

注意:
    - T-3.1 (mesh ステップ) は x86_64+RTX5090 別PCで実行する。
      DGX Spark では mesh ステップをスキップし、meshes_raw/ に GLB を転送してから続行。
    - Qwen3-VL-32B は別ターミナルで vLLM サーバーを起動しておくこと:
        bash scripts/start_vllm_server.sh
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from omegaconf import OmegaConf

from src.pipeline import AL3DGPipeline


def check_vllm_server(base_url: str = "http://localhost:8001/v1", timeout: int = 3) -> bool:
    """vLLM サーバーが起動しているか確認"""
    try:
        import requests
        response = requests.get(f"{base_url.replace('/v1', '')}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def start_vllm_server(script_path: str = "scripts/start_vllm_server.sh", max_wait: int = 900) -> bool:
    """vLLM サーバーを起動

    Args:
        script_path: vLLM起動スクリプトのパス
        max_wait: 最大待機時間（秒、デフォルト: 900秒 = 15分）
                  Qwen3-VL-32B (実使用 ~100GB) の起動には 600〜700秒かかることがある。
    """
    logger.info(f"vLLM サーバーを起動中（最大待機時間: {max_wait}秒）...")
    try:
        # ログファイルを準備
        log_file = Path("outputs/logs") / "vllm_startup.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # サーバーを起動（ログ出力を記録）
        with open(log_file, "w") as logf:
            proc = subprocess.Popen(
                ["bash", script_path],
                stdout=logf,
                stderr=subprocess.STDOUT,
            )

        logger.info(f"vLLM プロセス起動（PID: {proc.pid}）")

        # サーバーが起動するまで待機
        start_time = time.time()
        check_interval = 3  # チェック間隔を3秒に増加
        while time.time() - start_time < max_wait:
            if check_vllm_server():
                elapsed = int(time.time() - start_time)
                logger.info(f"✅ vLLM サーバーが起動しました（{elapsed}秒）")
                return True
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0:  # 30秒ごとにログ出力
                logger.info(f"  起動待機中... ({elapsed}秒経過)")
            time.sleep(check_interval)

        logger.warning(f"⚠️  vLLM サーバーが {max_wait} 秒以内に起動しませんでした")
        # 最終確認（ループ終了直前に起動が完了した可能性があるため）
        if check_vllm_server():
            elapsed = int(time.time() - start_time)
            logger.info(f"✅ vLLM サーバーが起動しました（最終確認: {elapsed}秒）")
            return True
        logger.warning(f"  起動ログを確認: {log_file}")
        if log_file.exists():
            with open(log_file, "r") as f:
                log_tail = f.read()[-500:]  # 最後の500文字
                if log_tail:
                    logger.warning(f"  ログの最後: {log_tail}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  vLLM サーバーの起動に失敗: {e}")
        return False


def stop_vllm_server() -> bool:
    """vLLM サーバーを停止し、メモリが完全に解放されるまで待機する。

    FLUX.1-schnell など別モデルをロードする前に呼び出す。
    （CLAUDE.md: モデル同時ロード禁止、逐次ロード戦略を厳守）

    プロセス終了の確認だけでは不十分: OS が CUDA メモリを回収するまでに
    わずかな遅延があるため、プロセス消滅後にシステムメモリが解放されるまで
    ポーリングで確認する。
    """
    import os
    import signal
    from src.utils.memory_guard import wait_until_free_gb

    # vLLM 停止後に要求する最小空きメモリ (GiB)
    # vLLM が ~100GB (モデル重量 ~65GB + KV キャッシュ + CUDA グラフ) 使用するため、
    # 解放後は 60GB 以上空くはず
    _VLLM_FREE_THRESHOLD_GB = 60.0
    _VLLM_MEMORY_TIMEOUT_S  = 60.0

    logger.info("vLLM サーバーを停止中（メモリ解放のため）...")
    try:
        # vLLM プロセスを複数パターンで検索する。
        # start_vllm_server.sh 経由（vllm serve CLI）と python -m 経由の両方に対応。
        #   パターン1: "vllm serve"  — vllm CLI から起動した親プロセス
        #   パターン2: "vllm.entrypoints.openai.api_server"  — python -m から起動した場合
        _VLLM_PATTERNS = [
            "vllm serve",
            "vllm.entrypoints.openai.api_server",
        ]
        all_pids: set[str] = set()
        for pat in _VLLM_PATTERNS:
            r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                all_pids.update(p for p in r.stdout.strip().split('\n') if p)

        if not all_pids:
            logger.info("vLLM サーバーは起動していませんでした")
            return True

        logger.info(f"  vLLM プロセス {sorted(all_pids)} を検出")
        for pid in all_pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                logger.info(f"  vLLM プロセス (PID: {pid}) に SIGTERM を送信")
            except ProcessLookupError:
                pass

        def _any_vllm_alive() -> bool:
            for pat in _VLLM_PATTERNS:
                r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
                if r.returncode == 0 and r.stdout.strip():
                    return True
            return False

        # プロセス終了を待機（最大 15 秒）
        process_dead = False
        for _ in range(15):
            time.sleep(1)
            if not _any_vllm_alive():
                process_dead = True
                break

        if not process_dead:
            # 15 秒で終了しなければ SIGKILL
            remaining: set[str] = set()
            for pat in _VLLM_PATTERNS:
                r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
                if r.returncode == 0 and r.stdout.strip():
                    remaining.update(p for p in r.stdout.strip().split('\n') if p)
            for pid in remaining:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info(f"  vLLM プロセス (PID: {pid}) を強制終了")
                except ProcessLookupError:
                    pass
            time.sleep(2)

        logger.info("vLLM プロセス終了を確認。システムメモリ解放を待機中...")

        # プロセス消滅 ≠ メモリ即時解放。
        # GB10 統合メモリでは OS がページを回収するまで遅延が生じる。
        # ポーリングで空きメモリが閾値を超えるまで待つ。
        wait_until_free_gb(
            target_gb=_VLLM_FREE_THRESHOLD_GB,
            timeout=_VLLM_MEMORY_TIMEOUT_S,
        )

        logger.info("✅ vLLM サーバーを停止しました")
        return True
    except Exception as e:
        logger.warning(f"⚠️  vLLM サーバーの停止に失敗: {e}")
        return False


# vLLM が必要なステップ / 不要なステップ
_STEPS_NEEDING_VLLM = {"prompt", "image_qa", "mesh_vlm_qa"}
_STEPS_NOT_NEEDING_VLLM = {"image", "mesh", "mesh_qa", "physics", "sim_export"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AL3DG フルパイプライン実行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline_config.yaml",
        help="設定ファイルパス (デフォルト: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--steps", "-s",
        nargs="+",
        choices=["prompt", "image", "image_qa", "mesh", "mesh_qa",
                 "mesh_vlm_qa", "physics", "sim_export"],
        default=None,
        help="実行するステップ (デフォルト: 設定ファイルの steps セクションに従う)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="中断再開を無効化して全件再実行する",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログレベル (デフォルト: INFO)",
    )
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=None,
        help="生成するプロンプトの総数 (デフォルト: 設定ファイルの prompt_generate_number)",
    )
    parser.add_argument(
        "--no-vllm-auto-start",
        action="store_true",
        default=False,
        help="vLLM サーバーの自動起動・停止を無効化",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ログ設定
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.add(
        "outputs/logs/pipeline_{time:YYYY-MM-DD_HH-mm-ss}.log",
        level="DEBUG",
        rotation="100 MB",
    )

    # 設定読み込み
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"設定ファイルが見つかりません: {cfg_path}")
        return 1

    cfg = OmegaConf.load(str(cfg_path))
    logger.info(f"設定ファイル: {cfg_path}")

    # コマンドラインオプションで設定を上書き
    if args.prompt_count is not None:
        logger.info(f"プロンプト生成数を上書き: {args.prompt_count}")
        cfg.generation.prompt_generate_number = args.prompt_count

    # 実行ステップを決定
    vllm_enabled = cfg.get("vllm", {}).get("enabled", True)
    vllm_base_url = cfg.get("models", {}).get("vlm", {}).get("base_url", "http://localhost:8001/v1")
    steps_to_run = args.steps or [
        step for step, enabled in cfg.steps.items() if enabled
    ]

    # CLAUDE.md: モデル同時ロード禁止、逐次ロード戦略を厳守
    # ステップごとに vLLM サーバーを起動/停止してメモリを管理する
    pipeline = AL3DGPipeline(cfg)
    results: dict = {}

    for step in steps_to_run:
        needs_vllm_for_step = vllm_enabled and step in _STEPS_NEEDING_VLLM

        if not args.no_vllm_auto_start:
            if needs_vllm_for_step:
                # このステップで vLLM が必要 → 起動確認
                if not check_vllm_server(vllm_base_url):
                    logger.info(f"[{step}] vLLM サーバーを起動します...")
                    if not start_vllm_server():
                        logger.warning(f"[{step}] vLLM サーバーの起動に失敗しました。ステップをスキップします。")
                        continue
                    logger.info(f"[{step}] vLLM サーバーの起動に成功しました")
                else:
                    logger.info(f"[{step}] ✅ vLLM サーバーは既に起動しています")
            else:
                # このステップで vLLM 不要 → プロセスを確実に停止してメモリ解放
                # （ヘルスチェック不要: 起動途中のプロセスも含めて強制終了する）
                logger.info(f"[{step}] vLLM プロセスを停止します（逐次ロード戦略）...")
                stop_vllm_server()

        # ステップ実行
        try:
            step_results = pipeline.run(
                steps=[step],
                resume=not args.no_resume,
            )
            results.update(step_results)
        except KeyboardInterrupt:
            logger.warning("パイプラインが中断されました (Ctrl-C)")
            return 130
        except Exception as e:
            logger.exception(f"[{step}] ステップ実行エラー: {e}")
            logger.warning(f"[{step}] エラーが発生しましたが、次のステップへ続行します")

    # 結果サマリー表示
    logger.info("=" * 60)
    logger.info("パイプライン完了サマリー:")
    for step, result in results.items():
        logger.info(f"  {step}: {result}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
