#!/usr/bin/env python3
"""
カテゴリ別合格率測定オーケストレータ

各カテゴリから均等に N 枚ずつ画像を生成・QA し、Wilson 95% CI で合格率を測定する。
測定結果から hard_suitcase の最終合格割合が目標値になる sampling 重みを推奨する。

使用例:
    # 小規模テスト（5枚/カテゴリ、合計 ~60枚）
    python scripts/measure_pass_rates.py --samples 5

    # 推奨スケール（20枚/カテゴリ、精度 ±22%、合計 ~240枚）
    python scripts/measure_pass_rates.py --samples 20

    # 既存 QA 結果から直接レポート生成（パイプライン実行をスキップ）
    python scripts/measure_pass_rates.py --skip-pipeline

    # 目標割合を指定
    python scripts/measure_pass_rates.py --samples 20 --target-hs-fraction 0.70

    # vLLM 自動起動・停止を無効化（手動で vLLM を管理する場合）
    python scripts/measure_pass_rates.py --no-vllm-auto-start

注意:
    - 既存 outputs/ が上書きされます。事前に backup_outputs.py で退避してください。
    - Qwen3-VL-32B は start_vllm_server.sh で起動済みか、--no-vllm-auto-start を付けて
      手動起動してください。
    - DGX Spark では mesh ステップ（TRELLIS.2）はスキップされます。
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from omegaconf import OmegaConf


# -----------------------------------------------------------------------
# 統計表示ヘルパー
# -----------------------------------------------------------------------

def _print_precision_table(n_per_cat: int, n_cats: int) -> None:
    """指定 N に対する統計精度と、他の精度レベルに必要な N を表示する"""
    logger.info("=" * 65)
    logger.info(f"統計精度ガイド（カテゴリ数: {n_cats}）")
    logger.info(f"{'精度':<12} {'N/カテゴリ':<12} {'総枚数':<10} {'生成時間':<10} {'QA時間':<10} {'合計'}")
    logger.info("-" * 65)
    for eps_pct, n_req in [(5, 385), (10, 97), (15, 43), (20, 25)]:
        total = n_req * n_cats
        gen_h = round(total * 35 / 3600, 1)
        qa_h = round(total * 100 / 3600, 1)
        marker = " ← 現在" if n_req <= n_per_cat else ""
        logger.info(
            f"  ±{eps_pct}%{'':<7} {n_req:<12} {total:<10} {gen_h}h{'':<6} {qa_h}h{'':<6} "
            f"{round(gen_h+qa_h, 1)}h{marker}"
        )
    eps_actual = math.ceil(1.96 * math.sqrt(0.25 / max(n_per_cat, 1)) * 100)
    logger.info("-" * 65)
    logger.info(
        f"  現在設定: N={n_per_cat}/カテゴリ × {n_cats} = {n_per_cat*n_cats} 枚、"
        f"達成精度 ±{eps_actual}% (95% CI, worst case p=0.5)"
    )
    logger.info("=" * 65)


# -----------------------------------------------------------------------
# vLLM 管理（run_pipeline.py と同じロジックを再利用）
# -----------------------------------------------------------------------

def _check_vllm(base_url: str = "http://localhost:8001/v1", timeout: int = 3) -> bool:
    try:
        import requests
        r = requests.get(f"{base_url.replace('/v1', '')}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _start_vllm(script_path: str = "scripts/start_vllm_server.sh", max_wait: int = 900) -> bool:
    logger.info(f"vLLM サーバーを起動中（最大待機時間: {max_wait}秒）...")
    import requests  # noqa: F401 — ensure import available before long wait
    try:
        log_file = Path("outputs/logs/vllm_startup_measure.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as logf:
            proc = subprocess.Popen(["bash", script_path], stdout=logf, stderr=subprocess.STDOUT)
        logger.info(f"vLLM プロセス起動（PID: {proc.pid}）")
        start = time.time()
        while time.time() - start < max_wait:
            if _check_vllm():
                logger.info(f"✅ vLLM 起動完了（{int(time.time()-start)}秒）")
                return True
            elapsed = int(time.time() - start)
            if elapsed % 30 == 0:
                logger.info(f"  起動待機中... ({elapsed}秒経過)")
            time.sleep(3)
        if _check_vllm():
            return True
        logger.warning(f"⚠️  vLLM が {max_wait} 秒以内に起動しませんでした")
        return False
    except Exception as e:
        logger.warning(f"⚠️  vLLM 起動失敗: {e}")
        return False


def _stop_vllm() -> None:
    import os
    import signal
    try:
        from src.utils.memory_guard import wait_until_free_gb
        _PATTERNS = ["vllm serve", "vllm.entrypoints.openai.api_server"]
        all_pids: set[str] = set()
        for pat in _PATTERNS:
            r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                all_pids.update(p for p in r.stdout.strip().split("\n") if p)
        if not all_pids:
            logger.info("vLLM サーバーは起動していません")
            return
        logger.info(f"  vLLM プロセス {sorted(all_pids)} を停止中...")
        for pid in all_pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _ in range(15):
            time.sleep(1)
            alive = any(
                subprocess.run(["pgrep", "-f", p], capture_output=True).returncode == 0
                for p in _PATTERNS
            )
            if not alive:
                break
        wait_until_free_gb(target_gb=60.0, timeout=60.0)
        logger.info("✅ vLLM 停止・メモリ解放完了")
    except Exception as e:
        logger.warning(f"⚠️  vLLM 停止中にエラー: {e}")


# -----------------------------------------------------------------------
# プロンプト生成（均等サンプリング）
# -----------------------------------------------------------------------

def _generate_uniform_prompts(cfg_dir: str, n_per_cat: int, output_path: Path) -> list[dict]:
    """各カテゴリから均等に n_per_cat 件のプロンプトを生成して保存する"""
    from src.prompt_generator import PromptGenerator
    logger.info(f"均等サンプリングプロンプト生成: {n_per_cat} 件/カテゴリ")
    gen = PromptGenerator(cfg_dir)
    prompts = gen.generate_uniform_per_category(n_per_cat)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    logger.info(f"プロンプト保存: {output_path} ({len(prompts)} 件)")
    return prompts


# -----------------------------------------------------------------------
# パイプライン実行（pipeline.py の run() を利用）
# -----------------------------------------------------------------------

def _run_pipeline_steps(cfg, steps: list[str], resume: bool = True) -> dict:
    from src.pipeline import AL3DGPipeline
    pipeline = AL3DGPipeline(cfg)
    return pipeline.run(steps=steps, resume=resume)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AL3DG カテゴリ別合格率測定",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=20,
        help="カテゴリあたりの測定サンプル数 (デフォルト: 20、精度 ±22%%)",
    )
    parser.add_argument(
        "--target-hs-fraction",
        type=float,
        default=0.70,
        help="最終合格のうち hard_suitcase が占める目標割合 (デフォルト: 0.70)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="レポート出力ディレクトリ (デフォルト: outputs/reports)",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline_config.yaml",
        help="パイプライン設定ファイル (デフォルト: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--qa-results",
        default="outputs/images_approved/image_qa_results.json",
        help="既存 QA 結果 JSON（--skip-pipeline 時に使用）",
    )
    parser.add_argument(
        "--prompts-output",
        default="outputs/prompts/prompts.json",
        help="プロンプト保存先 JSON パス (デフォルト: outputs/prompts/prompts.json)",
    )
    parser.add_argument(
        "--no-vllm-auto-start",
        action="store_true",
        default=False,
        help="vLLM サーバーの自動起動・停止を無効化",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        default=False,
        help="パイプライン実行をスキップし、既存 QA 結果からレポートのみ生成する",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    log_path = Path("outputs/logs/measure_pass_rates_{time:YYYY-MM-DD_HH-mm-ss}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="DEBUG", rotation="100 MB")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"設定ファイルが見つかりません: {cfg_path}")
        return 1
    cfg = OmegaConf.load(str(cfg_path))

    vllm_base_url = cfg.get("models", {}).get("vlm", {}).get("base_url", "http://localhost:8001/v1")
    cfg_dir = str(cfg_path.parent)

    # --skip-pipeline: 既存 QA 結果からレポートのみ生成
    if args.skip_pipeline:
        logger.info("--skip-pipeline: パイプラインをスキップしてレポート生成のみ実行")
        _run_report(args, cfg_path)
        return 0

    # カテゴリ数を取得して統計ガイドを表示
    try:
        from omegaconf import OmegaConf as OC
        cats_cfg = OC.load(str(cfg_path.parent / "luggage_categories.yaml"))
        n_cats = len(OC.to_container(cats_cfg.categories))
    except Exception:
        n_cats = 12  # デフォルト
    _print_precision_table(args.samples, n_cats)

    # -----------------------------------------------------------------------
    # Step 1: プロンプト生成（均等サンプリング）
    # -----------------------------------------------------------------------
    logger.info("[Step 1/3] プロンプト生成（均等サンプリング）")

    # プロンプト生成は組合せ生成のみで vLLM 不要 → 起動中なら停止してメモリ解放
    if not args.no_vllm_auto_start:
        _stop_vllm()

    prompts_path = Path(args.prompts_output)
    prompts = _generate_uniform_prompts(cfg_dir, args.samples, prompts_path)

    if not prompts:
        logger.error("プロンプト生成に失敗しました")
        return 1

    # -----------------------------------------------------------------------
    # Step 2: 画像生成（FLUX.1-schnell）
    # -----------------------------------------------------------------------
    logger.info("[Step 2/3] 画像生成（FLUX.1-schnell）")

    try:
        step_results = _run_pipeline_steps(cfg, steps=["image"], resume=False)
        logger.info(f"  画像生成完了: {step_results}")
    except KeyboardInterrupt:
        logger.warning("中断されました (Ctrl-C)")
        return 130
    except Exception as e:
        logger.exception(f"  画像生成エラー: {e}")
        return 1

    # -----------------------------------------------------------------------
    # Step 3: 画像 QA（Qwen3-VL-32B /think モード）
    # -----------------------------------------------------------------------
    logger.info("[Step 3/3] 画像 QA（Qwen3-VL-32B）")

    # QA 前に vLLM を起動（停止していた場合は再起動）
    if not _check_vllm(vllm_base_url):
        logger.info("vLLM サーバーを起動します...")
        if not _start_vllm():
            logger.error("vLLM の起動に失敗しました")
            return 1

    try:
        step_results = _run_pipeline_steps(cfg, steps=["image_qa"], resume=False)
        logger.info(f"  画像 QA 完了: {step_results}")
    except KeyboardInterrupt:
        logger.warning("中断されました (Ctrl-C)")
        return 130
    except Exception as e:
        logger.exception(f"  画像 QA エラー: {e}")
        return 1

    # QA 完了後に vLLM を停止
    if not args.no_vllm_auto_start:
        _stop_vllm()

    # -----------------------------------------------------------------------
    # レポート生成
    # -----------------------------------------------------------------------
    _run_report(args, cfg_path)
    return 0


def _run_report(args: argparse.Namespace, cfg_path: Path) -> None:
    """generate_report.py の analyze_pass_rates() を呼び出してレポートを生成する"""
    from scripts.generate_report import analyze_pass_rates

    output_dir = Path(args.output_dir)
    report = analyze_pass_rates(
        qa_results_path=Path(args.qa_results),
        prompts_path=Path(args.prompts_output),
        target_hs_fraction=args.target_hs_fraction,
        output_dir=output_dir,
        config_path=cfg_path,
    )
    if not report:
        logger.error("レポート生成に失敗しました")
        return

    meta = report["metadata"]
    guide = report["precision_guide"]
    cats = report["categories"]
    rec = report["weight_recommendations"]

    logger.info("=" * 65)
    logger.info("合格率測定レポート")
    logger.info(f"  全体合格率: {round(meta['overall_pass_rate']*100, 1)}% "
                f"({meta['total_passed']}/{meta['total_generated']})")
    logger.info(f"  統計精度: ±{guide['achieved_precision_pct']}% (95% CI)")
    logger.info("")
    logger.info(f"  {'カテゴリ':<22} {'合格/総数':<12} {'合格率':<10} {'95% CI'}")
    logger.info("  " + "-" * 60)
    for cat, v in sorted(cats.items(), key=lambda x: -x[1]["pass_rate"]):
        ci = f"[{round(v['ci_lower']*100, 1)}%, {round(v['ci_upper']*100, 1)}%]"
        logger.info(f"  {cat:<22} {v['passed']}/{v['samples']:<10} "
                    f"{round(v['pass_rate']*100, 1)}%{'':<6} {ci}")
    logger.info("")
    logger.info("推奨 category_weights:")
    for cat, w in sorted(rec["recommended_weights"].items(), key=lambda x: -x[1]):
        cur = cats.get(cat, {}).get("current_weight")
        cur_str = f"{cur:.4f}" if cur is not None else "—"
        logger.info(f"  {cat:<22} {cur_str} → {w:.4f}")
    logger.info("=" * 65)
    logger.info(f"レポート: {output_dir / 'pass_rate_report.json'}")
    logger.info(f"レポート: {output_dir / 'pass_rate_report.html'}")


if __name__ == "__main__":
    sys.exit(main())
