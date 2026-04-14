"""
T-0.3: GPU 動作確認テスト

各モデルが正常に推論できることを確認する。

実行方法（DGX Spark — aarch64）:
    # FLUX.1 + Qwen3 のみ（DGX Spark 標準）
    python tests/test_gpu_models.py
    python tests/test_gpu_models.py --models flux qwen

    # vLLM サーバーを自動起動してテスト（Qwen）
    python tests/test_gpu_models.py --models qwen --start-vllm

実行方法（x86_64 別 PC — TRELLIS.2 専用）:
    conda activate trellis2
    python tests/test_gpu_models.py --models trellis

確認内容（DGX Spark）:
    - FLUX.1-schnell で 1 枚画像生成（~12GB 使用確認）
    - Qwen3-VL-32B-Instruct でテキスト生成・画像理解（~65GB 使用確認, vLLM 経由）
    - 各ステップで GPU メモリ解放を確認

確認内容（x86_64 別 PC）:
    - TRELLIS.2-4B で 1 つの 3D モデル生成（~24GB 使用確認）

注意:
    - TRELLIS.2 は x86_64 + RTX 5090 専用。DGX Spark (aarch64) では動作しない
    - Qwen テストは vLLM サーバー (http://localhost:8001) が起動済みであること
      または --start-vllm フラグを使用すること
"""

import argparse
import base64
import gc
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# プロジェクトルートを sys.path に追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import setup_logger

from loguru import logger

setup_logger(log_file=PROJECT_ROOT / "outputs" / "reports" / "test_gpu_models.log")

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path.home() / "models"))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
MAX_MEMORY_GB = 128.0

# モデルパス解決（ローカルが優先、なければ HF ID で fallback）
def _resolve(local_name: str, hf_id: str) -> str:
    local = MODELS_DIR / local_name
    return str(local) if local.exists() else hf_id


# ============================================================
# GPU メモリユーティリティ
# ============================================================

def get_gpu_memory_gb() -> float:
    """現在の GPU メモリ使用量 (GB) を返す（torch 未ロード時も動作）"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        total_mib = sum(int(x.strip()) for x in result.stdout.strip().splitlines() if x.strip())
        return total_mib / 1024
    except Exception:
        return 0.0


def log_memory(label: str) -> float:
    used_gb = get_gpu_memory_gb()
    logger.info(f"[GPU MEM] {label}: {used_gb:.1f} GB")
    return used_gb


def free_torch_memory(obj=None) -> None:
    """torch オブジェクトを解放して GPU メモリをクリア"""
    try:
        import torch
        if obj is not None:
            del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    time.sleep(2)  # メモリ解放を待つ


# ============================================================
# テスト 1: FLUX.1-schnell
# ============================================================

def test_flux() -> dict:
    """FLUX.1-schnell で 1 枚画像を生成し、メモリ使用量を確認する"""
    logger.info("=" * 60)
    logger.info("TEST 1: FLUX.1-schnell 画像生成")
    logger.info("=" * 60)

    result = {"name": "FLUX.1-schnell", "passed": False, "peak_gb": 0.0, "error": None}

    mem_before = log_memory("FLUX ロード前")

    pipe = None
    try:
        import torch
        from diffusers import FluxPipeline

        model_path = _resolve("FLUX.1-schnell", "black-forest-labs/FLUX.1-schnell")
        logger.info(f"モデルパス: {model_path}")

        pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")

        mem_loaded = log_memory("FLUX ロード後")

        logger.info("画像生成中 (num_inference_steps=4, guidance_scale=0.0) ...")
        image = pipe(
            "A black hard-shell suitcase with wheels on white background, studio lighting",
            num_inference_steps=4,
            guidance_scale=0.0,
            height=1024,
            width=1024,
        ).images[0]

        mem_peak = log_memory("FLUX 生成後")
        result["peak_gb"] = mem_peak

        # 検証
        assert image is not None, "生成画像が None"
        assert image.size == (1024, 1024), f"解像度が不正: {image.size}"

        # 一時ファイルに保存して確認
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = Path(f.name)
        image.save(tmp_path)
        assert tmp_path.stat().st_size > 0, "画像ファイルサイズが 0"
        tmp_path.unlink()

        logger.info(f"✓ FLUX.1-schnell: 生成成功 (1024×1024, peak={mem_peak:.1f}GB)")
        result["passed"] = True

    except Exception as e:
        logger.error(f"✗ FLUX.1-schnell: {e}")
        result["error"] = str(e)
    finally:
        free_torch_memory(pipe)
        mem_after = log_memory("FLUX アンロード後")
        logger.info(f"  解放量: {result['peak_gb'] - mem_after:.1f} GB")

    return result


# ============================================================
# テスト 2: TRELLIS.2-4B（x86_64 別 PC 専用）
# ============================================================

def test_trellis() -> dict:
    """
    TRELLIS.2-4B で 1 つの 3D モデルを生成し、メモリ使用量を確認する。

    !! DGX Spark (aarch64) では実行不可 !!
    x86_64 + RTX 5090 の別 PC で以下のように実行する:
        conda activate trellis2
        python tests/test_gpu_models.py --models trellis
    """
    logger.info("=" * 60)
    logger.info("TEST 2: TRELLIS.2-4B 3D 生成（x86_64 別 PC 専用）")
    logger.info("=" * 60)

    result = {"name": "TRELLIS.2-4B", "passed": False, "peak_gb": 0.0, "error": None}

    log_memory("TRELLIS ロード前")

    gen = None
    try:
        # MeshGenerator 経由でテスト（本番コードと同じパスを通す）
        # trellis2 パッケージは conda 環境 "trellis2" 経由でインストール済みであること:
        #   cd ~/trellis2
        #   . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel
        from src.mesh_generator import MeshGenerator
        from PIL import Image

        model_path = _resolve("TRELLIS.2-4B", "microsoft/TRELLIS.2-4B")
        logger.info(f"モデルパス: {model_path}")

        gen = MeshGenerator(model_path)

        mem_loaded = log_memory("TRELLIS ロード後")

        # テスト用の白背景ダミー画像（グレーの直方体）
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_input.png"
            Image.new("RGB", (512, 512), color=(200, 200, 200)).save(str(img_path))

            glb_path = Path(tmpdir) / "test_output.glb"

            logger.info("3D 生成中 (seed=42) ...")
            gen.generate_single(str(img_path), seed=42, output_path=str(glb_path))

            mem_peak = log_memory("TRELLIS 生成後")
            result["peak_gb"] = mem_peak

            assert glb_path.exists(), "GLB ファイルが生成されなかった"
            assert glb_path.stat().st_size > 0, "GLB ファイルサイズが 0"

            import trimesh
            loaded_mesh = trimesh.load(str(glb_path))
            assert loaded_mesh is not None, "trimesh でメッシュを読み込めない"
            logger.info(
                f"  メッシュ頂点数: {len(loaded_mesh.vertices) if hasattr(loaded_mesh, 'vertices') else 'N/A'}"
            )

        logger.info(f"✓ TRELLIS.2-4B: 生成成功 (peak={mem_peak:.1f}GB)")
        result["passed"] = True

    except ImportError as e:
        msg = (
            f"trellis2 パッケージが見つかりません: {e}\n"
            "  このテストは x86_64 別 PC でのみ動作します。\n"
            "  conda 環境 'trellis2' がインストールされているか確認してください:\n"
            "    cd ~/trellis2\n"
            "    . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel\n"
            "  その後 conda activate trellis2 でこのスクリプトを実行してください。"
        )
        logger.error(msg)
        result["error"] = str(e)
    except Exception as e:
        logger.error(f"✗ TRELLIS.2-4B: {e}")
        result["error"] = str(e)
    finally:
        if gen is not None:
            gen.unload()
        mem_after = log_memory("TRELLIS アンロード後")
        logger.info(f"  解放量: {result['peak_gb'] - mem_after:.1f} GB")

    return result


# ============================================================
# テスト 3: Qwen3-VL-32B-Instruct (vLLM サーバー経由)
# ============================================================

def _start_vllm_server() -> subprocess.Popen:
    """vLLM サーバーをバックグラウンドで起動する"""
    model_path = _resolve("Qwen3-VL-32B-Instruct", "Qwen/Qwen3-VL-32B-Instruct")
    logger.info(f"vLLM サーバーを起動: {model_path}")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--dtype", "bfloat16",
            "--max-model-len", "8192",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--trust-remote-code",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    logger.info("vLLM サーバー起動待機中 (最大 120 秒)...")
    import urllib.request
    for _ in range(120):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://localhost:8001/health", timeout=2)
            logger.info("vLLM サーバー起動完了")
            return proc
        except Exception:
            pass
    raise TimeoutError("vLLM サーバーが 120 秒以内に起動しませんでした")


def _image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_qwen(start_vllm: bool = False) -> dict:
    """
    Qwen3-VL-32B-Instruct (vLLM サーバー経由) で以下をテスト:
      1. テキスト生成タスク（プロンプトリファイン相当）
      2. 画像理解タスク（画像検品相当）
    """
    logger.info("=" * 60)
    logger.info("TEST 3: Qwen3-VL-32B-Instruct (vLLM)")
    logger.info("=" * 60)

    result = {
        "name": "Qwen3-VL-32B-Instruct",
        "passed": False,
        "peak_gb": 0.0,
        "text_gen": False,
        "image_understanding": False,
        "error": None,
    }

    vllm_proc = None
    try:
        from openai import OpenAI
        from PIL import Image, ImageDraw

        if start_vllm:
            log_memory("vLLM ロード前")
            vllm_proc = _start_vllm_server()
            log_memory("vLLM ロード後")

        client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")

        # モデル名を vLLM サーバーから取得
        models = client.models.list()
        served_model = models.data[0].id if models.data else "Qwen/Qwen3-VL-32B-Instruct"
        logger.info(f"vLLM サービング中のモデル: {served_model}")

        mem_before = log_memory("テスト開始前")

        # ---- テスト 3a: テキスト生成（プロンプトリファイン相当）----
        logger.info("3a: テキスト生成テスト (/no_think)")
        response = client.chat.completions.create(
            model=served_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "/no_think\n"
                        "Refine the following image generation prompt for FLUX.1-schnell. "
                        "Keep it concise and return only the refined prompt:\n"
                        "A black suitcase on white background"
                    ),
                }
            ],
            max_tokens=128,
            temperature=0.0,
        )
        text_out = response.choices[0].message.content
        assert text_out and len(text_out) > 10, f"テキスト生成結果が短すぎる: {text_out!r}"
        logger.info(f"  出力: {text_out[:100]}...")
        result["text_gen"] = True
        logger.info("✓ テキスト生成 OK")

        # ---- テスト 3b: 画像理解（画像検品相当）----
        logger.info("3b: 画像理解テスト (画像+テキスト)")

        # テスト用ダミー画像を生成（白背景に灰色の矩形）
        img = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([60, 80, 196, 176], fill=(100, 100, 100))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_img_path = Path(f.name)
        img.save(tmp_img_path)

        img_b64 = _image_to_base64(tmp_img_path)
        tmp_img_path.unlink()

        response = client.chat.completions.create(
            model=served_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "/no_think\n"
                                "Evaluate this product image. "
                                "Reply in JSON with keys: realism_score (1-10), "
                                "background_clean (true/false), has_artifacts (true/false). "
                                "Reply only JSON."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=128,
            temperature=0.0,
        )
        vision_out = response.choices[0].message.content
        assert vision_out and len(vision_out) > 5, f"画像理解結果が空: {vision_out!r}"
        logger.info(f"  出力: {vision_out[:200]}")

        # JSON パース確認
        import json, re
        json_match = re.search(r"\{.*\}", vision_out, re.DOTALL)
        assert json_match, f"JSON が見つからない: {vision_out!r}"
        parsed = json.loads(json_match.group())
        assert "realism_score" in parsed, f"realism_score キーがない: {parsed}"
        result["image_understanding"] = True
        logger.info("✓ 画像理解 OK")

        mem_peak = log_memory("Qwen テスト後")
        result["peak_gb"] = mem_peak
        result["passed"] = True
        logger.info(f"✓ Qwen3-VL-32B-Instruct: 全テスト成功 (peak={mem_peak:.1f}GB)")

    except Exception as e:
        logger.error(f"✗ Qwen3-VL-32B-Instruct: {e}")
        result["error"] = str(e)
    finally:
        if vllm_proc is not None:
            logger.info("vLLM サーバーを停止します")
            vllm_proc.terminate()
            vllm_proc.wait(timeout=30)
            free_torch_memory()
            log_memory("vLLM アンロード後")

    return result


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="T-0.3: GPU 動作確認テスト")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["flux", "trellis", "qwen", "all"],
        default=["all"],
        help="テスト対象モデル (デフォルト: all)",
    )
    parser.add_argument(
        "--start-vllm",
        action="store_true",
        help="Qwen テスト前に vLLM サーバーを自動起動する",
    )
    args = parser.parse_args()

    targets = ["flux", "qwen"] if "all" in args.models else args.models

    logger.info("=" * 60)
    logger.info("T-0.3: GPU 動作確認テスト開始")
    logger.info(f"対象モデル: {targets}")
    logger.info(f"MODELS_DIR: {MODELS_DIR}")
    logger.info(f"VLLM_BASE_URL: {VLLM_BASE_URL}")
    logger.info("=" * 60)

    mem_start = log_memory("テスト開始時")

    results: list[dict] = []

    # 逐次ロード戦略：各モデルのテスト後に必ずアンロードしてから次へ
    if "flux" in targets:
        results.append(test_flux())

    if "trellis" in targets:
        results.append(test_trellis())

    if "qwen" in targets:
        results.append(test_qwen(start_vllm=args.start_vllm))

    # ---- サマリ ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("テスト結果サマリ")
    logger.info("=" * 60)

    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        peak = f"{r['peak_gb']:.1f}GB"
        err = f" — {r['error']}" if r.get("error") else ""
        logger.info(f"  [{status}] {r['name']} (peak: {peak}){err}")
        if not r["passed"]:
            all_passed = False

    mem_end = log_memory("テスト終了時")

    # 128GB 上限チェック
    all_peaks = [r["peak_gb"] for r in results if r["peak_gb"] > 0]
    max_peak = max(all_peaks) if all_peaks else 0.0
    if max_peak > MAX_MEMORY_GB:
        logger.error(f"  [FAIL] ピーク使用量 {max_peak:.1f}GB が上限 {MAX_MEMORY_GB}GB を超えました")
        all_passed = False
    else:
        logger.info(f"  [OK  ] ピーク使用量: {max_peak:.1f}GB / {MAX_MEMORY_GB}GB")

    logger.info("")
    if all_passed:
        logger.info("全テスト PASS — T-0.3 完了")
        sys.exit(0)
    else:
        logger.error("一部テストが FAIL — 上記ログを確認してください")
        sys.exit(1)


if __name__ == "__main__":
    main()
