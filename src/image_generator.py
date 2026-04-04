"""
T-2.1: 画像生成エンジン

FLUX.1-schnell を使った Text-to-Image 生成。
仕様:
  - num_inference_steps=4, guidance_scale=0.0
  - 解像度 1024×1024
  - BF16 ロード
  - バッチ生成は中断再開対応（既存ファイルスキップ）
  - 生成後は GPU メモリを解放して次工程（VLM等）に引き渡せる状態にする

使用例:
    gen = ImageGenerator("~/models/FLUX.1-schnell")
    image = gen.generate_single("A black suitcase on white background", seed=42)

    results = gen.generate_batch(
        prompts,           # list[dict] from PromptGenerator
        output_dir="outputs/images",
    )
    gen.unload()           # 次工程のために明示的にアンロード
"""

import gc
import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger

# PIL は常に利用可能想定
from PIL import Image


class ImageGenerator:
    """FLUX.1-schnell による Text-to-Image 生成エンジン"""

    DEFAULT_STEPS = 4
    DEFAULT_GUIDANCE = 0.0
    DEFAULT_HEIGHT = 1024
    DEFAULT_WIDTH = 1024

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ) -> None:
        """
        FLUX.1-schnell パイプラインを初期化する。

        Args:
            model_path: ローカルモデルパス または HF リポジトリ ID
                        例: "~/models/FLUX.1-schnell"
                            "black-forest-labs/FLUX.1-schnell"
            device: 使用デバイス ("cuda" / "cpu")
        """
        import torch
        from diffusers import FluxPipeline

        model_path = str(Path(model_path).expanduser())
        self.device = device
        self._pipe = None

        logger.info(f"FLUX.1-schnell ロード中: {model_path}")
        self._pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        self._pipe = self._pipe.to(device)
        logger.info("FLUX.1-schnell ロード完了")

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _make_filename(self, prompt_id: str, idx: int) -> str:
        """出力ファイル名を生成する"""
        safe_id = prompt_id[:12] if prompt_id else f"{idx:06d}"
        return f"{idx:06d}_{safe_id}.png"

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def generate_single(
        self,
        prompt: str,
        seed: Optional[int] = None,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
    ) -> Image.Image:
        """
        1 枚の画像を生成する。

        Args:
            prompt: 生成プロンプト文字列
            seed:   乱数シード（None の場合はランダム）
            height: 出力高さ（デフォルト 1024）
            width:  出力幅（デフォルト 1024）
            num_inference_steps: 推論ステップ数（デフォルト 4）
            guidance_scale:      CFG スケール（FLUX.1-schnell は 0.0）

        Returns:
            PIL.Image.Image
        """
        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        image = self._pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        return image

    def generate_batch(
        self,
        prompts: list[dict],
        output_dir: str,
        seeds: Optional[list[int]] = None,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        save_metadata: bool = True,
    ) -> list[dict]:
        """
        複数プロンプトからバッチ画像生成を行う。

        - 既存ファイルはスキップ（中断再開対応）
        - 生成失敗時は該当エントリを skip 扱いにして続行
        - メタデータを outputs/<output_dir>/generation_metadata.json に保存

        Args:
            prompts:    PromptGenerator.generate_combinatorial() の出力 list[dict]
            output_dir: 画像保存先ディレクトリ
            seeds:      各プロンプトに対応するシードのリスト（None の場合はインデックス値）
            save_metadata: True の場合、生成メタデータを JSON に保存

        Returns:
            list[dict] — 各エントリに以下が追加される:
            {
                "image_path": str,     # 保存先パス (失敗時は None)
                "seed": int,
                "status": "generated" | "skipped" | "failed",
                "error": str | None,
                **元の prompt/metadata フィールド
            }
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_dir / "generation_metadata.json"

        # 既存メタデータを読み込んで再開ポイントを確認
        existing: dict[str, dict] = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    for entry in json.load(f).get("results", []):
                        if entry.get("status") in ("generated", "skipped"):
                            pid = entry.get("metadata", {}).get("prompt_id")
                            if pid:
                                existing[pid] = entry
                logger.info(f"既存メタデータ読み込み: {len(existing)} 件スキップ対象")
            except Exception as e:
                logger.warning(f"既存メタデータの読み込みに失敗: {e}")

        results: list[dict] = []
        total = len(prompts)
        generated_count = 0
        skipped_count = 0
        failed_count = 0

        logger.info(f"バッチ画像生成開始: {total} 件 → {output_dir}")

        for idx, item in enumerate(prompts):
            prompt_text = item["prompt"]
            metadata = item.get("metadata", {})
            prompt_id = metadata.get("prompt_id", f"{idx:06d}")
            seed = seeds[idx] if seeds else idx

            filename = self._make_filename(prompt_id, idx)
            image_path = output_dir / filename

            # --- スキップ判定 ---
            if prompt_id in existing and image_path.exists():
                skipped_count += 1
                results.append(
                    {
                        **item,
                        "image_path": str(image_path),
                        "seed": seed,
                        "status": "skipped",
                        "error": None,
                    }
                )
                continue

            # --- 生成 ---
            try:
                image = self.generate_single(
                    prompt=prompt_text,
                    seed=seed,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                image.save(image_path)
                generated_count += 1

                results.append(
                    {
                        **item,
                        "image_path": str(image_path),
                        "seed": seed,
                        "status": "generated",
                        "error": None,
                    }
                )

            except Exception as e:
                logger.error(f"  [{idx}/{total}] 生成失敗 (prompt_id={prompt_id}): {e}")
                failed_count += 1
                results.append(
                    {
                        **item,
                        "image_path": None,
                        "seed": seed,
                        "status": "failed",
                        "error": str(e),
                    }
                )

            # 進捗表示
            done = generated_count + skipped_count + failed_count
            if done % 50 == 0 or done == total:
                logger.info(
                    f"  進捗: {done}/{total} "
                    f"(生成={generated_count}, スキップ={skipped_count}, 失敗={failed_count})"
                )

        # メタデータ保存
        if save_metadata:
            payload = {
                "total": total,
                "generated": generated_count,
                "skipped": skipped_count,
                "failed": failed_count,
                "results": results,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"メタデータ保存: {metadata_path}")

        logger.info(
            f"バッチ完了: 生成={generated_count}, スキップ={skipped_count}, 失敗={failed_count}"
        )
        return results

    def unload(self) -> None:
        """
        GPU メモリを解放する。

        次工程 (VLM / TRELLIS.2) をロードする前に必ず呼び出すこと。
        逐次ロード戦略に従い、同時ロードを防ぐ。
        """
        import torch

        if self._pipe is not None:
            logger.info("FLUX.1-schnell アンロード中...")
            del self._pipe
            self._pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("FLUX.1-schnell アンロード完了")

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
