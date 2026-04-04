"""
T-3.1: 3D モデル生成エンジン

TRELLIS.2-4B を使った Image-to-3D 生成。

TRELLIS.2 API（公式確認済み）:
  pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
  pipeline.cuda()
  mesh = pipeline.run(image)[0]          # PIL.Image → mesh オブジェクト
  mesh.simplify(target_vertices)         # メッシュ簡略化
  glb = o_voxel.postprocess.to_glb(     # GLB エクスポート
      vertices=mesh.vertices,
      faces=mesh.faces,
      attr_volume=mesh.attrs,
      coords=mesh.coords,
      attr_layout=mesh.layout,
      voxel_size=mesh.voxel_size,
      aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
      decimation_target=1_000_000,
      texture_size=1024,
      remesh=True,
      remesh_band=1,
      remesh_project=0,
      verbose=False,
  )
  glb.export("output.glb", extension_webp=True)

仕様パラメータ（plan/dgx_spark_construction_plan.html 記載）:
  simplify=0.95  → mesh.simplify(int(vertex_count * 0.95))
  texture_size=1024

注意:
  - TRELLIS.2 は conda 環境 "trellis2" でインストール
  - trellis2 / o_voxel パッケージが conda 環境に入っていること
  - run() に seed パラメータはない。torch.manual_seed() で制御する

使用例:
    gen = MeshGenerator("~/models/TRELLIS.2-4B")
    glb_path = gen.generate_single("outputs/images_approved/000001.png", seed=42)

    results = gen.generate_batch(
        image_dir="outputs/images_approved",
        output_dir="outputs/meshes_raw",
    )
    gen.unload()
"""

import gc
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger


# nvdiffrast の頂点数上限（公式ドキュメント記載値）
_NVDIFFRAST_VERTEX_LIMIT = 16_777_216

# GLB エクスポートの decimation_target（仕様書は simplify=0.95 なので 0.95 × 1_000_000）
_DEFAULT_DECIMATION_TARGET = int(1_000_000 * 0.95)  # = 950_000
_DEFAULT_TEXTURE_SIZE = 1024


class MeshGenerator:
    """TRELLIS.2-4B による Image-to-3D 生成エンジン"""

    def __init__(
        self,
        model_path: str = "microsoft/TRELLIS.2-4B",
    ) -> None:
        """
        TRELLIS.2 パイプラインを初期化する。

        Args:
            model_path: ローカルモデルパス または HF リポジトリ ID
                        例: "~/models/TRELLIS.2-4B"
                            "microsoft/TRELLIS.2-4B"

        Raises:
            ImportError: trellis2 / o_voxel パッケージが見つからない場合。
                         conda 環境 "trellis2" がインストールされているか確認すること:
                           cd ~/trellis2
                           . ./setup.sh --new-env --basic --flash-attn ...
        """
        # TRELLIS.2 推奨環境変数
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
        except ImportError as e:
            raise ImportError(
                f"trellis2 パッケージが見つかりません: {e}\n"
                "conda 環境 'trellis2' をインストールしてから起動してください:\n"
                "  cd ~/trellis2\n"
                "  . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel\n"
                "  conda activate trellis2"
            ) from e

        model_path = str(Path(model_path).expanduser())
        logger.info(f"TRELLIS.2-4B ロード中: {model_path}")
        self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        self._pipeline.cuda()
        self._pipeline_cls = Trellis2ImageTo3DPipeline
        logger.info("TRELLIS.2-4B ロード完了")

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _set_seed(self, seed: int) -> None:
        """torch の乱数シードを設定する（run() に seed 引数がないため）"""
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _export_glb(
        self,
        mesh,
        output_path: Path,
        simplify_ratio: float = 0.95,
        texture_size: int = _DEFAULT_TEXTURE_SIZE,
    ) -> Path:
        """
        メッシュオブジェクトを GLB ファイルにエクスポートする。

        Args:
            mesh:           pipeline.run() の出力メッシュオブジェクト
            output_path:    出力 GLB パス
            simplify_ratio: 頂点数削減率（0.95 = 元の 95% を維持）
            texture_size:   PBR テクスチャ解像度

        Returns:
            出力 GLB ファイルパス
        """
        import o_voxel

        # mesh.simplify() で頂点数を削減（nvdiffrast 上限以下に収める）
        if hasattr(mesh, "vertices") and mesh.vertices is not None:
            original_vertices = len(mesh.vertices)
            target_vertices = min(
                int(original_vertices * simplify_ratio),
                _NVDIFFRAST_VERTEX_LIMIT,
            )
            if target_vertices < original_vertices:
                logger.debug(
                    f"  メッシュ簡略化: {original_vertices:,} → {target_vertices:,} 頂点"
                )
                mesh.simplify(target_vertices)

        # GLB エクスポート
        decimation_target = int(_DEFAULT_DECIMATION_TARGET * simplify_ratio / 0.95)
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(output_path), extension_webp=True)
        return output_path

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def generate_single(
        self,
        image_path: str,
        seed: int = 42,
        simplify: float = 0.95,
        texture_size: int = _DEFAULT_TEXTURE_SIZE,
        output_path: Optional[str] = None,
    ) -> str:
        """
        1 枚の画像から 3D モデルを生成し GLB で保存する。

        Args:
            image_path:   入力画像ファイルパス
            seed:         乱数シード（再現性のため）
            simplify:     メッシュ簡略化率（0.95 = 元の 95% を維持）
            texture_size: PBR テクスチャ解像度（デフォルト 1024）
            output_path:  出力 GLB パス（None の場合は自動生成）

        Returns:
            出力 GLB ファイルパス（文字列）

        Raises:
            FileNotFoundError: 入力画像が存在しない場合
        """
        from PIL import Image

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"入力画像が見つかりません: {image_path}")

        if output_path is None:
            stem = image_path.stem
            output_path = image_path.parent / f"{stem}.glb"
        output_path = Path(output_path)

        logger.info(f"3D 生成: {image_path.name} (seed={seed})")

        # シード設定
        self._set_seed(seed)

        # 画像ロード
        image = Image.open(image_path).convert("RGB")

        # 3D 生成
        mesh = self._pipeline.run(image)[0]

        # GLB エクスポート
        self._export_glb(mesh, output_path, simplify_ratio=simplify, texture_size=texture_size)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  GLB 保存完了: {output_path} ({file_size_mb:.1f} MB)")
        return str(output_path)

    def generate_batch(
        self,
        image_dir: str,
        output_dir: str,
        simplify: float = 0.95,
        texture_size: int = _DEFAULT_TEXTURE_SIZE,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        seed_offset: int = 0,
    ) -> list[dict]:
        """
        ディレクトリ内の画像を一括で 3D 生成する。

        - 既存 GLB はスキップ（中断再開対応）
        - 生成失敗時は該当エントリを failed 扱いにして続行
        - 生成メタデータを output_dir/generation_metadata.json に保存

        Args:
            image_dir:    入力画像ディレクトリ
            output_dir:   GLB 出力ディレクトリ
            simplify:     メッシュ簡略化率
            texture_size: テクスチャ解像度
            extensions:   対象ファイル拡張子
            seed_offset:  シードの開始オフセット（インデックス + offset が seed）

        Returns:
            list[dict] — 各エントリ:
            {
                "image_path":  str,
                "glb_path":    str | None,
                "seed":        int,
                "status":      "generated" | "skipped" | "failed",
                "error":       str | None,
                "timestamp":   str,
            }
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_dir / "generation_metadata.json"

        # 既存メタデータ読み込み（再開用）
        existing: dict[str, dict] = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    for entry in json.load(f).get("results", []):
                        if entry.get("status") in ("generated", "skipped"):
                            existing[entry["image_path"]] = entry
                logger.info(f"既存メタデータ読み込み: {len(existing)} 件スキップ対象")
            except Exception as e:
                logger.warning(f"既存メタデータの読み込みに失敗: {e}")

        # 入力画像一覧
        image_files = sorted(
            f for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        total = len(image_files)
        logger.info(f"バッチ 3D 生成開始: {total} 件 → {output_dir}")

        results: list[dict] = []
        generated_count = skipped_count = failed_count = 0

        for idx, img_path in enumerate(image_files):
            img_key = str(img_path)
            seed = idx + seed_offset
            glb_path = output_dir / f"{img_path.stem}.glb"
            ts = datetime.now(timezone.utc).isoformat()

            # スキップ判定
            if img_key in existing and glb_path.exists():
                skipped_count += 1
                results.append(
                    {
                        **existing[img_key],
                        "status": "skipped",
                    }
                )
                continue

            # 生成
            try:
                out_path = self.generate_single(
                    image_path=str(img_path),
                    seed=seed,
                    simplify=simplify,
                    texture_size=texture_size,
                    output_path=str(glb_path),
                )
                generated_count += 1
                results.append(
                    {
                        "image_path": img_key,
                        "glb_path": out_path,
                        "seed": seed,
                        "status": "generated",
                        "error": None,
                        "timestamp": ts,
                    }
                )

            except Exception as e:
                logger.error(
                    f"  [{idx}/{total}] 3D 生成失敗 ({img_path.name}): {e}"
                )
                failed_count += 1
                results.append(
                    {
                        "image_path": img_key,
                        "glb_path": None,
                        "seed": seed,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": ts,
                    }
                )

            # 進捗表示
            done = generated_count + skipped_count + failed_count
            if done % 20 == 0 or done == total:
                logger.info(
                    f"  進捗: {done}/{total} "
                    f"(生成={generated_count}, スキップ={skipped_count}, 失敗={failed_count})"
                )

            # 中間保存（20 件ごと）
            if done % 20 == 0:
                self._save_metadata(
                    metadata_path, results, total,
                    generated_count, skipped_count, failed_count,
                )

        # 最終保存
        self._save_metadata(
            metadata_path, results, total,
            generated_count, skipped_count, failed_count,
        )

        logger.info(
            f"バッチ完了: 生成={generated_count}, スキップ={skipped_count}, "
            f"失敗={failed_count}"
        )
        return results

    def unload(self) -> None:
        """
        GPU メモリを解放する。

        次工程（VLM / メッシュ QA）をロードする前に呼び出すこと。
        """
        import torch

        if self._pipeline is not None:
            logger.info("TRELLIS.2-4B アンロード中...")
            del self._pipeline
            self._pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("TRELLIS.2-4B アンロード完了")

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------

    def _save_metadata(
        self,
        path: Path,
        results: list[dict],
        total: int,
        generated: int,
        skipped: int,
        failed: int,
    ) -> None:
        payload = {
            "total": total,
            "generated": generated,
            "skipped": skipped,
            "failed": failed,
            "results": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
