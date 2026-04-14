"""
T-3.1: 3D モデル生成エンジン

TRELLIS.2-4B (microsoft/TRELLIS.2-4B) を使った Image-to-3D 生成。

TRELLIS.2 公式 API:
  import sys; sys.path.insert(0, '~/trellis2_space')
  from trellis2.pipelines import Trellis2ImageTo3DPipeline

  pipeline = Trellis2ImageTo3DPipeline.from_pretrained("~/models/TRELLIS.2-4B")
  pipeline.cuda()
  results = pipeline.run(image, seed=1)   # List[MeshWithVoxel]
  mesh = results[0]
  # mesh.vertices, mesh.faces, mesh.attrs (voxel PBR), mesh.coords

GB10 対応:
  - SPARSE_CONV_BACKEND=torchsparse (flex_gemm は aarch64 非対応)
  - SPARSE_ATTN_BACKEND=sdpa (flash_attn は sm_121 非対応)
  - cumesh は lazy import で保護済み（fill_holes は無視）
  - o_voxel / flex_gemm 非対応のため scipy cKDTree 最近傍で頂点色を計算して trimesh で GLB 出力
    (旧 PyTorch trilinear dense-volume 実装は 0-contamination バグあり → cKDTree に置換済み)

使用例:
    gen = MeshGenerator("~/models/TRELLIS.2-4B")
    glb_path = gen.generate_single("outputs/images/000000_xxx.png", seed=42)

    results = gen.generate_batch(
        image_dir="outputs/images",
        output_dir="outputs/meshes_raw",
    )
    gen.unload()
"""

import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger


_DEFAULT_TEXTURE_SIZE = 2048
_DEFAULT_DECIMATION = 300_000


def _query_vertex_attrs_torch(mesh) -> "torch.Tensor":
    """
    Sparse voxel から頂点属性を cKDTree 最近傍補間で取得する。
    (o_voxel / flex_gemm を使わない GB10 対応代替実装)

    根本原因と修正経緯:
      旧実装は dense volume [1, C, D, H, W] (≈809M voxels) を構築して
      F.grid_sample で trilinear 補間していたが、occupied voxel は
      全体の 0.12% (989K/809M) のみ。99.9% がゼロ埋めになるため
      trilinear 補間結果が全頂点でほぼ 0 (真っ黒) になる「0-contamination」
      バグがあった。

      本実装は occupied voxel の coords から cKDTree を構築し、
      各頂点に最近傍の occupied voxel の属性を直接割り当てる。
      ゼロ埋め dense volume を一切使わないため 0-contamination がない。

      公式実装 (flex_gemm.ops.grid_sample_3d) との差異:
        - 公式: occupied voxel のみを使う sparse trilinear 補間
        - 本実装: occupied voxel からの最近傍 (nearest-neighbor) 割り当て
        - 差異: 本実装は trilinear でなく nearest なので境界が離散的になるが、
          voxel_size ≈ 1e-3 (1024 解像度) なので実用上の差は小さい。

    Returns:
        vertex_attrs: float32 Tensor [N_verts, C]  (values clamp to [0, 1])
    """
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    vertices = mesh.vertices.cpu().float().numpy()   # [N, 3] world coords
    coords   = mesh.coords.cpu().float().numpy()     # [K, 3] voxel indices (integer-valued)
    attrs    = mesh.attrs.cpu().float().numpy()      # [K, C]
    origin   = mesh.origin.cpu().float().numpy()     # e.g. [-0.5, -0.5, -0.5]

    # 頂点位置をボクセルインデックス空間に変換
    vert_voxel = (vertices - origin) / mesh.voxel_size  # [N, 3], same scale as coords

    # 最近傍の occupied voxel を検索
    tree = cKDTree(coords)
    _, idx = tree.query(vert_voxel, k=1, workers=-1)    # [N]

    # attrs は decode_tex_slat の出力 (* 0.5 + 0.5) で概ね [0, 1] だが
    # decoder に activation がないため [-1.66, 2.67] 程度まで逸脱する。
    # clamp で有効範囲に収める。
    result = torch.tensor(attrs[idx], dtype=torch.float32)
    return result.clamp(0.0, 1.0)


def _export_glb_trimesh(
    mesh,
    output_path: Path,
    decimation_target: int = _DEFAULT_DECIMATION,
) -> None:
    """
    MeshWithVoxel を頂点色付き GLB で trimesh エクスポートする。

    o_voxel が aarch64 で使えないため、PyTorch trilinear で頂点色を
    補間して trimesh で出力する。
    """
    import numpy as np
    import torch
    import trimesh

    verts_np = mesh.vertices.cpu().numpy()
    faces_np = mesh.faces.cpu().numpy()

    # 面に使われている頂点だけ残す（FDG は全ボクセルを頂点として出力するため）
    used = np.unique(faces_np)
    if len(used) < len(verts_np):
        remap = np.full(len(verts_np), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        verts_np = verts_np[used]
        faces_np = remap[faces_np]

    # 頂点 PBR 属性取得 [N, 6]: base_color(3), metallic(1), roughness(1), alpha(1)
    with torch.no_grad():
        vertex_attrs = _query_vertex_attrs_torch(mesh).cpu().numpy()  # [N_all, 6]
    vertex_attrs = vertex_attrs[used]  # 使用頂点のみ

    rgba = vertex_attrs[:, [0, 1, 2, 5]]  # base_color + alpha
    rgba_u8 = (rgba * 255).clip(0, 255).astype(np.uint8)
    # alpha チャンネルを 255 に固定する。
    # attrs[:, 5] (alpha) は [-1.04, 2.18] 程度の範囲を持ち clamp 後に 0 に近い値が
    # 多く存在するため、GLB ビューアーで頂点が透明になり「点群」に見える問題が生じる。
    rgba_u8[:, 3] = 255

    tri = trimesh.Trimesh(
        vertices=verts_np,
        faces=faces_np,
        vertex_colors=rgba_u8,
        process=False,
    )

    if decimation_target > 0 and len(faces_np) > decimation_target:
        try:
            from fast_simplification import simplify as _fs
            from scipy.spatial import cKDTree
            reduction = 1.0 - decimation_target / len(faces_np)
            v_out, f_out = _fs(verts_np.astype(np.float32), faces_np.astype(np.int32),
                               target_reduction=float(reduction))
            # 最近傍で頂点色を転送
            _, idx = cKDTree(verts_np).query(v_out, workers=-1)
            rgba_out = rgba_u8[idx]
            tri = trimesh.Trimesh(vertices=v_out, faces=f_out, vertex_colors=rgba_out, process=False)
        except Exception as e:
            logger.warning(f"mesh decimation skipped: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(trimesh.exchange.gltf.export_glb(tri))


class MeshGenerator:
    """TRELLIS.2-4B による Image-to-3D 生成エンジン"""

    def __init__(
        self,
        model_path: str = "~/models/TRELLIS.2-4B",
    ) -> None:
        """
        TRELLIS.2 パイプラインを初期化する。

        Args:
            model_path: ローカルモデルパス
                        例: "~/models/TRELLIS.2-4B"
        """
        # GB10 対応: flex_gemm / flash_attn を回避
        os.environ.setdefault("SPARSE_CONV_BACKEND",  "torchsparse")
        os.environ.setdefault("SPARSE_ATTN_BACKEND",  "sdpa")
        os.environ.setdefault("ATTN_BACKEND",         "sdpa")
        os.environ.setdefault("SPCONV_ALGO",          "native")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        trellis2_repo = str(Path("~/trellis2_space").expanduser())
        if trellis2_repo not in sys.path:
            sys.path.insert(0, trellis2_repo)

        try:
            from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
        except ImportError as e:
            raise ImportError(
                f"trellis2 パッケージが見つかりません: {e}\n"
                "~/trellis2_space が存在するか確認してください。"
            ) from e

        model_path = str(Path(model_path).expanduser())
        from src.utils.memory_guard import REQUIRED_GB, assert_memory_headroom
        assert_memory_headroom(REQUIRED_GB["TRELLIS.2-4B"], "TRELLIS.2-4B")
        logger.info(f"TRELLIS.2 ロード中: {model_path}")
        self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        self._pipeline.cuda()
        logger.info("TRELLIS.2 ロード完了")

    def generate_single(
        self,
        image_path: str,
        seed: int = 42,
        decimation_target: int = _DEFAULT_DECIMATION,
        texture_size: int = _DEFAULT_TEXTURE_SIZE,
        output_path: Optional[str] = None,
        sparse_structure_sampler_params: Optional[dict] = None,
        shape_slat_sampler_params: Optional[dict] = None,
        tex_slat_sampler_params: Optional[dict] = None,
    ) -> str:
        """
        1 枚の画像から 3D モデルを生成し GLB で保存する。

        Args:
            image_path:   入力画像ファイルパス
            seed:         乱数シード
            decimation_target: 最終メッシュの最大面数 (デフォルト 300_000)
            texture_size: (将来用) テクスチャ解像度 — 現在は vertex color 出力
            output_path:  出力 GLB パス（None の場合は自動生成）
            sparse_structure_sampler_params: Sparse Structure サンプラー追加パラメータ
            shape_slat_sampler_params:       Shape SLat サンプラー追加パラメータ
            tex_slat_sampler_params:         Texture SLat サンプラー追加パラメータ

        Returns:
            出力 GLB ファイルパス（文字列）
        """
        from PIL import Image

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"入力画像が見つかりません: {image_path}")

        if output_path is None:
            output_path = image_path.with_suffix(".glb")
        output_path = Path(output_path)

        logger.info(f"3D 生成: {image_path.name} (seed={seed})")

        image = Image.open(image_path).convert("RGB")

        results = self._pipeline.run(
            image,
            seed=seed,
            sparse_structure_sampler_params=sparse_structure_sampler_params or {},
            shape_slat_sampler_params=shape_slat_sampler_params or {},
            tex_slat_sampler_params=tex_slat_sampler_params or {},
            preprocess_image=True,
        )

        mesh = results[0]
        _export_glb_trimesh(mesh, output_path, decimation_target=decimation_target)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  GLB 保存完了: {output_path} ({size_mb:.1f} MB)")
        return str(output_path)

    def generate_batch(
        self,
        image_dir: str,
        output_dir: str,
        decimation_target: int = _DEFAULT_DECIMATION,
        texture_size: int = _DEFAULT_TEXTURE_SIZE,
        extensions: tuple = (".png", ".jpg", ".jpeg"),
        seed_offset: int = 0,
        limit: Optional[int] = None,
        sparse_structure_sampler_params: Optional[dict] = None,
        shape_slat_sampler_params: Optional[dict] = None,
        tex_slat_sampler_params: Optional[dict] = None,
    ) -> list:
        """
        ディレクトリ内の画像を一括で 3D 生成する。

        - 既存 GLB はスキップ（中断再開対応）
        - 生成失敗時は failed 扱いにして続行
        - 生成メタデータを output_dir/generation_metadata.json に保存

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
        image_dir  = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_dir / "generation_metadata.json"

        existing: dict = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    for entry in json.load(f).get("results", []):
                        if entry.get("status") in ("generated", "skipped"):
                            existing[entry["image_path"]] = entry
                logger.info(f"既存メタデータ読み込み: {len(existing)} 件スキップ対象")
            except Exception as e:
                logger.warning(f"既存メタデータの読み込みに失敗: {e}")

        image_files = sorted(
            f for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        if limit is not None:
            image_files = image_files[:limit]
        total = len(image_files)
        logger.info(f"バッチ 3D 生成開始: {total} 件 → {output_dir}")

        results: list = []
        generated_count = skipped_count = failed_count = 0

        for idx, img_path in enumerate(image_files):
            img_key  = str(img_path)
            seed     = idx + seed_offset
            glb_path = output_dir / f"{img_path.stem}.glb"
            ts       = datetime.now(timezone.utc).isoformat()

            if img_key in existing and glb_path.exists():
                skipped_count += 1
                results.append({**existing[img_key], "status": "skipped"})
                continue

            try:
                out_path = self.generate_single(
                    image_path=str(img_path),
                    seed=seed,
                    decimation_target=decimation_target,
                    texture_size=texture_size,
                    output_path=str(glb_path),
                    sparse_structure_sampler_params=sparse_structure_sampler_params,
                    shape_slat_sampler_params=shape_slat_sampler_params,
                    tex_slat_sampler_params=tex_slat_sampler_params,
                )
                generated_count += 1
                results.append({
                    "image_path": img_key,
                    "glb_path":   out_path,
                    "seed":       seed,
                    "status":     "generated",
                    "error":      None,
                    "timestamp":  ts,
                })
            except Exception as e:
                logger.error(f"  [{idx}/{total}] 3D 生成失敗 ({img_path.name}): {e}")
                failed_count += 1
                results.append({
                    "image_path": img_key,
                    "glb_path":   None,
                    "seed":       seed,
                    "status":     "failed",
                    "error":      str(e),
                    "timestamp":  ts,
                })

            done = generated_count + skipped_count + failed_count
            if done % 20 == 0 or done == total:
                logger.info(
                    f"  進捗: {done}/{total} "
                    f"(生成={generated_count}, スキップ={skipped_count}, 失敗={failed_count})"
                )

            if done % 20 == 0:
                self._save_metadata(
                    metadata_path, results, total,
                    generated_count, skipped_count, failed_count,
                )

        self._save_metadata(
            metadata_path, results, total,
            generated_count, skipped_count, failed_count,
        )
        logger.info(
            f"バッチ完了: 生成={generated_count}, スキップ={skipped_count}, 失敗={failed_count}"
        )
        return results

    def unload(self) -> None:
        """GPU メモリを解放する。次工程ロード前に呼び出すこと。"""
        if getattr(self, '_pipeline', None) is not None:
            logger.info("TRELLIS.2 アンロード中...")
            del self._pipeline
            self._pipeline = None
            from src.utils.memory_guard import flush_cuda_memory
            reserved, allocated = flush_cuda_memory()
            logger.info(
                f"TRELLIS.2 アンロード完了 "
                f"(CUDA reserved={reserved:.2f} GiB, allocated={allocated:.2f} GiB)"
            )

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    def _save_metadata(
        self,
        path: Path,
        results: list,
        total: int,
        generated: int,
        skipped: int,
        failed: int,
    ) -> None:
        payload = {
            "total":     total,
            "generated": generated,
            "skipped":   skipped,
            "failed":    failed,
            "results":   results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
