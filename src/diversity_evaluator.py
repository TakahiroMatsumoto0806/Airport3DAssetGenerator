"""
T-5.1: 多様性評価エンジン

OpenCLIP ViT-L/14 で全アセットの CLIP 埋め込みを計算し、
Vendi Score（エントロピーベース多様性指標）と近似重複検出を行う。

使用例:
    evaluator = DiversityEvaluator()
    evaluator.load_model()

    embeddings = evaluator.compute_clip_embeddings(image_paths)
    vendi = evaluator.compute_vendi_score(embeddings)
    dups = evaluator.find_near_duplicates(embeddings, threshold=0.95)
    report_path = evaluator.generate_report("outputs/reports")

    evaluator.unload()
"""

import gc
import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class DiversityEvaluator:
    """OpenCLIP ViT-L/14 + Vendi Score による多様性評価"""

    # OpenCLIP モデル設定（仕様書記載）
    _MODEL_NAME = "ViT-L-14"
    _PRETRAINED = "datacomp_xl_s13b_b90k"  # laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._device = None

    # ------------------------------------------------------------------
    # モデル管理
    # ------------------------------------------------------------------

    def load_model(self, device: str = "cuda") -> None:
        """OpenCLIP モデルをロードする"""
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                f"open_clip_torch が見つかりません: {e}\n"
                "uv pip install open_clip_torch でインストールしてください"
            ) from e
        import torch

        self._device = device
        logger.info(f"OpenCLIP {self._MODEL_NAME} ロード中...")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._MODEL_NAME, pretrained=self._PRETRAINED
        )
        self._model = self._model.to(device)
        self._model.eval()
        logger.info("OpenCLIP ロード完了")

    def unload(self) -> None:
        """GPU メモリを解放する"""
        import torch

        if self._model is not None:
            logger.info("OpenCLIP アンロード中...")
            del self._model
            self._model = None
            self._preprocess = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("OpenCLIP アンロード完了")

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # CLIP 埋め込み
    # ------------------------------------------------------------------

    def compute_clip_embeddings(
        self,
        image_paths: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        画像リストの CLIP 埋め込みを計算する（L2 正規化済み）。

        Args:
            image_paths: 画像ファイルパスのリスト
            batch_size:  バッチサイズ（メモリに合わせて調整）

        Returns:
            np.ndarray: shape (N, D)、L2 正規化済み埋め込みベクトル
        """
        if self._model is None:
            raise RuntimeError("モデルがロードされていません。load_model() を先に呼んでください")

        import torch
        from PIL import Image

        all_embeddings: list[np.ndarray] = []
        total = len(image_paths)

        for start in range(0, total, batch_size):
            batch_paths = image_paths[start:start + batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(self._preprocess(img))
                except Exception as e:
                    logger.warning(f"  画像ロード失敗 ({p}): {e} — ゼロベクトルで代替")
                    images.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(images).to(self._device)

            with torch.no_grad():
                features = self._model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().float().numpy())

            done = min(start + batch_size, total)
            if done % 200 == 0 or done == total:
                logger.debug(f"  CLIP 埋め込み: {done}/{total}")

        return np.vstack(all_embeddings)

    # ------------------------------------------------------------------
    # Vendi Score
    # ------------------------------------------------------------------

    def compute_vendi_score(self, embeddings: np.ndarray) -> float:
        """
        Vendi Score を計算する（エントロピーベース多様性指標）。

        VS = exp(H(K/n))  ここで K はコサイン類似度カーネル行列、H はフォン・ノイマンエントロピー

        高いほど多様。完全に重複していれば 1.0、完全に直交していれば N。

        Args:
            embeddings: L2 正規化済み埋め込み (N, D)

        Returns:
            float: Vendi Score（1.0 以上 N 以下）
        """
        n = embeddings.shape[0]
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0

        # コサイン類似度カーネル行列（L2 正規化済みなのでドット積 = コサイン類似度）
        K = embeddings @ embeddings.T
        K = K / n  # 正規化

        # 固有値計算
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # フォン・ノイマンエントロピー
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
        return float(np.exp(entropy))

    # ------------------------------------------------------------------
    # 近似重複検出
    # ------------------------------------------------------------------

    def find_near_duplicates(
        self,
        embeddings: np.ndarray,
        threshold: float = 0.95,
        image_paths: Optional[list[str]] = None,
    ) -> list[tuple]:
        """
        コサイン類似度が threshold 以上のペアを近似重複として検出する。

        Args:
            embeddings:   L2 正規化済み埋め込み (N, D)
            threshold:    類似度閾値（デフォルト 0.95）
            image_paths:  パスリスト（返り値にパス情報を含める場合）

        Returns:
            list[tuple]: (i, j, similarity) のリスト（i < j）
        """
        n = embeddings.shape[0]
        duplicates: list[tuple] = []

        # バッチ単位でコサイン類似度を計算（メモリ効率）
        batch = 512
        for i in range(0, n, batch):
            chunk = embeddings[i:i + batch]  # (B, D)
            sim_matrix = chunk @ embeddings.T  # (B, N)
            for bi, row in enumerate(sim_matrix):
                gi = i + bi
                for j in range(gi + 1, n):
                    if row[j] >= threshold:
                        entry = (gi, j, float(row[j]))
                        if image_paths:
                            entry = (image_paths[gi], image_paths[j], float(row[j]))
                        duplicates.append(entry)

        return duplicates

    # ------------------------------------------------------------------
    # サイズ・カテゴリ統計
    # ------------------------------------------------------------------

    def compute_size_diversity(self, mesh_info_list: list[dict]) -> dict:
        """
        アセットのサイズ/形状分布統計を計算する。

        Args:
            mesh_info_list: 各アセットの dict（scale.scaled_extents_mm を含む）

        Returns:
            {
                "count":         int,
                "short_side_mm": {"mean", "std", "min", "max"},
                "long_side_mm":  {"mean", "std", "min", "max"},
                "volume_mm3":    {"mean", "std", "min", "max"},
            }
        """
        short_sides = []
        long_sides = []
        volumes = []

        for info in mesh_info_list:
            scale = info.get("scale", {})
            extents = scale.get("scaled_extents_mm")
            if extents and len(extents) == 3:
                short_sides.append(min(extents))
                long_sides.append(max(extents))
                volumes.append(extents[0] * extents[1] * extents[2])

        def stats(arr: list) -> dict:
            if not arr:
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
            a = np.array(arr)
            return {
                "mean": round(float(a.mean()), 3),
                "std": round(float(a.std()), 3),
                "min": round(float(a.min()), 3),
                "max": round(float(a.max()), 3),
            }

        return {
            "count": len(short_sides),
            "short_side_mm": stats(short_sides),
            "long_side_mm": stats(long_sides),
            "volume_mm3": stats(volumes),
        }

    def compute_category_distribution(self, metadata_list: list[dict]) -> dict:
        """
        カテゴリ・材質・検出タイプの出現頻度分布を計算する。

        Args:
            metadata_list: 各アセットのメタデータ dict

        Returns:
            {
                "luggage_type":      {type: count},
                "material":          {material: count},
                "detected_type":     {type: count},
            }
        """
        from collections import Counter

        types = Counter()
        materials = Counter()
        detected = Counter()

        for meta in metadata_list:
            t = meta.get("luggage_type") or meta.get("detected_type") or "unknown"
            m = meta.get("material") or "unknown"
            d = meta.get("detected_type") or "unknown"
            types[t] += 1
            materials[m] += 1
            detected[d] += 1

        return {
            "luggage_type": dict(types),
            "material": dict(materials),
            "detected_type": dict(detected),
        }

    def check_size_realism(
        self,
        mesh_info_list: list[dict],
        category_refs: dict,
    ) -> dict:
        """
        アセットのサイズが現実的な範囲内かチェックする。

        Args:
            mesh_info_list: 各アセット情報 dict（luggage_type, scale.scaled_extents_mm を含む）
            category_refs:  カテゴリ別の基準サイズ {type: {min_mm, max_mm}}

        Returns:
            {
                "total":   int,
                "realistic": int,
                "unrealistic": int,
                "details": list[dict],
            }
        """
        total = realistic = unrealistic = 0
        details = []

        for info in mesh_info_list:
            luggage_type = info.get("luggage_type") or "unknown"
            scale = info.get("scale", {})
            extents = scale.get("scaled_extents_mm", [])
            total += 1

            ref = category_refs.get(luggage_type, {})
            min_mm = ref.get("min_mm", 0)
            max_mm = ref.get("max_mm", float("inf"))
            long_side = max(extents) if extents else 0

            is_realistic = min_mm <= long_side <= max_mm
            if is_realistic:
                realistic += 1
            else:
                unrealistic += 1

            details.append({
                "asset_id": info.get("asset_id", "?"),
                "luggage_type": luggage_type,
                "long_side_mm": long_side,
                "is_realistic": is_realistic,
            })

        return {
            "total": total,
            "realistic": realistic,
            "unrealistic": unrealistic,
            "details": details,
        }

    # ------------------------------------------------------------------
    # レポート生成
    # ------------------------------------------------------------------

    def generate_report(
        self,
        output_dir: str,
        embeddings: Optional[np.ndarray] = None,
        image_paths: Optional[list[str]] = None,
        metadata_list: Optional[list[dict]] = None,
        mesh_info_list: Optional[list[dict]] = None,
        near_dup_threshold: float = 0.95,
    ) -> str:
        """
        HTML 形式の多様性レポートを生成する。

        Args:
            output_dir:       レポート保存先ディレクトリ
            embeddings:       CLIP 埋め込み (N, D)（None の場合はスキップ）
            image_paths:      画像ファイルパスのリスト
            metadata_list:    各アセットのメタデータ
            mesh_info_list:   各アセットのメッシュ情報
            near_dup_threshold: 近似重複判定閾値

        Returns:
            str: 生成した HTML レポートのファイルパス
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_data: dict = {
            "vendi_score": None,
            "near_duplicates_count": None,
            "category_distribution": None,
            "size_diversity": None,
        }

        # Vendi Score
        if embeddings is not None and len(embeddings) > 0:
            report_data["vendi_score"] = round(self.compute_vendi_score(embeddings), 4)
            logger.info(f"  Vendi Score: {report_data['vendi_score']}")

        # 近似重複
        if embeddings is not None and len(embeddings) > 1:
            dups = self.find_near_duplicates(embeddings, near_dup_threshold, image_paths)
            report_data["near_duplicates_count"] = len(dups)
            report_data["near_duplicates"] = [
                {"i": str(d[0]), "j": str(d[1]), "similarity": round(d[2], 4)}
                for d in dups[:100]  # 先頭100件のみ記録
            ]
            logger.info(f"  近似重複ペア数: {len(dups)} (閾値={near_dup_threshold})")

        # カテゴリ分布
        if metadata_list:
            report_data["category_distribution"] = self.compute_category_distribution(metadata_list)

        # サイズ分布
        if mesh_info_list:
            report_data["size_diversity"] = self.compute_size_diversity(mesh_info_list)

        # JSON で保存（軽量）
        json_path = output_dir / "diversity_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # HTML レポート生成
        html_path = output_dir / "diversity_report.html"
        self._write_html_report(html_path, report_data)

        logger.info(f"多様性レポート生成完了: {html_path}")
        return str(html_path)

    def _write_html_report(self, path: Path, data: dict) -> None:
        """シンプルな HTML レポートを書き出す"""
        vendi = data.get("vendi_score", "N/A")
        dups = data.get("near_duplicates_count", "N/A")
        cat_dist = data.get("category_distribution") or {}
        size_div = data.get("size_diversity") or {}

        def dist_table(title: str, counts: dict) -> str:
            if not counts:
                return ""
            rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in sorted(counts.items(), key=lambda x: -x[1])
            )
            return f"<h3>{title}</h3><table border='1'><tr><th>カテゴリ</th><th>件数</th></tr>{rows}</table>"

        def stats_section(title: str, stats: dict) -> str:
            if not stats:
                return ""
            rows = ""
            for k in ("short_side_mm", "long_side_mm", "volume_mm3"):
                s = stats.get(k, {})
                rows += (
                    f"<tr><td>{k}</td>"
                    f"<td>{s.get('mean',''):.1f}</td>"
                    f"<td>{s.get('std',''):.1f}</td>"
                    f"<td>{s.get('min',''):.1f}</td>"
                    f"<td>{s.get('max',''):.1f}</td></tr>"
                )
            return (
                f"<h3>{title} (n={stats.get('count', 0)})</h3>"
                f"<table border='1'><tr><th>指標</th><th>平均</th><th>標準偏差</th>"
                f"<th>最小</th><th>最大</th></tr>{rows}</table>"
            )

        html = f"""<!DOCTYPE html>
<html lang="ja">
<head><meta charset="UTF-8"><title>AL3DG 多様性レポート</title>
<style>body{{font-family:sans-serif;margin:2em}} table{{border-collapse:collapse;margin:1em 0}}
td,th{{padding:0.4em 0.8em}} h2{{border-bottom:2px solid #333}}</style>
</head>
<body>
<h1>AL3DG 多様性評価レポート</h1>
<h2>サマリー</h2>
<table border="1">
  <tr><th>指標</th><th>値</th></tr>
  <tr><td>Vendi Score (CLIP ViT-L/14)</td><td><b>{vendi}</b></td></tr>
  <tr><td>近似重複ペア数 (cosine ≥ {data.get('near_dup_threshold', 0.95)})</td><td>{dups}</td></tr>
  <tr><td>アセット総数</td><td>{size_div.get('count', 'N/A')}</td></tr>
</table>

{dist_table("荷物タイプ分布", cat_dist.get("luggage_type", {}))}
{dist_table("材質分布", cat_dist.get("material", {}))}
{stats_section("スケール統計 (ミニチュア, mm)", size_div)}

<p><small>生成日時: {__import__('datetime').datetime.now().isoformat()}</small></p>
</body></html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
