"""
T-3.3: VLM マルチビュー 3D 検品エンジン

Qwen3-VL-32B-Instruct (vLLM サーバー経由) で 3D メッシュをマルチビュー評価する。
Thinking モード: /think（複雑な幾何・材質評価に精度向上）

処理フロー:
  1. GLB メッシュを pyrender で 4 方向からオフスクリーンレンダリング
  2. 4 枚の画像をすべて VLM に送信してスコアを取得
  3. スコア閾値で pass / review / fail を判定

合格基準（仕様書記載）:
  - geometry_score >= 7
  - texture_score  >= 6

使用例:
    qa = MeshVLMQA()
    render_paths = qa.render_multiview("outputs/meshes_raw/000001.glb", "tmp/renders")
    result = qa.evaluate_3d(render_paths, expected_type="hard_suitcase")
    # {"geometry_score": 8, "texture_score": 7, "pass": True, ...}

    summary = qa.evaluate_batch(
        mesh_dir="outputs/meshes_approved",
        output_json="outputs/meshes_approved/vlm_qa_results.json",
    )
"""

import base64
import json
import re
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger


# 合格基準（仕様書記載）
MIN_GEOMETRY_SCORE = 7
MIN_TEXTURE_SCORE = 6

# VLM モデルとエンドポイント
_DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
_DEFAULT_VLLM_URL = "http://localhost:8000/v1"

_SYSTEM_PROMPT = """You are a 3D asset quality inspector for a robotics training dataset.
Evaluate multi-view renders of a 3D luggage mesh for geometry quality, texture quality, and realism.
Respond ONLY with a valid JSON object. No markdown, no explanation."""

_USER_PROMPT_TEMPLATE = """/think
You are reviewing {n_views} rendered views (front, right, back, left) of a 3D luggage mesh.
Evaluate the mesh quality and return a JSON object with exactly these keys:

{{
  "geometry_score": <int 1-10, overall 3D geometry quality: clean topology, correct shape, no artifacts>,
  "texture_score": <int 1-10, texture/material quality: PBR realism, UV mapping, no seams>,
  "consistency_score": <int 1-10, visual consistency across all 4 views>,
  "is_realistic_luggage": <true if the object looks like real luggage, false otherwise>,
  "detected_type": <string: best matching luggage type, e.g. "hard_suitcase", "backpack", "duffel_bag">,
  "detected_material": <string: primary material, e.g. "polycarbonate", "nylon", "leather">,
  "issues": <list of strings: specific problems found, empty list if none>,
  "pass": <true if geometry_score>=7 AND texture_score>=6>
}}

{expected_type_hint}

Reply with only the JSON object."""


def _image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json_response(text: str) -> dict:
    """VLM の返答から JSON を抽出する"""
    # ```json ... ``` ブロックを除去
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # <think>...</think> ブロックを除去（Thinking モード）
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # JSON 部分だけを取り出す
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            return json.loads(m.group())
        raise


def _apply_defaults(result: dict) -> dict:
    """欠落フィールドにデフォルト値を補完する"""
    defaults = {
        "geometry_score": 1,
        "texture_score": 1,
        "consistency_score": 1,
        "is_realistic_luggage": False,
        "detected_type": "unknown",
        "detected_material": "unknown",
        "issues": [],
        "pass": False,
    }
    for k, v in defaults.items():
        result.setdefault(k, v)
    # pass を再計算（VLM の自己申告と閾値を両方満たす場合のみ）
    result["pass"] = (
        result["geometry_score"] >= MIN_GEOMETRY_SCORE
        and result["texture_score"] >= MIN_TEXTURE_SCORE
    )
    return result


class MeshVLMQA:
    """Qwen3-VL-32B (vLLM) + pyrender による マルチビュー 3D 検品エンジン"""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        vllm_base_url: str = _DEFAULT_VLLM_URL,
    ) -> None:
        """
        Args:
            model_name:     vLLM で提供しているモデル名
            vllm_base_url:  vLLM サーバーの base URL
        """
        self.model_name = model_name
        self.vllm_base_url = vllm_base_url
        self._client = None  # 遅延初期化

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=f"{self.vllm_base_url}",
                api_key="dummy",
            )
        return self._client

    # ------------------------------------------------------------------
    # レンダリング
    # ------------------------------------------------------------------

    def render_multiview(
        self,
        mesh_path: str,
        output_dir: str,
        views: int = 4,
    ) -> list[str]:
        """
        GLB メッシュを複数視点からオフスクリーンレンダリングする。

        Args:
            mesh_path:  入力 GLB ファイルパス
            output_dir: レンダリング画像の保存先ディレクトリ
            views:      レンダリング視点数（デフォルト 4）

        Returns:
            list[str]: PNG ファイルパスのリスト（views 件）
        """
        from src.utils.rendering import render_multiview as _render

        stem = Path(mesh_path).stem
        return _render(
            mesh_path=mesh_path,
            output_dir=output_dir,
            views=views,
            prefix=stem,
        )

    # ------------------------------------------------------------------
    # VLM 評価
    # ------------------------------------------------------------------

    def evaluate_3d(
        self,
        render_paths: list[str],
        expected_type: Optional[str] = None,
    ) -> dict:
        """
        マルチビュー画像を VLM で評価し品質スコアを返す。

        Args:
            render_paths:  レンダリング PNG パスのリスト（4 枚程度）
            expected_type: 期待する荷物タイプ（ヒントとして VLM に渡す）

        Returns:
            {
                "geometry_score":     int (1-10),
                "texture_score":      int (1-10),
                "consistency_score":  int (1-10),
                "is_realistic_luggage": bool,
                "detected_type":      str,
                "detected_material":  str,
                "issues":             list[str],
                "pass":               bool,
            }
        """
        client = self._get_client()

        # マルチビュー画像をメッセージに組み込む
        content: list[dict] = []

        # テキスト指示
        expected_hint = (
            f"The expected luggage type is: {expected_type}." if expected_type else ""
        )
        prompt_text = _USER_PROMPT_TEMPLATE.format(
            n_views=len(render_paths),
            expected_type_hint=expected_hint,
        )
        content.append({"type": "text", "text": prompt_text})

        # 画像を base64 で埋め込む
        for path in render_paths:
            b64 = _image_to_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        raw_text = response.choices[0].message.content or ""
        logger.debug(f"  VLM 応答 (先頭 200 文字): {raw_text[:200]}")

        try:
            result = _parse_json_response(raw_text)
        except Exception as e:
            logger.warning(f"  JSON パース失敗: {e} — デフォルト値で置換")
            result = {}

        return _apply_defaults(result)

    # ------------------------------------------------------------------
    # バッチ処理
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        mesh_dir: str,
        output_json: str,
        render_dir: Optional[str] = None,
        views: int = 4,
        extensions: tuple[str, ...] = (".glb",),
        resume: bool = True,
    ) -> dict:
        """
        ディレクトリ内の全 GLB メッシュをマルチビュー VLM 評価する。

        Args:
            mesh_dir:    入力メッシュディレクトリ
            output_json: 評価結果 JSON 保存先
            render_dir:  レンダリング画像保存先（None の場合は tmpdir 使用）
            views:       レンダリング視点数
            extensions:  対象拡張子
            resume:      True の場合、既存結果をスキップ

        Returns:
            {
                "total":   int,
                "passed":  int,
                "failed":  int,
                "results": list[dict],
            }
        """
        mesh_dir = Path(mesh_dir)
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        # 既存結果の読み込み（resume 用）
        existing: dict[str, dict] = {}
        if resume and output_json.exists():
            try:
                with open(output_json, encoding="utf-8") as f:
                    for entry in json.load(f).get("results", []):
                        existing[entry["mesh_path"]] = entry
                logger.info(f"既存結果読み込み: {len(existing)} 件スキップ対象")
            except Exception as e:
                logger.warning(f"既存結果の読み込みに失敗: {e}")

        mesh_files = sorted(
            f for f in mesh_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        total = len(mesh_files)
        logger.info(f"VLM 3D 検品バッチ開始: {total} 件 → {output_json}")

        results: list[dict] = list(existing.values())
        passed = sum(1 for r in results if r.get("pass"))
        failed = len(results) - passed

        use_tmpdir = render_dir is None
        tmpdir_ctx = tempfile.TemporaryDirectory() if use_tmpdir else None

        try:
            base_render_dir = Path(tmpdir_ctx.name if use_tmpdir else render_dir)

            for mesh_path in mesh_files:
                key = str(mesh_path)
                if key in existing:
                    continue

                render_out = base_render_dir / mesh_path.stem
                result_entry: dict = {"mesh_path": key}

                try:
                    render_paths = self.render_multiview(str(mesh_path), str(render_out), views)
                    eval_result = self.evaluate_3d(render_paths)
                    result_entry.update(eval_result)

                    if result_entry.get("pass"):
                        passed += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.error(f"  VLM 評価失敗 ({mesh_path.name}): {e}")
                    result_entry.update(_apply_defaults({}))
                    result_entry["error"] = str(e)
                    failed += 1

                results.append(result_entry)

                # 中間保存
                done = passed + failed
                if done % 20 == 0 or done == total:
                    logger.info(f"  進捗: {done}/{total} (合格={passed}, 不合格={failed})")
                    self._save_results(output_json, results, total, passed, failed)

        finally:
            if tmpdir_ctx:
                tmpdir_ctx.cleanup()

        self._save_results(output_json, results, total, passed, failed)
        logger.info(f"VLM 3D 検品完了: 合格={passed}, 不合格={failed}")

        return {"total": total, "passed": passed, "failed": failed, "results": results}

    def _save_results(
        self,
        path: Path,
        results: list[dict],
        total: int,
        passed: int,
        failed: int,
    ) -> None:
        payload = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
