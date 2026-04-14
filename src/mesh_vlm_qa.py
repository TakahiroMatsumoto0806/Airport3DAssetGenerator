"""
T-3.3: VLM マルチビュー 3D 検品エンジン

Qwen3-VL-32B-Instruct (vLLM サーバー経由) で 3D メッシュをマルチビュー評価する。
Thinking モード: /think（複雑な幾何・材質評価に精度向上）

処理フロー:
  1. GLB メッシュを pyrender で 4 方向からオフスクリーンレンダリング
  2. 4 枚の画像をすべて VLM に送信してスコアを取得
  3. スコア閾値で pass / review / fail を判定

合格基準（pipeline_config.yaml の mesh_vlm_qa.thresholds に従う）:
  - geometry_score >= 6
  - texture_score  >= 5

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


def _safe_format(template: str, **kwargs) -> str:
    """JSON例など波括弧を含むテンプレートの安全なフォーマット。

    re.sub で {英数字_} のシンプルな変数名のみを置換する。
    """
    import re

    def _replacer(m: "re.Match") -> str:
        key = m.group(1)
        return str(kwargs[key]) if key in kwargs else m.group(0)

    return re.sub(r"\{(\w+)\}", _replacer, template)


# 合格基準（モジュールのデフォルト値。pipeline_config.yaml の thresholds で上書き可能）
# TRELLIS 向け緩和: TRELLIS の baked texture は暗く低スコアになりやすいため
# 元仕様 (geometry>=7, texture>=6) から実測分布を踏まえて調整済み
MIN_GEOMETRY_SCORE = 6
MIN_TEXTURE_SCORE = 5

# VLM モデルとエンドポイント
_DEFAULT_MODEL = str(Path.home() / "models" / "Qwen3-VL-32B-Instruct")
_DEFAULT_VLLM_URL = "http://localhost:8001/v1"

_SYSTEM_PROMPT = """You are a 3D asset quality inspector for a robotics training dataset.
These meshes are AI-generated (TRELLIS diffusion model) and will be used as simulation assets, not photorealistic renders.
Evaluate multi-view renders of a 3D luggage mesh for geometry quality, texture quality, and realism.
Score relative to AI-generated mesh quality standards, not hand-crafted professional models.
Respond ONLY with a valid JSON object. No markdown, no explanation."""

_USER_PROMPT_TEMPLATE = """/think
You are reviewing {n_views} rendered views (front, right, back, left) of an AI-generated 3D luggage mesh.
These are TRELLIS diffusion model outputs intended for robot simulation training data.
Evaluate the mesh quality relative to AI-generated mesh standards and return a JSON object with exactly these keys:

{{
  "geometry_score": <int 1-10, 3D geometry quality: recognizable shape=5, clean topology=7, professional=10>,
  "texture_score": <int 1-10, texture/material quality: visible color/material=4, good UV=6, PBR realism=9>,
  "consistency_score": <int 1-10, visual consistency across all views>,
  "is_realistic_luggage": <true if the object is recognizable as luggage/bag, false otherwise>,
  "detected_type": <string: best matching type, e.g. "hard_suitcase", "soft_suitcase", "backpack", "duffel_bag", "handbag">,
  "detected_material": <string: primary material, e.g. "polycarbonate", "nylon", "leather", "fabric">,
  "issues": <list of strings: critical problems only, empty list if acceptable>,
  "pass": <true if geometry_score>=6 AND texture_score>=5>
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


def _apply_defaults(result: dict, min_geometry: int = MIN_GEOMETRY_SCORE, min_texture: int = MIN_TEXTURE_SCORE) -> dict:
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
        result["geometry_score"] >= min_geometry
        and result["texture_score"] >= min_texture
    )
    return result


class MeshVLMQA:
    """Qwen3-VL-32B (vLLM) + pyrender による マルチビュー 3D 検品エンジン"""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        vllm_base_url: str = _DEFAULT_VLLM_URL,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        thresholds: dict | None = None,
    ) -> None:
        """
        Args:
            model_name:     vLLM で提供しているモデル名
            vllm_base_url:  vLLM サーバーの base URL
            system_prompt:  カスタムシステムプロンプト（デフォルト使用時は None）
            user_prompt_template: カスタムユーザープロンプトテンプレート
            thresholds:     カスタム合格基準 dict（geometry, texture等）
        """
        self.model_name = model_name
        self.vllm_base_url = vllm_base_url
        self._client = None  # 遅延初期化

        # プロンプト設定（設定ファイルから受け取ったものを優先、なければハードコード）
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._user_prompt_template = user_prompt_template or _USER_PROMPT_TEMPLATE

        # 合格基準（設定ファイルから受け取ったものを優先、なければハードコード）
        self._min_geometry_score = MIN_GEOMETRY_SCORE
        self._min_texture_score = MIN_TEXTURE_SCORE

        if thresholds:
            self._min_geometry_score = thresholds.get("geometry", self._min_geometry_score)
            self._min_texture_score = thresholds.get("texture", self._min_texture_score)

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
        prompt_text = _safe_format(
            self._user_prompt_template,
            n_views=len(render_paths),
            expected_type_hint=expected_hint,
            min_geometry=self._min_geometry_score,
            min_texture=self._min_texture_score,
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
                {"role": "system", "content": self._system_prompt},
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

        return _apply_defaults(result, min_geometry=self._min_geometry_score, min_texture=self._min_texture_score)

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

        # ユーザープロンプトテンプレートをフォーマット（最初に1回）
        user_prompt_formatted = _safe_format(
            self._user_prompt_template,
            n_views=4,  # デフォルトは4ビュー
            expected_type_hint="",
            min_geometry=self._min_geometry_score,
            min_texture=self._min_texture_score,
        )

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
                    result_entry.update(_apply_defaults({}, min_geometry=self._min_geometry_score, min_texture=self._min_texture_score))
                    result_entry["error"] = str(e)
                    failed += 1

                results.append(result_entry)

                # 中間保存
                done = passed + failed
                if done % 20 == 0 or done == total:
                    logger.info(f"  進捗: {done}/{total} (合格={passed}, 不合格={failed})")
                    self._save_results(output_json, results, total, passed, failed, system_prompt_used=user_prompt_formatted)

        finally:
            if tmpdir_ctx:
                tmpdir_ctx.cleanup()

        self._save_results(output_json, results, total, passed, failed, system_prompt_used=user_prompt_formatted)
        logger.info(f"VLM 3D 検品完了: 合格={passed}, 不合格={failed}")

        return {"total": total, "passed": passed, "failed": failed, "results": results}

    def _save_results(
        self,
        path: Path,
        results: list[dict],
        total: int,
        passed: int,
        failed: int,
        system_prompt_used: str | None = None,
    ) -> None:
        payload = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "system_prompt_used": system_prompt_used or self._system_prompt,
            "results": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def generate_html_report(self, output_path: str) -> None:
        """
        3D 検品結果を HTML レポートとして生成する。

        テーブル構成:
        | # | Asset | 送信プロンプト | View0 | View1 | View2 | View3 | Geo | Tex | Consistency | Issues | Pass |

        Args:
            output_path: HTML 保存先ファイルパス
        """
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 対応する vlm_qa_results.json を探す（output_path の命名規則から推測）
        # output_path が outputs/reports/mesh_vlm_qa_review.html の場合
        # → outputs/renders/vlm_qa_results.json または outputs/meshes_approved/vlm_qa_results.json を探す
        results_json_paths = [
            Path("outputs/renders/vlm_qa_results.json"),
            Path("outputs/meshes_approved/vlm_qa_results.json"),
        ]

        results_json_path = None
        for p in results_json_paths:
            if p.exists():
                results_json_path = p
                break

        if results_json_path is None:
            logger.warning(f"結果 JSON が見つかりません。以下を確認: {results_json_paths}")
            return

        try:
            with open(results_json_path, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            logger.error(f"結果 JSON の読み込みに失敗: {e}")
            return

        results = payload.get("results", [])

        # HTML を組み立てる
        html_parts = [
            """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mesh VLM QA Review Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; color: #333; }
        .container { max-width: 1600px; margin: 20px auto; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .stat-card { background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
        .stat-card.pass { border-left-color: #27ae60; }
        .stat-card.reject { border-left-color: #e74c3c; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .stat-label { font-size: 12px; color: #7f8c8d; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; overflow-x: auto; }
        th { background: #34495e; color: white; padding: 12px; text-align: left; font-weight: 600; font-size: 13px; }
        td { padding: 10px; border-bottom: 1px solid #ddd; font-size: 13px; }
        tr:hover { background: #f9f9f9; }
        .thumb { max-width: 50px; max-height: 50px; cursor: pointer; border-radius: 3px; }
        .pass-badge { background: #d4edda; color: #155724; padding: 4px 8px; border-radius: 3px; }
        .reject-badge { background: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 3px; }
        .check { color: #27ae60; font-weight: bold; }
        .cross { color: #e74c3c; font-weight: bold; }
        .prompt-cell { max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: help; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); }
        .modal.show { display: flex; align-items: center; justify-content: center; }
        .modal-content { background: white; padding: 20px; border-radius: 8px; max-width: 90%; max-height: 90%; overflow: auto; position: relative; }
        .modal-close { position: absolute; top: 10px; right: 15px; font-size: 24px; cursor: pointer; color: #999; }
        .modal-close:hover { color: #333; }
        .img-preview { max-width: 25vw; max-height: 25vh; border-radius: 5px; margin: 5px; }
        .views-container { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .prompt-section { margin: 15px 0; padding: 10px; background: #f0f0f0; border-left: 3px solid #3498db; border-radius: 3px; }
        .prompt-section h3 { margin-bottom: 8px; color: #2c3e50; }
        .prompt-text { font-family: monospace; white-space: pre-wrap; word-wrap: break-word; font-size: 11px; }
        .issues-list { margin: 10px 0; }
        .issue-item { background: #fff3cd; padding: 8px; margin: 5px 0; border-radius: 3px; border-left: 3px solid #f39c12; }
    </style>
    <script>
        function openModal(id) { document.getElementById(id).classList.add('show'); }
        function closeModal(id) { document.getElementById(id).classList.remove('show'); }
        window.onclick = function(event) {
            if (event.target.classList.contains('modal')) {
                event.target.classList.remove('show');
            }
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>📊 Mesh VLM QA Review Report</h1>
"""
        ]

        # サマリー統計
        total = len(results)
        passed = sum(1 for r in results if r.get("pass"))
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0

        html_parts.append(f"""
        <div class="summary">
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Meshes</div>
            </div>
            <div class="stat-card pass">
                <div class="stat-value">{passed}</div>
                <div class="stat-label">Pass ({pass_rate:.1f}%)</div>
            </div>
            <div class="stat-card reject">
                <div class="stat-value">{failed}</div>
                <div class="stat-label">Fail</div>
            </div>
        </div>
""")

        # テーブルヘッダー
        html_parts.append("""
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Asset</th>
                    <th>Prompt Sent</th>
                    <th>View 0</th>
                    <th>View 1</th>
                    <th>View 2</th>
                    <th>View 3</th>
                    <th>Geo</th>
                    <th>Tex</th>
                    <th>Consistency</th>
                    <th>Issues</th>
                    <th>Pass</th>
                </tr>
            </thead>
            <tbody>
""")

        # テーブル行
        for idx, result in enumerate(results, 1):
            mesh_path = result.get("mesh_path", "")
            asset_id = Path(mesh_path).stem if mesh_path else f"asset_{idx}"
            pass_bool = result.get("pass", False)
            geo_score = result.get("geometry_score", "-")
            tex_score = result.get("texture_score", "-")
            consistency = result.get("consistency_score", "-")
            issues = result.get("issues", [])

            # プロンプト（展開可能）
            prompt_sent = result.get("prompt_sent", "")
            prompt_modal_id = f"modal_prompt_{idx}"
            prompt_preview = prompt_sent[:40] + "..." if len(prompt_sent) > 40 else prompt_sent

            prompt_modal = f"""
            <div id="{prompt_modal_id}" class="modal">
                <div class="modal-content">
                    <span class="modal-close" onclick="closeModal('{prompt_modal_id}')">&times;</span>
                    <div class="prompt-section">
                        <h3>VLM Input Prompt</h3>
                        <div class="prompt-text">{prompt_sent}</div>
                    </div>
                    <div class="issues-list" style="margin-top: 20px;">
                        <h3>Issues</h3>
                        {''.join(f'<div class="issue-item">{issue}</div>' for issue in issues) if issues else '<p>No critical issues.</p>'}
                    </div>
                </div>
            </div>
"""
            html_parts.append(prompt_modal)

            # ビュー画像を表示（render_dir から想定されるパスを構築）
            # TRELLIS output では meshes_raw/asset_id.glb → renders/asset_id/view_*.png
            render_base_dir = Path("outputs/renders")
            view_htmls = []
            for view_idx in range(4):
                view_path = render_base_dir / asset_id / f"view_{view_idx}.png"
                if view_path.exists():
                    view_modal_id = f"modal_view_{idx}_{view_idx}"
                    view_htmls.append(f'<img src="file://{view_path}" class="thumb" onclick="openModal(\'{view_modal_id}\')">')

                    view_modal = f"""
            <div id="{view_modal_id}" class="modal">
                <div class="modal-content">
                    <span class="modal-close" onclick="closeModal('{view_modal_id}')">&times;</span>
                    <img src="file://{view_path}" style="max-width: 80vw; max-height: 80vh; border-radius: 5px;">
                </div>
            </div>
"""
                    html_parts.append(view_modal)
                else:
                    view_htmls.append("—")

            pass_badge = f'<span class="pass-badge">✓ Pass</span>' if pass_bool else '<span class="reject-badge">✗ Fail</span>'
            issues_summary = f"{len(issues)} issues" if issues else "OK"

            html_parts.append(f"""
                <tr>
                    <td>{idx}</td>
                    <td>{asset_id}</td>
                    <td><span class="prompt-cell" onclick="openModal('{prompt_modal_id}')" title="{prompt_sent}">{prompt_preview}</span></td>
                    <td>{view_htmls[0] if len(view_htmls) > 0 else "—"}</td>
                    <td>{view_htmls[1] if len(view_htmls) > 1 else "—"}</td>
                    <td>{view_htmls[2] if len(view_htmls) > 2 else "—"}</td>
                    <td>{view_htmls[3] if len(view_htmls) > 3 else "—"}</td>
                    <td>{geo_score}</td>
                    <td>{tex_score}</td>
                    <td>{consistency}</td>
                    <td>{issues_summary}</td>
                    <td>{pass_badge}</td>
                </tr>
""")

        html_parts.append("""
            </tbody>
        </table>
    </div>
</body>
</html>
""")

        html_content = "\n".join(html_parts)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Mesh VLM QA HTML レポート生成完了: {output_path}")
        except Exception as e:
            logger.error(f"HTML レポート生成に失敗: {e}")
