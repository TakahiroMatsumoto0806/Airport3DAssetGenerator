"""
T-2.2: 画像検品エンジン（VLM）

Qwen3-VL-32B-Instruct (vLLM サーバー経由) で生成画像を品質評価する。
Thinking モード: /no_think（高速スクリーニング）

仕様:
  合格基準: realism_score >= 7, object_integrity >= 7, has_artifacts == False
  合格画像は outputs/images_approved/ にコピー
  結果は provenance JSON として保存（pass / review / reject）

使用例:
    qa = ImageQA()
    result = qa.evaluate_single("outputs/images/000001_abc.png")
    # {"realism_score": 8, "object_integrity": 7, "pass": True, ...}

    summary = qa.evaluate_batch(
        image_dir="outputs/images",
        output_json="outputs/images_approved/qa_results.json",
    )
    stats = qa.get_statistics(summary["results"])
"""

import base64
import json
import re
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger


def _safe_format(template: str, **kwargs) -> str:
    """JSON例など波括弧を含むテンプレートの安全なフォーマット。

    str.format() / format_map() は JSON 内の {"key": value} や
    {<type>} 形式を誤ってプレースホルダと解釈してしまう。
    本関数は re.sub で {英数字_} のシンプルな変数名のみを置換する。
    """
    import re

    def _replacer(m: "re.Match") -> str:
        key = m.group(1)
        return str(kwargs[key]) if key in kwargs else m.group(0)

    return re.sub(r"\{(\w+)\}", _replacer, template)


# 評価スキーマ（戻り値の型定義）
QA_RESULT_SCHEMA = {
    "realism_score": int,                      # 1–10
    "object_integrity": int,                   # 1–10
    "background_clean": bool,
    "luggage_type": str,
    "has_artifacts": bool,
    "handle_retracted": bool,                  # キャリーハンドルが完全収納されているか
    "is_bag_closed": bool,                     # 全開口部（ジッパー・バックル等）が閉じているか
    "is_checked_baggage_appropriate": bool,    # 受託手荷物として適切なサイズ・種別か
    "checked_in_ready": bool,                  # 今すぐチェックインカウンターに持って行ける状態か
    "material_estimate": str,
    "is_fully_visible": bool,                  # オブジェクト全体がフレーム内
    "contrast_sufficient": bool,               # 背景との十分なコントラスト
    "object_coverage_pct": int,                # 画面占有率(%) / 理想60-80
    "has_background_shadow": bool,             # 背景への影（TRELLIS で誤メッシュ化）
    "is_sharp_focus": bool,                    # シャープなエッジ（ブラーなし）
    "camera_angle_ok": bool,                   # 正面寄りカメラ（俯瞰・煽り禁止）
    "pass": bool,
    "verdict": str,                            # "pass" | "review" | "reject"
    "reason": str,                             # 不合格の場合の理由
}

# 合格基準
MIN_REALISM = 7
MIN_INTEGRITY = 7

# VLM へのプロンプト
_SYSTEM_PROMPT = """You are a quality control inspector for AI-generated product images.
Your task is to evaluate luggage product images for an airport robotics training dataset.
Images must depict items in CHECKED BAGGAGE condition: handles retracted, zippers closed, bag sealed.
Respond ONLY with a valid JSON object. No markdown, no explanation."""

_USER_PROMPT_TEMPLATE = """/think
Evaluate this product image of airport luggage for suitability as checked baggage in a robotics training dataset,
and as input to a 3D mesh generation model (TRELLIS).
Return a JSON object with exactly these keys:

{{
  "realism_score": <int 1-10, how photorealistic the image looks>,
  "object_integrity": <int 1-10, how complete and undistorted the luggage object is>,
  "background_clean": <true if background is solid white with no gradients, no textures, no patterns>,
  "luggage_type": <string: best matching type, e.g. "hard_suitcase", "backpack", "duffel_bag">,
  "has_artifacts": <true if image has visual glitches, extra unintended objects, or geometric distortions. NOTE: text or labels physically printed on or attached to the luggage (e.g. stickers, tags, branding) are NOT artifacts>,
  "handle_retracted": <true if ALL telescoping/carry handles are fully collapsed and stowed; false if any handle is extended or partially raised; null if the item has no telescoping handle>,
  "is_bag_closed": <true if the item is fully sealed: ALL zippers/clasps/buckles closed for bags, OR all flaps taped shut for cardboard boxes; false if any opening, compartment, or flap is visibly open or unsealed>,
  "is_checked_baggage_appropriate": <true if the item is an appropriate type and size for airline checked baggage — this includes: checked suitcases, large backpacks, duffel bags, equipment cases, golf bags, strollers, instrument cases, sealed cardboard shipping boxes, and any large packaged item typically checked at an airport; false ONLY if the item is clearly too small (small handbag, clutch, tiny purse, wallet, small accessory) OR is a loose unpackaged non-baggage object>,
  "checked_in_ready": <true if you could walk up to an airline check-in counter RIGHT NOW and hand this item to staff — handle retracted or absent, all openings sealed (zippers/clasps for bags, tape for boxes), item is large enough for check-in; false otherwise>,
  "material_estimate": <string: primary material, e.g. "polycarbonate", "nylon", "leather">,
  "is_fully_visible": <true ONLY if the ENTIRE object including all corners, wheels, handles, straps, and protruding parts is fully inside the frame with visible margin on all four sides; false if ANY edge or part of the object touches or goes beyond the image border>,
  "contrast_sufficient": <true if the object has clearly visible edges against the background; false if the object blends in due to similar color>,
  "object_coverage_pct": <int 0-100, estimated percentage of the image area occupied by the object; ideal is 60-80 for 3D generation>,
  "has_background_shadow": <true if there are shadows cast on the background or floor; these are harmful for 3D reconstruction>,
  "is_sharp_focus": <true if the object edges are sharp and in focus; false if depth-of-field blur or soft focus is visible>,
  "camera_angle_ok": <true if the camera is roughly frontal/eye-level (good for 3D); false if extremely top-down or low-angle>,
  "pass": <true if realism_score>={min_realism} AND object_integrity>={min_integrity} AND has_artifacts==false AND is_fully_visible==true AND contrast_sufficient==true AND object_coverage_pct>={min_coverage} AND has_background_shadow==false AND is_sharp_focus==true AND camera_angle_ok==true AND (handle_retracted==true OR handle_retracted==null) AND is_bag_closed==true AND is_checked_baggage_appropriate==true AND checked_in_ready==true>,
  "verdict": <"pass" if pass==true, "review" if borderline, "reject" if clearly bad>,
  "reason": <string: brief reason listing any issues if verdict is not "pass", empty string if "pass">
}}

Reply with only the JSON object."""


def _image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json_response(text: str) -> dict:
    """VLM 応答テキストから JSON を抽出・パースする"""
    # thinking タグを除去
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # コードブロックを除去
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    # 最初の { から最後の } までを抽出
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON が見つかりません: {text[:200]!r}")
    return json.loads(match.group())


def _validate_and_normalize(raw: dict) -> dict:
    """パース結果を検証・正規化してスキーマに合わせる"""
    result: dict = {}

    # スコア（1–10 の int に正規化）
    for key in ("realism_score", "object_integrity"):
        val = raw.get(key, 5)
        try:
            val = max(1, min(10, int(val)))
        except (TypeError, ValueError):
            val = 5
        result[key] = val

    # bool フィールド
    for key in ("background_clean", "has_artifacts"):
        val = raw.get(key, False)
        result[key] = bool(val)

    # handle_retracted: null（ハンドルなし）は合格扱い、False（ハンドル伸展中）は不合格
    hr = raw.get("handle_retracted")
    if hr is None:
        result["handle_retracted"] = None   # ハンドルなしアイテム（バックパック等）
    else:
        result["handle_retracted"] = bool(hr)

    # is_bag_closed: バッグ全体が閉じているか（デフォルト安全側 = True）
    result["is_bag_closed"] = bool(raw.get("is_bag_closed", True))

    # is_checked_baggage_appropriate: 受託手荷物として適切なサイズ・種別か（デフォルト安全側 = True）
    result["is_checked_baggage_appropriate"] = bool(raw.get("is_checked_baggage_appropriate", True))

    # checked_in_ready: 今すぐチェックインカウンターに持って行ける状態か（デフォルト安全側 = True）
    result["checked_in_ready"] = bool(raw.get("checked_in_ready", True))

    # 視認性・構図チェックフィールド（デフォルトは安全側 = True）
    result["is_fully_visible"] = bool(raw.get("is_fully_visible", True))
    result["contrast_sufficient"] = bool(raw.get("contrast_sufficient", True))
    result["is_sharp_focus"] = bool(raw.get("is_sharp_focus", True))
    result["camera_angle_ok"] = bool(raw.get("camera_angle_ok", True))
    result["has_background_shadow"] = bool(raw.get("has_background_shadow", False))

    # 画面占有率（0-100 に clamp）
    try:
        cov = int(raw.get("object_coverage_pct", 50))
        result["object_coverage_pct"] = max(0, min(100, cov))
    except (TypeError, ValueError):
        result["object_coverage_pct"] = 50

    # 文字列フィールド
    result["luggage_type"] = str(raw.get("luggage_type", "unknown"))
    result["material_estimate"] = str(raw.get("material_estimate", "unknown"))
    result["reason"] = str(raw.get("reason", ""))

    # pass を合格基準で再計算（VLM の判断を上書きしてルールを確実に適用）
    # handle_retracted: null（ハンドルなしアイテム）は合格、False（伸展中）は不合格
    handle_ok = result["handle_retracted"] is None or result["handle_retracted"] is True
    passed = (
        result["realism_score"] >= MIN_REALISM
        and result["object_integrity"] >= MIN_INTEGRITY
        and not result["has_artifacts"]
        and result["is_fully_visible"]
        and result["contrast_sufficient"]
        and result["object_coverage_pct"] >= 50
        and not result["has_background_shadow"]
        and result["is_sharp_focus"]
        and result["camera_angle_ok"]
        and handle_ok                                    # キャリーハンドルが収納されているか
        and result["is_bag_closed"]                      # バッグが閉じているか
        and result["is_checked_baggage_appropriate"]     # 受託手荷物として適切か
        and result["checked_in_ready"]                   # チェックインカウンターに持って行ける状態か
    )
    result["pass"] = passed

    # verdict
    # 受託手荷物条件違反・TRELLIS致命的問題は即 reject
    if passed:
        result["verdict"] = "pass"
    elif (not result["is_checked_baggage_appropriate"]
          or not result["checked_in_ready"]
          or result["has_artifacts"]
          or not result["is_fully_visible"]
          or not result["contrast_sufficient"]):
        result["verdict"] = "reject"
    elif (result["object_coverage_pct"] < 30
          or result["has_background_shadow"]
          or not result["is_sharp_focus"]
          or not result["camera_angle_ok"]
          or not result["is_bag_closed"]
          or not handle_ok):
        # 3D再構成に影響大、または受託手荷物状態違反 → reject
        result["verdict"] = "reject"
    elif (result["realism_score"] >= 6
          and result["object_integrity"] >= 6
          and result["object_coverage_pct"] >= 40):
        result["verdict"] = "review"
    else:
        result["verdict"] = "reject"

    return result


class ImageQA:
    """Qwen3-VL-32B-Instruct (vLLM) による画像検品エンジン"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        vllm_base_url: str = "http://localhost:8001/v1",
        max_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        thresholds: dict | None = None,
    ) -> None:
        """
        Args:
            model_name:     vLLM サーバーでサービング中のモデル名
            vllm_base_url:  vLLM サーバーの base URL
            max_tokens:     生成トークン上限
            temperature:    サンプリング温度（0.0 = 決定的）
            system_prompt:  カスタムシステムプロンプト（デフォルト使用時は None）
            user_prompt_template: カスタムユーザープロンプトテンプレート
            thresholds:     カスタム合格基準 dict（realism, integrity等）
        """
        from openai import OpenAI

        self._client = OpenAI(base_url=vllm_base_url, api_key="dummy")
        self._max_tokens = max_tokens
        self._temperature = temperature

        # プロンプト設定（設定ファイルから受け取ったものを優先、なければハードコード）
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._user_prompt_template = user_prompt_template or _USER_PROMPT_TEMPLATE

        # 合格基準（設定ファイルから受け取ったものを優先、なければハードコード）
        self._min_realism = MIN_REALISM
        self._min_integrity = MIN_INTEGRITY
        self._min_coverage_pct = 50
        self._require_fully_visible = True
        self._require_contrast_sufficient = True
        self._require_no_background_shadow = True
        self._require_sharp_focus = True
        self._require_camera_angle_ok = True
        self._require_no_artifacts = True
        self._require_handle_retracted = True
        self._require_bag_closed = True
        self._require_checked_baggage_appropriate = True
        self._require_checked_in_ready = True

        if thresholds:
            self._min_realism = thresholds.get("realism", self._min_realism)
            self._min_integrity = thresholds.get("integrity", self._min_integrity)
            self._min_coverage_pct = thresholds.get("min_coverage_pct", self._min_coverage_pct)
            self._require_fully_visible = thresholds.get("require_fully_visible", self._require_fully_visible)
            self._require_contrast_sufficient = thresholds.get("require_contrast_sufficient", self._require_contrast_sufficient)
            self._require_no_background_shadow = thresholds.get("require_no_background_shadow", self._require_no_background_shadow)
            self._require_sharp_focus = thresholds.get("require_sharp_focus", self._require_sharp_focus)
            self._require_camera_angle_ok = thresholds.get("require_camera_angle_ok", self._require_camera_angle_ok)
            self._require_no_artifacts = thresholds.get("require_no_artifacts", self._require_no_artifacts)
            self._require_handle_retracted = thresholds.get("require_handle_retracted", self._require_handle_retracted)
            self._require_bag_closed = thresholds.get("require_bag_closed", self._require_bag_closed)
            self._require_checked_baggage_appropriate = thresholds.get(
                "require_checked_baggage_appropriate", self._require_checked_baggage_appropriate
            )
            self._require_checked_in_ready = thresholds.get(
                "require_checked_in_ready", self._require_checked_in_ready
            )

        # サービング中のモデル名を取得（起動済みサーバーから確認）
        try:
            models = self._client.models.list()
            self._model_name = models.data[0].id if models.data else model_name
            logger.info(f"ImageQA 初期化: model={self._model_name}, url={vllm_base_url}")
        except Exception as e:
            logger.warning(f"vLLM サーバーへの接続確認に失敗: {e}")
            self._model_name = model_name

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def evaluate_single(self, image_path: str) -> dict:
        """
        1 枚の画像を VLM で評価する。

        Args:
            image_path: 評価する画像ファイルのパス

        Returns:
            {
                realism_score:    int (1–10),
                object_integrity: int (1–10),
                background_clean: bool,
                luggage_type:     str,
                has_artifacts:    bool,
                handle_retracted: bool,
                material_estimate:str,
                pass:             bool,
                verdict:          str ("pass"|"review"|"reject"),
                reason:           str,
            }

        Raises:
            FileNotFoundError: 画像ファイルが存在しない場合
            RuntimeError:      VLM 呼び出し・JSON パースに失敗した場合
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        img_b64 = _image_to_base64(image_path)
        ext = image_path.suffix.lstrip(".").lower()
        mime = "image/png" if ext == "png" else f"image/{ext}"

        # ユーザープロンプトテンプレートをフォーマット（閾値値を代入）
        user_msg_text = _safe_format(
            self._user_prompt_template,
            min_realism=self._min_realism,
            min_integrity=self._min_integrity,
            min_coverage=self._min_coverage_pct,
        )

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        },
                        {"type": "text", "text": user_msg_text},
                    ],
                },
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        raw_text = response.choices[0].message.content
        try:
            raw_json = _parse_json_response(raw_text)
            return _validate_and_normalize(raw_json)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"VLM 応答の JSON パースに失敗: {e}\n応答テキスト: {raw_text[:300]!r}"
            ) from e

    def evaluate_batch(
        self,
        image_dir: str,
        output_json: str,
        approved_dir: Optional[str] = None,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        resume: bool = True,
    ) -> dict:
        """
        ディレクトリ内の全画像を評価し、合格画像をコピーする。

        Args:
            image_dir:    評価対象画像ディレクトリ
            output_json:  結果 JSON 保存先
            approved_dir: 合格画像のコピー先（None の場合は image_dir の親 / images_approved）
            extensions:   対象とするファイル拡張子
            resume:       True の場合、既存結果 JSON を読み込んで処理済みをスキップ

        Returns:
            {
                "total": int,
                "passed": int,
                "reviewed": int,
                "rejected": int,
                "failed_eval": int,  # VLM 評価自体が失敗した件数
                "pass_rate": float,
                "results": list[dict],
            }
        """
        image_dir = Path(image_dir)
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        if approved_dir is None:
            approved_dir = image_dir.parent / "images_approved"
        approved_dir = Path(approved_dir)
        approved_dir.mkdir(parents=True, exist_ok=True)

        # 画像ファイル一覧を収集
        image_files = sorted(
            f for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        total = len(image_files)
        logger.info(f"画像検品開始: {total} 件 → {image_dir}")

        # 既存結果を読み込んで再開ポイントを確認
        existing: dict[str, dict] = {}
        if resume and output_json.exists():
            try:
                with open(output_json, encoding="utf-8") as f:
                    for entry in json.load(f).get("results", []):
                        existing[entry["image_path"]] = entry
                logger.info(f"既存結果読み込み: {len(existing)} 件スキップ対象")
            except Exception as e:
                logger.warning(f"既存結果の読み込みに失敗: {e}")

        results: list[dict] = []
        passed = reviewed = rejected = failed_eval = 0

        # ユーザープロンプトテンプレートをフォーマット（最初に1回）
        user_prompt_formatted = _safe_format(
            self._user_prompt_template,
            min_realism=self._min_realism,
            min_integrity=self._min_integrity,
            min_coverage=self._min_coverage_pct,
        )

        for idx, img_path in enumerate(image_files):
            img_key = str(img_path)

            # スキップ判定
            if img_key in existing:
                entry = existing[img_key]
                results.append(entry)
                v = entry.get("verdict", "reject")
                if v == "pass":
                    passed += 1
                elif v == "review":
                    reviewed += 1
                else:
                    rejected += 1
                continue

            # VLM 評価
            try:
                qa_result = self.evaluate_single(str(img_path))
                verdict = qa_result["verdict"]

                entry = {
                    "image_path": img_key,
                    "filename": img_path.name,
                    "prompt_sent": user_prompt_formatted,
                    **qa_result,
                }

                # 合格・要確認画像をコピー
                if verdict in ("pass", "review"):
                    dst = approved_dir / img_path.name
                    shutil.copy2(img_path, dst)
                    entry["approved_path"] = str(dst)
                else:
                    entry["approved_path"] = None

                if verdict == "pass":
                    passed += 1
                elif verdict == "review":
                    reviewed += 1
                else:
                    rejected += 1

            except Exception as e:
                logger.error(f"  [{idx}/{total}] 評価失敗 ({img_path.name}): {e}")
                entry = {
                    "image_path": img_key,
                    "filename": img_path.name,
                    "verdict": "reject",
                    "pass": False,
                    "eval_error": str(e),
                    "approved_path": None,
                }
                rejected += 1
                failed_eval += 1

            results.append(entry)

            # 進捗表示
            done = len(results)
            if done % 50 == 0 or done == total:
                pass_rate = passed / done if done > 0 else 0.0
                logger.info(
                    f"  進捗: {done}/{total} "
                    f"(pass={passed}, review={reviewed}, reject={rejected}) "
                    f"合格率={pass_rate:.1%}"
                )

            # 中間保存（50 件ごと）
            if done % 50 == 0:
                self._save_results(
                    results, output_json, total, passed, reviewed, rejected, failed_eval,
                    system_prompt_used=user_prompt_formatted
                )

        # 最終保存
        summary = self._save_results(
            results, output_json, total, passed, reviewed, rejected, failed_eval,
            system_prompt_used=user_prompt_formatted
        )

        pass_rate = passed / total if total > 0 else 0.0
        logger.info(
            f"画像検品完了: pass={passed}, review={reviewed}, reject={rejected} "
            f"合格率={pass_rate:.1%}"
        )

        return summary

    def get_statistics(self, results: list[dict]) -> dict:
        """
        評価結果の統計情報を返す。

        Args:
            results: evaluate_batch() の "results" リスト

        Returns:
            {
                "total": int,
                "passed": int,
                "reviewed": int,
                "rejected": int,
                "pass_rate": float,
                "avg_realism_score": float,
                "avg_object_integrity": float,
                "luggage_type_distribution": dict,
                "material_distribution": dict,
                "rejection_reasons": list[str],
            }
        """
        from collections import Counter

        total = len(results)
        passed = sum(1 for r in results if r.get("verdict") == "pass")
        reviewed = sum(1 for r in results if r.get("verdict") == "review")
        rejected = sum(1 for r in results if r.get("verdict") == "reject")

        realism_scores = [
            r["realism_score"] for r in results if "realism_score" in r
        ]
        integrity_scores = [
            r["object_integrity"] for r in results if "object_integrity" in r
        ]

        type_dist = dict(
            Counter(r.get("luggage_type", "unknown") for r in results)
        )
        material_dist = dict(
            Counter(r.get("material_estimate", "unknown") for r in results)
        )
        rejection_reasons = [
            r.get("reason", "") for r in results
            if r.get("verdict") == "reject" and r.get("reason")
        ]

        return {
            "total": total,
            "passed": passed,
            "reviewed": reviewed,
            "rejected": rejected,
            "pass_rate": passed / total if total > 0 else 0.0,
            "avg_realism_score": sum(realism_scores) / len(realism_scores) if realism_scores else 0.0,
            "avg_object_integrity": sum(integrity_scores) / len(integrity_scores) if integrity_scores else 0.0,
            "luggage_type_distribution": type_dist,
            "material_distribution": material_dist,
            "rejection_reasons": rejection_reasons[:20],  # 上位 20 件
        }

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _save_results(
        self,
        results: list[dict],
        output_json: Path,
        total: int,
        passed: int,
        reviewed: int,
        rejected: int,
        failed_eval: int,
        system_prompt_used: str | None = None,
    ) -> dict:
        pass_rate = passed / len(results) if results else 0.0
        payload = {
            "total": total,
            "evaluated": len(results),
            "passed": passed,
            "reviewed": reviewed,
            "rejected": rejected,
            "failed_eval": failed_eval,
            "pass_rate": round(pass_rate, 4),
            "system_prompt_used": system_prompt_used or self._system_prompt,
            "results": results,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    def generate_html_report(self, output_path: str) -> None:
        """
        評価結果を HTML レポートとして生成する。

        テーブル構成:
        | # | 画像 | Prompt | Real | Integ | Cov% | Conditions | Pass | Verdict | Reason |

        Args:
            output_path: HTML 保存先ファイルパス
        """
        import html as _html

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_json_path = Path("outputs/images_approved/image_qa_results.json")
        if not results_json_path.exists():
            logger.warning(f"結果 JSON が見つかりません: {results_json_path}")
            return

        try:
            with open(results_json_path, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            logger.error(f"結果 JSON の読み込みに失敗: {e}")
            return

        results = payload.get("results", [])

        # モーダルは tbody の外（</table> の後）にまとめて出力する。
        # tbody 内に <div> を置くと不正 HTML になりレイアウトが崩れる。
        modal_parts: list[str] = []

        html_parts = [
            """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image QA Review Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; color: #333; }
        .container { max-width: 1400px; margin: 20px auto; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .stat-card { background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
        .stat-card.pass   { border-left-color: #27ae60; }
        .stat-card.review { border-left-color: #f39c12; }
        .stat-card.reject { border-left-color: #e74c3c; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .stat-label { font-size: 12px; color: #7f8c8d; margin-top: 5px; }
        /* table-layout: fixed で列幅を明示制御し横伸びを防ぐ */
        table { width: 100%; border-collapse: collapse; margin-top: 20px; table-layout: fixed; }
        th { background: #34495e; color: white; padding: 8px 6px; text-align: left; font-weight: 600; white-space: nowrap; font-size: 13px; }
        td { padding: 8px 6px; border-bottom: 1px solid #ddd; vertical-align: top;
             word-break: break-word; overflow-wrap: break-word; white-space: normal; font-size: 13px; }
        tr:hover { background: #f9f9f9; }
        col.c-num     { width: 36px; }
        col.c-img     { width: 72px; }
        col.c-prompt  { width: 130px; }
        col.c-score   { width: 48px; }
        col.c-cond    { width: 108px; }
        col.c-pass    { width: 38px; }
        col.c-verdict { width: 64px; }
        /* c-reason は残り幅を自動取得 */
        td.center { text-align: center; }
        .thumb { max-width: 60px; max-height: 60px; cursor: pointer; border-radius: 3px; display: block; }
        .verdict-pass   { background: #d4edda; color: #155724; padding: 3px 6px; border-radius: 3px; display: inline-block; font-size: 12px; }
        .verdict-review { background: #fff3cd; color: #856404; padding: 3px 6px; border-radius: 3px; display: inline-block; font-size: 12px; }
        .verdict-reject { background: #f8d7da; color: #721c24; padding: 3px 6px; border-radius: 3px; display: inline-block; font-size: 12px; }
        .check    { color: #27ae60; font-weight: bold; }
        .cross    { color: #e74c3c; font-weight: bold; }
        .null-val { color: #bbb; }
        /* 条件アイコン: 3列グリッドで3行 */
        .cond-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px; font-size: 11px; }
        .cond-item { cursor: help; text-align: center; white-space: nowrap; }
        /* プロンプトセル: 3行クランプ、クリックで全文モーダル */
        .prompt-cell { font-size: 11px; color: #555; cursor: pointer;
                       display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
                       overflow: hidden; line-height: 1.4; text-decoration: underline dotted; }
        .reason-cell { font-size: 12px; color: #555; line-height: 1.5; }
        /* モーダル */
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); }
        .modal.show { display: flex; align-items: center; justify-content: center; }
        .modal-content { background: white; padding: 20px; border-radius: 8px; max-width: 90%; max-height: 90%; overflow: auto; position: relative; }
        .modal-close { position: absolute; top: 10px; right: 15px; font-size: 24px; cursor: pointer; color: #999; }
        .modal-close:hover { color: #333; }
        .img-preview { max-width: 80vw; max-height: 80vh; border-radius: 5px; display: block; }
        .prompt-section { margin: 10px 0; padding: 10px; background: #f0f0f0; border-left: 3px solid #3498db; border-radius: 3px; }
        .prompt-section h3 { margin-bottom: 8px; color: #2c3e50; font-size: 14px; }
        .prompt-text { font-family: monospace; white-space: pre-wrap; word-break: break-word; font-size: 12px; }
        /* Conditions 凡例 */
        .legend { margin: 20px 0; padding: 14px 16px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; }
        .legend-title { font-weight: 600; font-size: 13px; color: #2c3e50; margin-bottom: 10px; }
        .legend-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px 20px; }
        .legend-item { font-size: 12px; color: #444; line-height: 1.5; }
        .leg-label { display: inline-block; background: #34495e; color: white; font-size: 11px; font-weight: 600;
                     padding: 1px 5px; border-radius: 3px; margin-right: 4px; min-width: 28px; text-align: center; }
        .legend-note { margin-top: 10px; font-size: 11px; color: #888; border-top: 1px solid #dee2e6; padding-top: 8px; }
        /* フィルターバー */
        .filter-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
                      margin: 16px 0 8px; padding: 12px 14px; background: #f0f4f8;
                      border: 1px solid #cdd8e3; border-radius: 6px; }
        .filter-group { display: flex; align-items: center; gap: 6px; }
        .filter-group label { font-size: 12px; font-weight: 600; color: #555; white-space: nowrap; }
        .fbtn { font-size: 12px; padding: 3px 10px; border: 1px solid #bbb; border-radius: 3px;
                background: white; cursor: pointer; transition: background .15s; }
        .fbtn:hover { background: #e0e8f0; }
        .fbtn.active { background: #34495e; color: white; border-color: #34495e; }
        .fbtn.ng-active { background: #e74c3c; color: white; border-color: #c0392b; }
        #reason-search { font-size: 12px; padding: 3px 8px; border: 1px solid #bbb; border-radius: 3px;
                         width: 180px; outline: none; }
        #reason-search:focus { border-color: #3498db; }
        #row-count { font-size: 12px; color: #666; margin-left: auto; white-space: nowrap; }
        .reset-btn { font-size: 12px; padding: 3px 10px; border: 1px solid #e74c3c; border-radius: 3px;
                     background: white; color: #e74c3c; cursor: pointer; }
        .reset-btn:hover { background: #fdecea; }
        /* ソート可能列ヘッダー */
        th[data-col] { cursor: pointer; user-select: none; }
        th[data-col]:hover { background: #4a6070; }
        th[data-col]::after { content: ' ⇅'; font-size: 10px; opacity: .5; }
        th[data-col].sort-asc::after  { content: ' ▲'; opacity: 1; }
        th[data-col].sort-desc::after { content: ' ▼'; opacity: 1; }
    </style>
    <script>
        function openModal(id) { document.getElementById(id).classList.add('show'); }
        function closeModal(id) { document.getElementById(id).classList.remove('show'); }
        window.onclick = function(e) { if (e.target.classList.contains('modal')) e.target.classList.remove('show'); }

        /* ---- Sort ---- */
        let _sortCol = null, _sortAsc = true;
        function sortTable(col) {
            if (_sortCol === col) { _sortAsc = !_sortAsc; } else { _sortCol = col; _sortAsc = true; }
            document.querySelectorAll('th[data-col]').forEach(th => {
                th.classList.remove('sort-asc', 'sort-desc');
                if (th.dataset.col === col) th.classList.add(_sortAsc ? 'sort-asc' : 'sort-desc');
            });
            const tbody = document.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            rows.sort((a, b) => {
                const va = a.dataset[col] ?? '', vb = b.dataset[col] ?? '';
                const na = parseFloat(va), nb = parseFloat(vb);
                if (!isNaN(na) && !isNaN(nb)) return _sortAsc ? na - nb : nb - na;
                return _sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
            });
            rows.forEach(r => tbody.appendChild(r));
            updateCount();
        }

        /* ---- Filter ---- */
        const _f = { verdict: '', ngConds: new Set(), reason: '' };

        function setVerdict(v, btn) {
            _f.verdict = v;
            document.querySelectorAll('.verdict-fbtn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            applyFilters();
        }
        function toggleCond(cond, btn) {
            if (_f.ngConds.has(cond)) { _f.ngConds.delete(cond); btn.classList.remove('ng-active'); }
            else                       { _f.ngConds.add(cond);    btn.classList.add('ng-active'); }
            applyFilters();
        }
        function setReason(v) { _f.reason = v.toLowerCase(); applyFilters(); }

        function applyFilters() {
            const rows = document.querySelectorAll('tbody tr');
            let visible = 0;
            rows.forEach(row => {
                let show = true;
                if (_f.verdict && row.dataset.verdict !== _f.verdict) show = false;
                if (show && _f.ngConds.size > 0) {
                    // AND条件: 選択したNG条件をすべて満たす行のみ表示
                    for (const c of _f.ngConds) {
                        if (row.dataset[c] !== '0') { show = false; break; }
                    }
                }
                if (show && _f.reason) {
                    const txt = (row.querySelector('.reason-cell')?.textContent || '').toLowerCase();
                    if (!txt.includes(_f.reason)) show = false;
                }
                row.style.display = show ? '' : 'none';
                if (show) visible++;
            });
            updateCount(visible, rows.length);
        }
        function updateCount(v, t) {
            if (v === undefined) {
                const rows = document.querySelectorAll('tbody tr');
                v = Array.from(rows).filter(r => r.style.display !== 'none').length;
                t = rows.length;
            }
            document.getElementById('row-count').textContent = v + ' / ' + t + ' 件表示';
        }
        function resetFilters() {
            _f.verdict = ''; _f.ngConds.clear(); _f.reason = '';
            document.querySelectorAll('.verdict-fbtn').forEach(b => b.classList.remove('active'));
            document.querySelector('.verdict-fbtn[data-all]').classList.add('active');
            document.querySelectorAll('.cond-fbtn').forEach(b => b.classList.remove('ng-active'));
            document.getElementById('reason-search').value = '';
            document.querySelectorAll('tbody tr').forEach(r => r.style.display = '');
            updateCount();
        }
        window.addEventListener('DOMContentLoaded', updateCount);
    </script>
</head>
<body>
    <div class="container">
        <h1>Image QA Review Report</h1>
"""
        ]

        # サマリー統計
        total = len(results)
        passed = sum(1 for r in results if r.get("verdict") == "pass")
        reviewed = sum(1 for r in results if r.get("verdict") == "review")
        rejected = sum(1 for r in results if r.get("verdict") == "reject")
        pass_rate = (passed / total * 100) if total > 0 else 0

        html_parts.append(f"""
        <div class="summary">
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-card pass">
                <div class="stat-value">{passed}</div>
                <div class="stat-label">Pass ({pass_rate:.1f}%)</div>
            </div>
            <div class="stat-card review">
                <div class="stat-value">{reviewed}</div>
                <div class="stat-label">Review</div>
            </div>
            <div class="stat-card reject">
                <div class="stat-value">{rejected}</div>
                <div class="stat-label">Reject</div>
            </div>
        </div>
""")

        html_parts.append("""
        <div class="legend">
            <div class="legend-title">Conditions 列の見方</div>
            <div class="legend-grid">
                <div class="legend-item"><span class="leg-label">Vis</span> Fully Visible — 荷物が画面内に完全に収まっているか（見切れNG）</div>
                <div class="legend-item"><span class="leg-label">Ctr</span> Contrast — 背景に対して輪郭が明瞭に見えるか</div>
                <div class="legend-item"><span class="leg-label">Fcs</span> Focus — 被写体がシャープにピントが合っているか</div>
                <div class="legend-item"><span class="leg-label">Ang</span> Angle — カメラが正面/水平付近か（真上・真下NG）</div>
                <div class="legend-item"><span class="leg-label">Shd</span> No BG Shadow — 背景・床への落ち影がないか（3D再構成に有害）</div>
                <div class="legend-item"><span class="leg-label">Art</span> No Artifacts — 意図しない別物体・変形・ノイズがないか</div>
                <div class="legend-item"><span class="leg-label">Hdl</span> Handle Retracted — 伸縮ハンドルが完全収納されているか（null=ハンドルなし）</div>
                <div class="legend-item"><span class="leg-label">Cls</span> Bag Closed — 全ファスナー・留め具・フラップが閉じているか</div>
                <div class="legend-item"><span class="leg-label">Rdy</span> Check-in Ready — 今すぐ航空会社カウンターで受託手荷物として預けられる状態か</div>
            </div>
            <div class="legend-note">✓ = OK（緑）　✗ = NG（赤）　– = 該当なし（灰）　Prompt列クリックで送信プロンプト全文を表示</div>
        </div>

        <div class="filter-bar">
            <div class="filter-group">
                <label>Verdict:</label>
                <button class="fbtn verdict-fbtn active" data-all onclick="setVerdict('',this)">All</button>
                <button class="fbtn verdict-fbtn" onclick="setVerdict('pass',this)">Pass</button>
                <button class="fbtn verdict-fbtn" onclick="setVerdict('review',this)">Review</button>
                <button class="fbtn verdict-fbtn" onclick="setVerdict('reject',this)">Reject</button>
            </div>
            <div class="filter-group">
                <label>NG条件 (AND):</label>
                <button class="fbtn cond-fbtn" onclick="toggleCond('vis',this)">Vis</button>
                <button class="fbtn cond-fbtn" onclick="toggleCond('ctr',this)">Ctr</button>
                <button class="fbtn cond-fbtn" onclick="toggleCond('fcs',this)">Fcs</button>
                <button class="fbtn cond-fbtn" onclick="toggleCond('ang',this)">Ang</button>
                <button class="fbtn cond-fbtn" onclick="toggleCond('shd',this)">Shd</button>
                <button class="fbtn cond-fbtn" onclick="toggleCond('art',this)">Art</button>
                <button class="fbtn cond-fbtn" onclick="toggleCond('hdl',this)">Hdl</button>
                <button class="fbtn cond-fbtn" onclick="toggleCond('cls',this)">Cls</button>
                <button class="fbtn cond-fbtn" onclick="toggleCond('rdy',this)">Rdy</button>
            </div>
            <div class="filter-group">
                <label>Reason:</label>
                <input id="reason-search" type="text" placeholder="キーワード検索..." oninput="setReason(this.value)">
            </div>
            <span id="row-count"></span>
            <button class="reset-btn" onclick="resetFilters()">リセット</button>
        </div>

        <table>
            <colgroup>
                <col class="c-num"><col class="c-img"><col class="c-prompt">
                <col class="c-score"><col class="c-score"><col class="c-score">
                <col class="c-cond"><col class="c-pass"><col class="c-verdict">
                <col>
            </colgroup>
            <thead>
                <tr>
                    <th data-col="idx" onclick="sortTable('idx')">#</th>
                    <th>Image</th>
                    <th>Prompt</th>
                    <th data-col="realism" onclick="sortTable('realism')">Real</th>
                    <th data-col="integrity" onclick="sortTable('integrity')">Integ</th>
                    <th data-col="coverage" onclick="sortTable('coverage')">Cov%</th>
                    <th title="Vis/Ctr/Fcs / Ang/Shd/Art / Hdl/Cls/Rdy">Conditions</th>
                    <th data-col="pass" onclick="sortTable('pass')">Pass</th>
                    <th data-col="verdict" onclick="sortTable('verdict')">Verdict</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
""")

        def _ci(ok, label: str, ok_tip: str, ng_tip: str) -> str:
            """条件アイコン1個。ok=True→緑✓、False→赤✗、None→灰-"""
            if ok is None:
                return f'<span class="cond-item null-val" title="{label}: N/A">-</span>'
            cls = "check" if ok else "cross"
            sym = "✓" if ok else "✗"
            tip = ok_tip if ok else ng_tip
            return f'<span class="cond-item {cls}" title="{label}: {tip}">{sym}{label}</span>'

        for idx, result in enumerate(results, 1):
            image_path = result.get("image_path", "")
            verdict = result.get("verdict", "reject")
            realism = result.get("realism_score", "-")
            integrity = result.get("object_integrity", "-")
            coverage = result.get("object_coverage_pct", "-")
            pass_bool = result.get("pass", False)
            reason = _html.escape(result.get("reason", "") or "")

            is_fully_visible    = result.get("is_fully_visible", None)
            contrast_sufficient = result.get("contrast_sufficient", None)
            is_sharp_focus      = result.get("is_sharp_focus", None)
            camera_angle_ok     = result.get("camera_angle_ok", None)
            has_bg_shadow       = result.get("has_background_shadow", None)
            has_artifacts       = result.get("has_artifacts", None)
            handle_retracted    = result.get("handle_retracted", None)
            is_bag_closed       = result.get("is_bag_closed", None)
            checked_in_ready    = result.get("checked_in_ready", None)

            cond_html = (
                '<div class="cond-grid">'
                + _ci(is_fully_visible,    "Vis", "fully visible",  "cut off")
                + _ci(contrast_sufficient, "Ctr", "good contrast",  "low contrast")
                + _ci(is_sharp_focus,      "Fcs", "sharp",          "blurry")
                + _ci(camera_angle_ok,     "Ang", "angle OK",       "bad angle")
                + _ci(None if has_bg_shadow  is None else not has_bg_shadow,  "Shd", "no BG shadow", "BG shadow")
                + _ci(None if has_artifacts  is None else not has_artifacts,  "Art", "no artifacts", "artifacts")
                + _ci(handle_retracted,    "Hdl", "retracted",      "extended")
                + _ci(is_bag_closed,       "Cls", "closed",         "open")
                + _ci(checked_in_ready,    "Rdy", "ready",          "not ready")
                + '</div>'
            )

            verdict_class = f"verdict-{verdict}"
            modal_id = f"modal_img_{idx}"

            img_thumb = ""
            img_abs = Path(image_path).resolve() if image_path else None
            if img_abs and img_abs.exists():
                b64_data = base64.b64encode(img_abs.read_bytes()).decode()
                img_src = f"data:image/png;base64,{b64_data}"
                img_thumb = (
                    f'<img src="{img_src}" class="thumb"'
                    f' onclick="openModal(\'{modal_id}\')" title="Click to expand">'
                )
                modal_parts.append(f"""
    <div id="{modal_id}" class="modal">
        <div class="modal-content">
            <span class="modal-close" onclick="closeModal('{modal_id}')">&times;</span>
            <img src="{img_src}" class="img-preview">
        </div>
    </div>""")

            prompt_sent = result.get("prompt_sent", "")
            prompt_modal_id = f"modal_prompt_{idx}"
            prompt_preview = _html.escape(
                prompt_sent[:80] + "..." if len(prompt_sent) > 80 else prompt_sent
            )
            modal_parts.append(f"""
    <div id="{prompt_modal_id}" class="modal">
        <div class="modal-content">
            <span class="modal-close" onclick="closeModal('{prompt_modal_id}')">&times;</span>
            <div class="prompt-section">
                <h3>Prompt Sent to FLUX</h3>
                <div class="prompt-text">{_html.escape(prompt_sent)}</div>
            </div>
        </div>
    </div>""")

            pass_sym = '<span class="check">✓</span>' if pass_bool else '<span class="cross">✗</span>'

            def _dv(v):
                """data属性用: None→'null'、bool→0/1、その他そのまま"""
                if v is None: return 'null'
                if isinstance(v, bool): return '1' if v else '0'
                return str(v) if v != '-' else '-1'

            html_parts.append(f"""
                <tr data-verdict="{verdict}"
                    data-idx="{idx}"
                    data-realism="{_dv(realism)}"
                    data-integrity="{_dv(integrity)}"
                    data-coverage="{_dv(coverage)}"
                    data-pass="{_dv(pass_bool)}"
                    data-vis="{_dv(is_fully_visible)}"
                    data-ctr="{_dv(contrast_sufficient)}"
                    data-fcs="{_dv(is_sharp_focus)}"
                    data-ang="{_dv(camera_angle_ok)}"
                    data-shd="{_dv(None if has_bg_shadow is None else not has_bg_shadow)}"
                    data-art="{_dv(None if has_artifacts is None else not has_artifacts)}"
                    data-hdl="{_dv(handle_retracted)}"
                    data-cls="{_dv(is_bag_closed)}"
                    data-rdy="{_dv(checked_in_ready)}">
                    <td class="center">{idx}</td>
                    <td>{img_thumb}</td>
                    <td><span class="prompt-cell" onclick="openModal('{prompt_modal_id}')">{prompt_preview}</span></td>
                    <td class="center">{realism}</td>
                    <td class="center">{integrity}</td>
                    <td class="center">{coverage}</td>
                    <td>{cond_html}</td>
                    <td class="center">{pass_sym}</td>
                    <td><span class="{verdict_class}">{verdict}</span></td>
                    <td><span class="reason-cell">{reason}</span></td>
                </tr>
""")

        # tbody・table を閉じてからモーダルをまとめて出力（tbody 内 div は不正 HTML）
        html_parts.append("""
            </tbody>
        </table>
""")
        html_parts.extend(modal_parts)
        html_parts.append("""
    </div>
</body>
</html>
""")

        html_content = "\n".join(html_parts)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Image QA HTML レポート生成完了: {output_path}")
        except Exception as e:
            logger.error(f"HTML レポート生成に失敗: {e}")
