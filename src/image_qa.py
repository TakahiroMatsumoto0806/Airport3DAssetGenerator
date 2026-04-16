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
  "is_fully_visible": <true if the ENTIRE object is visible with no cropping or cut-off edges, false if any part extends beyond the frame>,
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
        | # | 画像 | 送信プロンプト | Realism | Integrity | Coverage% | 条件 | Pass | Verdict |

        Args:
            output_path: HTML 保存先ファイルパス
        """
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 結果 JSON を読み込む
        # 先ほど保存された結果から、system_prompt_used と results を取得
        # NOTE: evaluate_batch() が呼ばれた直後にこのメソッドを呼ぶ前提
        # （ただし、呼び出さない場合もあるので、JSON から直接読む）

        # 対応する image_qa_results.json を探す（output_path の命名規則から推測）
        # output_path が outputs/reports/image_qa_review.html の場合
        # → outputs/images_approved/image_qa_results.json を探す
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

        system_prompt_used = payload.get("system_prompt_used", "")
        results = payload.get("results", [])

        # HTML を組み立てる
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
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .stat-card { background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
        .stat-card.pass { border-left-color: #27ae60; }
        .stat-card.review { border-left-color: #f39c12; }
        .stat-card.reject { border-left-color: #e74c3c; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .stat-label { font-size: 12px; color: #7f8c8d; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th { background: #34495e; color: white; padding: 12px; text-align: left; font-weight: 600; }
        td { padding: 12px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f9f9f9; }
        .thumb { max-width: 60px; max-height: 60px; cursor: pointer; border-radius: 3px; }
        .verdict-pass { background: #d4edda; color: #155724; padding: 4px 8px; border-radius: 3px; }
        .verdict-review { background: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 3px; }
        .verdict-reject { background: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 3px; }
        .check { color: #27ae60; font-weight: bold; }
        .cross { color: #e74c3c; font-weight: bold; }
        .prompt-cell { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: help; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); }
        .modal.show { display: flex; align-items: center; justify-content: center; }
        .modal-content { background: white; padding: 20px; border-radius: 8px; max-width: 90%; max-height: 90%; overflow: auto; position: relative; }
        .modal-close { position: absolute; top: 10px; right: 15px; font-size: 24px; cursor: pointer; color: #999; }
        .modal-close:hover { color: #333; }
        .img-preview { max-width: 80vw; max-height: 80vh; border-radius: 5px; }
        .prompt-section { margin: 15px 0; padding: 10px; background: #f0f0f0; border-left: 3px solid #3498db; border-radius: 3px; }
        .prompt-section h3 { margin-bottom: 8px; color: #2c3e50; }
        .prompt-text { font-family: monospace; white-space: pre-wrap; word-wrap: break-word; font-size: 12px; }
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
        <h1>📊 Image QA Review Report</h1>
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

        # テーブルヘッダー
        html_parts.append("""
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Image</th>
                    <th>Prompt Sent</th>
                    <th>Realism</th>
                    <th>Integrity</th>
                    <th>Coverage%</th>
                    <th>Conditions</th>
                    <th>Pass</th>
                    <th>Verdict</th>
                </tr>
            </thead>
            <tbody>
""")

        # テーブル行
        for idx, result in enumerate(results, 1):
            filename = result.get("filename", "")
            image_path = result.get("image_path", "")
            verdict = result.get("verdict", "reject")
            realism = result.get("realism_score", "-")
            integrity = result.get("object_integrity", "-")
            coverage = result.get("object_coverage_pct", "-")
            pass_bool = result.get("pass", False)

            # 条件チェック
            is_fully_visible = result.get("is_fully_visible", True)
            contrast_sufficient = result.get("contrast_sufficient", True)
            is_sharp_focus = result.get("is_sharp_focus", True)
            camera_angle_ok = result.get("camera_angle_ok", True)
            has_background_shadow = result.get("has_background_shadow", False)
            has_artifacts = result.get("has_artifacts", False)

            conditions = (
                f"{'✓' if is_fully_visible else '✗'} "
                f"{'✓' if contrast_sufficient else '✗'} "
                f"{'✓' if is_sharp_focus else '✗'} "
                f"{'✓' if camera_angle_ok else '✗'} "
                f"{'✓' if not has_background_shadow else '✗'} "
                f"{'✓' if not has_artifacts else '✗'}"
            )

            verdict_class = f"verdict-{verdict}"
            modal_id = f"modal_img_{idx}"

            # 画像サムネイル生成（存在確認）
            # HTML 出力ディレクトリから画像への相対パスを算出（別 PC でも動作するよう絶対パス禁止）
            img_thumb = ""
            img_abs = Path(image_path).resolve() if image_path else None
            if img_abs and img_abs.exists():
                try:
                    import os as _os
                    rel_src = _os.path.relpath(img_abs, start=output_path.parent.resolve())
                except ValueError:
                    rel_src = str(img_abs)
                # HTML 内では URL 用にスラッシュ区切りを徹底
                rel_src = rel_src.replace("\\", "/")
                img_thumb = f'<img src="{rel_src}" class="thumb" onclick="openModal(\'{modal_id}\')" title="Click to expand">'
                img_modal = f"""
            <div id="{modal_id}" class="modal">
                <div class="modal-content">
                    <span class="modal-close" onclick="closeModal('{modal_id}')">&times;</span>
                    <img src="{rel_src}" class="img-preview">
                </div>
            </div>
"""
                html_parts.append(img_modal)

            # プロンプト（展開可能）
            prompt_sent = result.get("prompt_sent", "")
            prompt_modal_id = f"modal_prompt_{idx}"
            prompt_preview = prompt_sent[:50] + "..." if len(prompt_sent) > 50 else prompt_sent

            prompt_modal = f"""
            <div id="{prompt_modal_id}" class="modal">
                <div class="modal-content">
                    <span class="modal-close" onclick="closeModal('{prompt_modal_id}')">&times;</span>
                    <div class="prompt-section">
                        <h3>VLM Input Prompt</h3>
                        <div class="prompt-text">{prompt_sent}</div>
                    </div>
                </div>
            </div>
"""
            html_parts.append(prompt_modal)

            html_parts.append(f"""
                <tr>
                    <td>{idx}</td>
                    <td>{img_thumb}</td>
                    <td><span class="prompt-cell" onclick="openModal('{prompt_modal_id}')" title="{prompt_sent}">{prompt_preview}</span></td>
                    <td>{realism}</td>
                    <td>{integrity}</td>
                    <td>{coverage}</td>
                    <td>{conditions}</td>
                    <td>{"✓" if pass_bool else "✗"}</td>
                    <td><span class="{verdict_class}">{verdict}</span></td>
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
            logger.info(f"Image QA HTML レポート生成完了: {output_path}")
        except Exception as e:
            logger.error(f"HTML レポート生成に失敗: {e}")
