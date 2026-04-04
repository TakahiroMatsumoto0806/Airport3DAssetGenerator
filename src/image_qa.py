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

# 評価スキーマ（戻り値の型定義）
QA_RESULT_SCHEMA = {
    "realism_score": int,        # 1–10
    "object_integrity": int,     # 1–10
    "background_clean": bool,
    "luggage_type": str,
    "has_artifacts": bool,
    "handle_retracted": bool,    # スーツケースの場合
    "material_estimate": str,
    "pass": bool,
    "verdict": str,              # "pass" | "review" | "reject"
    "reason": str,               # 不合格の場合の理由
}

# 合格基準
MIN_REALISM = 7
MIN_INTEGRITY = 7

# VLM へのプロンプト
_SYSTEM_PROMPT = """You are a quality control inspector for AI-generated product images.
Your task is to evaluate luggage product images for a robotics training dataset.
Respond ONLY with a valid JSON object. No markdown, no explanation."""

_USER_PROMPT_TEMPLATE = """/no_think
Evaluate this product image of airport luggage. Return a JSON object with exactly these keys:

{
  "realism_score": <int 1-10, how photorealistic the image looks>,
  "object_integrity": <int 1-10, how complete and undistorted the luggage object is>,
  "background_clean": <true if background is clean white/neutral, false otherwise>,
  "luggage_type": <string: best matching type, e.g. "hard_suitcase", "backpack", "duffel_bag">,
  "has_artifacts": <true if image has visual glitches, text, extra objects, or distortions>,
  "handle_retracted": <true if suitcase handle is retracted/collapsed, false if extended, null if not a suitcase>,
  "material_estimate": <string: primary material, e.g. "polycarbonate", "nylon", "leather">,
  "pass": <true if realism_score>=7 AND object_integrity>=7 AND has_artifacts==false>,
  "verdict": <"pass" if pass==true, "review" if borderline (score 6), "reject" if clearly bad>,
  "reason": <string: brief reason if verdict is "review" or "reject", empty string if "pass">
}

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

    # handle_retracted は None 許容
    hr = raw.get("handle_retracted")
    result["handle_retracted"] = bool(hr) if hr is not None else False

    # 文字列フィールド
    result["luggage_type"] = str(raw.get("luggage_type", "unknown"))
    result["material_estimate"] = str(raw.get("material_estimate", "unknown"))
    result["reason"] = str(raw.get("reason", ""))

    # pass を合格基準で再計算（VLM の判断を上書きしてルールを確実に適用）
    passed = (
        result["realism_score"] >= MIN_REALISM
        and result["object_integrity"] >= MIN_INTEGRITY
        and not result["has_artifacts"]
    )
    result["pass"] = passed

    # verdict
    # has_artifacts == True は即 reject（スコアに関わらず）
    raw_verdict = str(raw.get("verdict", "")).lower()
    if passed:
        result["verdict"] = "pass"
    elif result["has_artifacts"]:
        result["verdict"] = "reject"
    elif raw_verdict == "review" or (
        result["realism_score"] >= 6 and result["object_integrity"] >= 6
    ):
        result["verdict"] = "review"
    else:
        result["verdict"] = "reject"

    return result


class ImageQA:
    """Qwen3-VL-32B-Instruct (vLLM) による画像検品エンジン"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        vllm_base_url: str = "http://localhost:8000/v1",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        """
        Args:
            model_name:     vLLM サーバーでサービング中のモデル名
            vllm_base_url:  vLLM サーバーの base URL
            max_tokens:     生成トークン上限
            temperature:    サンプリング温度（0.0 = 決定的）
        """
        from openai import OpenAI

        self._client = OpenAI(base_url=vllm_base_url, api_key="dummy")
        self._max_tokens = max_tokens
        self._temperature = temperature

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

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        },
                        {"type": "text", "text": _USER_PROMPT_TEMPLATE},
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
                    results, output_json, total, passed, reviewed, rejected, failed_eval
                )

        # 最終保存
        summary = self._save_results(
            results, output_json, total, passed, reviewed, rejected, failed_eval
        )

        pass_rate = passed / total if total > 0 else 0.0
        logger.info(
            f"画像検品完了: pass={passed}, review={reviewed}, reject={rejected} "
            f"合格率={pass_rate:.1%}"
        )
        if pass_rate < 0.70:
            logger.warning(
                f"合格率 {pass_rate:.1%} が目標 70% を下回っています。"
                "プロンプトテンプレートや生成パラメータの見直しを検討してください。"
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
            "results": results,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload
