#!/usr/bin/env python3
"""
T-6.2: レポート生成スクリプト

パイプライン全体を再実行せずにレポートだけ更新したい場合に使用。

使用例:
    # 画像 QA リジェクト分析レポートを生成（デフォルト）
    python scripts/generate_report.py

    # 同上、明示的に指定
    python scripts/generate_report.py --step image_qa

    # カテゴリ別合格率レポートを生成（最終合格の 70% を hard_suitcase にする重み推奨）
    python scripts/generate_report.py --step pass_rate
    python scripts/generate_report.py --step pass_rate --target-hs-fraction 0.70
    python scripts/generate_report.py --step pass_rate \\
        --qa-results outputs/images_approved/image_qa_results.json \\
        --prompts outputs/prompts/prompts.json \\
        --output-dir outputs/reports

    # 出力ディレクトリを指定
    python scripts/generate_report.py --output-dir outputs/reports/v2

出力ファイル:
    outputs/reports/image_qa_rejection_analysis.json — 画像QAリジェクト分析（--step image_qa 時）
    outputs/reports/image_qa_rejection_analysis.html — 画像QAリジェクト分析 HTML（--step image_qa 時）
    outputs/reports/pass_rate_report.json          — カテゴリ別合格率レポート（--step pass_rate 時）
    outputs/reports/pass_rate_report.html          — カテゴリ別合格率レポート HTML（--step pass_rate 時）
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from omegaconf import OmegaConf

def analyze_image_qa_rejections(
    qa_results_path: Path,
    output_dir: Path,
) -> dict:
    """画像 QA 結果 JSON からリジェクト理由を集計してレポートを生成する。

    Args:
        qa_results_path: image_qa_results.json のパス
        output_dir:      レポート出力先ディレクトリ

    Returns:
        集計結果 dict（JSON に書き出す内容と同一）
    """
    if not qa_results_path.exists():
        logger.error(f"QA 結果ファイルが見つかりません: {qa_results_path}")
        return {}

    with open(qa_results_path, encoding="utf-8") as f:
        raw = json.load(f)

    # results キーがある場合はそちらを使う（evaluate_batch の出力形式）
    results: list[dict] = raw.get("results", raw) if isinstance(raw, dict) else raw

    total = len(results)
    pass_count = sum(1 for r in results if r.get("verdict") == "pass")
    review_count = sum(1 for r in results if r.get("verdict") == "review")
    reject_count = sum(1 for r in results if r.get("verdict") == "reject")
    pass_rate = round(pass_count / total, 4) if total > 0 else 0.0

    # 条件別失敗カウント（reject + review の中での内訳）
    non_pass = [r for r in results if r.get("verdict") != "pass"]
    condition_counts: dict[str, int] = {
        "realism_low": sum(1 for r in non_pass if r.get("realism_score", 10) < 7),
        "integrity_low": sum(1 for r in non_pass if r.get("object_integrity", 10) < 7),
        "has_artifacts": sum(1 for r in non_pass if r.get("has_artifacts", False)),
        "handle_extended": sum(
            1 for r in non_pass
            if r.get("handle_retracted") is False  # None (no handle) は除外
        ),
        "bag_open": sum(1 for r in non_pass if not r.get("is_bag_closed", True)),
        "not_checked_baggage_appropriate": sum(
            1 for r in non_pass if not r.get("is_checked_baggage_appropriate", True)
        ),
        "not_checked_in_ready": sum(
            1 for r in non_pass if not r.get("checked_in_ready", True)
        ),
        "shadow": sum(1 for r in non_pass if r.get("has_background_shadow", False)),
        "blur": sum(1 for r in non_pass if not r.get("is_sharp_focus", True)),
        "coverage_low": sum(
            1 for r in non_pass if r.get("object_coverage_pct", 100) < 50
        ),
        "bad_angle": sum(1 for r in non_pass if not r.get("camera_angle_ok", True)),
        "not_fully_visible": sum(
            1 for r in non_pass if not r.get("is_fully_visible", True)
        ),
        "low_contrast": sum(
            1 for r in non_pass if not r.get("contrast_sufficient", True)
        ),
    }

    # カテゴリ別合格率
    cat_stats: dict[str, dict] = {}
    for r in results:
        cat = r.get("luggage_type", "unknown")
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "pass": 0}
        cat_stats[cat]["total"] += 1
        if r.get("verdict") == "pass":
            cat_stats[cat]["pass"] += 1
    category_pass_rates = {
        cat: {
            "total": v["total"],
            "pass": v["pass"],
            "pass_rate": round(v["pass"] / v["total"], 4) if v["total"] > 0 else 0.0,
        }
        for cat, v in sorted(cat_stats.items())
    }

    # Top 10 reject reasons（reason フィールドのテキストを単語・フレーズ単位で頻出順に）
    reason_counter: collections.Counter = collections.Counter()
    for r in non_pass:
        reason_text = r.get("reason", "").strip()
        if reason_text:
            reason_counter[reason_text] += 1
    top_reasons = [
        {"reason": reason, "count": cnt}
        for reason, cnt in reason_counter.most_common(10)
    ]

    summary = {
        "total": total,
        "pass": pass_count,
        "review": review_count,
        "reject": reject_count,
        "pass_rate": pass_rate,
        "condition_counts": condition_counts,
        "category_pass_rates": category_pass_rates,
        "top_reject_reasons": top_reasons,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 出力
    json_path = output_dir / "image_qa_rejection_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"  リジェクト分析 JSON: {json_path}")

    # HTML 出力
    html_path = output_dir / "image_qa_rejection_analysis.html"
    _write_rejection_html(summary, html_path)
    logger.info(f"  リジェクト分析 HTML: {html_path}")

    return summary


def _write_rejection_html(summary: dict, html_path: Path) -> None:
    """リジェクト分析結果を CSS のみの HTML に書き出す"""
    cond = summary["condition_counts"]
    non_pass_total = summary["review"] + summary["reject"]

    # 条件別バーチャート行を生成
    max_count = max(cond.values(), default=1)

    def _bar_row(label: str, key: str) -> str:
        cnt = cond.get(key, 0)
        pct = round(cnt / non_pass_total * 100, 1) if non_pass_total > 0 else 0
        bar_width = round(cnt / max_count * 100) if max_count > 0 else 0
        return (
            f"<tr><td class='label'>{label}</td>"
            f"<td class='bar-cell'><div class='bar' style='width:{bar_width}%'></div></td>"
            f"<td class='count'>{cnt}</td>"
            f"<td class='pct'>{pct}%</td></tr>"
        )

    cond_rows = "".join([
        _bar_row("リアリズム低 (< 7)", "realism_low"),
        _bar_row("オブジェクト完全性低 (< 7)", "integrity_low"),
        _bar_row("アーティファクト", "has_artifacts"),
        _bar_row("ハンドル伸展中", "handle_extended"),
        _bar_row("バッグ開放", "bag_open"),
        _bar_row("受託手荷物不適切", "not_checked_baggage_appropriate"),
        _bar_row("チェックイン不可状態", "not_checked_in_ready"),
        _bar_row("背景影", "shadow"),
        _bar_row("ピンボケ", "blur"),
        _bar_row("画面占有率低 (< 50%)", "coverage_low"),
        _bar_row("カメラ角度NG", "bad_angle"),
        _bar_row("オブジェクトはみ出し", "not_fully_visible"),
        _bar_row("コントラスト不足", "low_contrast"),
    ])

    # カテゴリ別合格率テーブル行
    cat_rows = ""
    for cat, v in summary["category_pass_rates"].items():
        pct = round(v["pass_rate"] * 100, 1)
        bar_w = round(v["pass_rate"] * 100)
        cat_rows += (
            f"<tr><td class='label'>{cat}</td>"
            f"<td class='bar-cell'><div class='bar bar-cat' style='width:{bar_w}%'></div></td>"
            f"<td class='count'>{v['pass']} / {v['total']}</td>"
            f"<td class='pct'>{pct}%</td></tr>"
        )

    # Top reasons テーブル行
    reason_rows = "".join(
        f"<tr><td>{i+1}</td><td>{item['reason']}</td><td>{item['count']}</td></tr>"
        for i, item in enumerate(summary["top_reject_reasons"])
    )

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>AL3DG — Image QA Rejection Analysis</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; background: #fafafa; color: #222; }}
  h1 {{ color: #333; }}
  h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(4, auto); gap: 16px; margin: 16px 0; }}
  .stat-box {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px 20px; text-align: center; }}
  .stat-box .num {{ font-size: 2em; font-weight: bold; color: #333; }}
  .stat-box .lbl {{ font-size: 0.85em; color: #888; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 800px; }}
  th {{ background: #eee; text-align: left; padding: 8px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
  td.label {{ width: 220px; }}
  td.bar-cell {{ width: 300px; }}
  td.count {{ width: 60px; text-align: right; }}
  td.pct {{ width: 60px; text-align: right; color: #666; }}
  .bar {{ height: 18px; background: #e05; border-radius: 3px; min-width: 2px; }}
  .bar-cat {{ background: #28a; }}
  .pass-rate {{ font-size: 1.2em; font-weight: bold; color: #22a; }}
</style>
</head>
<body>
<h1>AL3DG — Image QA Rejection Analysis</h1>

<h2>全体サマリー</h2>
<div class="summary-grid">
  <div class="stat-box"><div class="num">{summary["total"]}</div><div class="lbl">総数</div></div>
  <div class="stat-box"><div class="num" style="color:#282">{summary["pass"]}</div><div class="lbl">合格 (pass)</div></div>
  <div class="stat-box"><div class="num" style="color:#a80">{summary["review"]}</div><div class="lbl">要確認 (review)</div></div>
  <div class="stat-box"><div class="num" style="color:#e05">{summary["reject"]}</div><div class="lbl">不合格 (reject)</div></div>
</div>
<p class="pass-rate">合格率: {round(summary["pass_rate"]*100, 1)}%</p>

<h2>条件別失敗カウント (reject + review, 合計 {non_pass_total} 件)</h2>
<table>
  <thead><tr><th>条件</th><th>バー</th><th>件数</th><th>割合</th></tr></thead>
  <tbody>{cond_rows}</tbody>
</table>

<h2>カテゴリ別合格率</h2>
<table>
  <thead><tr><th>カテゴリ</th><th>バー</th><th>合格/総数</th><th>合格率</th></tr></thead>
  <tbody>{cat_rows}</tbody>
</table>

<h2>Top 10 リジェクト理由</h2>
<table>
  <thead><tr><th>#</th><th>理由</th><th>件数</th></tr></thead>
  <tbody>{reason_rows}</tbody>
</table>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson スコア 95% 信頼区間を計算する（scipy 不要）。

    Args:
        k: 成功数
        n: 試行数
        z: 正規分布のz値（95%CI なら 1.96）

    Returns:
        (lower, upper) — 0.0〜1.0 の範囲にクランプ済み
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z / (1 + z2 / n)) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def _calc_recommended_weights(
    cat_pass_rates: dict[str, float],
    current_weights: dict[str, float],
    target_hs_fraction: float,
    min_weight_zero: float = 0.005,
) -> dict[str, float]:
    """目標 hard_suitcase 最終合格割合を達成するサンプリング重みを計算する。

    目標条件:
        w_hs * p_hs / sum(w_i * p_i) = target_hs_fraction

    解法:
      - 合格率 0% カテゴリ: w = min_weight_zero（探索継続のため最低重みを残す）
      - 非 hard_suitcase の active カテゴリ: w_i = k * current_weight_i (比率維持)
      - hard_suitcase: 連立方程式を解いて w_hs を算出

    Args:
        cat_pass_rates:      {category: pass_rate}（0.0〜1.0）
        current_weights:     {category: current_weight}（合計 1.0 を想定）
        target_hs_fraction:  最終合格のうち hard_suitcase が占める目標割合
        min_weight_zero:     合格率 0% カテゴリに割り当てる最低重み

    Returns:
        {category: new_weight}（合計 1.0 に正規化済み）
    """
    p_hs = cat_pass_rates.get("hard_suitcase", 0.0)
    f = target_hs_fraction
    eps = min_weight_zero

    if p_hs <= 0.0:
        logger.warning("hard_suitcase の合格率が 0% のため重み計算をスキップします")
        return current_weights.copy()

    active_nonhs = {
        c: p for c, p in cat_pass_rates.items()
        if c != "hard_suitcase" and p > 0.0
    }
    zero_cats = [c for c in cat_pass_rates if cat_pass_rates[c] == 0.0 and c != "hard_suitcase"]
    reserved = len(zero_cats) * eps

    S = sum(current_weights.get(c, 0.0) for c in active_nonhs)
    Q = sum(current_weights.get(c, 0.0) * p for c, p in active_nonhs.items())

    # 非ゼロカテゴリが hard_suitcase だけの極端なケースを回避
    if Q == 0.0 and S == 0.0:
        weights: dict[str, float] = {"hard_suitcase": 1.0 - reserved}
        for cat in zero_cats:
            weights[cat] = eps
        total = sum(weights.values())
        return {c: round(w / total, 4) for c, w in weights.items()}

    denominator = f * Q / (p_hs * (1.0 - f)) + S
    if denominator == 0.0:
        logger.warning("重み計算の分母がゼロです。現在の重みをそのまま返します。")
        return current_weights.copy()

    k_scale = (1.0 - reserved) / denominator
    w_hs = f * k_scale * Q / (p_hs * (1.0 - f))

    weights = {"hard_suitcase": w_hs}
    for cat in active_nonhs:
        weights[cat] = k_scale * current_weights.get(cat, 0.0)
    for cat in zero_cats:
        weights[cat] = eps

    # カテゴリが設定に存在するが pass_rates に含まれない場合（測定対象外）
    for cat in current_weights:
        if cat not in weights:
            weights[cat] = eps

    total = sum(weights.values())
    return {c: round(w / total, 4) for c, w in weights.items()}


def analyze_pass_rates(
    qa_results_path: Path,
    prompts_path: Path,
    target_hs_fraction: float,
    output_dir: Path,
    config_path: Path,
) -> dict:
    """カテゴリ別合格率を集計し、Wilson CI・推奨重みをレポートとして出力する。

    Args:
        qa_results_path:     image_qa_results.json のパス
        prompts_path:        prompts.json のパス（prompt_id → luggage_type マップ構築用）
        target_hs_fraction:  最終合格のうち hard_suitcase が占める目標割合
        output_dir:          レポート出力先ディレクトリ
        config_path:         設定ファイルパス（現在の category_weights 読み取り用）

    Returns:
        レポート dict（JSON に書き出す内容と同一）
    """
    if not qa_results_path.exists():
        logger.error(f"QA 結果ファイルが見つかりません: {qa_results_path}")
        return {}

    # --- prompts.json から {prompt_id → luggage_type} マップを構築 ---
    pid_to_cat: dict[str, str] = {}
    if prompts_path.exists():
        with open(prompts_path, encoding="utf-8") as f:
            prompts_raw = json.load(f)
        prompts_list: list[dict] = (
            prompts_raw if isinstance(prompts_raw, list)
            else prompts_raw.get("prompts", [])
        )
        for item in prompts_list:
            meta = item.get("metadata", {})
            pid = meta.get("prompt_id", "")
            cat = meta.get("luggage_type", "")
            if pid and cat:
                pid_to_cat[pid] = cat
        logger.info(f"prompts.json からカテゴリマップ構築: {len(pid_to_cat)} 件")
    else:
        logger.warning(f"prompts.json が見つかりません: {prompts_path}（QA 結果の luggage_type フィールドを使用）")

    # --- 現在のカテゴリ重みを設定ファイルから読み取る ---
    current_weights: dict[str, float] = {}
    if config_path.exists():
        try:
            from omegaconf import OmegaConf
            templates_path = config_path.parent / "prompt_templates.yaml"
            if templates_path.exists():
                tmpl = OmegaConf.load(str(templates_path))
                cw = OmegaConf.to_container(tmpl.get("sampling", {}).get("category_weights", {}))
                current_weights = {k: float(v) for k, v in cw.items()}
                logger.info(f"category_weights 読み取り: {len(current_weights)} カテゴリ")
        except Exception as e:
            logger.warning(f"category_weights 読み取りエラー: {e}")

    # --- QA 結果を読み込む ---
    with open(qa_results_path, encoding="utf-8") as f:
        raw = json.load(f)
    results: list[dict] = raw.get("results", raw) if isinstance(raw, dict) else raw

    # --- カテゴリ別に集計 ---
    cat_stats: dict[str, dict[str, int]] = {}
    for r in results:
        # luggage_type: QA 結果の直接フィールド → prompt_id マップ → "unknown" の優先順
        cat = r.get("luggage_type", "")
        if not cat:
            image_path = r.get("image_path", "")
            stem = Path(image_path).stem if image_path else ""
            # prompt_id はファイル名末尾 12 文字（MD5 先頭 12 文字）
            pid = stem[-12:] if len(stem) >= 12 else stem
            cat = pid_to_cat.get(pid, pid_to_cat.get(stem, "unknown"))
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "passed": 0}
        cat_stats[cat]["total"] += 1
        if r.get("verdict") == "pass":
            cat_stats[cat]["passed"] += 1

    total_generated = sum(v["total"] for v in cat_stats.values())
    total_passed = sum(v["passed"] for v in cat_stats.values())

    # --- Wilson CI の計算 ---
    categories_detail: dict[str, dict] = {}
    cat_pass_rates: dict[str, float] = {}

    for cat in sorted(cat_stats.keys()):
        v = cat_stats[cat]
        n, k = v["total"], v["passed"]
        p = k / n if n > 0 else 0.0
        ci_lo, ci_hi = _wilson_ci(k, n)
        ci_half = (ci_hi - ci_lo) / 2.0
        status = (
            "zero_pass_investigate" if p == 0.0 and n > 0
            else "active" if p > 0.0
            else "no_samples"
        )
        categories_detail[cat] = {
            "samples": n,
            "passed": k,
            "pass_rate": round(p, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "ci_half_width": round(ci_half, 4),
            "status": status,
            "current_weight": current_weights.get(cat, None),
        }
        cat_pass_rates[cat] = p

    # --- 推奨重み計算 ---
    rec_weights: dict[str, float] = {}
    yaml_snippet: str = ""
    excluded_cats: list[str] = []
    if current_weights and "hard_suitcase" in cat_pass_rates:
        rec_weights = _calc_recommended_weights(
            cat_pass_rates=cat_pass_rates,
            current_weights=current_weights,
            target_hs_fraction=target_hs_fraction,
        )
        # カテゴリ別に推奨重みをマージ
        for cat in categories_detail:
            categories_detail[cat]["recommended_weight"] = rec_weights.get(cat)

        excluded_cats = [
            c for c, p in cat_pass_rates.items() if p == 0.0 and c != "hard_suitcase"
        ]

        # YAML スニペット生成
        lines = ["sampling:", "  category_weights:"]
        for cat, w in sorted(rec_weights.items(), key=lambda x: -x[1]):
            lines.append(f"    {cat}: {w}")
        yaml_snippet = "\n".join(lines)

    # --- 精度ガイド ---
    n_current = total_generated // len(cat_stats) if cat_stats else 0
    achieved_eps = math.ceil(1.96 * math.sqrt(0.25 / max(n_current, 1)) * 100) if n_current > 0 else 0
    n_cats = len(cat_stats)
    precision_table = []
    for eps_pct, n_req in [(5, 385), (10, 97), (15, 43), (20, 25)]:
        total_imgs = n_req * n_cats
        est_h_gen = round(total_imgs * 35 / 3600, 1)
        est_h_qa = round(total_imgs * 100 / 3600, 1)
        precision_table.append({
            "epsilon_pct": eps_pct,
            "n_required": n_req,
            "total": total_imgs,
            "est_gen_hours": est_h_gen,
            "est_qa_hours": est_h_qa,
            "est_total_hours": round(est_h_gen + est_h_qa, 1),
        })

    report = {
        "metadata": {
            "run_date": "2026-04-12",
            "n_per_category": n_current,
            "target_hs_fraction": target_hs_fraction,
            "total_generated": total_generated,
            "total_passed": total_passed,
            "overall_pass_rate": round(total_passed / total_generated, 4) if total_generated > 0 else 0.0,
        },
        "precision_guide": {
            "current_n_per_category": n_current,
            "achieved_precision_pct": achieved_eps,
            "n_categories": n_cats,
            "table": precision_table,
        },
        "categories": categories_detail,
        "weight_recommendations": {
            "target_hs_fraction": target_hs_fraction,
            "recommended_weights": rec_weights,
            "excluded_from_optimization": excluded_cats,
            "yaml_snippet": yaml_snippet,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "pass_rate_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"  合格率レポート JSON: {json_path}")

    html_path = output_dir / "pass_rate_report.html"
    _write_pass_rate_html(report, html_path)
    logger.info(f"  合格率レポート HTML: {html_path}")

    return report


def _write_pass_rate_html(report: dict, html_path: Path) -> None:
    """合格率レポートを CSS のみの HTML に書き出す"""
    meta = report["metadata"]
    guide = report["precision_guide"]
    cats = report["categories"]
    rec = report["weight_recommendations"]

    # 精度ガイドテーブル行
    guide_rows = ""
    for row in guide["table"]:
        current_marker = " ★" if row["n_required"] <= guide["current_n_per_category"] else ""
        guide_rows += (
            f"<tr><td>±{row['epsilon_pct']}%</td>"
            f"<td>{row['n_required']}</td>"
            f"<td>{row['total']}</td>"
            f"<td>{row['est_gen_hours']}h</td>"
            f"<td>{row['est_qa_hours']}h</td>"
            f"<td>{row['est_total_hours']}h{current_marker}</td></tr>"
        )

    # カテゴリ別合格率テーブル行
    cat_rows = ""
    zero_cats_warnings = []
    for cat, v in sorted(cats.items(), key=lambda x: -x[1]["pass_rate"]):
        pct = round(v["pass_rate"] * 100, 1)
        ci_lo = round(v["ci_lower"] * 100, 1)
        ci_hi = round(v["ci_upper"] * 100, 1)
        bar_w = round(v["pass_rate"] * 100)
        ci_bar_lo = round(v["ci_lower"] * 100)
        ci_bar_hi = round(v["ci_upper"] * 100)
        status_cls = "status-zero" if v["status"] == "zero_pass_investigate" else "status-active"
        cur_w = f"{v['current_weight']:.4f}" if v.get("current_weight") is not None else "—"
        rec_w = f"{v['recommended_weight']:.4f}" if v.get("recommended_weight") is not None else "—"
        if v["status"] == "zero_pass_investigate":
            zero_cats_warnings.append(cat)
        cat_rows += (
            f"<tr>"
            f"<td class='label'>{cat}</td>"
            f"<td class='{status_cls}'>{v['status']}</td>"
            f"<td>{v['passed']} / {v['samples']}</td>"
            f"<td>"
            f"  <div class='ci-container'>"
            f"    <div class='bar bar-cat' style='width:{bar_w}%'></div>"
            f"    <div class='ci-bar' style='left:{ci_bar_lo}%;width:{max(1, ci_bar_hi - ci_bar_lo)}%'></div>"
            f"  </div>"
            f"</td>"
            f"<td class='pct'>{pct}%</td>"
            f"<td class='pct'>[{ci_lo}%, {ci_hi}%]</td>"
            f"<td>{cur_w}</td>"
            f"<td>{rec_w}</td>"
            f"</tr>"
        )

    # 0%カテゴリ警告
    zero_warn_html = ""
    if zero_cats_warnings:
        items = "".join(f"<li>{c}</li>" for c in zero_cats_warnings)
        zero_warn_html = (
            f"<div class='warn-box'>"
            f"<strong>警告: 合格率 0% カテゴリ</strong><br>"
            f"以下のカテゴリは合格率が 0% です。category_clip_prefix の見直しや、"
            f"生成設定の見直しが必要な可能性があります。<ul>{items}</ul>"
            f"</div>"
        )

    # 推奨重みテーブル行
    weight_rows = ""
    for cat, w in sorted(rec["recommended_weights"].items(), key=lambda x: -x[1]):
        cur_w = cats.get(cat, {}).get("current_weight")
        cur_str = f"{cur_w:.4f}" if cur_w is not None else "—"
        delta = w - cur_w if cur_w is not None else None
        delta_str = (f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}") if delta is not None else "—"
        delta_cls = "delta-up" if (delta or 0) > 0.001 else "delta-down" if (delta or 0) < -0.001 else ""
        weight_rows += (
            f"<tr><td class='label'>{cat}</td>"
            f"<td>{cur_str}</td>"
            f"<td><strong>{w:.4f}</strong></td>"
            f"<td class='{delta_cls}'>{delta_str}</td></tr>"
        )

    # YAML スニペット
    yaml_block = rec.get("yaml_snippet", "")
    yaml_html = f"<pre class='yaml-block'>{yaml_block}</pre>" if yaml_block else ""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>AL3DG — Pass Rate Report</title>
<style>
  body {{ font-family: sans-serif; margin: 20px; background: #fafafa; color: #222; }}
  h1 {{ color: #333; }}
  h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(5, auto); gap: 12px; margin: 16px 0; }}
  .stat-box {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 16px; text-align: center; }}
  .stat-box .num {{ font-size: 1.8em; font-weight: bold; color: #333; }}
  .stat-box .lbl {{ font-size: 0.82em; color: #888; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 960px; }}
  th {{ background: #eee; text-align: left; padding: 8px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: middle; }}
  td.label {{ width: 180px; font-weight: bold; }}
  td.pct {{ width: 80px; text-align: right; }}
  .ci-container {{ position: relative; width: 200px; height: 18px; background: #f0f0f0; border-radius: 3px; }}
  .bar {{ height: 18px; background: #28a; border-radius: 3px; min-width: 2px; }}
  .bar-cat {{ position: absolute; top: 0; height: 18px; background: #28a; border-radius: 3px; min-width: 2px; }}
  .ci-bar {{ position: absolute; top: 5px; height: 8px; background: rgba(0,0,0,0.25); border-radius: 2px; }}
  .status-zero {{ color: #e00; font-weight: bold; }}
  .status-active {{ color: #282; }}
  .warn-box {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 12px 16px; margin: 16px 0; }}
  .yaml-block {{ background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 0.9em; }}
  .delta-up {{ color: #282; font-weight: bold; }}
  .delta-down {{ color: #e05; }}
  .precision-note {{ color: #666; font-style: italic; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>AL3DG — カテゴリ別合格率レポート</h1>
<p>実行日: {meta['run_date']} ／ 目標 hard_suitcase 割合: {round(rec['target_hs_fraction'] * 100)}%
   （target_hs_fraction = {rec['target_hs_fraction']}）</p>

<h2>全体サマリー</h2>
<div class="summary-grid">
  <div class="stat-box"><div class="num">{meta['total_generated']}</div><div class="lbl">生成総数</div></div>
  <div class="stat-box"><div class="num" style="color:#282">{meta['total_passed']}</div><div class="lbl">合格数</div></div>
  <div class="stat-box"><div class="num">{round(meta['overall_pass_rate']*100, 1)}%</div><div class="lbl">全体合格率</div></div>
  <div class="stat-box"><div class="num">{meta['n_per_category']}</div><div class="lbl">N/カテゴリ</div></div>
  <div class="stat-box"><div class="num">±{guide['achieved_precision_pct']}%</div><div class="lbl">統計精度 (95%CI)</div></div>
</div>

<h2>統計精度ガイド（N/カテゴリと精度の関係）</h2>
<p class="precision-note">現在: N={guide['current_n_per_category']}/カテゴリ × {guide['n_categories']} カテゴリ
   = {meta['total_generated']} 枚、精度 ±{guide['achieved_precision_pct']}%</p>
<table>
  <thead><tr><th>精度 (95%CI)</th><th>N/カテゴリ</th><th>総枚数</th><th>生成時間</th><th>QA時間</th><th>合計時間</th></tr></thead>
  <tbody>{guide_rows}</tbody>
</table>

{zero_warn_html}

<h2>カテゴリ別合格率（バー = 合格率、灰色帯 = 95% Wilson CI）</h2>
<table>
  <thead>
    <tr>
      <th>カテゴリ</th><th>ステータス</th><th>合格/総数</th>
      <th>合格率バー (0-100%)</th><th>合格率</th><th>95% CI</th>
      <th>現在重み</th><th>推奨重み</th>
    </tr>
  </thead>
  <tbody>{cat_rows}</tbody>
</table>

<h2>推奨 sampling 重み（目標: 最終合格の {round(rec['target_hs_fraction']*100)}% を hard_suitcase に）</h2>
<table>
  <thead><tr><th>カテゴリ</th><th>現在重み</th><th>推奨重み</th><th>変化量</th></tr></thead>
  <tbody>{weight_rows}</tbody>
</table>

<h2>コピペ用 YAML スニペット</h2>
<p>以下の内容を <code>configs/prompt_templates.yaml</code> の <code>sampling.category_weights</code> に貼り付けてください。</p>
{yaml_html}
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AL3DG 多様性評価レポート生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        default="image_qa",
        choices=["image_qa", "pass_rate"],
        help=(
            "生成するレポートの種類 "
            "(image_qa: QAリジェクト分析, pass_rate: カテゴリ別合格率)"
        ),
    )
    parser.add_argument(
        "--qa-results",
        default="outputs/images_approved/image_qa_results.json",
        help=(
            "--step image_qa / pass_rate 時の QA 結果 JSON パス "
            "(デフォルト: outputs/images_approved/image_qa_results.json)"
        ),
    )
    parser.add_argument(
        "--prompts",
        default="outputs/prompts/prompts.json",
        help="--step pass_rate 時の prompts.json パス (デフォルト: outputs/prompts/prompts.json)",
    )
    parser.add_argument(
        "--target-hs-fraction",
        type=float,
        default=0.70,
        help="--step pass_rate 時の hard_suitcase 目標合格割合 (デフォルト: 0.70)",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline_config.yaml",
        help="設定ファイルパス",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="レポート出力ディレクトリ (デフォルト: outputs/reports)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # --step pass_rate: カテゴリ別合格率レポートのみ実行して終了
    if args.step == "pass_rate":
        output_dir_str = args.output_dir or "outputs/reports"
        report = analyze_pass_rates(
            qa_results_path=Path(args.qa_results),
            prompts_path=Path(args.prompts),
            target_hs_fraction=args.target_hs_fraction,
            output_dir=Path(output_dir_str),
            config_path=Path(args.config),
        )
        if not report:
            return 1
        meta = report["metadata"]
        logger.info(
            f"合格率レポート完了: 全体合格率 {round(meta['overall_pass_rate']*100, 1)}% "
            f"({meta['total_passed']}/{meta['total_generated']}), "
            f"精度 ±{report['precision_guide']['achieved_precision_pct']}%"
        )
        return 0

    # --step image_qa: QAリジェクト分析のみ実行して終了
    if args.step == "image_qa":
        output_dir_str = args.output_dir or "outputs/reports"
        summary = analyze_image_qa_rejections(
            qa_results_path=Path(args.qa_results),
            output_dir=Path(output_dir_str),
        )
        if not summary:
            return 1
        logger.info(
            f"画像QAリジェクト分析完了: 合格率 {round(summary['pass_rate']*100, 1)}% "
            f"({summary['pass']}/{summary['total']})"
        )
        return 0

    logger.error(f"不明なステップ: {args.step}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
