"""
T-1.2: プロンプト生成エンジン

空港荷物カテゴリ・属性テンプレートから FLUX.1-schnell 向けプロンプトを生成する。
Qwen3-VL-32B-Instruct (vLLM サーバー経由) による LLM リファインもサポート。

使用例:
    gen = PromptGenerator("configs")
    prompts = gen.generate_combinatorial(n=1500)
    refined  = gen.generate_with_llm_refinement(prompts)
    gen.save(refined, "outputs/prompts/prompts.json")
"""

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf

from src.utils.logging_utils import setup_logger


class PromptGenerator:
    """属性の組み合わせおよび LLM リファインによるプロンプト生成"""

    def __init__(self, config_dir: str = "configs", seed: int = 42) -> None:
        """
        Args:
            config_dir: luggage_categories.yaml / prompt_templates.yaml を含むディレクトリ
            seed: 再現性のための乱数シード
        """
        config_dir = Path(config_dir)
        self._categories = OmegaConf.load(config_dir / "luggage_categories.yaml")
        self._templates = OmegaConf.load(config_dir / "prompt_templates.yaml")
        self._rng = random.Random(seed)

        # 固定フレーズ
        # FLUX.1-schnell の CLIP エンコーダは 77 トークン制限あり。
        # プロンプトは以下の順序で組み立て、重要指示が CLIP に届くよう先頭配置する:
        #   [prefix: フレーミング] → [core: 属性] → [bg_brief: 背景色] → [suffix_full: T5向け詳細]
        fixed = self._templates.fixed
        self._fixed_prefix = fixed.get("prefix", "single product photo, object centered,")
        self._fixed_bg_brief = fixed.get("bg_brief", "solid white background, no shadows, sharp focus")
        self._fixed_suffix = f"{fixed.lighting}, {fixed.view}, {fixed.quality}"

        # カテゴリ重みリスト（サンプリング用）
        weights_cfg = self._templates.sampling.category_weights
        self._cat_keys = list(OmegaConf.to_container(weights_cfg).keys())
        self._cat_weights = [weights_cfg[k] for k in self._cat_keys]

        # 色重みリスト
        color_weights_cfg = self._templates.sampling.color_weights
        self._color_groups = list(OmegaConf.to_container(color_weights_cfg).keys())
        self._color_group_weights = [color_weights_cfg[g] for g in self._color_groups]

        # 状態重みリスト
        cond_weights_cfg = self._templates.sampling.condition_weights
        self._cond_keys = list(OmegaConf.to_container(cond_weights_cfg).keys())
        self._cond_weights = [cond_weights_cfg[k] for k in self._cond_keys]

        # 淡色リスト（YAML から読み込み、フォールバック付き）
        light_colors_cfg = self._templates.get("light_colors", [
            "white", "ivory", "cream", "beige", "off-white",
            "pearl", "champagne", "silver", "light gray"
        ])
        self._light_colors = frozenset(OmegaConf.to_container(light_colors_cfg))

        # CLIP 先頭保証フレーズ（LLM リファイン後に先頭付加）
        # LLM が出力プロンプトのどこにハンドル格納指示を置いても、
        # この短縮フレーズを先頭に付加することで CLIP 77トークン枠内に必ず収める。
        cat_clip_cfg = self._templates.get("category_clip_prefix", {})
        self._category_clip_prefix: dict[str, str] = (
            OmegaConf.to_container(cat_clip_cfg) if cat_clip_cfg else {}
        )

        # サブカテゴリサイズ重み（カテゴリ → {subcategory: weight}）
        # sampling.subcategory_size_weights から読み込む（設定ファイルで制御可能）。
        # 設定されたカテゴリは重み付きサンプリング、未設定は均等サンプリング。
        size_weights_cfg = self._templates.sampling.get("subcategory_size_weights", {})
        self._subcategory_size_weights: dict[str, dict[str, float]] = (
            OmegaConf.to_container(size_weights_cfg) if size_weights_cfg else {}
        )

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _pick(self, lst: list) -> Any:
        return self._rng.choice(lst)

    def _pick_weighted(self, keys: list, weights: list) -> str:
        return self._rng.choices(keys, weights=weights, k=1)[0]

    def _flatten_list(self, obj: Any) -> list[str]:
        """OmegaConf のネストリストを str のフラットリストに変換"""
        result = []
        container = OmegaConf.to_container(obj) if hasattr(obj, "_metadata") else obj
        if isinstance(container, list):
            for item in container:
                if isinstance(item, (list, dict)):
                    result.extend(self._flatten_list(item))
                else:
                    result.append(str(item))
        elif isinstance(container, dict):
            for v in container.values():
                result.extend(self._flatten_list(v))
        else:
            result.append(str(container))
        return result

    def _sample_type_phrase(self, category: str) -> tuple[str, str]:
        """
        カテゴリ名からタイプ記述フレーズとサブカテゴリ名を返す。

        subcategory_size_weights が設定されているカテゴリは重み付きサンプリング、
        未設定のカテゴリは均等サンプリング。

        Returns:
            (phrase, subcategory)
        """
        type_cfg = self._templates.attributes.type
        if category not in type_cfg:
            # カテゴリが type 定義にない場合は category 名をそのまま使う
            cat_en = self._categories.categories[category].name_en
            return cat_en, "default"

        sub_cfg = type_cfg[category]
        sub_keys = list(OmegaConf.to_container(sub_cfg).keys())

        # subcategory_size_weights が設定されているカテゴリは重み付きサンプリング
        if category in self._subcategory_size_weights:
            weights_map = self._subcategory_size_weights[category]
            sub_weights = [weights_map.get(k, 1.0) for k in sub_keys]
            sub_key = self._pick_weighted(sub_keys, sub_weights)
        else:
            sub_key = self._pick(sub_keys)

        phrases = OmegaConf.to_container(sub_cfg[sub_key])
        return self._pick(phrases), sub_key

    def _sample_color(self) -> str:
        group = self._pick_weighted(self._color_groups, self._color_group_weights)
        colors = OmegaConf.to_container(self._templates.attributes.color[group])
        return self._pick(colors)

    def _sample_material_phrase(self, category: str) -> tuple[str, str]:
        """
        Returns:
            (phrase, material_key)
        """
        mat_cfg = self._templates.attributes.material
        # カテゴリに対応する典型材質を優先
        cat_info = self._categories.categories.get(category)
        if cat_info and "typical_materials" in cat_info:
            typical = OmegaConf.to_container(cat_info.typical_materials)
            # typical_materials の中で phrase 定義があるものを使う
            candidates = []
            for mat_key in typical:
                for group_key, group_val in OmegaConf.to_container(mat_cfg).items():
                    if isinstance(group_val, dict) and mat_key in group_val:
                        phrases = group_val[mat_key]
                        candidates.append((self._pick(phrases), mat_key))
                        break
                    elif isinstance(group_val, list) and mat_key in [
                        "cardboard",
                        "nonwoven_fabric",
                    ]:
                        # トップレベルリスト型の材質
                        pass
            # cardboard など直接リスト型
            for mat_key in typical:
                if mat_key in mat_cfg:
                    v = mat_cfg[mat_key]
                    if isinstance(OmegaConf.to_container(v), list):
                        phrases = OmegaConf.to_container(v)
                        candidates.append((self._pick(phrases), mat_key))
            if candidates:
                return self._pick(candidates)

        # フォールバック: ランダムに選ぶ
        mat_container = OmegaConf.to_container(mat_cfg)
        all_pairs: list[tuple[str, str]] = []
        for group_val in mat_container.values():
            if isinstance(group_val, dict):
                for mat_key, phrases in group_val.items():
                    all_pairs.append((self._pick(phrases), mat_key))
            elif isinstance(group_val, list):
                all_pairs.append((self._pick(group_val), "unknown"))
        return self._pick(all_pairs)

    def _sample_texture(self) -> str:
        tex_cfg = self._templates.attributes.texture
        groups = list(OmegaConf.to_container(tex_cfg).keys())
        group = self._pick(groups)
        return self._pick(OmegaConf.to_container(tex_cfg[group]))

    def _sample_style(self) -> str:
        style_cfg = self._templates.attributes.style
        groups = list(OmegaConf.to_container(style_cfg).keys())
        group = self._pick(groups)
        return self._pick(OmegaConf.to_container(style_cfg[group]))

    def _sample_condition(self) -> str:
        cond_key = self._pick_weighted(self._cond_keys, self._cond_weights)
        return self._pick(
            OmegaConf.to_container(self._templates.attributes.condition[cond_key])
        )


    def _build_prompt(
        self,
        type_phrase: str,
        color: str,
        material_phrase: str,
        texture: str,
        style: str,
        condition: str,
        category: str = "",
    ) -> str:
        """属性を組み合わせて最終プロンプト文字列を生成。

        プロンプト順序（CLIP 77トークン制限を考慮）:
          1. prefix（フレーミング指示）← CLIP の先頭トークン
          2. type_phrase, color, material（オブジェクト説明）
          3. texture, style, condition
          4. bg_brief（背景色、短く）← CLIPの77トークン前後
          5. suffix_full（詳細指示）← T5エンコーダが処理
        """
        core = (
            f"{type_phrase}, {color}, {material_phrase}, "
            f"{texture}, {style}, {condition}"
        )
        suffix = self._fixed_suffix
        # 淡色の場合: 輪郭強調（影は禁止のためcast shadow不使用）
        if any(lc in color.lower() for lc in self._light_colors):
            suffix = suffix + ", clearly defined object outline, strong edge contrast against background"
        # カテゴリ別 CLIP プレフィクス（ハンドル収納・ジッパー閉鎖などの必須指示）
        clip_prefix = self._category_clip_prefix.get(category, "").strip().rstrip(",").strip()
        # 順序: [prefix] [clip_prefix] [core] [bg_brief] [suffix_full]
        parts = [self._fixed_prefix]
        if clip_prefix:
            parts.append(clip_prefix)
        parts.extend([core, self._fixed_bg_brief, suffix])
        return ", ".join(parts)

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def generate_all(self, total: int = 100) -> list[dict]:
        """
        指定した総数のプロンプトを生成する。
        各プロンプトのカテゴリは category_weights に従い確率的に決まる。

        Args:
            total: 生成するプロンプトの総数

        Returns:
            プロンプトリスト（長さ = total）
        """
        logger.info(f"generate_all: 総数 {total} 件（category_weights 比率で割り当て）")
        return self.generate_combinatorial(n=total)

    def generate_uniform_per_category(self, n_per_cat: int) -> list[dict]:
        """
        各カテゴリから均等に n_per_cat 件ずつプロンプトを生成する。

        通常の generate_all() は加重ランダムサンプリングを使うため、
        weight の小さいカテゴリ（ski_bag=0.02 など）はほとんど選ばれない。
        合格率測定など全カテゴリを均等に評価したい場合はこのメソッドを使う。

        Args:
            n_per_cat: カテゴリあたりのプロンプト数

        Returns:
            全カテゴリ分のプロンプトリスト（カテゴリ内はランダム、最後にシャッフル）
        """
        cat_keys = list(
            OmegaConf.to_container(self._categories.categories).keys()
        )
        total_target = len(cat_keys) * n_per_cat
        logger.info(
            f"均等サンプリング開始: {len(cat_keys)} カテゴリ × {n_per_cat} = {total_target} 件"
        )

        all_results: list[dict] = []
        for category in cat_keys:
            generated = 0
            seen_local: set[str] = set()
            max_try = n_per_cat * 10
            for _ in range(max_try):
                if generated >= n_per_cat:
                    break
                type_phrase, subcategory = self._sample_type_phrase(category)
                color = self._sample_color()
                material_phrase, material_key = self._sample_material_phrase(category)
                texture = self._sample_texture()
                style = self._sample_style()
                condition = self._sample_condition()
                prompt = self._build_prompt(
                    type_phrase, color, material_phrase, texture,
                    style, condition, category
                )
                h = self._prompt_hash(prompt)
                if h in seen_local:
                    continue
                seen_local.add(h)
                all_results.append({
                    "prompt": prompt,
                    "metadata": {
                        "luggage_type": category,
                        "subcategory": subcategory,
                        "color": color,
                        "material": material_key,
                        "material_phrase": material_phrase,
                        "texture": texture,
                        "style": style,
                        "condition": condition,
                        "prompt_id": h,
                        "refined": False,
                    },
                })
                generated += 1
            if generated < n_per_cat:
                logger.warning(
                    f"  {category}: {generated}/{n_per_cat} 件しか生成できませんでした"
                )

        self._rng.shuffle(all_results)
        logger.info(f"均等サンプリング完了: {len(all_results)} 件")
        return all_results

    def generate_combinatorial(self, n: int, max_attempts_ratio: float = 5.0) -> list[dict]:
        """
        属性の組み合わせから n 個のプロンプトを生成する。

        Args:
            n: 生成するプロンプト数
            max_attempts_ratio: 重複除去のための最大試行倍率 (n * ratio)

        Returns:
            list of {
                "prompt": str,
                "metadata": {
                    "luggage_type": str,
                    "subcategory": str,
                    "color": str,
                    "material": str,
                    "texture": str,
                    "style": str,
                    "condition": str,
                    "prompt_id": str,
                }
            }
        """
        results: list[dict] = []
        seen_hashes: set[str] = set()
        max_attempts = int(n * max_attempts_ratio)
        attempts = 0

        logger.info(f"組み合わせ生成開始: 目標 {n} 件")

        while len(results) < n and attempts < max_attempts:
            attempts += 1

            # 属性サンプリング
            category = self._pick_weighted(self._cat_keys, self._cat_weights)
            type_phrase, subcategory = self._sample_type_phrase(category)
            color = self._sample_color()
            material_phrase, material_key = self._sample_material_phrase(category)
            texture = self._sample_texture()
            style = self._sample_style()
            condition = self._sample_condition()

            prompt = self._build_prompt(
                type_phrase, color, material_phrase, texture, style, condition,
                category=category,
            )

            # 重複チェック
            h = self._prompt_hash(prompt)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            results.append(
                {
                    "prompt": prompt,
                    "metadata": {
                        "luggage_type": category,
                        "subcategory": subcategory,
                        "color": color,
                        "material": material_key,
                        "material_phrase": material_phrase,
                        "texture": texture,
                        "style": style,
                        "condition": condition,
                        "prompt_id": h,
                    },
                }
            )

            if len(results) % 100 == 0:
                logger.info(f"  生成済み: {len(results)} / {n}")

        if len(results) < n:
            logger.warning(
                f"目標 {n} 件に対して {len(results)} 件のみ生成できました "
                f"(試行: {attempts} 回)。属性バリエーションを増やすことを検討してください。"
            )
        else:
            logger.info(f"生成完了: {len(results)} 件 (試行: {attempts} 回)")

        return results

    @staticmethod
    def _wait_for_vllm(
        base_url: str,
        timeout: float = 300.0,
        poll_interval: float = 10.0,
    ) -> None:
        """
        vLLM サーバーが起動するまで待機する。

        Args:
            base_url:      vLLM サーバーの base URL（例: "http://localhost:8001/v1"）
            timeout:       最大待機時間（秒）。デフォルト 300 秒
            poll_interval: ヘルスチェック間隔（秒）

        Raises:
            RuntimeError: タイムアウトまでに vLLM に到達できなかった場合
        """
        import time

        try:
            import requests as _requests
        except ImportError:
            _requests = None

        health_url = base_url.removesuffix("/v1").rstrip("/") + "/health"
        elapsed = 0.0
        attempt = 0

        logger.info(f"vLLM サーバー待機中: {health_url} (最大 {timeout:.0f} 秒)")

        while elapsed < timeout:
            attempt += 1
            try:
                if _requests is not None:
                    resp = _requests.get(health_url, timeout=5)
                    if resp.status_code == 200:
                        logger.info(f"✅ vLLM サーバーに到達しました ({elapsed:.0f} 秒後)")
                        return
                else:
                    # requests が使えない場合は urllib で代替
                    import urllib.request
                    with urllib.request.urlopen(health_url, timeout=5) as r:
                        if r.status == 200:
                            logger.info(f"✅ vLLM サーバーに到達しました ({elapsed:.0f} 秒後)")
                            return
            except Exception as e:
                if attempt == 1 or elapsed % 60 < poll_interval:
                    logger.info(f"  vLLM 未起動 ({elapsed:.0f}s 経過): {e}")

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise RuntimeError(
            f"[PromptGenerator] vLLM サーバーに接続できませんでした。\n"
            f"  URL      : {health_url}\n"
            f"  待機時間 : {timeout:.0f} 秒\n"
            f"  対処方法 : vLLM サーバーを起動してから再実行してください。\n"
            f"    bash scripts/start_vllm_server.sh\n"
            f"  または run_pipeline.py を使うと自動起動されます。"
        )

    def generate_with_llm_refinement(
        self,
        base_prompts: list[dict],
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        vllm_base_url: str = "http://localhost:8001/v1",
        batch_size: int = 10,
        max_tokens: int = 100,
        vllm_wait_timeout: float = 300.0,
    ) -> list[dict]:
        """
        Qwen3-VL-32B-Instruct (vLLM サーバー経由) でプロンプトをリファイン。

        - テキストのみのリクエスト（画像なし）
        - Thinking モード無効 (/no_think) で高速化
        - vLLM サーバーが起動していない場合は最大 vllm_wait_timeout 秒待機し、
          接続できなければ RuntimeError を送出して異常終了する

        Args:
            base_prompts: generate_combinatorial() の出力
            model_name: vLLM でサービング中のモデル名
            vllm_base_url: vLLM サーバーの base URL
            batch_size: 同時送信するバッチサイズ
            max_tokens: リファイン後の最大トークン数
            vllm_wait_timeout: vLLM 起動待機の最大秒数（デフォルト 300 秒）

        Returns:
            元と同じ構造のリスト（prompt フィールドがリファイン後の値に更新される）

        Raises:
            ImportError: openai パッケージが未インストールの場合
            RuntimeError: vllm_wait_timeout 秒以内に vLLM に接続できなかった場合
        """
        from openai import OpenAI

        # vLLM サーバーが起動しているか確認（起動待機）
        self._wait_for_vllm(vllm_base_url, timeout=vllm_wait_timeout)

        client = OpenAI(base_url=vllm_base_url, api_key="dummy")

        # テンプレートを取得（既に文字列の場合はそのまま使用）
        system_prompt_raw = self._templates.templates.llm_refine_system
        user_template_raw = self._templates.templates.llm_refine_user

        system_prompt = (
            OmegaConf.to_container(system_prompt_raw)
            if isinstance(system_prompt_raw, (dict, list)) or hasattr(system_prompt_raw, '_metadata')
            else system_prompt_raw
        )
        user_template = (
            OmegaConf.to_container(user_template_raw)
            if isinstance(user_template_raw, (dict, list)) or hasattr(user_template_raw, '_metadata')
            else user_template_raw
        )

        results = list(base_prompts)  # コピー
        total = len(results)
        logger.info(f"LLM リファイン開始: {total} 件 (model={model_name})")

        for i in range(0, total, batch_size):
            batch = results[i : i + batch_size]
            for j, item in enumerate(batch):
                idx = i + j
                base_prompt = item["prompt"]
                user_msg = user_template.replace("{base_prompt}", base_prompt)

                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.7,
                    )
                    refined = response.choices[0].message.content.strip()
                    # 空またはごく短い場合は元プロンプトを維持
                    if len(refined) > 20:
                        vlm_output = refined  # ログ用に raw LLM 出力を保存

                        # CLIP 先頭保証: カテゴリ別短縮フレーズを先頭に付加
                        # これにより CLIP 77トークン枠の先頭にハンドル格納指示が必ず届く
                        cat = item.get("metadata", {}).get("luggage_type", "")
                        clip_prefix = self._category_clip_prefix.get(cat, "")
                        if clip_prefix:
                            refined = f"{clip_prefix} {refined}"

                        # 65語ハードトリム（CLIP 77トークン ≈ 60-65語）
                        # suffix_full の T5 向け詳細指示が末尾から削除される
                        words = refined.split()
                        if len(words) > 65:
                            refined = " ".join(words[:65])

                        results[idx] = {
                            **item,
                            "prompt": refined,
                            "metadata": {
                                **item["metadata"],
                                "refined": True,
                                "original_prompt": base_prompt,
                                "vlm_input": user_msg,  # VLM 入力プロンプト（中間ファイル用）
                                "vlm_output": vlm_output,  # raw VLM 出力（トリム前）
                            },
                        }
                    else:
                        logger.warning(
                            f"  [{idx}] リファイン結果が短すぎるため元プロンプトを維持: {refined!r}"
                        )
                        results[idx]["metadata"]["refined"] = False

                except Exception as e:
                    logger.warning(f"  [{idx}] リファイン失敗 (元プロンプトを維持): {e}")
                    results[idx]["metadata"]["refined"] = False

            logger.info(f"  リファイン済み: {min(i + batch_size, total)} / {total}")

        refined_count = sum(1 for r in results if r["metadata"].get("refined"))
        logger.info(f"LLM リファイン完了: {refined_count} / {total} 件成功")
        return results

    def save(self, prompts: list[dict], output_path: str) -> Path:
        """
        プロンプトリストを JSON 形式で保存する。

        Args:
            prompts: generate_combinatorial() または generate_with_llm_refinement() の出力
            output_path: 保存先ファイルパス

        Returns:
            保存先の Path オブジェクト
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "total": len(prompts),
            "prompts": prompts,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"プロンプト保存完了: {output_path} ({len(prompts)} 件)")
        return output_path

    def get_statistics(self, prompts: list[dict]) -> dict:
        """
        生成済みプロンプトの統計情報を返す。

        Returns:
            {
                "total": int,
                "unique_prompts": int,
                "category_distribution": dict,
                "color_distribution": dict,
                "material_distribution": dict,
                "condition_distribution": dict,
                "refined_count": int,
            }
        """
        from collections import Counter

        stats: dict[str, Any] = {"total": len(prompts)}

        prompts_set = {p["prompt"] for p in prompts}
        stats["unique_prompts"] = len(prompts_set)

        metas = [p["metadata"] for p in prompts]
        stats["category_distribution"] = dict(
            Counter(m["luggage_type"] for m in metas)
        )
        stats["color_distribution"] = dict(Counter(m["color"] for m in metas))
        stats["material_distribution"] = dict(Counter(m["material"] for m in metas))
        stats["condition_distribution"] = dict(Counter(m["condition"] for m in metas))
        stats["refined_count"] = sum(1 for m in metas if m.get("refined"))

        return stats

    # ------------------------------------------------------------------
    # CSV 出力 / 入力
    # ------------------------------------------------------------------

    def _save_vlm_input_prompts_csv(self, prompts: list[dict], output_path: str) -> Path:
        """
        VLM 入力プロンプト用 CSV を保存する。

        ユーザーが vlm_input_prompt カラムを編集して T-1.2 を再実行する際に使用される。

        Args:
            prompts: 生成済みプロンプトリスト
            output_path: 保存先ファイルパス

        Returns:
            保存先の Path オブジェクト
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "luggage_type", "subcategory", "base_prompt", "vlm_input_prompt"
            ])
            writer.writeheader()

            for idx, prompt_dict in enumerate(prompts):
                meta = prompt_dict["metadata"]
                base_prompt = meta.get("original_prompt", prompt_dict["prompt"])

                # VLM 入力メッセージを再構成（templates から読み込み）
                user_template: str = str(self._templates.templates.llm_refine_user)
                vlm_input = user_template.replace("{base_prompt}", base_prompt)

                writer.writerow({
                    "id": idx,
                    "luggage_type": meta.get("luggage_type", ""),
                    "subcategory": meta.get("subcategory", ""),
                    "base_prompt": base_prompt,
                    "vlm_input_prompt": vlm_input,
                })

        logger.info(f"VLM入力プロンプトCSV保存完了: {output_path}")
        return output_path

    def _save_image_generation_prompts_csv(self, prompts: list[dict], output_path: str) -> Path:
        """
        画像生成プロンプト用 CSV を保存する。

        ユーザーが final_prompt_for_flux カラムを編集して T-2.1 を再実行する際に使用される。

        Args:
            prompts: 生成済みプロンプトリスト（各要素に prompt フィールドを持つ）
            output_path: 保存先ファイルパス

        Returns:
            保存先の Path オブジェクト
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "asset_id", "vlm_output_prompt", "final_prompt_for_flux"
            ])
            writer.writeheader()

            for idx, prompt_dict in enumerate(prompts):
                meta = prompt_dict["metadata"]
                asset_id = f"{idx:06d}_{meta.get('prompt_id', '000000')}"

                writer.writerow({
                    "id": idx,
                    "asset_id": asset_id,
                    "vlm_output_prompt": meta.get("original_prompt", ""),  # VLM 出力（参考）
                    "final_prompt_for_flux": prompt_dict["prompt"],  # 最終プロンプト（編集可能）
                })

        logger.info(f"画像生成プロンプトCSV保存完了: {output_path}")
        return output_path

    def _load_vlm_input_prompts_csv(self, csv_path: str) -> dict[int, str]:
        """
        vlm_input_prompts.csv から修正済みプロンプトを読み込む。

        Returns:
            {id: vlm_input_prompt} の辞書
        """
        result = {}
        csv_path = Path(csv_path)

        if not csv_path.exists():
            return result

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    result[int(row["id"])] = row["vlm_input_prompt"]
                except (KeyError, ValueError):
                    pass

        logger.info(f"VLM入力プロンプトCSV読み込み完了: {len(result)} 件")
        return result

    def _load_image_generation_prompts_csv(self, csv_path: str) -> dict[int, str]:
        """
        image_generation_prompts.csv から修正済みプロンプトを読み込む。

        Returns:
            {id: final_prompt_for_flux} の辞書
        """
        result = {}
        csv_path = Path(csv_path)

        if not csv_path.exists():
            return result

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    result[int(row["id"])] = row["final_prompt_for_flux"]
                except (KeyError, ValueError):
                    pass

        logger.info(f"画像生成プロンプトCSV読み込み完了: {len(result)} 件")
        return result

    def generate_batch(
        self,
        total: int = 100,
        config_dir: str = "configs",
        prompts_csv_dir: str = "outputs/prompts",
        images_csv_dir: str = "outputs/images",
        check_csv_for_resume: bool = True,
        vllm_enabled: bool = True,
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        vllm_base_url: str = "http://localhost:8001/v1",
    ) -> list[dict]:
        """
        FullPipeline for T-1.2: プロンプト生成 → VLM リファイン → CSV 出力

        CSV ファイルが存在する場合、そこから修正済みプロンプトを読み込んで使用する。
        これにより、ユーザーがプロンプトを微修正して再実行することが可能。

        Args:
            total: 生成するプロンプトの総数
            config_dir: 設定ファイルディレクトリ
            prompts_csv_dir: 出力 CSV ディレクトリ（prompts用）
            images_csv_dir: 出力 CSV ディレクトリ（images用）
            check_csv_for_resume: CSV から修正済みプロンプトを読み込むかどうか
            vllm_enabled: VLM リファインを有効化するかどうか
            model_name: vLLM モデル名
            vllm_base_url: vLLM サーバーベース URL

        Returns:
            最終的なプロンプトリスト（JSON と同じ構造）
        """
        logger.info("=" * 70)
        logger.info("T-1.2: プロンプト生成バッチ開始")
        logger.info("=" * 70)

        prompts_csv_dir = Path(prompts_csv_dir)
        images_csv_dir = Path(images_csv_dir)
        vlm_input_csv = prompts_csv_dir / "vlm_input_prompts.csv"
        image_gen_csv = images_csv_dir / "image_generation_prompts.csv"

        # ステップ1: CSV から修正済みプロンプトを読み込む（再実行時）
        vlm_input_overrides = {}
        final_prompt_overrides = {}

        if check_csv_for_resume:
            if vlm_input_csv.exists():
                vlm_input_overrides = self._load_vlm_input_prompts_csv(str(vlm_input_csv))
                logger.info(f"VLM入力プロンプトCSVから {len(vlm_input_overrides)} 件の修正を読み込みました")

            if image_gen_csv.exists():
                final_prompt_overrides = self._load_image_generation_prompts_csv(str(image_gen_csv))
                logger.info(f"画像生成プロンプトCSVから {len(final_prompt_overrides)} 件の修正を読み込みました")

        # ステップ2: ベースプロンプト生成
        logger.info(f"ステップ1: ベースプロンプト生成（総数: {total} 件）")
        base_prompts = self.generate_all(total=total)
        logger.info(f"  生成完了: {len(base_prompts)} 件")

        # ステップ3: VLM リファイン（修正がない場合）
        if vllm_enabled and not vlm_input_overrides:
            logger.info("ステップ2: VLMリファイン")
            refined_prompts = self.generate_with_llm_refinement(
                base_prompts,
                model_name=model_name,
                vllm_base_url=vllm_base_url,
            )
        elif vlm_input_overrides:
            # CSV から修正済みプロンプトを読み込んだ場合、リファイン処理をスキップ
            logger.info("ステップ2: VLMリファイン（CSV から修正を読み込むためスキップ）")
            refined_prompts = base_prompts
        else:
            logger.info("ステップ2: VLMリファイン（無効化）")
            refined_prompts = base_prompts

        # ステップ4: 最終プロンプトへ編集を反映
        if final_prompt_overrides:
            logger.info(f"ステップ3: 最終プロンプトへ {len(final_prompt_overrides)} 件の編集を反映")
            for idx, final_prompt in final_prompt_overrides.items():
                if idx < len(refined_prompts):
                    refined_prompts[idx]["prompt"] = final_prompt

        # ステップ5: CSV 出力
        logger.info("ステップ4: CSV 出力")
        self._save_vlm_input_prompts_csv(refined_prompts, str(vlm_input_csv))
        self._save_image_generation_prompts_csv(refined_prompts, str(image_gen_csv))

        logger.info("=" * 70)
        logger.info(f"T-1.2: プロンプト生成バッチ完了 ({len(refined_prompts)} 件)")
        logger.info("=" * 70)

        return refined_prompts

    def generate_html_report(
        self,
        prompts_json_path: str,
        vlm_input_csv_path: str,
        image_gen_csv_path: str,
        images_dir: str = "outputs/images",
        output_path: str = "outputs/reports/prompt_review.html",
    ) -> Path:
        """
        プロンプト生成レビュー用 HTML レポートを生成する。

        VLM 入力プロンプト、VLM 出力プロンプト（または最終プロンプト）、生成画像を表示。
        CSV で編集がある場合は、編集後の値を優先表示。

        Args:
            prompts_json_path: prompts.json ファイルパス
            vlm_input_csv_path: vlm_input_prompts.csv ファイルパス
            image_gen_csv_path: image_generation_prompts.csv ファイルパス
            images_dir: 生成画像ディレクトリ
            output_path: HTML 出力先

        Returns:
            HTML ファイルパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # JSON と CSV を読み込む
        try:
            with open(prompts_json_path, "r", encoding="utf-8") as f:
                prompts_data = json.load(f)
                # prompts_data が直接リストの場合と、辞書で "prompts" キーを持つ場合の両方に対応
                if isinstance(prompts_data, list):
                    prompts = prompts_data
                else:
                    prompts = prompts_data.get("prompts", [])
        except FileNotFoundError:
            logger.warning(f"prompts.json が見つかりません: {prompts_json_path}")
            prompts = []

        # CSV から修正情報を読み込む
        vlm_input_overrides = self._load_vlm_input_prompts_csv(vlm_input_csv_path)
        final_prompt_overrides = self._load_image_generation_prompts_csv(image_gen_csv_path)

        # HTML 生成
        images_dir = Path(images_dir)
        html_lines = [
            "<!DOCTYPE html>",
            '<html lang="ja">',
            "<head>",
            '    <meta charset="UTF-8">',
            "    <title>AL3DG プロンプトレビュー</title>",
            "    <style>",
            "        * { box-sizing: border-box; }",
            "        body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }",
            "        h1 { color: #333; }",
            "        .table-container { overflow-x: auto; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 4px; }",
            "        table { width: 100%; border-collapse: collapse; }",
            "        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; min-width: 150px; }",
            "        th { background: #2c3e50; color: white; font-weight: bold; position: sticky; top: 0; }",
            "        th:first-child, td:first-child { min-width: 50px; }",
            "        tr:hover { background: #f9f9f9; }",
            "        .prompt { word-break: break-word; word-wrap: break-word; white-space: normal; line-height: 1.5; font-size: 0.95em; }",
            "        .image-cell { text-align: center; min-width: 180px; }",
            "        .image-cell img { max-width: 160px; max-height: 160px; cursor: pointer; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s; }",
            "        .image-cell img:hover { transform: scale(1.05); }",
            "        .modified { background: #fff3cd; }",
            "        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); }",
            "        .modal-content { margin: auto; display: block; max-width: 90%; max-height: 90%; margin-top: 2%; object-fit: contain; }",
            "        .modal.show { display: flex; align-items: center; justify-content: center; }",
            "        .close { position: absolute; top: 20px; right: 40px; color: white; font-size: 40px; font-weight: bold; cursor: pointer; }",
            "        .close:hover { color: #f0f0f0; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>AL3DG プロンプトレビュー</h1>",
            "    <p>VLM 入力プロンプト → VLM 出力 / 最終プロンプト → 生成画像</p>",
            "    <div class=\"table-container\">",
            "    <table>",
            "        <thead>",
            "            <tr>",
            "                <th>#</th>",
            "                <th>荷物タイプ</th>",
            "                <th>VLM 入力プロンプト</th>",
            "                <th>最終プロンプト</th>",
            "                <th>生成画像</th>",
            "            </tr>",
            "        </thead>",
            "        <tbody>",
        ]

        for idx, prompt_dict in enumerate(prompts):
            meta = prompt_dict["metadata"]
            luggage_type = meta.get("luggage_type", "unknown")

            # VLM 入力プロンプト（CSV で修正あれば優先）
            if idx in vlm_input_overrides:
                vlm_input = vlm_input_overrides[idx]
                vlm_input_row_class = ' class="modified"'
            else:
                # VLMリファイン実行時: meta.get("vlm_input") を使用
                # VLMリファイン非実行時: ベースプロンプト（prompt_dict["prompt"]）を使用
                vlm_input = meta.get("vlm_input", meta.get("original_prompt", prompt_dict.get("prompt", "（利用不可）")))
                vlm_input_row_class = ""

            # 最終プロンプト（CSV で修正あれば優先）
            if idx in final_prompt_overrides:
                final_prompt = final_prompt_overrides[idx]
                final_prompt_row_class = ' class="modified"'
            else:
                final_prompt = prompt_dict["prompt"]
                final_prompt_row_class = ""

            # 生成画像パス
            image_path = images_dir / f"{idx:06d}_{meta.get('prompt_id', '000000')}.png"
            image_exists = image_path.exists()
            image_rel_path = f"../images/{image_path.name}" if image_exists else None

            # プロンプト文字列を全文表示（改行対応）
            vlm_input_escaped = vlm_input.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            final_prompt_escaped = final_prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

            html_lines.append(f'            <tr{vlm_input_row_class}{final_prompt_row_class}>')
            html_lines.append(f"                <td>{idx}</td>")
            html_lines.append(f"                <td>{luggage_type}</td>")
            html_lines.append(f'                <td class="prompt">{vlm_input_escaped}</td>')
            html_lines.append(f'                <td class="prompt">{final_prompt_escaped}</td>')

            if image_exists:
                html_lines.append(f'                <td class="image-cell"><img src="{image_rel_path}" onclick="openModal(this.src)" alt="image {idx}"></td>')
            else:
                html_lines.append(f'                <td class="image-cell">（未生成）</td>')

            html_lines.append("            </tr>")

        html_lines.extend([
            "        </tbody>",
            "    </table>",
            "    </div>",
            '    <div id="imageModal" class="modal">',
            '        <div style="position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">',
            '            <span class="close" onclick="closeModal()">&times;</span>',
            '            <img class="modal-content" id="modalImage" alt="">',
            "        </div>",
            "    </div>",
            "    <script>",
            "        function openModal(src) {",
            "            document.getElementById('imageModal').classList.add('show');",
            "            document.getElementById('modalImage').src = src;",
            "        }",
            "        function closeModal() {",
            "            document.getElementById('imageModal').classList.remove('show');",
            "        }",
            "        window.onclick = function(event) {",
            "            const modal = document.getElementById('imageModal');",
            "            if (event.target == modal) closeModal();",
            "        }",
            "        // Escキーでモーダルを閉じる",
            "        document.addEventListener('keydown', function(event) {",
            "            if (event.key === 'Escape') closeModal();",
            "        });",
            "    </script>",
            "</body>",
            "</html>",
        ])

        html_content = "\n".join(html_lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML レポート生成完了: {output_path}")
        return output_path
