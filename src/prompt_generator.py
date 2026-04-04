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
        fixed = self._templates.fixed
        self._fixed_suffix = (
            f"{fixed.background}, {fixed.lighting}, {fixed.view}, {fixed.quality}"
        )

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
    ) -> str:
        """属性を組み合わせて最終プロンプト文字列を生成"""
        core = (
            f"{type_phrase}, {color}, {material_phrase}, "
            f"{texture}, {style}, {condition}"
        )
        return f"{core}, {self._fixed_suffix}"

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

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
                type_phrase, color, material_phrase, texture, style, condition
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

    def generate_with_llm_refinement(
        self,
        base_prompts: list[dict],
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        vllm_base_url: str = "http://localhost:8000/v1",
        batch_size: int = 10,
        max_tokens: int = 150,
    ) -> list[dict]:
        """
        Qwen3-VL-32B-Instruct (vLLM サーバー経由) でプロンプトをリファイン。

        - テキストのみのリクエスト（画像なし）
        - Thinking モード無効 (/no_think) で高速化
        - リファイン失敗時は元プロンプトを維持

        Args:
            base_prompts: generate_combinatorial() の出力
            model_name: vLLM でサービング中のモデル名
            vllm_base_url: vLLM サーバーの base URL
            batch_size: 同時送信するバッチサイズ
            max_tokens: リファイン後の最大トークン数

        Returns:
            元と同じ構造のリスト（prompt フィールドがリファイン後の値に更新される）
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai パッケージが見つかりません: pip install openai")
            return base_prompts

        client = OpenAI(base_url=vllm_base_url, api_key="dummy")
        system_prompt = OmegaConf.to_container(
            self._templates.templates.llm_refine_system
        )
        user_template: str = OmegaConf.to_container(
            self._templates.templates.llm_refine_user
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
                        results[idx] = {
                            **item,
                            "prompt": refined,
                            "metadata": {
                                **item["metadata"],
                                "refined": True,
                                "original_prompt": base_prompt,
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
