"""
T-1.2: プロンプト生成エンジン

空港荷物カテゴリ・属性テンプレートから FLUX.1-schnell 向けプロンプトを生成する。

使用例:
    gen = PromptGenerator("configs")
    prompts = gen.generate_combinatorial(n=1500)
    gen.save(prompts, "outputs/prompts/prompts.json")
"""

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf


class PromptGenerator:
    """属性の組み合わせによる FLUX.1-schnell 向けプロンプト生成"""

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

        # CLIP 先頭保証フレーズ（ベースプロンプト先頭に付加）
        # ハンドル格納・ジッパー閉鎖などの必須指示を CLIP 77 トークン枠の先頭に確実に届ける。
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

    def save(self, prompts: list[dict], output_path: str) -> Path:
        """
        プロンプトリストを JSON 形式で保存する。

        Args:
            prompts: generate_combinatorial() または generate_all() の出力
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

        return stats

    # ------------------------------------------------------------------
    # CSV 入力
    # ------------------------------------------------------------------

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

    def generate_html_report(
        self,
        prompts_json_path: str,
        image_gen_csv_path: str,
        images_dir: str = "outputs/images",
        output_path: str = "outputs/reports/prompt_review.html",
    ) -> Path:
        """
        プロンプト生成レビュー用 HTML レポートを生成する。

        プロンプトと生成画像を表示。
        image_generation_prompts.csv で編集がある場合は、編集後の値を優先表示。

        Args:
            prompts_json_path: prompts.json ファイルパス
            image_gen_csv_path: image_generation_prompts.csv ファイルパス
            images_dir: 生成画像ディレクトリ
            output_path: HTML 出力先

        Returns:
            HTML ファイルパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

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

        final_prompt_overrides = self._load_image_generation_prompts_csv(image_gen_csv_path)

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
            "    <p>プロンプト → 生成画像（image_generation_prompts.csv で編集された行は黄色でハイライト）</p>",
            "    <div class=\"table-container\">",
            "    <table>",
            "        <thead>",
            "            <tr>",
            "                <th>#</th>",
            "                <th>荷物タイプ</th>",
            "                <th>プロンプト</th>",
            "                <th>生成画像</th>",
            "            </tr>",
            "        </thead>",
            "        <tbody>",
        ]

        for idx, prompt_dict in enumerate(prompts):
            meta = prompt_dict["metadata"]
            luggage_type = meta.get("luggage_type", "unknown")

            if idx in final_prompt_overrides:
                final_prompt = final_prompt_overrides[idx]
                row_class = ' class="modified"'
            else:
                final_prompt = prompt_dict["prompt"]
                row_class = ""

            image_path = images_dir / f"{idx:06d}_{meta.get('prompt_id', '000000')}.png"
            image_exists = image_path.exists()

            final_prompt_escaped = final_prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

            html_lines.append(f'            <tr{row_class}>')
            html_lines.append(f"                <td>{idx}</td>")
            html_lines.append(f"                <td>{luggage_type}</td>")
            html_lines.append(f'                <td class="prompt">{final_prompt_escaped}</td>')

            if image_exists:
                import base64 as _b64
                b64_data = _b64.b64encode(image_path.read_bytes()).decode()
                img_src = f"data:image/png;base64,{b64_data}"
                html_lines.append(f'                <td class="image-cell"><img src="{img_src}" onclick="openModal(this.src)" alt="image {idx}"></td>')
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
