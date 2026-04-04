# prompt_and_config

## 役割

荷物カテゴリ・プロンプトテンプレート・材質物理プロパティの設定ファイルを作成し、Qwen3-VL-32B（vLLM 経由）を使ったプロンプト生成エンジンを実装する。パイプライン全体の入力となるプロンプト群を生成する。

---

## 担当ファイル

### T-1.1 設定ファイル作成
- `configs/luggage_categories.yaml`
- `configs/prompt_templates.yaml`
- `configs/material_properties.yaml`

### T-1.2 プロンプト生成エンジン
- `src/prompt_generator.py`
- `tests/test_prompt_generator.py`

---

## 入力

- `plan/dgx_spark_construction_plan.html`（Phase 1 タスク詳細）
- T-0.1 完了済みの環境（`uv` 環境、OmegaConf/Hydra 利用可能）
- T-0.3 完了済みの vLLM サーバー（`http://localhost:8000/v1`）

---

## 出力

### configs/luggage_categories.yaml
- 空港で扱う荷物の全カテゴリ（日英名、サブカテゴリ）
- 典型的なサイズ範囲（cm）、材質、色、重量範囲（kg）

### configs/prompt_templates.yaml
- 属性軸：タイプ / 色 / 材質 / テクスチャ / スタイル / 状態
- 各軸のバリエーション一覧
- プロンプト組み立てテンプレート文
- 固定部分：白背景・スタジオ照明

### configs/material_properties.yaml
- 材質ごとの物理プロパティ（密度・静摩擦・動摩擦・反発係数）
- ランダム変動範囲

### src/prompt_generator.py

```python
class PromptGenerator:
    def __init__(self, config_path: str): ...
    def generate_combinatorial(self, n: int) -> list[dict]: ...
    def generate_with_llm_refinement(self, base_prompts: list,
                                      model_name: str = "Qwen/Qwen3-VL-32B") -> list: ...
    def save(self, prompts: list, output_path: str): ...
```

- LLM リファイン時は Qwen3-VL-32B（vLLM サーバー経由）を使用
- テキストのみのリクエスト（画像なし）
- Thinking モード無効（`/no_think`）で高速化
- 各プロンプトは `{prompt: str, metadata: dict}` 形式
- metadata には `luggage_type`, `color`, `material` 等を含む
- 出力先：`outputs/prompts/`（JSON 形式）

---

## 完了条件

- `configs/luggage_categories.yaml` が存在し、複数カテゴリとサブカテゴリを含む
- `configs/prompt_templates.yaml` が存在し、全属性軸のバリエーションを含む
- `configs/material_properties.yaml` が存在し、全材質の物理プロパティを含む
- `tests/test_prompt_generator.py` が PASS する
  - 100 個のプロンプトを生成し全てユニーク
  - 全プロンプトに `luggage_type` メタデータがある
  - 全プロンプトに白背景指定が含まれている

---

## 他エージェントへの引き継ぎ条件

- T-1.1 完了 → `evaluation_and_tests` に通知（`material_properties.yaml` を使う）
- T-1.2 完了 → `image_pipeline` に通知（プロンプト群が `outputs/prompts/` に用意された）
- `configs/pipeline_config.yaml` は T-6.1（`evaluation_and_tests` エージェント）が作成するが、`vlm` セクションの値を本エージェントが定義した設定と整合させること

---

## やってはいけないこと

- Qwen3-VL-32B 以外の LLM を使ってリファインする（8B は使用禁止）
- プロンプトに Hunyuan 系・非推奨モデル向けの記法を混入する
- `generate_with_llm_refinement` で Thinking モードを有効にする（高速化のため `/no_think` 固定）
- 設定ファイルを `src/` や `scripts/` 配下に置く（`configs/` のみ）
- 物理プロパティをハードコーディングする（必ず YAML から読む）
