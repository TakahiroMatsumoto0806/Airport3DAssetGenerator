# VLM設定ファイル仕様

AL3DGパイプラインでVLM（Qwen3-VL-32B-Instruct）を使用するステップは3つある。
各ステップのVLMへの依頼内容と、それを制御する設定ファイルを整理する。

---

## 共通設定: VLMサーバー接続

**設定ファイル**: `configs/pipeline_config.yaml`

```yaml
models:
  vlm:
    base_url: "http://localhost:8001/v1"
    model_name: "/home/ntt/models/Qwen3-VL-32B-Instruct"  # vLLMはローカルパスをモデルIDとして使用
    temperature: 0.1
    max_tokens: 2048
```

**サーバー起動**: `bash scripts/start_vllm_server.sh`
- `--gpu-memory-utilization 0.90`（DGX Spark 128GBで必須: モデル重量~65GBのため0.55では起動不可）
- `--max-model-len 8192`
- `--dtype bfloat16`

---

## T-1.2: プロンプト生成（LLMリファイン）

### 何を依頼するか
FLUX.1-schnell用の商品画像プロンプトを改善する。
ベーステンプレートで組み立てたプロンプトをQwen3-VLに渡し、より詳細・効果的な英語プロンプトに書き直させる。

### 設定ファイル: `configs/prompt_templates.yaml`

**システムプロンプト** (`templates.llm_refine_system`):
```
You are a professional product photographer and prompt engineer specializing
in generating high-quality product images with FLUX.1-schnell diffusion model.
Your task is to refine image generation prompts for airport luggage items.
Output only the refined prompt, no explanations.
```

**ユーザープロンプト** (`templates.llm_refine_user`):
```
/no_think
Refine this product image prompt for FLUX.1-schnell.
Keep it under 120 words. Preserve all factual attributes.
Ensure it specifies: pure white background, studio lighting,
product photography angle, photorealistic quality.
Input prompt: {base_prompt}
```

**重要**: `/no_think` トークンで思考モードをOFF（高速スクリーニング用）

### プロンプトの組み立て元

`configs/prompt_templates.yaml` の `attributes` と `fixed` セクションが元データ:

| セクション | 内容 |
|-----------|------|
| `fixed.prefix` | 全プロンプト共通のフレーミング指示（CLIP 77トークン制限内に収める） |
| `fixed.bg_brief` | 背景指示（CLIPに届かせる短い記述） |
| `fixed.lighting`, `fixed.view`, `fixed.quality` | T5エンコーダ向け詳細指示 |
| `attributes.type` | 荷物タイプ別の英語記述（carry_on/medium_check_in/large_check_in等サブカテゴリあり） |
| `attributes.color` | 色バリエーション（neutrals/blues/reds_pinks/greens/browns_golds/others） |
| `attributes.material` | 材質記述（polycarbonate/nylon/leather等） |
| `attributes.texture` | テクスチャ記述（smooth/textured/fabric/worn） |
| `attributes.style` | スタイル（modern/classic/sporty/luxury/casual） |
| `attributes.condition` | 状態（new/used/worn） |

**サンプリング重み** (`sampling`):
- `category_weights`: 荷物カテゴリの生成比率（hard_suitcase=0.25が最多）
- `color_weights`: 色グループの選択比率（neutrals=0.40が最多）
- `condition_weights`: 状態の選択比率（new=0.50が最多）

**荷物カテゴリ定義**: `configs/luggage_categories.yaml`
- 11カテゴリ: hard_suitcase, soft_suitcase, backpack, duffel_bag, briefcase, cardboard_box, hard_case, golf_bag, ski_bag, stroller, instrument_case
- 各カテゴリにサイズ範囲（mm）・標準材質・色傾向・物理プロパティのデフォルト値を定義

### VLM プロンプト出力の確認・微修正（CSV 編集）

T-1.2 実行後、以下 2 つの CSV ファイルが生成され、ユーザーが微修正して再実行できます。

| CSV ファイル | 用途 | 編集対象カラム |
|-----------|------|------------|
| `outputs/prompts/vlm_input_prompts.csv` | VLM 入力プロンプトの確認・微修正 | `vlm_input_prompt` |
| `outputs/images/image_generation_prompts.csv` | FLUX.1 最終プロンプトの確認・微修正 | `final_prompt_for_flux` |

**再実行フロー:**

```bash
# 1. T-1.2 実行
python scripts/run_pipeline.py --steps prompt

# 2. 生成された CSV を確認・編集
# outputs/prompts/vlm_input_prompts.csv の vlm_input_prompt カラムを編集

# 3. T-1.2 を再実行（CSV の修正値を使用）
python scripts/run_pipeline.py --steps prompt

# 4. 修正後のプロンプトで画像生成・以降のパイプラインを実行
python scripts/run_pipeline.py --steps image image_qa
```

---

## T-2.2: 画像QA（VLM画像検品）

### 何を依頼するか
FLUX.1-schnellで生成した荷物画像1枚をVLMに見せ、荷物として使えるクオリティか評価させる。

### 設定ファイル: `configs/pipeline_config.yaml` - `image_qa` セクション

```yaml
image_qa:
  # VLM プロンプト設定
  system_prompt: |
    You are a quality control inspector for AI-generated product images.
    Your task is to evaluate luggage product images for a robotics training dataset.
    Respond ONLY with a valid JSON object. No markdown, no explanation.

  user_prompt_template: |
    /no_think
    Evaluate this product image of airport luggage for suitability as input to a 3D mesh generation model (TRELLIS).
    Return a JSON object with exactly these keys:
    {
      "realism_score": <int 1-10>,
      "object_integrity": <int 1-10>,
      ...
      "pass": <true if realism_score>={min_realism} AND object_integrity>={min_integrity} AND ...>
    }

  # スコア・定性的な合格基準
  thresholds:
    realism: 7
    integrity: 7
    min_coverage_pct: 50
    require_fully_visible: true
    require_contrast_sufficient: true
    require_no_background_shadow: true
    require_sharp_focus: true
    require_camera_angle_ok: true
    require_no_artifacts: true

  thinking_mode: false   # /no_think（高速スクリーニング）
  batch_size: 16
  resume: true
  output_dir: "outputs/images_approved"
```

### VLM プロンプト内の動的置換

`user_prompt_template` 内のプレースホルダ `{min_realism}`, `{min_integrity}`, `{min_coverage}` は
実行時に `thresholds` セクションの値で自動置換されます。

例：
```
"pass": <true if realism_score>={min_realism} AND object_integrity>={min_integrity} AND object_coverage_pct>={min_coverage} ...>
```
↓
```
"pass": <true if realism_score>=7 AND object_integrity>=7 AND object_coverage_pct>=50 ...>
```

> **実装メモ**: テンプレートは JSON 形式のサンプル（`{` `}` を含む）のため、通常の `str.format()` ではなく
> `_safe_format()` を使用している（`src/image_qa.py`）。`{英数字_}` 形式のシンプルな変数名のみを
> 置換し、JSON の構造記号はそのまま保持する。

### 合否判定ロジック

| 指標 | 合格閾値 | 説明 |
|------|---------|------|
| `realism_score` | >= 7 (default) | 画像のリアリティ・品質 |
| `object_integrity` | >= 7 (default) | 荷物として認識できるか・形状の整合性 |
| `is_fully_visible` | true | オブジェクト全体が画面内に入っているか |
| `contrast_sufficient` | true | 背景とのコントラストは十分か |
| `object_coverage_pct` | >= 50% (default) | 画面占有率 |
| `has_background_shadow` | false | 背景への影がないか |
| `is_sharp_focus` | true | ピントは鋭いか |
| `camera_angle_ok` | true | カメラ角度は適正か |
| `has_artifacts` | false | 生成エラー・ノイズはないか |

全ての条件を満たす場合のみ `pass: true`。合格画像は `outputs/images_approved/` に移動。

---

## T-3.3: VLMマルチビュー3D検品

### 何を依頼するか
3DメッシュのGLBを4方向（正面・右・背面・左）からレンダリングした4枚の画像を
まとめてVLMに送り、3Dアセットとしての品質スコアを返させる。

### 設定ファイル: `configs/pipeline_config.yaml` - `mesh_vlm_qa` セクション

```yaml
mesh_vlm_qa:
  # VLM プロンプト設定
  system_prompt: |
    You are a 3D asset quality inspector for a robotics training dataset.
    These meshes are AI-generated (TRELLIS diffusion model) and will be used as simulation assets.
    Evaluate multi-view renders of a 3D luggage mesh for geometry quality, texture quality, and realism.
    Score relative to AI-generated mesh quality standards, not hand-crafted professional models.
    Respond ONLY with a valid JSON object. No markdown, no explanation.

  user_prompt_template: |
    /think
    You are reviewing {n_views} rendered views of an AI-generated 3D luggage mesh (TRELLIS diffusion model).
    Evaluate the mesh quality relative to AI-generated mesh standards and return a JSON object:
    {
      "geometry_score": <int 1-10, recognizable shape=5, clean topology=7, professional=10>,
      "texture_score": <int 1-10, visible color/material=4, good UV=6, PBR realism=9>,
      "consistency_score": <int 1-10, visual consistency across all views>,
      "is_realistic_luggage": <true if the object is recognizable as luggage/bag>,
      "detected_type": <string: e.g. "hard_suitcase", "backpack", "duffel_bag">,
      "detected_material": <string: e.g. "polycarbonate", "nylon", "leather">,
      "issues": <list of strings: critical problems only>,
      "pass": <true if geometry_score>={min_geometry} AND texture_score>={min_texture}>
    }

  # スコア閾値（最終確定値）
  thresholds:
    geometry: 6
    texture: 5

  thinking_mode: true    # /think（複雑な3D幾何・材質評価で精度向上）
  azimuths: [0, 90, 180, 270]   # レンダリング方位角（度）
  render_size: [512, 512]        # レンダリング解像度
  output_dir: "outputs/meshes_approved"
  render_dir: "outputs/renders"
  resume: true
```

### VLM プロンプト内の動的置換

`user_prompt_template` 内のプレースホルダ `{min_geometry}`, `{min_texture}` は
実行時に `thresholds` セクションの値で自動置換されます。

例：
```
"pass": <true if geometry_score>={min_geometry} AND texture_score>={min_texture}>
```
↓
```
"pass": <true if geometry_score>=6 AND texture_score>=5>
```

また、`{n_views}` はレンダリング視点数（デフォルト4）で置換されます。

### 合否判定ロジック

| 指標 | 合格閾値 | 説明 |
|------|---------|------|
| `geometry_score` | >= 6 | 3D形状品質（recognizable=5, clean=7, professional=10） |
| `texture_score` | >= 5 | テクスチャ品質（visible color=4, good UV=6, PBR realism=9） |
| `is_realistic_luggage` | true | 荷物として認識できるか |

`geometry_score >= min_geometry AND texture_score >= min_texture` を満たす場合のみ `pass: true`。

### 返答スキーマ

| フィールド | 型 | 説明 |
|-----------|---|------|
| `geometry_score` | int 1-10 | 形状品質（recognizable=5, clean=7, professional=10） |
| `texture_score` | int 1-10 | テクスチャ品質（visible color=4, good UV=6, PBR realism=9） |
| `consistency_score` | int 1-10 | 4視点間の視覚的一貫性 |
| `is_realistic_luggage` | bool | 荷物として認識できるか |
| `detected_type` | str | 検出された荷物タイプ |
| `detected_material` | str | 検出された主材質 |
| `issues` | list[str] | 重大な問題点（許容範囲なら空リスト） |
| `pass` | bool | geometry_score >= min_geometry AND texture_score >= min_texture の場合true |

### レンダリング設定（`src/utils/rendering.py`）

| パラメータ | 値 | 説明 |
|-----------|---|------|
| 解像度 | 512×512 | レンダリング画像サイズ |
| 視点数 | 4 | 方位角0°, 90°, 180°, 270° |
| 仰角 | 20° | カメラの俯角 |
| カメラ距離 | BBmax × 2.2 | バウンディングボックス最大辺の2.2倍 |
| ライト | DirectionalLight×3 | メイン(4.0) + フィル(1.5) + アンビエント(0.8) |
| レンダラー | pyrender + OSMesa | PBRマテリアル・UVテクスチャを正しく反映 |
| 環境変数 | `PYOPENGL_PLATFORM=osmesa` | Python起動前に設定必須 |

---

## HTML レビューレポート

各 VLM 検品ステップ完了後、VLM に送ったプロンプト、得られたスコア、結果を一覧表示する HTML レポートが自動生成されます。

| ステップ | HTML レポート | 表示内容 |
|---------|------------|---------|
| T-1.2 | `outputs/reports/prompt_review.html` | VLM 入力プロンプト、VLM 出力（最終プロンプト）、生成画像サムネイル |
| T-2.2 | `outputs/reports/image_qa_review.html` | 検品プロンプト、Realism/Integrity スコア、各条件チェック、合格判定、画像サムネイル |
| T-3.3 | `outputs/reports/mesh_vlm_qa_review.html` | 検品プロンプト、Geometry/Texture/Consistency スコア、4 ビュー画像、問題点リスト、合格判定 |

### レポート表示機能

- **サマリー統計**: 合格率、総数、パス/フェイル数を表示
- **テーブル表示**: 各評価結果を行ごとに表示
- **モーダルダイアログ**: プロンプトテキストや画像をクリックで拡大表示
- **色分け**: Pass（緑）/ Review（黄）/ Reject（赤）で視覚的に区別

### ブラウザで確認

```bash
# パイプライン実行後
python scripts/run_pipeline.py

# ブラウザで開く
open outputs/reports/prompt_review.html
open outputs/reports/image_qa_review.html
open outputs/reports/mesh_vlm_qa_review.html
```

---

## サマリー

| ステップ | Thinkingモード | 入力 | 出力 | レポート |
|---------|--------------|------|------|---------|
| T-1.2 プロンプトリファイン | `/no_think`（高速） | テキストプロンプト | 改善済みプロンプト | prompt_review.html |
| T-2.2 画像QA | `/no_think`（高速） | 画像1枚 | realism/integrity スコア + pass/fail | image_qa_review.html |
| T-3.3 VLM 3D検品 | `/think`（精度重視） | 4方向レンダリング画像 | geometry/texture/consistency スコア + pass/fail | mesh_vlm_qa_review.html |
