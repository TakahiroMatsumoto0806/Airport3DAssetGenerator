# AL3DG — Airport Luggage 3D Asset Generator

空港荷物コンテナ詰込ロボット向け Physical AI 学習データ用の 3D アセットを大量自動生成するパイプライン。

**目標**: 任意の数の PBR テクスチャ付き 3D アセット（Isaac Sim USD 形式、コリジョン・物理プロパティ付き）

---

## 目次

- [インストール](#インストール)
- [前提条件](#前提条件)
- [モデル構成](#モデル構成)
- [パイプライン構成](#パイプライン構成)
- [vLLM サーバー管理](#vllm-サーバー管理)
- [パイプライン実行](#パイプライン実行)
- [出力ディレクトリ構成](#出力ディレクトリ構成)
- [設定ファイル](#設定ファイル)
- [プロンプトの調整](#プロンプトの調整)
- [カテゴリ別合格率の測定](#カテゴリ別合格率の測定)
- [レポートの確認](#レポートの確認)
- [バックアップ](#バックアップ)
- [DGX Spark 運用上の注意点](#dgx-spark-運用上の注意点)

---

## インストール

> [!IMPORTANT]
> インストールを始める前に「[前提条件](#前提条件)」セクションを必ず確認してください。

### Method A: GitHub + Hugging Face（オンライン）

```bash
git clone https://github.com/TakahiroMatsumoto0806/Airport3DAssetGenerator.git
cd Airport3DAssetGenerator/al3dg

# 依存パッケージ・Python 仮想環境のセットアップ
bash scripts/setup_environment.sh

# 仮想環境をアクティベート（以降のコマンドはすべてこの venv 内で実行）
source .venv/bin/activate

# Hugging Face にログイン（モデルダウンロード前に必須）
huggingface-cli login

# モデルダウンロード（~117GB）
python scripts/download_models.py

# パイプライン実行
python scripts/run_pipeline.py
```

### Method B: ZIP ファイル経由（オフライン・推奨）

別の DGX Spark にモデルを渡す場合。Hugging Face アカウント不要。

送り側 DGX Spark で `al3dg_share.zip`（モデルウェイト約 117GB を含む）を作成し、
USB ドライブや外付け SSD 等の物理メディアで受け取り側へ運搬します。

```bash
# 【送り側 DGX Spark で実行】— ZIP の作成
cd /path/to/Airport3DAssetGenerator
zip -r -0 al3dg_share.zip share/
# → al3dg_share.zip (~117GB) が作成される
# その後、USB / 外付け SSD などへコピーして受け取り側へ運搬

# 【受け取り側 DGX Spark で実行】— ZIP の展開とセットアップ
mkdir -p /data/al3dg_share
cd /data/al3dg_share
unzip /path/to/al3dg_share.zip
# → /data/al3dg_share/share/ 配下にモデルが展開される

git clone https://github.com/TakahiroMatsumoto0806/Airport3DAssetGenerator.git
cd Airport3DAssetGenerator/al3dg

# モデルコピー + 仮想環境セットアップ（自動実行）
bash scripts/setup_from_share.sh /data/al3dg_share/share

source .venv/bin/activate
python scripts/run_pipeline.py
```

> **ディスク要件（受け取り側）**:
> - ZIP ファイル本体: ~117GB
> - 展開後の `share/`: ~117GB
> - `~/models/` へコピー後: ~117GB
> - 一時的に最大 ~351GB の空き容量が必要（展開と `~/models/` コピー後に ZIP と `share/` を削除可能）

詳細は `share/README.md` を参照。

### Method C: Docker（コンテナ実行）

```bash
git clone https://github.com/TakahiroMatsumoto0806/Airport3DAssetGenerator.git
cd Airport3DAssetGenerator/al3dg

# ~/models/ にモデルを用意（Method A or B で取得）

# イメージをビルド
docker compose build

# パイプライン実行（vLLM はコンテナ内の run_pipeline.py が自動起動・停止）
docker compose run --rm al3dg python scripts/run_pipeline.py

# 個別ステップの実行例
docker compose run --rm al3dg python scripts/run_step.py --step mesh_qa
```

> [!NOTE]
> NVIDIA Container Toolkit がインストール済みであること。
> ```bash
> sudo apt install nvidia-container-toolkit
> sudo systemctl restart docker
> ```

> [!WARNING]
> `docker compose up` はデフォルトで `--help` を表示するだけです。
> 実際のパイプライン実行には `docker compose run --rm al3dg python scripts/run_pipeline.py` を使用してください。

---

## 前提条件

### DGX Spark（メイン実行環境）

| 項目 | 要件 |
|------|------|
| ハードウェア | NVIDIA DGX Spark (Grace Blackwell GB10) |
| アーキテクチャ | aarch64 (ARM64) |
| 統合メモリ | 128GB 以上 |
| OS | Ubuntu 22.04 LTS |
| CUDA | 12.4 以上 |
| Python | 3.11（`uv venv` で自動インストール） |
| ディスク空き容量 | Method A: モデル ~117GB + 出力 ~10GB = **合計 130GB 以上**<br>Method B: ZIP 展開・コピー一時領域含む **合計 360GB 以上**（セットアップ後は 130GB に縮小） |
| sudo 権限 | 必要（`apt-get` でシステムパッケージをインストール） |
| git / curl | 事前にインストール済みであること（`sudo apt install git curl`） |
| ネットワーク | Method A: Hugging Face へのアクセス / Method B: 共有ドライブ or SCP |
| ポート 8001 | vLLM サーバー用（他プロセスと競合しないこと） |

### Hugging Face アカウント（Method A のみ）

1. [Hugging Face](https://huggingface.co) にアカウント登録
2. 以下モデルのライセンスに同意:
   - [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)（要ライセンス同意）
   - [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
3. アクセストークンを発行し、`huggingface-cli login` で認証

### Docker 環境（Method C のみ）

- Docker Engine 24.0 以上
- NVIDIA Container Toolkit（`nvidia-docker2`）

```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
docker info | grep -i runtime  # → nvidia が表示されることを確認
```

### 3D 生成（Step 3）は本プロジェクトの対象範囲外

Step 3（画像 → 3D メッシュ生成）は別 PC（x86_64 + RTX 5090）で別プロジェクトとして実施します。
本リポジトリのセットアップ・ドキュメントは Step 3 を**含みません**。
Step 3 で生成した GLB ファイルを `outputs/meshes_raw/` に配置した状態から、
本パイプラインのメッシュ QA 以降（QA-2a, QA-2b, Step 4, Step 5）を実行してください。

---

## モデル構成

DGX Spark でロードするモデル：

| モデル | 用途 | メモリ使用量 |
|--------|------|--------------|
| FLUX.1-schnell | Text-to-Image 生成 | ~12GB |
| Qwen3-VL-32B (vLLM) | 画像 QA・マルチビュー 3D QA（QA 用途に統一） | ~100GB |

> プロンプト生成は `configs/prompt_templates.yaml` / `luggage_categories.yaml` の組合せ生成のみで、vLLM / GPU を使用しません。

> **Image-to-3D（Step 3）は本プロジェクトの対象範囲外です（別 PC で別プロジェクトとして実施）。
> GLB を受け取って以降の QA・物理付与・エクスポートのみを本プロジェクトで担います。**

全てのモデルを一度に GPU メモリに展開するとメモリ不足になるため、モデルは逐次ロード（同時ロード禁止）とします。
１つずつロードしてアンロードを繰り返すことで、ピーク使用量は常に 128GB 以下に保たれます。

---

## パイプライン構成

```
Step 1: プロンプト生成    (組合せ生成、GPU 不使用)
    ↓
Step 2: 画像生成          (FLUX.1-schnell, 1024×1024, steps=4, ランダムシード)
    ↓
QA-1:  画像 QA            (Qwen3-VL-32B, /no_think, realism≥7, integrity≥7)
    ↓
Step 3: 3D 生成           ※ 本プロジェクトの対象範囲外（別 PC で実施、GLB を受け取る）
    ↓
QA-2a: メッシュ QA        (Open3D + trimesh, ルールベース)
QA-2b: マルチビュー 3D QA (Qwen3-VL-32B, /think,
                          geometry≥6, texture≥5, consistency≥6, reality≥6)
    ↓
Step 4: 物理プロパティ    (CoACD 凸分解 + material_properties.yaml)
    ↓
Step 5: エクスポート      (Isaac Sim USDA + メタデータ JSON。
                          collisions Xform は visibility=invisible)
```

---

## vLLM サーバー管理

**`run_pipeline.py` はvLLM の起動・停止を自動管理するため、フルパイプライン実行時に手動操作は不要です。**

個別ステップ（`run_step.py`）を手動実行する場合のみ、別ターミナルで起動してください:

```bash
bash scripts/start_vllm_server.sh &
curl http://localhost:8001/health   # 起動確認（"ok" が返るまで待つ）
```

| 項目 | 内容 |
|------|------|
| ポート | 8001（8000 は Docker proxy が使用中のため） |
| 初回起動時間 | **約 15 分**（14 シャード読み込み + CUDA グラフ生成） |
| メモリ使用量 | 約 100GB（KV キャッシュ・CUDA グラフ等を含む）。FLUX.1 ロード前に必ず停止すること |

---

## パイプライン実行

### 通常の運用フロー

```bash
# フルパイプライン実行（プロンプト生成〜エクスポートまで一括）
# ※ vLLM（Qwen3-VL-32B）の初回起動に約 15 分かかります
python scripts/run_pipeline.py
```

完了後は「[レポートの確認](#レポートの確認)」セクションの手順でブラウザから結果を確認できます。

### Step 3（3D 生成）は本プロジェクトの対象範囲外

Step 3（画像 → 3D メッシュ生成）は別 PC・別プロジェクトで実施します。本プロジェクトでは、
受け取った GLB ファイルを `outputs/meshes_raw/` に配置した状態から後続ステップを実行します。

```bash
# 本 PC（DGX Spark）での実行：プロンプト生成〜画像 QA まで
python scripts/run_pipeline.py --steps prompt image image_qa

# → 画像 QA 合格分（outputs/images_approved/）を別 PC へ引き渡す
# → 別 PC 側で生成された GLB を outputs/meshes_raw/ に配置した後、後続ステップを実行

python scripts/run_pipeline.py --steps mesh_qa mesh_vlm_qa physics sim_export
```

### よく使うオプション

```bash
# 特定ステップのみ実行
python scripts/run_pipeline.py --steps prompt image image_qa

# 全件再実行（中断再開をリセット）
python scripts/run_pipeline.py --no-resume

# 設定ファイルを指定
python scripts/run_pipeline.py --config configs/pipeline_config.yaml

# vLLM を手動管理する場合（自動起動・停止を無効化）
python scripts/run_pipeline.py --no-vllm-auto-start
```

### 個別ステップ実行

```bash
# 任意のステップを単体で実行
python scripts/run_step.py --step <ステップ名>

# 例: 物理プロパティ付与のみ
python scripts/run_step.py --step physics
```

利用可能なステップ一覧:

| ステップ名 | 処理内容 |
|-----------|---------|
| `prompt` | プロンプト生成（組合せ生成） |
| `image` | 画像生成（FLUX.1-schnell） |
| `image_qa` | 画像 QA（Qwen3-VL-32B） |
| `mesh` | 3D メッシュ生成 — ※ 本プロジェクト対象外（別 PC で実施、スキップ可） |
| `mesh_qa` | メッシュ品質チェック（trimesh + Open3D、ルールベース） |
| `mesh_vlm_qa` | マルチビュー 3D QA（Qwen3-VL-32B） |
| `physics` | 物理プロパティ付与（CoACD 凸分解） |
| `sim_export` | Isaac Sim USD エクスポートメタデータ生成 |

---

## 出力ディレクトリ構成

```
outputs/
├── prompts/
│   └── prompts.json                      # 生成済みプロンプト一覧
├── images/                               # FLUX.1 生成画像（PNG）
├── images_approved/
│   ├── *.png                             # 画像 QA 合格画像
│   └── image_qa_results.json            # QA スコア・合否記録
├── meshes_raw/                           # 別 PC で生成された GLB の受け取り先
├── meshes_approved/
│   ├── *.glb                             # メッシュ QA + VLM QA 合格メッシュ
│   └── mesh_vlm_qa_results.json         # VLM QA スコア記録
├── assets_final/
│   └── <asset_id>/                       # アセット毎のディレクトリ
│       ├── visual.glb                    # ビジュアル GLB
│       ├── collisions/*.stl              # CoACD 凸分解コリジョンメッシュ
│       ├── physics.json                  # 物理プロパティ
│       ├── <asset_id>.usda               # Isaac Sim USDA
│       └── <asset_id>_usd_meta.json      # Isaac Sim USD エクスポートメタデータ
├── renders/                              # マルチビュー 3D QA 用レンダリング画像
├── reports/
│   ├── prompt_review.html                # プロンプト一覧・生成画像サムネイル（HTML）
│   ├── image_qa_review.html              # 画像 QA レポート（HTML）
│   ├── mesh_vlm_qa_review.html           # マルチビュー 3D QA レポート（HTML）
│   ├── pass_rate_report.html             # カテゴリ別合格率レポート（HTML）
│   └── pass_rate_report.json            # カテゴリ別合格率サマリー（JSON）
└── logs/                                 # 実行ログ
```

各アセットには provenance（pass / review / reject）が JSON で記録される。

---

## 設定ファイル

### `configs/pipeline_config.yaml` — メイン設定

主要な調整項目:

```yaml
generation:
  prompt_generate_number: 20    # 生成するプロンプトの総数（category_weights の比率で割り当て）
                                # 初回動作確認向けの既定値。本番は必要数に合わせて増やす

models:
  vlm:
    base_url: "http://localhost:8001/v1"  # vLLM サーバーの URL
    # model_name: vLLM serve 起動時と同じパスを指定する。
    # vLLM は "~/" を自動展開しないため、ホームディレクトリ配下に置く場合も
    # 絶対パスで指定する（scripts/start_vllm_server.sh も同じパスを使用）
    model_name: "/home/ntt/models/Qwen3-VL-32B-Instruct"

image_qa:
  thresholds:
    realism: 7      # 画像リアリティ閾値（1-10、高いほど厳しい）
    integrity: 7    # 形状整合性閾値（1-10）

mesh_vlm_qa:
  thresholds:
    geometry: 6     # 3D 形状品質閾値（1-10）
    texture: 5      # テクスチャ品質閾値（1-10）
    consistency: 6  # 画像とのカテゴリ一貫性閾値（1-10）
    reality: 6      # 実在感・現実性閾値（1-10）
```

### `configs/prompt_templates.yaml` — プロンプト・カテゴリ重み

カテゴリごとの生成比率（`sampling.category_weights`）を調整することで、特定カテゴリの生成数を増減できる（合計が 1.0 になるよう設定）:

```yaml
sampling:
  category_weights:
    hard_suitcase: 0.27   # 最終合格の約 27% を hard_suitcase にする場合
    soft_suitcase: 0.16
    backpack: 0.16
    # ...（11 カテゴリ、合計 1.00）
```

### `configs/luggage_categories.yaml` — カテゴリ定義

各カテゴリのサイズ・材質・色などを定義する。カテゴリを追加・削除する際は以下 3 ファイルを同時に更新すること:

1. `configs/luggage_categories.yaml` — カテゴリ本体
2. `configs/prompt_templates.yaml` — `attributes.type` にテンプレート追加 + `category_weights` に重み追加
3. `configs/material_properties.yaml` — `category_material_mapping` にデフォルト材質追加

---

## プロンプトの調整

本パイプラインには 3 種類のプロンプトがあり、それぞれ編集箇所と再実行方法が異なります。まず全体像を把握してから個別セクションに進んでください。

| # | プロンプト | 役割 | 編集箇所 | 再実行ステップ |
|---|-----------|------|----------|---------------|
| ① | **画像生成プロンプト** | FLUX.1-schnell に渡す英文を組合せ生成する | `configs/prompt_templates.yaml` | `prompt` → `image` |
| ② | **画像 QA プロンプト** | Qwen3-VL-32B が画像 QA 結果を JSON で返すための指示 | `src/image_qa.py`（モジュール定数） | `image_qa` |
| ③ | **マルチビュー 3D QA プロンプト** | Qwen3-VL-32B がマルチビュー 3D QA 結果を JSON で返すための指示 | `src/mesh_vlm_qa.py`（モジュール定数） | `mesh_vlm_qa` |

> [!TIP]
> ① は YAML 編集のみで完結します。② ③ は Python ソースのモジュール定数を編集するだけで、再インストールや再ビルドは不要です。編集後、該当ステップだけを単体再実行すれば反映されます（`python scripts/run_step.py --step <ステップ名>`）。

### ① 画像生成プロンプト（`configs/prompt_templates.yaml`）

FLUX.1-schnell に渡す英文プロンプトは、YAML 内の属性軸（タイプ・色・材質・テクスチャ・スタイル・状態）を `src/prompt_generator.py` の `_build_prompt()` が組合せて生成します。vLLM / GPU は使用しません。

**主な編集ポイント**

| 目的 | 編集箇所 |
|------|---------|
| カテゴリを追加・削除 | `attributes.type.<カテゴリ>.<サブカテゴリ>` に英文フレーズを追加／削除<br>※ 同時に `sampling.category_weights` と `configs/luggage_categories.yaml`・`configs/material_properties.yaml` も更新 |
| 色・材質・テクスチャ・スタイル・状態のバリエーションを増やす | `attributes.{color, material, texture, style, condition}` の各配列にフレーズを追加 |
| カテゴリの生成比率を調整 | `sampling.category_weights`（合計 1.0） |
| 特定カテゴリのサイズ比率を調整 | `sampling.subcategory_size_weights` |
| 背景・照明・構図を一括変更 | `fixed.{prefix, bg_brief, lighting, view, quality}` |
| カテゴリ固有の短い先頭フレーズを追加（CLIP 77 トークン枠の先頭に必ず届く） | `category_clip_prefix.<カテゴリ>`（例: ハンドル格納・バッグ閉鎖など QA-1 で弾かれやすい指示を優先） |

**再実行**

```bash
# プロンプト再生成から画像 QA まで
python scripts/run_pipeline.py --steps prompt image image_qa --no-resume
```

> [!WARNING]
> FLUX.1-schnell は negative prompt に対応していません。`fixed.negative_hint` は空文字で固定し、変更しないでください。

### ② 画像 QA プロンプト（`src/image_qa.py`）

Qwen3-VL-32B への指示は `src/image_qa.py` の 2 つのモジュール定数にまとまっています。

| 定数 | 役割 |
|------|------|
| `_SYSTEM_PROMPT` | モデルのロール指定（「checked baggage 条件を評価」「JSON のみ返す」等） |
| `_USER_PROMPT_TEMPLATE` | 評価軸と JSON スキーマの定義。`{min_realism}` / `{min_integrity}` / `{min_coverage}` のプレースホルダには `configs/pipeline_config.yaml > image_qa.thresholds` の閾値が自動挿入される |

**主な編集ポイント**

| 目的 | 編集方法 |
|------|---------|
| 合格閾値のみ変えたい | `configs/pipeline_config.yaml > image_qa.thresholds` を編集（Python 側は触らない） |
| 評価軸を追加・削除したい | `_USER_PROMPT_TEMPLATE` の JSON スキーマにキーを追加／削除し、`_validate_and_normalize()` と `_apply_defaults()` のロジックを対応させる |
| 合否判定ロジックを変えたい | `_validate_and_normalize()` 内の `passed = (...)` 条件式を編集 |
| モデルのロール指定・前提条件を変えたい | `_SYSTEM_PROMPT` を編集 |

**再実行**

```bash
# 既存画像に対して画像 QA のみを再実行
python scripts/run_step.py --step image_qa
```

> [!NOTE]
> プロンプトは英語で記述されています（Qwen3-VL-32B は多言語対応ですが、英語のほうが JSON 整形精度が安定）。日本語化する場合は応答スキーマも日本語化し、`_parse_json_response()` の正規表現が追従できることを確認してください。

### ③ マルチビュー 3D QA プロンプト（`src/mesh_vlm_qa.py`）

構成は ② と同じで、モジュール定数 `_SYSTEM_PROMPT` / `_USER_PROMPT_TEMPLATE` を編集します。front / right / back / left の 4 方向レンダリングを入力とし、`geometry_score` / `texture_score` / `consistency_score` / `reality_score` の 4 軸で評価します。

**主な編集ポイント**

| 目的 | 編集方法 |
|------|---------|
| 合格閾値のみ変えたい | `configs/pipeline_config.yaml > mesh_vlm_qa.thresholds` を編集（Python 側は触らない） |
| スコア軸を追加・削除したい | `_USER_PROMPT_TEMPLATE` の JSON スキーマを編集し、`_apply_defaults()` の閾値チェックを対応させる |
| 評価基準の語り口を変えたい（例: 「AI-generated mesh 基準」を「プロ品質基準」に変えて厳しくする） | `_SYSTEM_PROMPT` と `_USER_PROMPT_TEMPLATE` の本文を書き換え |
| 検品対象ビュー数を変えたい | `{n_views}` プレースホルダは呼び出し側のレンダラから自動挿入されるため、レンダラ側（ビュー数生成箇所）も合わせて変更 |

**再実行**

```bash
# 既存メッシュに対してマルチビュー 3D QA のみを再実行
python scripts/run_step.py --step mesh_vlm_qa
```

> [!TIP]
> `reality_score` はテクスチャ品質を除外した「形状としての実在感」を問う軸です。TRELLIS 出力はテクスチャが不完全なことがあるため、形状だけで判定できる軸を別途用意しています。厳しくしすぎるとテクスチャの悪いメッシュが通ってしまうので、他の 3 軸と組合せて調整してください。

---

## カテゴリ別合格率の測定

**目的**: 本番の大量生成を開始する前に、カテゴリごとの通過率を小規模サンプルで測定し、
`configs/prompt_templates.yaml` の `category_weights` を最適化する。

各カテゴリの画像 QA 通過率にばらつきがあるため、通過率の低いカテゴリの重みを下げて
最終アセットの分布が目標比率に近くなるよう調整する。

```bash
# 20 枚/カテゴリで測定（精度 ±22%、推奨スケール）
python scripts/measure_pass_rates.py --samples 20

# 5 枚/カテゴリで簡易テスト
python scripts/measure_pass_rates.py --samples 5

# 既存 QA 結果からレポートのみ再生成
python scripts/measure_pass_rates.py --skip-pipeline
```

> [!CAUTION]
> 既存の `outputs/` が上書きされます。事前に `scripts/backup_outputs.py` でバックアップを取ってください。

実行完了後、`outputs/reports/pass_rate_report.html` にカテゴリ別合格率と推奨 `category_weights` が表示される。

---

## レポートの確認

DGX Spark はヘッドレスサーバーのため、`open` コマンドは使用できません。
ローカル PC のブラウザからアクセスするには以下のいずれかの方法を使用してください。

```bash
# 方法 1: HTTP サーバーを起動してブラウザアクセス（推奨）
python3 -m http.server 8080 --directory outputs/
# → ローカル PC のブラウザで http://<DGX-Spark-IP>:8080/reports/ を開く

# 方法 2: ローカル PC に scp でコピー
scp -r user@dgx-spark:~/Airport3DAssetGenerator/al3dg/outputs/ ~/al3dg_outputs/
# → ローカル PC のブラウザで al3dg_outputs/reports/prompt_review.html を開く
```

| レポート | 内容 |
|---------|------|
| `outputs/reports/prompt_review.html` | プロンプト一覧・生成画像サムネイル |
| `outputs/reports/image_qa_review.html` | 画像 QA スコア・合否・サムネイル |
| `outputs/reports/mesh_vlm_qa_review.html` | マルチビュー 3D QA スコア・レンダリング画像・問題点 |
| `outputs/reports/physics_report.html` | 物理プロパティ付与結果（質量・摩擦・コリジョン数） |
| `outputs/reports/pass_rate_report.html` | カテゴリ別合格率・推奨 `category_weights` |

### レポートのみ再生成（パイプライン再実行なし）

```bash
# 画像 QA リジェクト分析
python scripts/generate_report.py --step image_qa

# カテゴリ別合格率レポート
python scripts/generate_report.py --step pass_rate
```

---

## バックアップ

`measure_pass_rates.py` や `--no-resume` 実行は既存の `outputs/` を上書きするため、重要な出力は事前にバックアップする。

```bash
python scripts/backup_outputs.py
```

バックアップ先: `outputs_backup/YYYY-MM-DD_HHMMSS/`

---

## DGX Spark 運用上の注意点

### 逐次ロード戦略（必須）

DGX Spark は 128GB 統合メモリ。**モデルを同時にロードしない**こと。

| 実行中のモデル | 停止しておくもの |
|--------------|----------------|
| Qwen3-VL-32B（~100GB） | FLUX.1-schnell |
| FLUX.1-schnell（~12GB） | vLLM サーバー |

`run_pipeline.py` はこの切り替えを自動管理します。`run_step.py` で個別実行する場合は上記制約を手動で守ってください。

### 3D 生成（Step 3）は本プロジェクトの対象範囲外

Step 3 は別 PC・別プロジェクトで実施します。生成された GLB を `outputs/meshes_raw/` に配置した状態から、
本プロジェクトのメッシュ QA 以降を実行してください。
