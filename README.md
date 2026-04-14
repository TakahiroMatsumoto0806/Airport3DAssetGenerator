# AL3DG — Airport Luggage 3D Asset Generator

空港荷物コンテナ詰込ロボット向け Physical AI 学習データ用の 3D アセットを大量自動生成するパイプライン。

**目標**: 任意の数の PBR テクスチャ付き 3D アセット（Isaac Sim USD 形式、コリジョン・物理プロパティ付き）

---

## インストール

> **事前確認**: インストールを始める前に「[前提条件](#前提条件)」セクションを必ず確認してください。

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

### Method B: 共有ドライブから（オフライン・推奨）

別の DGX Spark からモデルを受け取る場合。Hugging Face アカウント不要。

`share/` はモデルウェイト（~75GB）を含むオフライン配布パッケージです。
送り側 DGX Spark の `share/` フォルダを受け取り側へ転送してから以下を実行します。

```bash
# 【送り側 DGX Spark で実行】
rsync -avz --progress \
  /path/to/Airport3DAssetGenerator/share/ \
  user@target-dgx:/data/al3dg_share/

# 【受け取り側 DGX Spark で実行】
git clone https://github.com/TakahiroMatsumoto0806/Airport3DAssetGenerator.git
cd Airport3DAssetGenerator/al3dg

# モデルコピー + 仮想環境セットアップ（自動実行）
bash scripts/setup_from_share.sh /data/al3dg_share

source .venv/bin/activate
python scripts/run_pipeline.py
```

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

> **前提**: NVIDIA Container Toolkit インストール済み
> ```bash
> sudo apt install nvidia-container-toolkit
> sudo systemctl restart docker
> ```

> **注意**: `docker compose up` はデフォルトで `--help` を表示するだけです。
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
| ディスク空き容量 | モデル ~117GB + 出力 ~10GB = **合計 130GB 以上** |
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

### TRELLIS.2 専用 PC（Step 3: 3D 生成）

TRELLIS.2-4B は aarch64 では動作しない。別途 x86_64 PC が必要。

| 項目 | 要件 |
|------|------|
| アーキテクチャ | x86_64 |
| GPU | NVIDIA RTX 5090 推奨（VRAM 24GB 以上） |
| conda | Miniconda / Anaconda（TRELLIS.2 の `setup.sh` が conda 環境を作成） |
| ネットワーク | DGX Spark と SSH/rsync でファイル転送可能なこと |

> **注意:** DGX Spark（aarch64）では TRELLIS.2 は実行できません。
> GLB ファイルを生成後、`rsync` で `outputs/meshes_raw/` に転送してください。

---

## モデル構成

| モデル | 用途 | サイズ |
|--------|------|--------|
| FLUX.1-schnell | Text-to-Image 生成 | ~12GB |
| TRELLIS.2-4B (Microsoft) | Image-to-3D 生成（別 PC で実行） | ~24GB |
| Qwen3-VL-32B (vLLM) | プロンプト生成・画像検品・3D 検品（全タスク統一） | ~65GB |

モデルは逐次ロード（同時ロード禁止）。ピーク使用量は常に 128GB 以下に保つ。

---

## パイプライン構成

```
Step 1: プロンプト生成  (Qwen3-VL-32B, /no_think)
    ↓
Step 2: 画像生成        (FLUX.1-schnell, 1024×1024, steps=4)
    ↓
QA-1:  画像検品         (Qwen3-VL-32B, /no_think, realism≥7, integrity≥7)
    ↓
Step 3: 3D 生成         (TRELLIS.2-4B, 別 PC で実行 → GLB を転送)
    ↓
QA-2a: メッシュ QA      (Open3D + trimesh, ルールベース)
QA-2b: VLM 3D 検品      (Qwen3-VL-32B, /think, geometry≥6, texture≥5)
    ↓
Step 4: 物理プロパティ  (CoACD 凸分解 + material_properties.yaml)
    ↓
Step 5: エクスポート    (Isaac Sim USD メタデータ)
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
| メモリ使用量 | 約 65GB。FLUX.1 / TRELLIS.2 ロード前に必ず停止すること |

---

## パイプライン実行

### 通常の運用フロー

```bash
# フルパイプライン実行（プロンプト生成〜エクスポートまで一括）
# ※ vLLM（Qwen3-VL-32B）の初回起動に約 15 分かかります
python scripts/run_pipeline.py
```

完了後は「[レポートの確認](#レポートの確認)」セクションの手順でブラウザから結果を確認できます。

### Step 3（3D 生成）は別 PC で実行

TRELLIS.2-4B は x86_64 + RTX 5090 の別 PC で実行する。DGX Spark では mesh ステップをスキップし、生成済み GLB を転送してから続行する。

```bash
# DGX Spark での実行（mesh ステップを除く）
python scripts/run_pipeline.py --steps prompt image image_qa

# 別 PC で TRELLIS.2 を実行後、GLB を転送
rsync -avz user@trellis-pc:~/outputs/meshes_raw/ outputs/meshes_raw/

# 転送後、後続ステップを実行
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
| `prompt` | プロンプト生成（Qwen3-VL-32B リファイン） |
| `image` | 画像生成（FLUX.1-schnell） |
| `image_qa` | 画像検品（Qwen3-VL-32B） |
| `mesh` | 3D メッシュ生成（TRELLIS.2、別 PC 推奨） |
| `mesh_qa` | メッシュ品質チェック（trimesh + Open3D） |
| `mesh_vlm_qa` | VLM マルチビュー 3D 検品（Qwen3-VL-32B） |
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
├── meshes_raw/                           # TRELLIS.2 生成 GLB（別 PC から転送）
├── meshes_approved/
│   ├── *.glb                             # メッシュ QA + VLM QA 合格メッシュ
│   └── mesh_vlm_qa_results.json         # VLM QA スコア記録
├── assets_final/
│   ├── *.glb                             # 物理プロパティ付き最終メッシュ
│   ├── metadata/
│   │   └── *.json                        # Isaac Sim USD エクスポートメタデータ
│   └── collisions/
│       └── *.obj                         # CoACD 凸分解コリジョンメッシュ
├── renders/                              # VLM 3D 検品用マルチビューレンダリング画像
├── reports/
│   ├── prompt_review.html                # プロンプト一覧・VLM リファイン入出力（HTML）
│   ├── image_qa_review.html              # 画像 QA 検品レポート（HTML）
│   ├── mesh_vlm_qa_review.html           # 3D 検品レポート（HTML）
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
  target_count: 1000          # 目標生成数（最終的に assets_final に残るアセット数）
  prompt_count_per_category: 10  # カテゴリあたりのプロンプト生成数

models:
  vlm:
    base_url: "http://localhost:8001/v1"  # vLLM サーバーの URL
    # model_name: vLLM serve 起動時と同じパスを指定する（~/は自動展開される）
    # モデルを ~/models/ 以外に置く場合は絶対パスまたは ~/... で指定
    model_name: "~/models/Qwen3-VL-32B-Instruct"

image_qa:
  thresholds:
    realism: 7      # 画像リアリティ閾値（1-10、高いほど厳しい）
    integrity: 7    # 形状整合性閾値（1-10）

mesh_vlm_qa:
  thresholds:
    geometry: 6     # 3D 形状品質閾値（1-10）
    texture: 5      # テクスチャ品質閾値（1-10）
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

## カテゴリ別合格率の測定

**目的**: 本番の大量生成（1,000件）を開始する前に、カテゴリごとの通過率を小規模サンプルで測定し、
`configs/prompt_templates.yaml` の `category_weights` を最適化する。

各カテゴリの画像 QA 通過率にばらつきがあるため、通過率の低いカテゴリの重みを下げて
最終アセットの分布が目標比率に近くなるよう調整する。

```bash
# 20 枚/カテゴリで測定（精度 ±22%、推奨スケール）
python scripts/measure_pass_rates.py --samples 20

# 5 枚/カテゴリで簡易テスト
python scripts/measure_pass_rates.py --samples 5 --no-llm-refine

# 既存 QA 結果からレポートのみ再生成
python scripts/measure_pass_rates.py --skip-pipeline
```

> **注意**: 既存の `outputs/` が上書きされる。事前に `backup_outputs.py` でバックアップを取ること。

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
| `outputs/reports/prompt_review.html` | プロンプト一覧・VLM リファイン入出力・生成画像サムネイル |
| `outputs/reports/image_qa_review.html` | 画像 QA 検品スコア・合否・サムネイル |
| `outputs/reports/mesh_vlm_qa_review.html` | 3D 検品マルチビュー画像・スコア・問題点 |
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

バックアップ先: `outputs_backup/YYYYMMDD_HHMMSS/`

---

## DGX Spark 運用上の注意点

### 逐次ロード戦略（必須）

DGX Spark は 128GB 統合メモリ。**モデルを同時にロードしない**こと。

| 実行中のモデル | 停止しておくもの |
|--------------|----------------|
| Qwen3-VL-32B（~65GB） | FLUX.1 / TRELLIS.2 |
| FLUX.1-schnell（~12GB） | vLLM サーバー |
| TRELLIS.2-4B（~24GB） | vLLM サーバー |

`run_pipeline.py` はこの切り替えを自動管理します。`run_step.py` で個別実行する場合は上記制約を手動で守ってください。

### TRELLIS.2 は別 PC で実行

TRELLIS.2-4B は x86_64 アーキテクチャが必要です。DGX Spark（aarch64）では動作しません。
GLB ファイルを生成後、`outputs/meshes_raw/` へ転送してから後続ステップを実行します。
