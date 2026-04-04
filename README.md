# AL3DG — Airport Luggage 3D Asset Generator

空港荷物コンテナ詰込ロボット向け Physical AI 学習データ用の 3D アセットを大量自動生成するパイプライン。

**目標**: 1,000 個以上の PBR テクスチャ付き 3D アセット（USD / MJCF 形式、コリジョン・物理プロパティ付き）

---

## 実行環境

| 項目 | 内容 |
|------|------|
| ハードウェア | NVIDIA DGX Spark (Grace Blackwell GB10) |
| メモリ | 128GB 統合メモリ |
| OS | Ubuntu Linux |
| Python | 3.11 |
| パッケージ管理 | uv |

---

## モデル構成

| モデル | 用途 | サイズ |
|--------|------|--------|
| FLUX.1-schnell | Text-to-Image 生成 | ~12GB |
| TRELLIS.2-4B (Microsoft) | Image-to-3D 生成 | ~24GB |
| Qwen3-VL-32B (vLLM) | プロンプト生成・画像検品・3D検品（全タスク統一） | ~65GB |
| CLIP ViT-L/14 (OpenCLIP) | 多様性評価 | ~2GB |

モデルは逐次ロード（同時ロード禁止）。ピーク使用量は常に 128GB 以下に保つ。

---

## セットアップ

```bash
# 1. 環境構築（初回のみ）
bash scripts/setup_environment.sh

# 2. 仮想環境を有効化
source .venv/bin/activate

# 3. モデルダウンロード
python scripts/download_models.py

# 4. GPU 動作確認
python tests/test_gpu_models.py
```

---

## パイプライン実行

```bash
# フルパイプライン（全ステップ順次実行）
python scripts/run_pipeline.py

# 特定ステップのみ実行
python scripts/run_pipeline.py --steps prompt image image_qa

# 中断再開なし（全件再実行）
python scripts/run_pipeline.py --no-resume

# 個別ステップ実行
python scripts/run_step.py --step physics

# 多様性レポートのみ生成
python scripts/generate_report.py --no-clip
python scripts/generate_report.py --assets-dir outputs/assets_final/
```

---

## パイプライン構成

```
Step 1: プロンプト生成  (Qwen3-VL-32B, /no_think)
    ↓
Step 2: 画像生成        (FLUX.1-schnell, 1024×1024, steps=4)
    ↓
QA-1:  画像検品         (Qwen3-VL-32B, /no_think, realism≥7, integrity≥7)
    ↓
Step 3: 3D生成          (TRELLIS.2-4B, simplify=0.95, texture=1024)
    ↓
QA-2a: メッシュQA       (Open3D + trimesh, ルールベース主判定)
QA-2b: VLM 3D検品       (Qwen3-VL-32B, /think, geometry≥7, texture≥6)
    ↓
Step 4: 物理プロパティ  (CoACD凸分解 + material_properties.yaml)
    ↓
Step 5: エクスポート    (MJCF + Isaac Sim USD メタデータ)
    ↓
Step 6: 多様性評価      (OpenCLIP + Vendi Score)
```

---

## 実装フェーズ

| Phase | タスクID | 状態 | ドキュメント |
|-------|---------|------|------------|
| 0 | T-0.1 基盤環境セットアップ | ✅ | — |
| 0 | T-0.2 モデルダウンロード | ✅ | — |
| 0 | T-0.3 GPU 動作確認 | ✅ | — |
| 1 | T-1.1 設定ファイル | ✅ | — |
| 1 | T-1.2 プロンプト生成エンジン | ✅ | — |
| 2 | T-2.1 画像生成エンジン | ✅ | — |
| 2 | T-2.2 画像検品 VLM | ✅ | — |
| 3 | T-3.1 3D 生成エンジン | ✅ | — |
| 3 | T-3.2 メッシュ品質チェック | ✅ | [docs/t32_mesh_qa.md](docs/t32_mesh_qa.md) |
| 3 | T-3.3 VLM マルチビュー検品 | ✅ | [docs/t33_mesh_vlm_qa.md](docs/t33_mesh_vlm_qa.md) |
| 4 | T-4.1 物理プロパティ付与 | ✅ | — |
| 4 | T-4.2 シミュレータエクスポート | ✅ | — |
| 5 | T-5.1 多様性評価 | ✅ | — |
| 6 | T-6.1 パイプライン統合 | ✅ | — |
| 6 | T-6.2 CLI スクリプト | ✅ | — |

---

## 各コンポーネントのセットアップ手順

各フェーズで必要なインストール・設定・実行手順を個別ドキュメントにまとめている。
DGX Spark 固有の注意点（OSMesa, vLLM, TRELLIS.2 conda 環境 等）を詳述。

| ドキュメント | 内容 |
|------------|------|
| [docs/t32_mesh_qa.md](docs/t32_mesh_qa.md) | T-3.2 メッシュ品質チェック（Open3D + trimesh） |
| [docs/t33_mesh_vlm_qa.md](docs/t33_mesh_vlm_qa.md) | T-3.3 VLM マルチビュー検品（pyrender + OSMesa + vLLM） |

---

## DGX Spark 運用上の注意点

### 逐次ロード戦略（必須）

同時に複数の大規模モデルをロードしないこと。ピークメモリは常に 128GB 以下に保つ。

```
VLM（~65GB）使用中 → FLUX.1 / TRELLIS.2 は必ずアンロード済みであること
TRELLIS.2（~24GB）使用中 → VLM は必ずアンロード（vLLM サーバーは停止 or 別プロセス）
```

各クラスの `unload()` メソッドを明示的に呼び出してから次のモデルをロードすること。

### vLLM サーバー

Qwen3-VL-32B は全タスク（T-1.2 プロンプト生成 / T-2.2 画像検品 / T-3.3 3D 検品）で共用する。

```bash
# バックグラウンドで起動
bash scripts/start_vllm_server.sh &

# ヘルスチェック
curl http://localhost:8000/health
```

### TRELLIS.2 専用 conda 環境

TRELLIS.2 は pip 配布なし。専用の conda 環境 `trellis2` が必要。

```bash
cd ~/trellis2
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel
conda activate trellis2
# TRELLIS.2 の推論はこの環境で実行する
```

### pyrender / OSMesa（T-3.3 で必要）

```bash
# OSMesa ライブラリ（初回のみ）
sudo apt-get install -y libosmesa6-dev

# 環境変数（必ず Python 起動前に設定）
export PYOPENGL_PLATFORM=osmesa
```

---

## 正本仕様

`plan/dgx_spark_construction_plan.html` を唯一の正本仕様とする。
