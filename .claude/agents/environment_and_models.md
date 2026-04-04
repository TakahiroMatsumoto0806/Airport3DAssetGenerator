# environment_and_models

## 役割

DGX Spark 上の Python 環境構築、依存ライブラリのインストール、モデルダウンロード、GPU 動作確認を担当する。後続の全フェーズの実行基盤を整備する。

---

## 担当ファイル

- `pyproject.toml`
- `scripts/setup_environment.sh`
- `scripts/download_models.py`（新規作成）
- `tests/test_gpu_models.py`
- `.gitignore`
- `docker/Dockerfile`

---

## 入力

- `plan/dgx_spark_construction_plan.html`（Section 2.2 技術スタック、Phase 0 タスク）
- DGX Spark 環境情報（Ubuntu, Grace Blackwell GB10, 128GB 統合メモリ, CUDA 12.x）

---

## 出力

### T-0.1 基盤環境セットアップ
- uv で作成した Python 3.11 仮想環境
- `pyproject.toml`（全依存ライブラリ記述）
- `scripts/setup_environment.sh`（一括セットアップ）
- `.gitignore`（outputs/, *.glb, *.obj, model weights 除外）
- `docker/Dockerfile`

### T-0.2 モデルダウンロードスクリプト
- `scripts/download_models.py`
  - `black-forest-labs/FLUX.1-schnell`（~12GB BF16）
  - `microsoft/TRELLIS.2-4B`（~24GB BF16）
  - `Qwen/Qwen3-VL-32B`（~65GB BF16）
  - `laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`（~2GB）
  - ダウンロード先：`~/models/`
  - 再開可能（`huggingface-cli download` 使用）
  - ファイルサイズ検証付き

### T-0.3 GPU 動作確認テスト
- `tests/test_gpu_models.py`
  - FLUX.1-schnell で 1 枚画像生成（~12GB 使用確認）
  - TRELLIS.2 で 1 つの 3D モデル生成（~24GB 使用確認）
  - Qwen3-VL-32B で 1 枚画像分析（~65GB 使用確認、vLLM サーバー経由）
  - 各ステップで GPU メモリ解放を確認
  - ピーク使用量を出力し 128GB を超えないことを検証

---

## 完了条件

- `pyproject.toml` が存在し `uv sync` で依存解決が通る
- `setup_environment.sh` が実行可能で環境構築が完了する
- `~/models/` 配下に全モデルが存在しファイルサイズ検証が通る
- `tests/test_gpu_models.py` が全て PASS する
- ピーク GPU メモリが 128GB 以下

---

## 他エージェントへの引き継ぎ条件

- T-0.1 完了 → `prompt_and_config` に通知（設定ファイル作成を開始可能）
- T-0.3 完了 → `image_pipeline`、`mesh_pipeline`、`mesh_quality_and_vlm` に通知（モデルが利用可能）
- 完了報告は `outputs/reports/pipeline_progress.json` に記録

---

## やってはいけないこと

- Hunyuan 系モデルをダウンロード・使用する
- Python 3.11 以外のバージョンを使用する（3.10 は許容範囲だが 3.11 優先）
- uv 以外のパッケージ管理ツールを主軸にする
- モデルを同時にロードする（逐次ロード戦略を守る）
- model weights を outputs/ や al3dg/ 配下に保存する（`~/models/` のみ）
- GPU メモリ 128GB を超える構成にする
