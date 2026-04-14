# T-3.3: VLM マルチビュー 3D 検品 — セットアップと実行手順

対象環境: **DGX Spark (Grace Blackwell GB10, Ubuntu, 128GB 統合メモリ)**

---

## 概要

GLB メッシュを 4 方向からオフスクリーンレンダリングし、Qwen3-VL-32B-Instruct（vLLM
サーバー経由）で幾何品質・テクスチャ品質を自動評価する。

| 項目 | 内容 |
|------|------|
| レンダラー | pyrender + OSMesa (headless OpenGL) |
| VLM | Qwen3-VL-32B-Instruct（vLLM、localhost:8001） |
| Thinking モード | `/think`（3D 幾何の精密評価に有効） |
| 合格基準 | geometry_score ≥ 7 **かつ** texture_score ≥ 6 |
| 出力解像度 | 512 × 512 px × 4 視点 |

---

## 1. 前提条件

### 1-1. OSMesa（ヘッドレス OpenGL）

DGX Spark はディスプレイ非接続の headless 環境のため、OpenGL には OSMesa が必要。
EGL は Grace Blackwell の環境では動作しない場合がある。

```bash
# OSMesa 開発ライブラリのインストール（初回のみ）
sudo apt-get install -y libosmesa6-dev freeglut3-dev
```

インストール確認:
```bash
dpkg -l libosmesa6-dev | grep ii   # "ii" が表示されれば OK
```

### 1-2. pyrender のインストール

```bash
source .venv/bin/activate
uv pip install pyrender
```

`setup_environment.sh` を実行済みであれば `pyproject.toml` の依存に含まれているため、
`uv pip install -e ".[dev]"` でインストール済みのはず。

インストール確認:
```bash
source .venv/bin/activate
python -c "import pyrender; print(pyrender.__version__)"
```

### 1-3. 環境変数の設定（重要）

pyrender がヘッドレス環境で OSMesa を使うために **インポート前**に設定が必要。

```bash
# セッション全体に適用（ターミナルを開くたびに設定が必要）
export PYOPENGL_PLATFORM=osmesa

# ~/.bashrc に恒久設定（推奨）
echo 'export PYOPENGL_PLATFORM=osmesa' >> ~/.bashrc
source ~/.bashrc
```

> **注意**: この環境変数を設定せずに pyrender をインポートすると
> `EGL initialization error` などのエラーが発生する。
> `src/utils/rendering.py` は `os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")` で
> 自動設定するが、pyrender が既に import 済みの場合は効かない。
> シェル環境変数で設定しておく方が確実。

### 1-4. 動作確認（pyrender 単体テスト）

```python
# Python インタラクティブシェルで確認
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # 必ず先に設定
import pyrender
import numpy as np

scene = pyrender.Scene()
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
scene.add(camera)
r = pyrender.OffscreenRenderer(512, 512)
color, depth = r.render(scene)
r.delete()
print(f"レンダリング成功: color shape = {color.shape}")  # (512, 512, 3) が出ればOK
```

---

## 2. vLLM サーバーの起動

T-3.3 は Qwen3-VL-32B-Instruct を vLLM サーバー経由で使用する。
T-2.2（画像検品）と同じサーバーインスタンスを共用できる。

### 2-1. 起動コマンド

```bash
# scripts/start_vllm_server.sh が自動生成されているので使用する
bash scripts/start_vllm_server.sh
```

手動で起動する場合:
```bash
MODEL="${HOME}/models/Qwen3-VL-32B-Instruct"
# モデルが未ダウンロードの場合は HuggingFace ID を直接指定
# MODEL="Qwen/Qwen3-VL-32B-Instruct"

vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8001 \
    --trust-remote-code
```

### 2-2. 起動確認

```bash
# ヘルスチェック（起動に 2〜3 分かかる）
curl http://localhost:8001/health
# "OK" が返ればサーバー起動済み

# モデル一覧確認
curl http://localhost:8001/v1/models | python -m json.tool
```

### 2-3. メモリ使用量

Qwen3-VL-32B-Instruct (BF16) は約 **65 GB** を使用。
T-3.3 実行中は TRELLIS.2 および FLUX.1 を**アンロードしておくこと**（逐次ロード戦略）。

---

## 3. 実行方法

### 3-1. Python API による直接実行

```python
from src.mesh_vlm_qa import MeshVLMQA

qa = MeshVLMQA(
    model_name="Qwen/Qwen3-VL-32B-Instruct",   # vLLM に渡すモデル名
    vllm_base_url="http://localhost:8001/v1",
)

# 単体メッシュ評価
render_paths = qa.render_multiview(
    "outputs/meshes_raw/000001.glb",
    output_dir="outputs/renders/000001",
    views=4,
)
result = qa.evaluate_3d(render_paths, expected_type="hard_suitcase")
print(result)
# {
#   "geometry_score": 8,
#   "texture_score": 7,
#   "consistency_score": 8,
#   "is_realistic_luggage": True,
#   "detected_type": "hard_suitcase",
#   "detected_material": "polycarbonate",
#   "issues": [],
#   "pass": True
# }

# バッチ評価（中断再開対応）
summary = qa.evaluate_batch(
    mesh_dir="outputs/meshes_raw",
    output_json="outputs/meshes_approved/vlm_qa_results.json",
    render_dir="outputs/renders",     # None にすると tmpdir 使用（ディスク節約）
    views=4,
    resume=True,                      # 既存結果をスキップして再開
)
print(f"合格: {summary['passed']}, 不合格: {summary['failed']}")
```

### 3-2. レンダリングのみ実行

```python
from src.utils.rendering import render_multiview

paths = render_multiview(
    mesh_path="outputs/meshes_raw/000001.glb",
    output_dir="outputs/renders/000001",
    views=4,
    width=512, height=512,
    elevation_deg=20.0,   # カメラ仰角（20° = 少し上から見下ろし）
    distance=2.0,         # バウンディングボックス最大辺に乗じる倍率
)
# → ["outputs/renders/000001/000001_0.png", ..., "000001_3.png"]
```

---

## 4. 出力形式

### 4-1. `evaluate_3d()` の返り値

```json
{
  "geometry_score":     8,
  "texture_score":      7,
  "consistency_score":  8,
  "is_realistic_luggage": true,
  "detected_type":      "hard_suitcase",
  "detected_material":  "polycarbonate",
  "issues":             [],
  "pass":               true
}
```

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `geometry_score` | int 1–10 | 3D 幾何品質（形状正確性・位相清潔さ） |
| `texture_score` | int 1–10 | テクスチャ品質（PBR リアリズム・UV 継ぎ目） |
| `consistency_score` | int 1–10 | 4 視点間の一貫性 |
| `is_realistic_luggage` | bool | 実際の荷物に見えるか |
| `detected_type` | str | 検出された荷物タイプ |
| `detected_material` | str | 推定主材質 |
| `issues` | list[str] | 具体的な問題点（なければ空リスト） |
| `pass` | bool | geometry≥7 かつ texture≥6 で true |

### 4-2. `evaluate_batch()` の JSON 出力

```json
{
  "total":   100,
  "passed":  72,
  "failed":  28,
  "results": [
    {
      "mesh_path": "/path/to/outputs/meshes_raw/000001.glb",
      "geometry_score": 8,
      "texture_score": 7,
      "pass": true,
      ...
    }
  ]
}
```

保存先: `output_json` 引数で指定したパス（例: `outputs/meshes_approved/vlm_qa_results.json`）

---

## 5. トラブルシューティング

### `EGL initialization error` / `cannot open display`

**原因**: `PYOPENGL_PLATFORM=osmesa` が未設定のまま pyrender がロードされた。

**対処**:
```bash
export PYOPENGL_PLATFORM=osmesa
# その後 Python プロセスを再起動して再実行
```

### `OSError: libGL.so.1: cannot open shared object file`

**原因**: OpenGL ライブラリ不足。

**対処**:
```bash
sudo apt-get install -y libgl1-mesa-glx libosmesa6
```

### `RuntimeError: メッシュロード失敗`

**原因**: GLB ファイルの破損または非対応フォーマット。T-3.2（MeshQA）でチェック済みの
メッシュを入力すること。

**確認方法**:
```python
import trimesh
mesh = trimesh.load("outputs/meshes_raw/000001.glb", force="scene")
print(mesh)
```

### `ConnectionRefusedError: http://localhost:8001`

**原因**: vLLM サーバーが起動していない。

**対処**:
```bash
bash scripts/start_vllm_server.sh &
# 起動完了まで 2〜3 分待機
curl http://localhost:8001/health
```

### VLM が有効な JSON を返さない / スコアが常に低い

**原因 1**: Thinking モード (`/think`) のトークン出力がタイムアウト。
`--max-model-len` を増やすか、`max_tokens` を増やして再試行。

**原因 2**: モデルが温まっていない（最初の数リクエストは遅い）。
数件で試してから本番実行する。

**確認方法**:
```bash
# vLLM ログを確認
bash scripts/start_vllm_server.sh 2>&1 | tee vllm.log
```

### メモリ不足 (`CUDA out of memory`)

**原因**: vLLM（Qwen3-VL-32B, ~65GB）と TRELLIS.2（~24GB）が同時にロードされている。

**対処**: TRELLIS.2 の `MeshGenerator.unload()` を呼んでから vLLM サーバーを起動する。
パイプライン全体での逐次ロード戦略を守ること（`CLAUDE.md` 参照）。

---

## 6. テスト

```bash
cd al3dg
source .venv/bin/activate

# T-3.3 ユニットテスト（vLLM 不要、モック使用）
python -m pytest tests/test_mesh_vlm_qa.py -v

# T-3.2 + T-3.3 まとめて実行
python -m pytest tests/test_mesh_qa.py tests/test_mesh_vlm_qa.py -v

# 全ユニットテスト（GPU テストを除く）
python -m pytest tests/ --ignore=tests/test_gpu_models.py -v
```

全テストはモックを使用するため、vLLM サーバーや GPU がなくても実行可能。

---

## 7. パイプライン内での位置づけ

```
QA-2a: メッシュQA (T-3.2)
    src/mesh_qa.py — Open3D + trimesh
    watertight / manifold / face count (5K–100K) / degenerate / aspect ratio

            ↓ 合格 (pass=True) のみ通過

QA-2b: VLM 3D 検品 (T-3.3)   ← ここ
    src/mesh_vlm_qa.py
    pyrender で 4 方向レンダリング
    → Qwen3-VL-32B /think で評価
    → geometry≥7 AND texture≥6 で合格

            ↓ 合格のみ
    outputs/meshes_approved/  （T-4.1 へ）
```

T-3.2 通過後の GLB を入力にすること。T-3.2 未通過のメッシュは
ルールベースで既に品質不足と判定されているため VLM コストを無駄にしない。

---

## 8. 実装ファイル一覧

| ファイル | 役割 |
|----------|------|
| [src/mesh_vlm_qa.py](../src/mesh_vlm_qa.py) | MeshVLMQA クラス本体 |
| [src/utils/rendering.py](../src/utils/rendering.py) | pyrender オフスクリーンレンダリング |
| [tests/test_mesh_vlm_qa.py](../tests/test_mesh_vlm_qa.py) | ユニットテスト（23 件） |
| [scripts/start_vllm_server.sh](../scripts/start_vllm_server.sh) | vLLM サーバー起動スクリプト |
| [configs/material_properties.yaml](../configs/material_properties.yaml) | 材質プロパティ定義 |
