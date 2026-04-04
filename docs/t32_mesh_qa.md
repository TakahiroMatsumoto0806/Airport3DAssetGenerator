# T-3.2: メッシュ品質チェック — セットアップと実行手順

対象環境: **DGX Spark (Grace Blackwell GB10, Ubuntu, 128GB 統合メモリ)**

---

## 概要

TRELLIS.2 で生成した GLB メッシュを Open3D + trimesh でルールベース品質チェックし、
修復可能なものは自動修復して後段（T-3.3 VLM 検品 → T-4.1 物理付与）に流す。

| チェック項目 | 基準 |
|-------------|------|
| face count | 5,000 ≤ faces ≤ 100,000 |
| watertight | 穴なし（閉じたメッシュ） |
| edge manifold | 各エッジを共有する面が 2 面以下 |
| vertex manifold | 各頂点が単一の連結面集合を持つ |
| degenerate faces | 縮退面 = 0 |
| normal consistency | ウィンディング方向が統一されている |
| aspect ratio | 最長辺 / 最短辺 ≤ 20 |

---

## 1. 前提条件

### 1-1. 依存パッケージ

`setup_environment.sh` を実行済みであれば `pyproject.toml` の依存に含まれる。

```bash
source .venv/bin/activate

# 手動インストールが必要な場合
uv pip install trimesh open3d networkx
```

確認:
```bash
python -c "import trimesh, open3d; print('trimesh', trimesh.__version__, '/ open3d', open3d.__version__)"
```

> **注意**: Open3D は ARM64 (aarch64) 向けのバイナリが提供されている。
> `uv pip install open3d` で自動的に正しいホイールがインストールされる。
> インストールに数分かかる場合がある。

---

## 2. 実行方法

### 2-1. Python API による直接実行

```python
from src.mesh_qa import MeshQA

qa = MeshQA()

# 単体チェック
result = qa.check_single("outputs/meshes_raw/000001.glb")
print(result)
# {
#   "face_count": 52480,
#   "face_count_ok": True,
#   "is_watertight": True,
#   "is_edge_manifold": True,
#   "is_vertex_manifold": True,
#   "degenerate_count": 0,
#   "is_normal_consistent": True,
#   "aspect_ratio": 1.8,
#   "aspect_ratio_ok": True,
#   "pass": True,
#   "issues": []
# }

# 修復
repair_result = qa.repair(
    "outputs/meshes_raw/000001.glb",
    "outputs/meshes_approved/000001.glb",
)
print(f"修復成功: {repair_result['repaired']}")

# バッチチェック（中断再開なし、全件処理）
summary = qa.check_batch(
    mesh_dir="outputs/meshes_raw",
    output_json="outputs/meshes_approved/mesh_qa_results.json",
    approved_dir="outputs/meshes_approved",   # 合格・修復済みをコピー
    attempt_repair=True,                      # 不合格時に修復を試みる
)
print(f"合格: {summary['passed']}, 修復: {summary['repaired']}, 不合格: {summary['failed']}")
```

### 2-2. コマンドラインから（スクリプト化する場合）

```bash
source .venv/bin/activate
python -c "
from src.mesh_qa import MeshQA
qa = MeshQA()
s = qa.check_batch(
    'outputs/meshes_raw',
    'outputs/meshes_approved/mesh_qa_results.json',
    approved_dir='outputs/meshes_approved',
)
print(s)
"
```

---

## 3. 出力形式

### 3-1. `check_single()` の返り値

```json
{
  "mesh_path": "/path/to/mesh.glb",
  "is_watertight": true,
  "is_edge_manifold": true,
  "is_vertex_manifold": true,
  "has_self_intersection": false,
  "face_count": 52480,
  "face_count_ok": true,
  "degenerate_count": 0,
  "is_normal_consistent": true,
  "aspect_ratio": 1.82,
  "aspect_ratio_ok": true,
  "pass": true,
  "issues": [],
  "timestamp": "2026-04-04T12:00:00+00:00"
}
```

### 3-2. `check_batch()` の JSON 出力

```json
{
  "total":   100,
  "passed":  68,
  "repaired": 12,
  "failed":   20,
  "results": [...]
}
```

各エントリに `"status": "passed" | "repaired" | "failed"` が付与される。

---

## 4. 修復処理の詳細

`repair()` は以下の順序で修復を試みる:

1. **degenerate faces 除去** — `mesh.nondegenerate_faces()` でマスク → `update_faces()`
2. **重複頂点・面の除去** — `merge_vertices()` + `unique_faces()` + `remove_unreferenced_vertices()`
3. **穴埋め (hole filling)** — `trimesh.repair.fill_holes()`
4. **法線修正** — `trimesh.repair.fix_winding()` + `fix_normals()`

修復後に再度 `check_single()` を実行し、`pass=True` になれば `repaired` とみなす。
修復後も `pass=False` の場合は `failed` 扱いとなり、後段には進まない。

> **制限**: `fill_holes()` は小さい穴（単純なポリゴンで閉じられる穴）のみ対応。
> 大きく開いたメッシュや多数の穴があるメッシュは修復不能で `failed` となる。

---

## 5. トラブルシューティング

### `ModuleNotFoundError: No module named 'networkx'`

trimesh の穴埋め処理（`fill_holes`）に networkx が必要。

```bash
source .venv/bin/activate
uv pip install networkx
```

### `Open3D: メッシュが空です`

GLB に複数マテリアルが含まれる場合、Open3D が読み込めないことがある。
trimesh 側のチェック結果（watertight, degenerate 等）は依然として有効。
`is_edge_manifold` / `is_vertex_manifold` のみ `False` として記録され、
issues に `"Open3D マニフォールド判定失敗"` が追加される。

### face_count が 5K 未満 / 100K 超

TRELLIS.2 の出力が稀に極端な face count になることがある。

- **5K 未満**: `simplify_ratio` が高すぎる（デフォルト 0.95 で OK のはず）
- **100K 超**: `decimation_target` が大きすぎる。`MeshGenerator._export_glb()` の
  `_DEFAULT_DECIMATION_TARGET` 定数を確認する

---

## 6. テスト

```bash
cd al3dg
source .venv/bin/activate
python -m pytest tests/test_mesh_qa.py -v
# 22 tests → passed
```

---

## 7. パイプライン内での位置づけ

```
Step 3: 3D 生成 (T-3.1)
    outputs/meshes_raw/*.glb

        ↓

QA-2a: メッシュ品質チェック (T-3.2)  ← ここ
    src/mesh_qa.py
    check_batch() → 合格・修復済みを outputs/meshes_approved/ にコピー
    mesh_qa_results.json に provenance 保存

        ↓ pass OR repaired のみ

QA-2b: VLM 3D 検品 (T-3.3)
    outputs/meshes_approved/*.glb を入力
```

---

## 8. 実装ファイル一覧

| ファイル | 役割 |
|----------|------|
| [src/mesh_qa.py](../src/mesh_qa.py) | MeshQA クラス本体 |
| [src/utils/mesh_repair.py](../src/utils/mesh_repair.py) | 低レベル修復ユーティリティ |
| [tests/test_mesh_qa.py](../tests/test_mesh_qa.py) | ユニットテスト（22 件） |
