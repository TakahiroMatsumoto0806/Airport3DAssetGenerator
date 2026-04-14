# T-3.2: メッシュ品質チェック — セットアップと実行手順

対象環境: **DGX Spark (Grace Blackwell GB10, Ubuntu, 128GB 統合メモリ)**

---

## 概要

TRELLIS.2 で生成した GLB メッシュを Open3D + trimesh でルールベース品質チェックし、
修復可能なものは自動修復して後段（T-3.3 VLM 検品 → T-4.1 物理付与）に流す。

> **前提**: T-3.1 (TRELLIS.2-4B 3D生成) は x86_64 + RTX 5090 の別PCで実行される。
> 生成した GLB は DGX Spark の `outputs/meshes_raw/` に配置済みであることを前提とする。
> ファイルが揃っていれば T-3.2 はそのまま実行可能。

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

1. **degenerate faces 除去** — `mesh.update_faces(mesh.nondegenerate_faces())`
2. **重複頂点・面の除去** — `merge_vertices()` + `unique_faces()` + `remove_unreferenced_vertices()`
3. **法線修正** — `trimesh.repair.fix_normals()`

修復後に再度 `check_single()` を実行し、`pass=True` になれば `repaired` とみなす。
修復後も `pass=False` の場合は `failed` 扱いとなり、後段には進まない。

> **注意**: `fill_holes()` は TRELLIS メッシュでは逆効果（non-manifold エッジを生成）のため
> repair フローに含めていない（2026-04-05 調査結果）。

---

## 5. トラブルシューティング

### [2026-04-05 調査] TRELLIS 出力は全件 "非 watertight" になる

**現象**: TRELLIS で生成した 161 件の GLB を `check_single()` にかけると、
161/161 件が `is_watertight=False` となり、元の pass 条件では全件 `failed` になった。

**原因**: TRELLIS は Sparse SDF → マーチングキューブスでメッシュを生成するため、
境界面（open boundary）が大量に残る。これは生成手法の構造的特性であり避けられない。

**対処**: `pass` 判定条件から `is_watertight` を除外した（下記「pass 条件の変更」参照）。
watertight はメトリクスとして記録し続けるが、合否には影響しない。
コリジョンメッシュは T-4.1 CoACD で別途生成するため、視覚メッシュの閉性は不要。

---

### [2026-04-05 調査] Open3D の `is_edge_manifold()` が aarch64 で Segfault

**現象**: `o3d_mesh.is_edge_manifold(allow_boundary_edges=False)` を呼ぶと
Python インタプリタが Segfault (exit code 139) でクラッシュする。

**環境**: Open3D 0.18.0 / aarch64 (DGX Spark GB10)

**対処**: Open3D によるマニフォールド判定を廃止し、trimesh の `edges_sorted` を使った
エッジカウントで代替した。

```python
from collections import Counter
edge_face_counts = Counter(map(tuple, tm.edges_sorted))
# edge_face_counts[e] > 2 → 非マニフォールドエッジ
non_manifold_edges = sum(1 for v in edge_face_counts.values() if v > 2)
boundary_edges     = sum(1 for v in edge_face_counts.values() if v == 1)
```

---

### [2026-04-05 調査] `trimesh.repair.fill_holes()` がトポロジーを悪化させる

**現象**: `repair()` 内で `fill_holes()` を呼ぶと、非マニフォールドエッジが
0 → 591 件に急増し、修復前より pass 条件を悪化させた。

**原因**: `fill_holes()` は単純なポリゴンでホールを埋めるが、TRELLIS メッシュのように
オープンバウンダリが多い場合、内部でポリゴンが重なり non-manifold エッジを生成する。

**対処**: `fill_holes()` を repair フローから完全に除外した。
現在の repair 手順は以下のみ:
1. `update_faces(nondegenerate_faces())` — 縮退面除去
2. `merge_vertices()` + `unique_faces()` + `remove_unreferenced_vertices()` — 重複除去
3. `fix_normals()` — 法線修正

---

### pass 条件の変更（2026-04-05 確定版）

| 条件 | 変更前 | 変更後 | 理由 |
|------|--------|--------|------|
| is_watertight | **必須** | 参考値のみ | TRELLIS 全出力が非 watertight |
| is_edge_manifold | 必須 | 必須 (trimesh で代替) | Open3D segfault 回避 |
| is_vertex_manifold | **必須** | 参考値のみ | open boundary があれば常に False |
| degenerate_count == 0 | 必須 | 必須 | 修復で対処可能 |
| is_normal_consistent | 必須 | 必須 | 変更なし |
| aspect_ratio_ok | 必須 | 必須 | 変更なし |
| face_count_ok | 必須 | 必須 | 変更なし |

---

### [2026-04-05] 実測結果（161 メッシュ）

| 状態 | 件数 | 割合 |
|------|------|------|
| pass（修復なし合格） | 45 | 28% |
| repaired（修復後合格） | 100 | 62% |
| failed（修復不能） | 16 | 10% |
| **通過率合計** | **145** | **90.1%** |

failed 16 件の主要原因: non-manifold エッジが修復後も残存（主に edge_manifold=False）

---

### `ModuleNotFoundError: No module named 'networkx'`

`fill_holes()` を使っていた頃は networkx が必要だったが、現在は不要。

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
| [tests/test_mesh_qa.py](../tests/test_mesh_qa.py) | ユニットテスト（22 件） |
