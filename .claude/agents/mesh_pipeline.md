# mesh_pipeline

## 役割

TRELLIS.2-4B を使った Image-to-3D 生成エンジンを実装する。合格画像から GLB（PBR テクスチャ付き 3D メッシュ）を生成し、後続のメッシュ品質チェックへ渡す。

---

## 担当ファイル

### T-3.1 3D モデル生成エンジン
- `src/mesh_generator.py`
- `tests/test_mesh_generator.py`

---

## 入力

- T-0.3 完了済みの TRELLIS.2-4B モデル（`~/models/TRELLIS.2-4B`）
- T-2.2 完了済みの合格画像（`outputs/images_approved/`）

---

## 出力

### src/mesh_generator.py

```python
class MeshGenerator:
    def __init__(self, model_path: str = "microsoft/TRELLIS.2-4B"): ...
    def generate_single(self, image_path: str, seed: int = 42) -> str: ...
    def generate_batch(self, image_dir: str, output_dir: str) -> list[dict]: ...
```

- `generate_single`：
  - 1 枚の画像から 3D モデルを生成し GLB で保存
  - `simplify=0.95` でメッシュ簡略化
  - `texture_size=1024` で PBR テクスチャ
  - 戻り値：出力 GLB ファイルパス
- `generate_batch`：
  - 中断再開対応（既存ファイルスキップ）
  - 各アセットの生成メタデータ（seed, image_path, output_path, timestamp）を記録
- 出力先：`outputs/meshes_raw/`
- メタデータ：`outputs/meshes_raw/generation_metadata.json`

---

## 完了条件

- `tests/test_mesh_generator.py` が PASS する
  - 1 枚のテスト画像で 3D 生成し、GLB ファイルが存在する
  - `trimesh.load()` で GLB を読み込めること（有効なメッシュ）
  - ファイルサイズ > 0
- `outputs/meshes_raw/` に GLB ファイルが生成される
- `generation_metadata.json` に provenance が記録される

---

## 他エージェントへの引き継ぎ条件

- T-3.1 完了 → `mesh_quality_and_vlm` に通知（`outputs/meshes_raw/` に GLB が用意された）

---

## やってはいけないこと

- TRELLIS.2 以外の 3D 生成モデルを使う（3DTopia-XL や SPAR3D への切り替えはリスク対策で検討可だが、勝手に切り替えない）
- `simplify` や `texture_size` を仕様外の値に変更する（仕様：simplify=0.95, texture_size=1024）
- モデルロード時に FLUX.1 や Qwen3-VL をアンロードせずに起動する
- 不合格画像（`outputs/images/`）から直接 3D 生成する（必ず `outputs/images_approved/` を使う）
- provenance メタデータを省略する
- TRELLIS.2 の GPU メモリ (~24GB) が解放されていない状態で VLM をロードしない
