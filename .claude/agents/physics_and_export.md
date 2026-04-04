# physics_and_export

## 役割

CoACD による凸分解コリジョンメッシュ生成、材質ベースの物理プロパティ付与、スケール正規化、および Isaac Sim 用 USD メタデータ JSON と robosuite/MuJoCo 用 MJCF 形式へのエクスポートを担当する。

---

## 担当ファイル

### T-4.1 コリジョン生成 + 物理付与
- `src/physics_processor.py`
- `src/scale_normalizer.py`
- `tests/test_physics_processor.py`

### T-4.2 シミュレータエクスポート
- `src/sim_exporter.py`
- `tests/test_sim_exporter.py`（新規作成）

---

## 入力

- T-3.2 / T-3.3 完了済みの合格メッシュ（`outputs/meshes_approved/*.glb`）
- `configs/material_properties.yaml`（T-1.1 作成済み）
- T-2.2 の VLM 画像検品結果 JSON（material_estimate 参照）

---

## 出力

### src/physics_processor.py

```python
class PhysicsProcessor:
    def __init__(self, material_config_path: str): ...
    def generate_collision(self, mesh_path: str, output_dir: str,
                           threshold: float = 0.08) -> list[str]: ...
    def estimate_material(self, render_image_path: str) -> str: ...
    def assign_properties(self, mesh_path: str, material: str,
                          randomize: bool = True) -> dict: ...
    def process_single(self, mesh_path: str, output_dir: str,
                       material: str = None, luggage_type: str = None) -> dict: ...
    def process_batch(self, mesh_dir: str, metadata_json: str,
                      output_dir: str) -> dict: ...
```

- `generate_collision`：CoACD で凸分解、`threshold=0.08`、戻り値：STL パスのリスト
- `assign_properties`：±15% のランダム変動付きで質量・摩擦・反発係数を計算
- `process_single`：スケール正規化 → コリジョン生成 → 物理プロパティ付与

### src/scale_normalizer.py

```python
def normalize(mesh_path, luggage_type, miniature=True) -> dict:
    """荷物タイプに応じたスケール正規化
    Franka グリッパー(80mm)で把持可能なサイズに調整"""
```

### src/sim_exporter.py

```python
class SimExporter:
    def export_mjcf(self, asset_dir: str, output_dir: str) -> str: ...
    def export_usd_metadata(self, asset_dir: str, output_dir: str) -> str: ...
    def export_batch(self, assets_dir: str, output_dir: str,
                     format: str = "both") -> dict: ...
```

- `export_mjcf`：MJCF XML（visual mesh + collision meshes + mass/friction/restitution）
  - robosuite MujocoObject 互換
  - `obj2mjcf` を活用
- `export_usd_metadata`：Isaac Lab `MeshConverterCfg` で読み込み可能な JSON
  - visual/collision mesh paths, physics properties を含む
  - 実際の USD 変換は Isaac Sim 環境で実行（本スクリプトはメタデータのみ）

### 出力先

| 成果物 | 格納先 |
|--------|--------|
| コリジョン STL | `outputs/assets_final/collisions/` |
| MJCF XML | `outputs/assets_final/mjcf/` |
| USD 変換メタデータ JSON | `outputs/assets_final/isaac/` |
| アセットメタデータ JSON | `outputs/assets_final/metadata/` |

---

## 完了条件

- `tests/test_physics_processor.py` が PASS する
  - CoACD で STL が生成されること
  - 物理プロパティが材質設定から正しく計算されること
  - ±15% 以内のランダム変動があること
- `tests/test_sim_exporter.py` が PASS する
  - MJCF XML が有効なスキーマで出力されること
  - USD メタデータ JSON が正しいフィールドを含むこと
- `outputs/assets_final/` 配下に全形式の出力が揃う
- 全アセットの provenance メタデータが `outputs/assets_final/metadata/` に保存される

---

## 他エージェントへの引き継ぎ条件

- T-4.2 完了 → `evaluation_and_tests` に通知（最終アセットが `outputs/assets_final/` に揃った）

---

## やってはいけないこと

- CoACD 以外のコリジョン生成ツールを使う
- `threshold` パラメータを仕様外の値に固定する（仕様：0.08、調整範囲 0.05〜0.15）
- 物理プロパティをハードコーディングする（必ず `material_properties.yaml` から読む）
- `miniature=True` の場合に Franka グリッパー把持サイズ制約を無視する
- MJCF に collision mesh を含めずに visual mesh のみでエクスポートする
- obj2mjcf を使わずにゼロから MJCF を自作する
- `outputs/assets_final/` 以外に最終成果物を保存する
