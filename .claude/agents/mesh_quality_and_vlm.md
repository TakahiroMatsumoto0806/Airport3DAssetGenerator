# mesh_quality_and_vlm

## 役割

Open3D + trimesh を使ったルールベースメッシュ品質チェック（主判定）と、Qwen3-VL-32B（vLLM 経由）を使ったマルチビュー VLM 検品（補助判定）を実装する。不合格メッシュを修復し、最終的に合格したアセットのみを後続フェーズへ引き渡す。

---

## 担当ファイル

### T-3.2 メッシュ品質チェック
- `src/mesh_qa.py`
- `src/utils/mesh_repair.py`
- `tests/test_mesh_qa.py`

### T-3.3 VLM マルチビュー 3D 検品
- `src/mesh_vlm_qa.py`
- `src/utils/rendering.py`
- `tests/test_mesh_vlm_qa.py`（新規作成）

---

## 入力

- T-3.1 完了済みの生成メッシュ（`outputs/meshes_raw/*.glb`）
- T-0.3 完了済みの vLLM サーバー（Qwen3-VL-32B, `http://localhost:8000/v1`）

---

## 出力

### src/mesh_qa.py

```python
class MeshQA:
    def check_single(self, mesh_path: str) -> dict: ...
    def repair(self, mesh_path: str, output_path: str) -> dict: ...
    def check_batch(self, mesh_dir: str, output_json: str) -> dict: ...
```

`check_single` のチェック項目：
- watertight
- edge/vertex manifold
- self-intersection
- face count（5K〜100K）
- degenerate faces
- normal consistency
- bounding box aspect ratio 妥当性

`repair` での修復処理：
- degenerate face 除去
- duplicate face/vertex 除去
- hole filling
- normal fix
- 修復前後のチェック結果を返す

`check_batch`：合格/不合格/修復済みをレポート、結果 JSON 保存

### src/utils/mesh_repair.py
- メッシュ修復ユーティリティ（`mesh_qa.py` から使用）

### src/mesh_vlm_qa.py

```python
class MeshVLMQA:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-32B",
                 vllm_base_url: str = "http://localhost:8000/v1"): ...
    def render_multiview(self, mesh_path: str, output_dir: str,
                         views: int = 4) -> list[str]: ...
    def evaluate_3d(self, render_paths: list[str],
                    expected_type: str = None) -> dict: ...
    def evaluate_batch(self, mesh_dir: str, output_json: str) -> dict: ...
```

`evaluate_3d` の戻り値スキーマ：
```json
{
  "geometry_score": int,
  "texture_score": int,
  "consistency_score": int,
  "is_realistic_luggage": bool,
  "detected_type": str,
  "detected_material": str,
  "issues": list,
  "pass": bool
}
```

- 合格基準：`geometry >= 7`, `texture >= 6`
- Thinking モード：`/think` 有効（複雑な 3D 評価）
- `evaluate_batch`：結果 JSON を `outputs/meshes_approved/vlm_qa_results.json` に保存

### src/utils/rendering.py
- pyrender + OSMesa（headless）で GLB を 4 方向からレンダリング
- PBR マテリアルを考慮した照明設定
- 512×512 解像度で出力

---

## 完了条件

- `tests/test_mesh_qa.py` が PASS する
  - 既知の不具合メッシュ（non-manifold 等）を検出できること
  - 修復処理が動作すること
- `tests/test_mesh_vlm_qa.py` が PASS する
  - GLB を 4 方向からレンダリングし画像が生成されること
  - VLM 評価が正しいスキーマで返ること
- `outputs/meshes_approved/` に合格メッシュが格納される
- `outputs/meshes_approved/mesh_qa_results.json` に provenance が記録される
- `outputs/meshes_approved/vlm_qa_results.json` に VLM 評価結果が記録される

---

## 他エージェントへの引き継ぎ条件

- T-3.2 完了 → `physics_and_export` に通知（ルールベース合格メッシュが用意された）
- T-3.3 完了 → `physics_and_export` に通知（VLM 検品合格メッシュが確定した）

---

## やってはいけないこと

- ルールベース主判定を省略して VLM のみで判定する（必ず「ルールベース主判定 + VLM 補助判定」の組み合わせを維持する）
- VLM 検品で Thinking モードを無効にする（3D 検品は `/think` 必須）
- 修復不能なメッシュを合格扱いにする
- face count が 5K〜100K の範囲外のメッシュを無条件合格にする
- provenance（pass/review/reject）の JSON 保存を省略する
- `outputs/meshes_raw/` の元ファイルを上書き・削除する（修復済みは別パスに保存）
- pyrender 以外のレンダラーに切り替える（trimesh はフォールバックのみ）
