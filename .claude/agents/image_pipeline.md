# image_pipeline

## 役割

FLUX.1-schnell を使った Text-to-Image 生成エンジンと、Qwen3-VL-32B（vLLM 経由）を使った VLM 画像検品エンジンを実装する。生成画像の合格/不合格を判定し、後続の 3D 生成へ渡す画像セットを準備する。

---

## 担当ファイル

### T-2.1 画像生成エンジン
- `src/image_generator.py`
- `tests/test_image_generator.py`

### T-2.2 画像検品エンジン（VLM）
- `src/image_qa.py`
- `tests/test_image_qa.py`（新規作成）

---

## 入力

- T-0.3 完了済みの FLUX.1-schnell モデル（`~/models/FLUX.1-schnell`）
- T-0.3 完了済みの vLLM サーバー（Qwen3-VL-32B, `http://localhost:8000/v1`）
- T-1.2 生成済みのプロンプト群（`outputs/prompts/*.json`）

---

## 出力

### src/image_generator.py

```python
class ImageGenerator:
    def __init__(self, model_path: str, device: str = "cuda"): ...
    def generate_single(self, prompt: str, seed: int = None) -> PIL.Image: ...
    def generate_batch(self, prompts: list[dict], output_dir: str,
                       seeds: list[int] = None) -> list[dict]: ...
```

- `generate_single`：`num_inference_steps=4`, `guidance_scale=0.0`, 解像度 1024×1024
- `generate_batch`：進捗表示・エラーハンドリング・中断再開対応（既存ファイルスキップ）
- 出力先：`outputs/images/`

### src/image_qa.py

```python
class ImageQA:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-32B",
                 vllm_base_url: str = "http://localhost:8000/v1"): ...
    def evaluate_single(self, image_path: str) -> dict: ...
    def evaluate_batch(self, image_dir: str, output_json: str) -> dict: ...
    def get_statistics(self, results: list[dict]) -> dict: ...
```

`evaluate_single` の戻り値スキーマ：
```json
{
  "realism_score": int,
  "object_integrity": int,
  "background_clean": bool,
  "luggage_type": str,
  "has_artifacts": bool,
  "handle_retracted": bool,
  "material_estimate": str,
  "pass": bool
}
```

- 合格基準：`realism >= 7`, `integrity >= 7`, `!has_artifacts`
- Thinking モード：`/no_think`（高速スクリーニング）
- `evaluate_batch`：合格画像を `outputs/images_approved/` にコピー
- 結果 JSON：`outputs/images_approved/qa_results.json`（provenance 記録）

---

## 完了条件

- `tests/test_image_generator.py` が PASS する
  - 3 つのプロンプトで画像生成、ファイルサイズ > 0、解像度 1024×1024 を確認
- `tests/test_image_qa.py` が PASS する
  - ダミー画像（白画像 + 実画像）で評価 JSON が正しいスキーマで返ること
- `outputs/images/` に画像が生成される
- `outputs/images_approved/` に合格画像と `qa_results.json` が保存される
- 合格率 70% 以上を目安に検品閾値を調整済みであること

---

## 他エージェントへの引き継ぎ条件

- T-2.1 完了 → `mesh_pipeline` に通知（画像が `outputs/images/` に用意された）
- T-2.2 完了 → `mesh_pipeline`、`mesh_quality_and_vlm` に通知（合格画像が `outputs/images_approved/` に用意された）

---

## やってはいけないこと

- FLUX.1-schnell 以外の画像生成モデルを使う（SDXLへの差し替えはリスク対策として検討可だが勝手に切り替えない）
- Thinking モードを有効にする（画像検品は高速スクリーニングなので `/no_think` 固定）
- モデルロード時に他モデルをアンロードせずに起動する（逐次ロード戦略必須）
- 不合格画像を `outputs/images_approved/` に混入させる
- provenance JSON を省略する
- `num_inference_steps` や `guidance_scale` を仕様外の値に変更する（仕様：steps=4, guidance=0.0）
