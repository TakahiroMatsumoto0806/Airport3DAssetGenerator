# evaluation_and_tests

## 役割

OpenCLIP を使った多様性評価エンジンの実装（T-5.1）、メインパイプライン統合（T-6.1）、CLI スクリプト（T-6.2）を担当する。また、プロジェクト全体の横断テスト管理と最終レポート生成も行う。

---

## 担当ファイル

### T-5.1 多様性評価エンジン
- `src/diversity_evaluator.py`
- `tests/test_diversity_evaluator.py`（新規作成）

### T-6.1 メインパイプライン
- `src/pipeline.py`
- `configs/pipeline_config.yaml`

### T-6.2 実行スクリプト・CLI
- `scripts/run_pipeline.py`
- `scripts/run_step.py`
- `scripts/generate_report.py`

---

## 入力

- T-1.1 完了済みの設定ファイル群（`configs/`）
- T-4.2 完了済みの最終アセット（`outputs/assets_final/`）
- 各エージェントが生成した provenance JSON 群

---

## 出力

### src/diversity_evaluator.py

```python
class DiversityEvaluator:
    def __init__(self): ...  # OpenCLIP ViT-L/14 初期化
    def compute_clip_embeddings(self, image_paths: list[str]) -> np.ndarray: ...
    def compute_vendi_score(self, embeddings: np.ndarray) -> float: ...
    def find_near_duplicates(self, embeddings: np.ndarray,
                             threshold: float = 0.95) -> list[tuple]: ...
    def compute_size_diversity(self, mesh_info_list: list[dict]) -> dict: ...
    def compute_category_distribution(self, metadata_list: list[dict]) -> dict: ...
    def check_size_realism(self, mesh_info_list: list[dict],
                           category_refs: dict) -> dict: ...
    def generate_report(self, output_dir: str) -> str: ...
```

- `generate_report`：HTML 形式レポート
  - Vendi Score
  - 近似重複数（threshold=0.95）
  - カテゴリ分布チャート
  - サイズ分布ヒストグラム
  - t-SNE / UMAP 埋め込み可視化
- 出力先：`outputs/reports/diversity_report.html`

### src/pipeline.py

```python
class AssetGenerationPipeline:
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"): ...
    def run_full(self, n_assets: int = 1000): ...
    def run_step(self, step_name: str, **kwargs): ...
    def resume(self): ...
```

`run_full` の実行順序：
1. プロンプト生成（目標数 × 1.5 個、不合格分を見込む）
2. 画像生成
3. 画像検品 → 合格画像のみ続行
4. 3D 生成
5. メッシュ品質チェック → 修復 → 再チェック
6. VLM マルチビュー検品 → 合格のみ続行
7. 物理プロパティ付与
8. シミュレータ形式エクスポート
9. 多様性評価レポート生成

### configs/pipeline_config.yaml

```yaml
image_generator:
    model: "black-forest-labs/FLUX.1-schnell"
    resolution: [1024, 1024]
    num_inference_steps: 4
mesh_generator:
    model: "microsoft/TRELLIS.2-4B"
    texture_size: 1024
    simplify: 0.95
vlm:
    model: "Qwen/Qwen3-VL-32B"
    vllm_base_url: "http://localhost:8000/v1"
    dtype: "bfloat16"
    max_model_len: 8192
    thinking_mode: false
image_qa:
    model: "Qwen/Qwen3-VL-32B"
    thinking_mode: false       # /no_think — 高速スクリーニング
    min_realism: 7
    min_integrity: 7
prompt_refine:
    model: "Qwen/Qwen3-VL-32B"
    thinking_mode: false       # /no_think — 高速生成
mesh_vlm_qa:
    model: "Qwen/Qwen3-VL-32B"
    thinking_mode: true        # /think — 複雑な 3D 評価
    min_geometry: 7
    min_texture: 6
physics:
    collision_threshold: 0.08
    randomize_properties: true
    randomize_range: 0.15
export:
    formats: ["mjcf", "usd_meta"]
    miniature: true
```

### scripts/run_pipeline.py
- argparse CLI
  - `--config`：設定ファイルパス
  - `--n-assets`：目標アセット数
  - `--step`：特定ステップのみ実行
  - `--resume`：中断再開
  - `--dry-run`：各ステップの計画のみ表示

### scripts/generate_report.py
- 多様性レポートを単独生成
- `--assets-dir outputs/assets_final/`

---

## 完了条件

- `tests/test_diversity_evaluator.py` が PASS する
  - CLIP 埋め込みが取得できること
  - Vendi Score が計算されること
  - 近似重複検出が動作すること
- `scripts/run_pipeline.py` で `--dry-run` が正常に動作すること
- `scripts/run_pipeline.py --n-assets 100` で 100 アセットの E2E 生成が成功すること
- `outputs/reports/diversity_report.html` が生成されること
- 本番の大量生成が正常完了すること

---

## 他エージェントへの引き継ぎ条件

このエージェントはパイプライン末尾を担当するため、他エージェントへの引き継ぎは基本的に発生しない。

完了後は `repo_orchestrator` に最終完了報告を行い、`outputs/reports/pipeline_progress.json` を更新する。

---

## やってはいけないこと

- `pipeline_config.yaml` のモデル名を仕様外の値に変更する
- VLM の `thinking_mode` を仕様と異なる設定にする（画像QA: false、3D QA: true を維持）
- `min_realism`（7）や `min_geometry`（7）などの品質閾値を勝手に下げる
- Vendi Score の計算に CLIP 以外のモデルを使う（仕様：`laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`）
- 各ステップの完了状態チェックを省略して `resume` 機能を実装する
- `--dry-run` なしで本番実行を強制する
