# AL3DG — Airport Luggage 3D Asset Generator

## プロジェクト目的

空港荷物コンテナ詰込ロボット向け Physical AI 学習データ用の 3D アセットを大量自動生成するパイプラインを DGX Spark 上に構築する。

目標：PBR テクスチャ付き 3D アセットを大量自動生成（Isaac Sim USD 形式、コリジョン・物理プロパティ付き）

---

## 正本仕様

**`plan/dgx_spark_construction_plan.html`** を唯一の正本仕様とする。

- 他の md / yaml / コードコメントと矛盾した場合は必ず `dgx_spark_construction_plan.html` を優先する
- モデル構成を勝手に変更しない
- 主構成は `FLUX.1-schnell + TRELLIS.2 + Qwen3-VL-32B + CLIP + CoACD` に固定

---

## 実装フェーズ順序

| Phase | タスクID | 内容 | 所要 |
|-------|---------|------|------|
| 0 | T-0.1, T-0.2, T-0.3 | 環境構築・モデルDL・GPU動作確認 | 0.5〜1日 |
| 1 | T-1.1, T-1.2 | 設定ファイル・プロンプト生成エンジン | 0.5日 |
| 2 | T-2.1, T-2.2 | 画像生成・VLM画像検品 | 1〜2日 |
| 3 | T-3.1, T-3.2, T-3.3 | 3D生成・メッシュQA・VLMマルチビュー検品 | 2〜3日 |
| 4 | T-4.1, T-4.2 | 物理プロパティ付与・シミュレータエクスポート | 1〜2日 |
| 5 | T-5.1 | 多様性評価レポート | 1日 |
| 6 | T-6.1, T-6.2 | パイプライン統合・CLI | 1〜2日 |

実装順序：T-0.1 → T-0.2 → T-0.3 → T-1.1 → T-1.2 → T-2.1 → T-2.2 → T-3.1 → T-3.2 → T-3.3 → T-4.1 → T-4.2 → T-5.1 → T-6.1 → T-6.2

---

## 技術スタック

| カテゴリ | ツール | 用途 |
|---------|-------|------|
| 言語 | Python 3.11 | 実装言語 |
| パッケージ管理 | uv | 依存解決 |
| 画像生成 | diffusers + FLUX.1-schnell | Text-to-Image |
| 3D生成 | TRELLIS.2-4B (Microsoft) | Image-to-3D |
| VLM/LLM 統一 | vLLM + Qwen3-VL-32B | プロンプト生成・画像検品・3D検品（全タスク共通） |
| メッシュ処理 | trimesh, Open3D | 品質チェック・修復 |
| 凸分解 | CoACD | コリジョンメッシュ生成 |
| 多様性評価 | open_clip (ViT-L/14) | CLIP埋め込み・Vendi Score |
| レンダリング | pyrender / trimesh | オフスクリーンレンダリング |
| 設定管理 | Hydra / OmegaConf | YAML設定 |
| ログ | loguru | 構造化ログ |

**採用しないモデル**：Hunyuan 系は一切使用しない。

---

## DGX Spark 制約

- ハードウェア：Grace Blackwell GB10、128GB 統合メモリ、Ubuntu Linux
- **Step 3（3D 生成）は本プロジェクトの対象範囲外**：別 PC（x86_64 + RTX 5090）で別プロジェクトとして実施する。
  本リポジトリでは 3D 生成の手順・フローを記述しない（古い記述は混乱を招くため削除ポリシー）。
  生成された GLB は `outputs/meshes_raw/` に配置された状態から QA 以降を実行する。
- モデル同時ロード禁止：逐次ロード戦略を厳守
  - Step1（VLM）→ Step2（FLUX.1 ロード、VLM アンロード）→ QA-1（VLM 再ロード、FLUX.1 アンロード）→ QA-2（VLM で 3D メッシュ検品）
  - モデル切り替え時は `del model; torch.cuda.empty_cache()` を必ず実行
- ピーク GPU メモリ 128GB を超えないこと（Qwen3-VL-32B 実使用 ≈ 100GB、FLUX.1 ≈ 12GB）
- Qwen3-VL-32B は vLLM サーバーとして起動：`vllm serve Qwen/Qwen3-VL-32B --dtype bfloat16`
- モデルダウンロード先：`~/models/`

---

## テスト方針

- 各タスク完了ごとに対応するテストを実行し、通過を確認してから次のタスクへ進む
- テストが失敗したまま次フェーズへ進まない
- テストファイルは `tests/` 配下に配置
- 最低限のテスト要件は各タスクの仕様（HTML）に記載されている内容に従う

---

## 受け入れ条件

| 項目 | 基準 |
|------|------|
| 生成アセット数 | prompt_generate_number に応じた任意の数 |
| メッシュ品質 | watertight、manifold、face count 5K〜100K |
| 3D検品 | geometry ≥ 7、texture ≥ 6 |
| 多様性 | Vendi Score および近似重複検出でレポート確認 |
| エクスポート形式 | Isaac Sim USD 変換メタデータ JSON |
| provenance | 全アセットに pass/review/reject を JSON で記録 |

---

## ディレクトリ構成

```
al3dg/
├── README.md
├── pyproject.toml
├── configs/
│   ├── pipeline_config.yaml
│   ├── prompt_templates.yaml
│   ├── material_properties.yaml
│   └── luggage_categories.yaml
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── prompt_generator.py
│   ├── image_generator.py
│   ├── image_qa.py
│   ├── mesh_generator.py
│   ├── mesh_qa.py
│   ├── mesh_vlm_qa.py
│   ├── physics_processor.py
│   ├── scale_normalizer.py
│   ├── diversity_evaluator.py
│   ├── sim_exporter.py
│   └── utils/
│       ├── rendering.py
│       ├── mesh_repair.py
│       └── logging_utils.py
├── scripts/
│   ├── setup_environment.sh
│   ├── run_pipeline.py
│   ├── run_step.py
│   └── generate_report.py
├── tests/
│   ├── test_prompt_generator.py
│   ├── test_image_generator.py
│   ├── test_mesh_generator.py
│   ├── test_mesh_qa.py
│   └── test_physics_processor.py
├── outputs/
│   ├── prompts/
│   ├── images/
│   ├── images_approved/
│   ├── meshes_raw/
│   ├── meshes_approved/
│   ├── assets_final/
│   │   ├── isaac/
│   │   ├── collisions/
│   │   └── metadata/
│   └── reports/
└── docker/
    └── Dockerfile
```

---

## サブエージェント役割一覧

| エージェント | タスクID | 担当内容 |
|------------|---------|---------|
| `repo_orchestrator` | 全体 | 全体計画・依存関係管理・フェーズ進行・レビュー |
| `environment_and_models` | T-0.1, T-0.2, T-0.3 | 環境構築・モデルDL・GPU動作確認 |
| `prompt_and_config` | T-1.1, T-1.2 | 設定ファイル・プロンプト生成エンジン |
| `image_pipeline` | T-2.1, T-2.2 | 画像生成・VLM画像検品 |
| `mesh_pipeline` | T-3.1 | TRELLIS.2 3D生成エンジン |
| `mesh_quality_and_vlm` | T-3.2, T-3.3 | メッシュQA・VLMマルチビュー検品 |
| `physics_and_export` | T-4.1, T-4.2 | 物理プロパティ付与・エクスポート |
| `evaluation_and_tests` | T-5.1, T-6.1, T-6.2 + 各種テスト | 多様性評価・パイプライン統合・CLI |

---

## 重要ルール

1. HTML 仕様書に記載のモデル名・パラメータ・閾値を変更しない
2. VLM は全タスクで Qwen3-VL-32B に統一（8B は使用しない）
3. Qwen3-VL-32B のThinkingモード：QA-1（画像検品）は `/no_think`、QA-2（3D検品）は `/think`
4. 生成物の provenance（pass/review/reject）を必ず JSON で保存
5. 出力先は HTML 記載の `outputs/` 配下に従う
6. 中断再開対応（既存ファイルスキップ）を全バッチ処理に実装する
