"""
T-6.1: パイプライン統合

AL3DG の全ステップを統括するオーケストレーター。
DGX Spark の逐次ロード制約に従い、1 モデルずつロード→実行→アンロード。

ステップ順序:
  T-1.2  プロンプト生成
  T-2.1  画像生成 (FLUX.1-schnell)
  T-2.2  画像 QA   (Qwen3-VL-32B via vLLM)
  T-3.1  3D 生成   ※ 本プロジェクトの対象範囲外（別 PC で実施、GLB を受け取る）
  T-3.2  メッシュ QA (trimesh / open3d)
  T-3.3  VLM マルチビュー QA (Qwen3-VL-32B + pyrender)
  T-4.1  物理プロパティ付与 (CoACD)
  T-4.2  シミュレータエクスポート (Isaac Sim USD)

使用例:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/pipeline_config.yaml")
    pipeline = AL3DGPipeline(cfg)
    pipeline.run()
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger
from omegaconf import DictConfig, OmegaConf


class AL3DGPipeline:
    """
    AL3DG フルパイプライン オーケストレーター

    Attributes:
        cfg: OmegaConf DictConfig（pipeline_config.yaml）
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._output_dir = Path(cfg.paths.get("reports_dir", "outputs/reports"))

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def run(
        self,
        steps: Optional[list[str]] = None,
        resume: bool = True,
    ) -> dict:
        """
        パイプラインを実行する。

        Args:
            steps:  実行するステップ名のリスト。
                    None の場合は cfg.steps に従う。
                    例: ["prompt", "image", "image_qa"]
            resume: True の場合、各ステップで既存出力をスキップ。

        Returns:
            dict: 各ステップの結果サマリー
        """
        enabled = self._resolve_steps(steps)
        logger.info(f"AL3DG パイプライン開始: 有効ステップ={enabled}")

        results: dict[str, dict] = {}

        if "prompt" in enabled:
            results["prompt"] = self.run_prompt_generation()

        if "image" in enabled:
            results["image"] = self.run_image_generation(resume=resume)

        if "image_qa" in enabled:
            results["image_qa"] = self.run_image_qa(resume=resume)

        if "mesh" in enabled:
            results["mesh"] = self.run_mesh_generation(resume=resume)

        if "mesh_qa" in enabled:
            results["mesh_qa"] = self.run_mesh_qa(resume=resume)

        if "mesh_vlm_qa" in enabled:
            results["mesh_vlm_qa"] = self.run_mesh_vlm_qa(resume=resume)

        if "physics" in enabled:
            results["physics"] = self.run_physics(resume=resume)

        if "sim_export" in enabled:
            results["sim_export"] = self.run_sim_export(resume=resume)

        logger.info("AL3DG パイプライン完了")
        self._save_run_summary(results)
        return results

    # ------------------------------------------------------------------
    # ステップ実装
    # ------------------------------------------------------------------

    def run_prompt_generation(self) -> dict:
        """T-1.2: プロンプト生成"""
        logger.info("=== T-1.2 プロンプト生成 ===")
        from src.prompt_generator import PromptGenerator

        pg_cfg = self.cfg.get("prompt_generation", {})
        gen = PromptGenerator(
            config_dir=pg_cfg.get("configs_dir", "configs"),
            seed=pg_cfg.get("seed", 42),
        )
        output_file = pg_cfg.get("output_file", "outputs/prompts/prompts.json")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        prompts = gen.generate_all(
            total=self.cfg.generation.get("prompt_generate_number", 100)
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)

        logger.info(f"  プロンプト生成完了: {len(prompts)} 件 → {output_file}")

        # prompt_review.html 生成（CSV が無い場合でも JSON とカテゴリ情報のみで出力される）
        try:
            report_dir = Path(self.cfg.paths.get("reports_dir", "outputs/reports"))
            report_dir.mkdir(parents=True, exist_ok=True)
            images_dir = self.cfg.get("image_generation", {}).get(
                "output_dir", "outputs/images"
            )
            images_csv_dir = Path(
                self.cfg.paths.get("images_csv_dir", "outputs/images_csv")
            )
            gen.generate_html_report(
                prompts_json_path=output_file,
                image_gen_csv_path=str(images_csv_dir / "image_generation_prompts.csv"),
                images_dir=images_dir,
                output_path=str(report_dir / "prompt_review.html"),
            )
            logger.info(f"  プロンプトレビュー: {report_dir / 'prompt_review.html'}")
        except Exception as e:
            logger.warning(f"  prompt_review.html 生成エラー: {e}")

        return {"count": len(prompts), "output_file": output_file}

    def run_image_generation(self, resume: bool = True) -> dict:
        """T-2.1: 画像生成 (FLUX.1-schnell)"""
        logger.info("=== T-2.1 画像生成 ===")
        from src.image_generator import ImageGenerator

        img_cfg = self.cfg.get("image_generation", {})
        output_dir = img_cfg.get("output_dir", "outputs/images")

        # プロンプト一覧を読み込む
        prompt_file = self.cfg.prompt_generation.get(
            "output_file", "outputs/prompts/prompts.json"
        )
        with open(prompt_file, encoding="utf-8") as f:
            prompts = json.load(f)

        gen = ImageGenerator(
            model_path=str(Path(self.cfg.models.flux.get(
                "model_dir", "~/models/FLUX.1-schnell"
            )).expanduser()),
        )
        try:
            result_list = gen.generate_batch(
                prompts=prompts,
                output_dir=output_dir,
                num_inference_steps=img_cfg.get("num_inference_steps", 4),
                guidance_scale=img_cfg.get("guidance_scale", 0.0),
                width=img_cfg.get("width", 1024),
                height=img_cfg.get("height", 1024),
            )
        finally:
            gen.unload()

        result = {
            "success": sum(1 for r in result_list if r.get("status") == "generated"),
            "skipped": sum(1 for r in result_list if r.get("status") == "skipped"),
            "failed":  sum(1 for r in result_list if r.get("status") == "failed"),
            "total":   len(result_list),
        }
        logger.info(f"  画像生成完了: 成功={result['success']}, 失敗={result['failed']}")
        return result

    def run_image_qa(self, resume: bool = True) -> dict:
        """T-2.2: 画像 QA (Qwen3-VL-32B)"""
        logger.info("=== T-2.2 画像 QA ===")
        from src.image_qa import ImageQA

        qa_cfg = self.cfg.get("image_qa", {})
        img_dir = self.cfg.image_generation.get("output_dir", "outputs/images")
        approved_dir = qa_cfg.get("output_dir", "outputs/images_approved")
        output_json = str(Path(approved_dir) / "image_qa_results.json")

        qa = ImageQA(
            vllm_base_url=self.cfg.models.vlm.get("base_url", "http://localhost:8001/v1"),
            model_name=str(Path(self.cfg.models.vlm.get(
                "model_name", str(Path.home() / "models" / "Qwen3-VL-32B-Instruct")
            )).expanduser()),
            thresholds=dict(qa_cfg.get("thresholds", {"realism": 7, "integrity": 7})),
        )
        result = qa.evaluate_batch(
            image_dir=img_dir,
            output_json=output_json,
            approved_dir=approved_dir,
            resume=resume,
        )

        logger.info(
            f"  画像QA完了: 合格={result.get('passed')}, "
            f"不合格={result.get('rejected')}"
        )

        # image_qa_review.html 生成
        try:
            report_dir = Path(self.cfg.paths.get("reports_dir", "outputs/reports"))
            report_dir.mkdir(parents=True, exist_ok=True)
            qa.generate_html_report(str(report_dir / "image_qa_review.html"))
            logger.info(f"  画像QAレビュー: {report_dir / 'image_qa_review.html'}")
        except Exception as e:
            logger.warning(f"  image_qa_review.html 生成エラー: {e}")

        return result

    def run_mesh_generation(self, resume: bool = True) -> dict:
        """T-3.1: 3D メッシュ生成 (TRELLIS.2)"""
        logger.info("=== T-3.1 3D メッシュ生成 ===")
        from src.mesh_generator import MeshGenerator

        mesh_cfg = self.cfg.get("mesh_generation", {})
        approved_dir = self.cfg.image_qa.get("output_dir", "outputs/images_approved")

        gen = MeshGenerator(
            model_path=self.cfg.models.trellis.get(
                "model_dir", "~/models/TRELLIS.2-4B"
            ),
        )
        try:
            result_list = gen.generate_batch(
                image_dir=approved_dir,
                output_dir=mesh_cfg.get("output_dir", "outputs/meshes_raw"),
                seed_offset=mesh_cfg.get("seed", 42),
            )
        finally:
            gen.unload()

        result = {
            "success": sum(1 for r in result_list if r.get("status") == "generated"),
            "skipped": sum(1 for r in result_list if r.get("status") == "skipped"),
            "failed":  sum(1 for r in result_list if r.get("status") == "failed"),
            "total":   len(result_list),
        }
        logger.info(
            f"  3D生成完了: 成功={result['success']}, 失敗={result['failed']}"
        )
        return result

    def run_mesh_qa(self, resume: bool = True) -> dict:
        """T-3.2: メッシュ QA"""
        logger.info("=== T-3.2 メッシュ QA ===")
        from src.mesh_qa import MeshQA

        qa_cfg = self.cfg.get("mesh_qa", {})
        mesh_dir = self.cfg.mesh_generation.get("output_dir", "outputs/meshes_raw")
        approved_dir = qa_cfg.get("output_dir", "outputs/meshes_approved")

        qa = MeshQA()
        result = qa.check_batch(
            mesh_dir=mesh_dir,
            output_json=str(Path(approved_dir) / "mesh_qa_results.json"),
            approved_dir=approved_dir,
            attempt_repair=qa_cfg.get("repair", True),
        )

        logger.info(
            f"  メッシュQA完了: 合格={result.get('passed')}, "
            f"不合格={result.get('failed')}"
        )
        return result

    def run_mesh_vlm_qa(self, resume: bool = True) -> dict:
        """T-3.3: VLM マルチビュー QA"""
        logger.info("=== T-3.3 VLM マルチビュー QA ===")
        from src.mesh_vlm_qa import MeshVLMQA

        qa_cfg = self.cfg.get("mesh_vlm_qa", {})
        mesh_dir = qa_cfg.get("output_dir", "outputs/meshes_approved")
        output_json = str(Path(mesh_dir) / "vlm_qa_results.json")

        qa = MeshVLMQA(
            vllm_base_url=self.cfg.models.vlm.get("base_url", "http://localhost:8001/v1"),
            model_name=str(Path(self.cfg.models.vlm.get(
                "model_name", str(Path.home() / "models" / "Qwen3-VL-32B-Instruct")
            )).expanduser()),
            thresholds=dict(qa_cfg.get("thresholds", {"geometry": 6, "texture": 5})),
        )
        result = qa.evaluate_batch(
            mesh_dir=mesh_dir,
            output_json=output_json,
            render_dir=qa_cfg.get("render_dir", "outputs/renders"),
            views=len(list(qa_cfg.get("azimuths", [0, 90, 180, 270]))),
            resume=resume,
        )

        logger.info(
            f"  VLM 3D QA完了: 合格={result.get('passed')}, "
            f"不合格={result.get('failed')}"
        )

        # mesh_vlm_qa_review.html 生成
        try:
            report_dir = Path(self.cfg.paths.get("reports_dir", "outputs/reports"))
            report_dir.mkdir(parents=True, exist_ok=True)
            qa.generate_html_report(str(report_dir / "mesh_vlm_qa_review.html"))
            logger.info(f"  3D QAレビュー: {report_dir / 'mesh_vlm_qa_review.html'}")
        except Exception as e:
            logger.warning(f"  mesh_vlm_qa_review.html 生成エラー: {e}")

        return result

    def run_physics(self, resume: bool = True) -> dict:
        """T-4.1: 物理プロパティ付与"""
        logger.info("=== T-4.1 物理プロパティ付与 ===")
        from src.physics_processor import PhysicsProcessor

        phys_cfg = self.cfg.get("physics", {})
        mesh_dir = self.cfg.mesh_vlm_qa.get("output_dir", "outputs/meshes_approved")
        vlm_json = str(Path(mesh_dir) / "vlm_qa_results.json")
        configs_dir = self.cfg.prompt_generation.get("configs_dir", "configs")

        proc = PhysicsProcessor(
            material_config_path=str(Path(configs_dir) / "material_properties.yaml"),
            seed=42,
        )
        coacd_cfg = phys_cfg.get("coacd", {})
        result = proc.process_batch(
            mesh_dir=mesh_dir,
            output_dir=phys_cfg.get("output_dir", "outputs/assets_final"),
            metadata_json=vlm_json if Path(vlm_json).exists() else None,
            coacd_threshold=coacd_cfg.get("threshold", 0.01),
            coacd_max_convex_hull=coacd_cfg.get("max_convex_hull", 64),
            coacd_max_ch_vertex=coacd_cfg.get("max_ch_vertex", 2048),
            coacd_resolution=coacd_cfg.get("resolution", 8000),
            coacd_mcts_iterations=coacd_cfg.get("mcts_iterations", 300),
            resume=resume,
        )

        logger.info(
            f"  物理付与完了: 成功={result.get('success')}, 失敗={result.get('failed')}"
        )
        return result

    def run_sim_export(self, resume: bool = True) -> dict:
        """T-4.2: シミュレータエクスポート"""
        logger.info("=== T-4.2 シミュレータエクスポート ===")
        from src.sim_exporter import SimExporter

        exp_cfg = self.cfg.get("sim_export", {})
        assets_dir = self.cfg.physics.get("output_dir", "outputs/assets_final")

        exporter = SimExporter()
        result = exporter.export_batch(
            assets_dir=assets_dir,
            output_dir=exp_cfg.get("output_dir", "outputs/assets_final"),
            resume=resume,
        )

        logger.info(
            f"  エクスポート完了: 成功={result.get('success')}, 失敗={result.get('failed')}"
        )
        return result

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _resolve_steps(self, steps: Optional[list[str]]) -> list[str]:
        """有効なステップ名リストを解決する"""
        step_map = {
            "t1_prompt_generation": "prompt",
            "t2_image_generation": "image",
            "t2_image_qa": "image_qa",
            "t3_mesh_generation": "mesh",
            "t3_mesh_qa": "mesh_qa",
            "t3_mesh_vlm_qa": "mesh_vlm_qa",
            "t4_physics": "physics",
            "t4_sim_export": "sim_export",
        }
        if steps is not None:
            return steps

        enabled = []
        cfg_steps = self.cfg.get("steps", {})
        for cfg_key, step_name in step_map.items():
            if cfg_steps.get(cfg_key, True):
                enabled.append(step_name)
        return enabled

    def _save_run_summary(self, results: dict) -> None:
        """実行サマリーを JSON で保存する"""
        output_dir = Path(self.cfg.paths.get("reports_dir", "outputs/reports"))
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "pipeline_run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"実行サマリー保存: {summary_path}")
