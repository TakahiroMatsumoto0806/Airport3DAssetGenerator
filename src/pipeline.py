"""
T-6.1: パイプライン統合

AL3DG の全ステップを統括するオーケストレーター。
DGX Spark の逐次ロード制約に従い、1 モデルずつロード→実行→アンロード。

ステップ順序:
  T-1.2  プロンプト生成
  T-2.1  画像生成 (FLUX.1-schnell)
  T-2.2  画像 QA   (Qwen3-VL-32B via vLLM)
  T-3.1  3D 生成   (TRELLIS.2)
  T-3.2  メッシュ QA (trimesh / open3d)
  T-3.3  VLM マルチビュー QA (Qwen3-VL-32B + pyrender)
  T-4.1  物理プロパティ付与 (CoACD)
  T-4.2  シミュレータエクスポート (MJCF / USD)
  T-5.1  多様性評価レポート (OpenCLIP)

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

        if "diversity" in enabled:
            results["diversity"] = self.run_diversity_report()

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
            configs_dir=pg_cfg.get("configs_dir", "configs"),
            seed=pg_cfg.get("seed", 42),
        )
        output_file = pg_cfg.get("output_file", "outputs/prompts/prompts.json")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        prompts = gen.generate_all(
            count_per_category=self.cfg.generation.get("prompt_count_per_category", 10)
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)

        logger.info(f"  プロンプト生成完了: {len(prompts)} 件 → {output_file}")
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
            model_id=self.cfg.models.flux.get(
                "model_id", "black-forest-labs/FLUX.1-schnell"
            ),
        )
        gen.load_model()
        try:
            result = gen.generate_batch(
                prompts=prompts,
                output_dir=output_dir,
                num_inference_steps=img_cfg.get("num_inference_steps", 4),
                guidance_scale=img_cfg.get("guidance_scale", 0.0),
                width=img_cfg.get("width", 1024),
                height=img_cfg.get("height", 1024),
                resume=resume,
            )
        finally:
            gen.unload()

        logger.info(f"  画像生成完了: 成功={result['success']}, 失敗={result['failed']}")
        return result

    def run_image_qa(self, resume: bool = True) -> dict:
        """T-2.2: 画像 QA (Qwen3-VL-32B)"""
        logger.info("=== T-2.2 画像 QA ===")
        from src.image_qa import ImageQA

        qa_cfg = self.cfg.get("image_qa", {})
        img_dir = self.cfg.image_generation.get("output_dir", "outputs/images")
        image_paths = sorted(Path(img_dir).glob("*.png"))

        qa = ImageQA(
            base_url=self.cfg.models.vlm.get("base_url", "http://localhost:8000/v1"),
            model_name=self.cfg.models.vlm.get(
                "model_name", "Qwen/Qwen3-VL-32B-Instruct"
            ),
        )
        result = qa.evaluate_batch(
            image_paths=[str(p) for p in image_paths],
            output_dir=qa_cfg.get("output_dir", "outputs/images_approved"),
            thresholds=dict(qa_cfg.get("thresholds", {"realism": 7, "integrity": 7})),
            resume=resume,
        )

        logger.info(
            f"  画像QA完了: 合格={result.get('approved')}, "
            f"不合格={result.get('rejected')}"
        )
        return result

    def run_mesh_generation(self, resume: bool = True) -> dict:
        """T-3.1: 3D メッシュ生成 (TRELLIS.2)"""
        logger.info("=== T-3.1 3D メッシュ生成 ===")
        from src.mesh_generator import MeshGenerator

        mesh_cfg = self.cfg.get("mesh_generation", {})
        approved_dir = self.cfg.image_qa.get(
            "output_dir", "outputs/images_approved"
        )
        image_paths = sorted(Path(approved_dir).glob("*.png"))

        gen = MeshGenerator(
            model_dir=self.cfg.models.trellis.get(
                "model_dir", "~/models/TRELLIS-image-large"
            ),
        )
        gen.load_model()
        try:
            result = gen.generate_batch(
                image_paths=[str(p) for p in image_paths],
                output_dir=mesh_cfg.get("output_dir", "outputs/meshes_raw"),
                seed=mesh_cfg.get("seed", 42),
                resume=resume,
            )
        finally:
            gen.unload()

        logger.info(
            f"  3D生成完了: 成功={result.get('success')}, 失敗={result.get('failed')}"
        )
        return result

    def run_mesh_qa(self, resume: bool = True) -> dict:
        """T-3.2: メッシュ QA"""
        logger.info("=== T-3.2 メッシュ QA ===")
        from src.mesh_qa import MeshQA

        qa_cfg = self.cfg.get("mesh_qa", {})
        mesh_dir = self.cfg.mesh_generation.get(
            "output_dir", "outputs/meshes_raw"
        )
        mesh_paths = sorted(Path(mesh_dir).glob("**/*.glb"))

        qa = MeshQA(
            min_faces=qa_cfg.get("thresholds", {}).get("min_faces", 5000),
            max_faces=qa_cfg.get("thresholds", {}).get("max_faces", 100000),
            max_aspect_ratio=qa_cfg.get("thresholds", {}).get("max_aspect_ratio", 20.0),
        )
        result = qa.check_batch(
            mesh_paths=[str(p) for p in mesh_paths],
            output_dir=qa_cfg.get("output_dir", "outputs/meshes_approved"),
            repair=qa_cfg.get("repair", True),
            resume=resume,
        )

        logger.info(
            f"  メッシュQA完了: 合格={result.get('approved')}, "
            f"不合格={result.get('rejected')}"
        )
        return result

    def run_mesh_vlm_qa(self, resume: bool = True) -> dict:
        """T-3.3: VLM マルチビュー QA"""
        logger.info("=== T-3.3 VLM マルチビュー QA ===")
        from src.mesh_vlm_qa import MeshVLMQA

        qa_cfg = self.cfg.get("mesh_vlm_qa", {})
        mesh_dir = self.cfg.mesh_qa.get(
            "output_dir", "outputs/meshes_approved"
        )
        mesh_paths = sorted(Path(mesh_dir).glob("**/*.glb"))

        qa = MeshVLMQA(
            base_url=self.cfg.models.vlm.get("base_url", "http://localhost:8000/v1"),
            model_name=self.cfg.models.vlm.get(
                "model_name", "Qwen/Qwen3-VL-32B-Instruct"
            ),
        )
        result = qa.evaluate_batch(
            mesh_paths=[str(p) for p in mesh_paths],
            output_dir=qa_cfg.get("output_dir", "outputs/meshes_approved"),
            render_dir=qa_cfg.get("render_dir", "outputs/renders"),
            thresholds=dict(
                qa_cfg.get("thresholds", {"geometry": 7, "texture": 6})
            ),
            azimuths=list(qa_cfg.get("azimuths", [0, 90, 180, 270])),
            render_size=tuple(qa_cfg.get("render_size", [512, 512])),
            resume=resume,
        )

        logger.info(
            f"  VLM 3D QA完了: 合格={result.get('approved')}, "
            f"不合格={result.get('rejected')}"
        )
        return result

    def run_physics(self, resume: bool = True) -> dict:
        """T-4.1: 物理プロパティ付与"""
        logger.info("=== T-4.1 物理プロパティ付与 ===")
        from src.physics_processor import PhysicsProcessor

        phys_cfg = self.cfg.get("physics", {})
        # T-3.3 合格済みメッシュ
        mesh_dir = self.cfg.mesh_vlm_qa.get(
            "output_dir", "outputs/meshes_approved"
        )
        mesh_paths = sorted(Path(mesh_dir).glob("**/*.glb"))

        proc = PhysicsProcessor(
            configs_dir=self.cfg.prompt_generation.get("configs_dir", "configs"),
            coacd_threshold=phys_cfg.get("coacd_threshold", 0.08),
            max_convex_hulls=phys_cfg.get("max_convex_hulls", 16),
            miniature=phys_cfg.get("miniature", True),
        )
        result = proc.process_batch(
            mesh_paths=[str(p) for p in mesh_paths],
            output_dir=phys_cfg.get("output_dir", "outputs/assets_final"),
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
            format=exp_cfg.get("format", "both"),
            resume=resume,
        )

        logger.info(
            f"  エクスポート完了: 成功={result.get('success')}, 失敗={result.get('failed')}"
        )
        return result

    def run_diversity_report(self) -> dict:
        """T-5.1: 多様性評価レポート"""
        logger.info("=== T-5.1 多様性評価レポート ===")
        from src.diversity_evaluator import DiversityEvaluator

        div_cfg = self.cfg.get("diversity", {})
        assets_dir = self.cfg.physics.get("output_dir", "outputs/assets_final")
        output_dir = div_cfg.get("output_dir", "outputs/reports")

        evaluator = DiversityEvaluator()

        # アセット画像パス（visual.glb のレンダリング済み画像があれば使用）
        image_paths = sorted(Path("outputs/renders").glob("**/*_0.png"))

        embeddings = None
        if image_paths:
            evaluator.load_model(device=self.cfg.models.clip.get("device", "cuda"))
            try:
                embeddings = evaluator.compute_clip_embeddings(
                    [str(p) for p in image_paths],
                    batch_size=div_cfg.get("embed_batch_size", 32),
                )
            finally:
                evaluator.unload()

        # メタデータ収集
        metadata_list: list[dict] = []
        mesh_info_list: list[dict] = []
        for phys_json in sorted(Path(assets_dir).glob("*/physics.json")):
            with open(phys_json, encoding="utf-8") as f:
                phys = json.load(f)
            metadata_list.append(phys)
            extents = phys.get("scale", {}).get("scaled_extents_mm")
            if extents:
                mesh_info_list.append({"scale": {"scaled_extents_mm": extents},
                                       "luggage_type": phys.get("luggage_type")})

        html_path = evaluator.generate_report(
            output_dir=output_dir,
            embeddings=embeddings,
            image_paths=[str(p) for p in image_paths] if image_paths else None,
            metadata_list=metadata_list or None,
            mesh_info_list=mesh_info_list or None,
            near_dup_threshold=div_cfg.get("near_dup_threshold", 0.95),
        )

        logger.info(f"  多様性レポート完了: {html_path}")
        return {"html_path": html_path}

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
            "t5_diversity_report": "diversity",
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
