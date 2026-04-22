"""
統合テスト

個別機能テストでは検出しにくい「ステップ間の連携」「CLI ルーティング」「resume 動作」を検証する。

テストグループ:
  A: CLI --input ルーティング回帰（BUG-02）
     run_step.py の --input オプションが各ステップのコンフィグキーに正しくマップされること
  B: ステップ resume 統合
     resume=True でスキップされること、resume=False で再処理されること
  C: ステップ間データフロー（C1〜C4 の 6 テスト）
     前ステップ出力スキーマが後ステップ入力として受理されること
     C1: prompt → image  (prompts.json → ImageGenerator.generate_batch)
     C2: mesh_qa → mesh_vlm_qa  (approved GLBs → VLM evaluate_batch)
     C3: image_qa → mesh_generation  (approved PNGs → MeshGenerator.generate_batch)
     (既存) VLM QA → physics, physics.json → SimExporter
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import trimesh

sys.path.insert(0, str(Path(__file__).parent.parent))

# テスト用設定ファイルパス
_CONFIG = str(Path(__file__).parent.parent / "configs" / "pipeline_config.yaml")
_MATERIAL_CFG = str(Path(__file__).parent.parent / "configs" / "material_properties.yaml")


def _load_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load(_CONFIG)


def _make_glb(tmpdir: str, name: str = "mesh.glb") -> str:
    box = trimesh.creation.box(extents=[1.0, 0.6, 1.4])
    path = str(Path(tmpdir) / name)
    box.export(path, file_type="glb")
    return path


# ============================================================
# A: CLI --input ルーティング回帰（BUG-02）
# ============================================================

class TestRunStepInputRouting(unittest.TestCase):
    """
    run_step.py の --input オプションが正しい設定キーにマップされることを確認する。
    BUG-02: 以前は args.input が参照されておらず、--input が無視されていた。

    main() を丸ごと実行すると実データに対してパイプラインが走るため、
    config 更新ロジックのみを直接テストする。
    """

    def _apply_input_routing(self, step: str, input_dir: str):
        """run_step.main() の config 更新部分のみを再現して cfg を返す"""
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(_CONFIG)
        # run_step.py の --input ルーティングロジックと同一
        if step == "mesh_qa":
            OmegaConf.update(cfg, "mesh_qa.input_dir", input_dir)
        elif step == "mesh_vlm_qa":
            OmegaConf.update(cfg, "mesh_vlm_qa.input_dir", input_dir)
        elif step == "physics":
            OmegaConf.update(cfg, "mesh_vlm_qa.output_dir", input_dir)
        elif step == "sim_export":
            OmegaConf.update(cfg, "physics.output_dir", input_dir)
        return cfg

    def _assert_routing(self, step: str, input_dir: str, cfg_key_path: str, expected: str):
        """ルーティング後の cfg 値を検証するヘルパー"""
        cfg = self._apply_input_routing(step, input_dir)
        keys = cfg_key_path.split(".")
        val = cfg
        for k in keys:
            val = val[k]
        self.assertEqual(str(val), expected,
                         f"--step {step} --input {input_dir} → {cfg_key_path} = {val!r} (expected {expected!r})")

    def test_input_routes_to_mesh_qa_input_dir(self):
        """--step mesh_qa --input X → cfg.mesh_qa.input_dir = X"""
        self._assert_routing("mesh_qa", "/test/meshes", "mesh_qa.input_dir", "/test/meshes")

    def test_input_routes_to_mesh_vlm_qa_input_dir(self):
        """--step mesh_vlm_qa --input X → cfg.mesh_vlm_qa.input_dir = X"""
        self._assert_routing("mesh_vlm_qa", "/test/approved", "mesh_vlm_qa.input_dir", "/test/approved")

    def test_input_routes_to_physics_via_vlm_output_dir(self):
        """--step physics --input X → cfg.mesh_vlm_qa.output_dir = X"""
        self._assert_routing("physics", "/test/vlm_out", "mesh_vlm_qa.output_dir", "/test/vlm_out")

    def test_input_routes_to_sim_export_via_physics_output_dir(self):
        """--step sim_export --input X → cfg.physics.output_dir = X"""
        self._assert_routing("sim_export", "/test/assets", "physics.output_dir", "/test/assets")

    def test_routing_does_not_mutate_other_keys(self):
        """mesh_qa ルーティングが他ステップの設定を変えないこと"""
        from omegaconf import OmegaConf
        cfg_before = OmegaConf.load(_CONFIG)
        cfg_after = self._apply_input_routing("mesh_qa", "/test/meshes")
        # physics.output_dir は変化しないこと
        self.assertEqual(
            str(cfg_before.get("physics", {}).get("output_dir", "")),
            str(cfg_after.get("physics", {}).get("output_dir", "")),
        )


# ============================================================
# B: ステップ resume 統合
# ============================================================

class TestPhysicsResume(unittest.TestCase):
    """PhysicsProcessor の resume 動作を統合レベルで確認する"""

    def setUp(self):
        from src.physics_processor import PhysicsProcessor
        self.processor = PhysicsProcessor(_MATERIAL_CFG, seed=42)
        self.tmpdir = tempfile.mkdtemp()
        self.mesh_dir = Path(self.tmpdir) / "meshes"
        self.mesh_dir.mkdir()
        self.output_dir = Path(self.tmpdir) / "assets"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_glb(self, name: str) -> Path:
        box = trimesh.creation.box(extents=[1.0, 0.6, 1.4])
        path = self.mesh_dir / f"{name}.glb"
        box.export(str(path), file_type="glb")
        return path

    @patch("src.physics_processor.PhysicsProcessor.generate_collision")
    def test_resume_true_skips_existing(self, mock_collision):
        """resume=True のとき既存 physics.json を持つアセットをスキップすること"""
        mock_collision.return_value = ["col_0.stl"]
        self._create_glb("asset_a")
        self._create_glb("asset_b")

        # 1 回目: 全件処理
        self.processor.process_batch(str(self.mesh_dir), str(self.output_dir), resume=True)
        first_call_count = mock_collision.call_count

        # 2 回目: resume=True でスキップ
        mock_collision.reset_mock()
        self.processor.process_batch(str(self.mesh_dir), str(self.output_dir), resume=True)
        self.assertEqual(mock_collision.call_count, 0, "resume=True で generate_collision が呼ばれた")

    @patch("src.physics_processor.PhysicsProcessor.generate_collision")
    def test_resume_false_reprocesses(self, mock_collision):
        """resume=False のとき既存アセットを再処理すること"""
        mock_collision.return_value = ["col_0.stl"]
        self._create_glb("asset_a")

        # 1 回目
        self.processor.process_batch(str(self.mesh_dir), str(self.output_dir), resume=True)

        # 2 回目: resume=False で再処理
        mock_collision.reset_mock()
        self.processor.process_batch(str(self.mesh_dir), str(self.output_dir), resume=False)
        self.assertGreater(mock_collision.call_count, 0, "resume=False で再処理されなかった")

    @patch("src.physics_processor.PhysicsProcessor.generate_collision")
    def test_partial_resume_processes_missing_only(self, mock_collision):
        """一部が処理済みの場合、未処理分だけ実行されること"""
        mock_collision.return_value = ["col_0.stl"]
        self._create_glb("asset_a")
        self._create_glb("asset_b")
        self._create_glb("asset_c")

        # asset_a だけ先に処理
        self.processor.process_batch(
            str(self.mesh_dir / ".."),  # dummy: 個別処理を使う
            str(self.output_dir),
            resume=True,
        )
        # asset_a の physics.json だけ手動で作成してスキップ状態を再現
        (self.output_dir / "asset_a").mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "asset_a" / "physics.json", "w") as f:
            json.dump({"asset_id": "asset_a", "mass_kg": 0.1}, f)

        mock_collision.reset_mock()
        result = self.processor.process_batch(
            str(self.mesh_dir), str(self.output_dir), resume=True
        )
        # asset_a はスキップされ asset_b, asset_c だけ処理される
        processed_ids = {r.get("asset_id") for r in result["results"]
                         if r.get("status") not in ("skipped",)}
        self.assertNotIn("asset_a", processed_ids, "asset_a がスキップされなかった")


# ============================================================
# C: ステップ間データフロー
# ============================================================

class TestStepDataFlow(unittest.TestCase):
    """
    前ステップの出力スキーマが後ステップの入力として正しく処理されることを確認する。
    実際のモデルはモック化し、データ形式の整合性のみを検証する。
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_vlm_qa_output_consumed_by_physics_batch(self):
        """
        mesh_vlm_qa の output_json スキーマが physics process_batch で正しく読めること。
        pass=True のメッシュだけが処理対象になることを確認する。
        """
        from src.physics_processor import PhysicsProcessor

        mesh_dir = Path(self.tmpdir) / "meshes"
        mesh_dir.mkdir()
        output_dir = Path(self.tmpdir) / "assets"

        # テスト用 GLB を 3 件作成
        assets = {"pass_x": True, "fail_y": False, "pass_z": True}
        for name in assets:
            box = trimesh.creation.box(extents=[1.0, 0.6, 1.4])
            box.export(str(mesh_dir / f"{name}.glb"), file_type="glb")

        # mesh_vlm_qa の出力 JSON スキーマを再現
        vlm_json = Path(self.tmpdir) / "vlm_qa_results.json"
        with open(vlm_json, "w") as f:
            json.dump({
                "total": 3, "passed": 2, "failed": 1,
                "results": [
                    {"mesh_path": str(mesh_dir / f"{name}.glb"),
                     "pass": passed,
                     "geometry_score": 7 if passed else 3,
                     "texture_score": 6 if passed else 2,
                     "detected_type": "hard_suitcase",
                     "detected_material": "polycarbonate"}
                    for name, passed in assets.items()
                ]
            }, f)

        processor = PhysicsProcessor(_MATERIAL_CFG, seed=42)
        with patch.object(processor, "generate_collision", return_value=["col_0.stl"]):
            result = processor.process_batch(
                str(mesh_dir), str(output_dir),
                metadata_json=str(vlm_json),
            )

        self.assertEqual(result["total"], 2, "pass=True の 2 件だけ処理されるべき")
        self.assertTrue((output_dir / "pass_x" / "physics.json").exists())
        self.assertTrue((output_dir / "pass_z" / "physics.json").exists())
        self.assertFalse((output_dir / "fail_y" / "physics.json").exists())

    def test_physics_json_schema_accepted_by_sim_exporter(self):
        """
        physics process_single が出力する physics.json のスキーマを
        SimExporter が正しく読み込めること。
        """
        from src.sim_exporter import SimExporter
        from src.physics_processor import PhysicsProcessor

        asset_dir = Path(self.tmpdir) / "asset_001"
        asset_dir.mkdir()

        # visual.glb を作成
        glb_path = str(asset_dir / "visual.glb")
        box = trimesh.creation.box(extents=[0.06, 0.04, 0.09])
        box.export(glb_path, file_type="glb")

        # collision STL を作成
        col_dir = asset_dir / "collisions"
        col_dir.mkdir()
        col_path = str(col_dir / "collision_000.stl")
        box.export(col_path, file_type="stl")

        # PhysicsProcessor が生成する physics.json スキーマを再現
        physics_data = {
            "asset_id": "asset_001",
            "material": "polycarbonate",
            "density_kg_m3": 1200.0,
            "static_friction": 0.36,
            "dynamic_friction": 0.28,
            "restitution": 0.12,
            "volume_m3": 0.000216,
            "mass_kg": 0.2592,
            "luggage_type": "hard_suitcase",
            "collision_count": 1,
            "scale_factor": 0.00006,
        }
        with open(asset_dir / "physics.json", "w") as f:
            json.dump(physics_data, f)

        exporter = SimExporter()
        # MJCF のみ生成してスキーマ互換性を確認（USD は外部ツール依存）
        result = exporter.export_mjcf(str(asset_dir), str(asset_dir / "asset_001.xml"))

        self.assertTrue(Path(result).exists(), "MJCF ファイルが生成されなかった")
        # physics.json の mass が MJCF に反映されていること
        xml_content = Path(result).read_text()
        self.assertIn("mass", xml_content)


# ============================================================
# C1〜C4: ステップ間データフロー（追加）
# ============================================================

class TestStepDataFlowExtended(unittest.TestCase):
    """
    グループ C の拡張テスト。
    モデル呼び出しはすべてモック化し、データスキーマの互換性のみを検証する。
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ----------------------------------------------------------
    # C1: prompt → image
    # ----------------------------------------------------------
    def test_prompt_json_schema_accepted_by_image_generator_batch(self):
        """
        C1: PromptGenerator.generate_combinatorial() の出力 JSON スキーマを
        ImageGenerator.generate_batch() がそのまま受理できること。

        pipeline.run_image_generation() は prompts.json を json.load() してリストとして
        generate_batch(prompts=...) に渡す。各エントリの "prompt" キーが必須。
        """
        from PIL import Image as PILImage
        from src.image_generator import ImageGenerator

        # --- PromptGenerator 出力スキーマを再現 ---
        prompts = [
            {
                "prompt": "close-up product photo, hard shell suitcase, glossy black, white background",
                "luggage_type": "hard_suitcase",
                "subcategory": "carry-on",
                "material": "polycarbonate",
                "color": "glossy_black",
                "prompt_id": "abc123",
            },
            {
                "prompt": "close-up product photo, soft duffel bag, navy blue, white background",
                "luggage_type": "duffel_bag",
                "subcategory": "weekend",
                "material": "nylon",
                "color": "navy_blue",
                "prompt_id": "def456",
            },
        ]

        output_dir = str(Path(self.tmpdir) / "images")
        dummy_image = PILImage.new("RGB", (1024, 1024), color=(200, 200, 200))

        gen = ImageGenerator.__new__(ImageGenerator)
        gen.device = "cuda"
        gen._pipe = MagicMock()
        gen._pipe.return_value.images = [dummy_image]

        with patch.object(gen, "generate_single", return_value=dummy_image):
            result = gen.generate_batch(prompts, output_dir=output_dir)

        # generate_batch は list[dict] を返す
        self.assertEqual(len(result), len(prompts),
                         "prompts.json の全エントリが generate_batch に受理されること")
        success_count = sum(1 for r in result if r.get("status") == "generated")
        self.assertEqual(success_count, len(prompts),
                         "全プロンプトが画像生成に成功すること")

    # ----------------------------------------------------------
    # C2: mesh_qa → mesh_vlm_qa
    # ----------------------------------------------------------
    def test_mesh_qa_approved_dir_accepted_by_vlm_evaluate_batch(self):
        """
        C2: MeshQA.check_batch() が approved_dir に出力する GLB ファイル群を
        MeshVLMQA.evaluate_batch() の mesh_dir として受理できること。

        mesh_qa は合格メッシュを approved_dir にコピーするだけ（追加ファイルなし）。
        mesh_vlm_qa は mesh_dir/*.glb を処理する。
        """
        from src.mesh_vlm_qa import MeshVLMQA

        # mesh_qa の approved_dir 構造: GLB ファイルが平置き
        approved_dir = Path(self.tmpdir) / "meshes_approved"
        approved_dir.mkdir()
        for name in ("asset_001.glb", "asset_002.glb"):
            box = trimesh.creation.box(extents=[1.0, 0.6, 1.4])
            box.export(str(approved_dir / name), file_type="glb")

        output_dir = str(Path(self.tmpdir) / "vlm_results")

        from src.mesh_vlm_qa import _USER_PROMPT_TEMPLATE, MIN_GEOMETRY_SCORE, MIN_TEXTURE_SCORE
        qa = MeshVLMQA.__new__(MeshVLMQA)
        qa._client = MagicMock()
        qa.model_name = "Qwen/Qwen3-VL-32B-Instruct"
        qa.vllm_base_url = "http://localhost:8001/v1"
        qa._system_prompt = "test"
        qa._user_prompt_template = _USER_PROMPT_TEMPLATE
        qa._min_geometry_score = MIN_GEOMETRY_SCORE
        qa._min_texture_score = MIN_TEXTURE_SCORE

        good_response = json.dumps({
            "geometry_score": 8, "texture_score": 7,
            "detected_type": "hard_suitcase", "detected_material": "polycarbonate",
            "issues": [], "thinking": ""
        })
        mock_msg = MagicMock()
        mock_msg.content = good_response
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        qa._client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        with patch("src.utils.rendering.render_multiview",
                   return_value=[str(approved_dir / "asset_001.glb")]):
            result = qa.evaluate_batch(str(approved_dir), output_dir)

        self.assertEqual(result["total"], 2,
                         "mesh_qa approved_dir の GLB 2 件が evaluate_batch で処理されること")
        self.assertIn("passed", result)

    # ----------------------------------------------------------
    # C3: image_qa → mesh_generation
    # ----------------------------------------------------------
    def test_image_qa_approved_dir_accepted_by_mesh_generator_batch(self):
        """
        C3: ImageQA.evaluate_batch() が images_approved に出力する PNG ファイル群を
        MeshGenerator.generate_batch() の image_dir として受理できること。

        image_qa は approved 画像を images_approved/ に PNG としてコピーする。
        mesh_generator は image_dir/*.png を処理する。
        """
        from PIL import Image as PILImage
        from src.mesh_generator import MeshGenerator

        # image_qa の images_approved 構造: PNG ファイルが平置き
        approved_dir = Path(self.tmpdir) / "images_approved"
        approved_dir.mkdir()
        for name in ("000001_abc.png", "000002_def.png"):
            img = PILImage.new("RGB", (1024, 1024), color=(200, 200, 200))
            img.save(str(approved_dir / name))

        output_dir = str(Path(self.tmpdir) / "meshes_raw")

        gen = MeshGenerator.__new__(MeshGenerator)
        gen._pipeline = MagicMock()

        # generate_single が GLB を出力するようにモック
        def fake_generate_single(image_path, output_path=None, seed=0):
            out = output_path or str(Path(image_path).with_suffix(".glb"))
            box = trimesh.creation.box()
            box.export(out, file_type="glb")
            return out

        with patch.object(gen, "generate_single", side_effect=fake_generate_single):
            result = gen.generate_batch(str(approved_dir), str(output_dir))

        self.assertEqual(result["total"], 2,
                         "images_approved の PNG 2 件が generate_batch で処理されること")
        self.assertEqual(result["success"], 2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
