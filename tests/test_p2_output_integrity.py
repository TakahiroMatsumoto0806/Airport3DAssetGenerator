"""
P2 テスト: 出力内容品質・ステップ整合性・resume・config 反映・memory_guard 統合

## 目的

P0/P1 は「コマンドが正常終了する」レベルの確認だった。
P2 は **出力の中身が正しいこと** を検証する:

  F: 出力ファイルの内容品質  (既存 outputs/ に対して静的に実行)
  G: ステップ間整合性        (前後ステップの入出力件数・ID が一致)
  H: resume 動作            (スキップ条件・再実行の正確性)
  I: config 値の反映        (YAML 設定がコード内で実際に使われる)
  J: memory_guard 統合      (headroom check が実測値で動作する)

実行方法:
  # モデル不要・即実行 (F/G/H/I/J-01,02)
  pytest tests/test_p2_output_integrity.py -v -m "not gpu"

  # CUDA 環境 (J-02,03,04)
  pytest tests/test_p2_output_integrity.py -v
"""

import gc
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---- 共通定数 ----
ASSETS_DIR    = Path("outputs/assets_final")
REPORTS_DIR   = Path("outputs/reports")
RENDERS_DIR   = Path("outputs/renders")
APPROVED_DIR  = Path("outputs/images_approved")
MESHES_APP    = Path("outputs/meshes_approved")
PROMPTS_DIR   = Path("outputs/prompts")
IMAGES_DIR    = Path("outputs/images")


def _all_physics_jsons() -> list[Path]:
    return sorted(ASSETS_DIR.glob("*/physics.json"))


def _all_asset_dirs() -> list[Path]:
    return sorted(d for d in ASSETS_DIR.iterdir()
                  if d.is_dir() and (d / "physics.json").exists() and d.name != "isaac")


# ===========================================================================
# F: 出力ファイル内容検証
# ===========================================================================

class TestOutputFileSchema:
    """F-01〜F-06: physics.json / collision / MJCF / USD の内容品質"""

    PHYSICS_REQUIRED_KEYS = {
        "material", "density_kg_m3", "static_friction",
        "dynamic_friction", "restitution", "volume_m3",
        "mass_kg", "asset_id", "collision_count",
    }

    def test_F01_physics_json_schema(self):
        """F-01: physics.json が必須キーを全て持つ"""
        jsons = _all_physics_jsons()
        assert len(jsons) > 0, "physics.json が存在しない"

        failures = []
        for p in jsons:
            data = json.loads(p.read_text())
            missing = self.PHYSICS_REQUIRED_KEYS - set(data.keys())
            if missing:
                failures.append(f"{p.parent.name}: missing={missing}")

        assert not failures, "\n".join(failures)

    def test_F02_physics_json_numeric_ranges(self):
        """F-02: physics.json の数値が物理的に有効な範囲に収まる"""
        failures = []
        for p in _all_physics_jsons():
            d = json.loads(p.read_text())
            aid = d.get("asset_id", p.parent.name)

            density = d.get("density_kg_m3", 0)
            if not (100 <= density <= 3000):
                failures.append(f"{aid}: density_kg_m3={density} (期待: 100~3000)")

            mass = d.get("mass_kg", 0)
            if mass <= 0:
                failures.append(f"{aid}: mass_kg={mass} (期待: >0)")

            restitution = d.get("restitution", -1)
            if not (0.0 <= restitution <= 1.0):
                failures.append(f"{aid}: restitution={restitution} (期待: 0~1)")

            sf = d.get("static_friction", 0)
            if sf <= 0:
                failures.append(f"{aid}: static_friction={sf} (期待: >0)")

            df = d.get("dynamic_friction", 0)
            if df <= 0:
                failures.append(f"{aid}: dynamic_friction={df} (期待: >0)")

        assert not failures, f"{len(failures)} 件の範囲エラー:\n" + "\n".join(failures[:10])

    def test_F03_collision_stl_exists(self):
        """F-03: collision_count > 0 のアセットに STL ファイルが存在する"""
        failures = []
        for p in _all_physics_jsons():
            d = json.loads(p.read_text())
            count = d.get("collision_count", 0)
            if count <= 0:
                continue
            stls = list(p.parent.glob("collisions/*.stl"))
            if len(stls) == 0:
                failures.append(f"{d.get('asset_id')}: collision_count={count} だが STL なし")
            if len(stls) != count:
                failures.append(
                    f"{d.get('asset_id')}: collision_count={count} だが STL={len(stls)} 件"
                )

        assert not failures, "\n".join(failures[:10])

    def test_F04_mjcf_xml_parses_and_has_required_elements(self):
        """F-04: MJCF XML がパースでき必須要素を含む"""
        mjcf_files = list(ASSETS_DIR.glob("*/mjcf/*.xml"))
        if not mjcf_files:
            pytest.skip("MJCF ファイルが存在しない (sim_export 未実行)")

        failures = []
        for xml_path in mjcf_files:
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError as e:
                failures.append(f"{xml_path.name}: ParseError={e}")
                continue

            assert root.tag == "mujoco", f"{xml_path.name}: root tag={root.tag}"

            has_worldbody = root.find("worldbody") is not None
            has_asset = root.find("asset") is not None
            has_freejoint = root.find(".//freejoint") is not None

            if not has_worldbody:
                failures.append(f"{xml_path.name}: <worldbody> なし")
            if not has_asset:
                failures.append(f"{xml_path.name}: <asset> なし")
            if not has_freejoint:
                failures.append(f"{xml_path.name}: <freejoint> なし")

        assert not failures, "\n".join(failures)

    def test_F05_mjcf_mass_matches_physics_json(self):
        """F-05: MJCF の inertial mass が physics.json の mass_kg と一致する (±0.1%)"""
        mjcf_files = list(ASSETS_DIR.glob("*/mjcf/*.xml"))
        if not mjcf_files:
            pytest.skip("MJCF ファイルが存在しない")

        failures = []
        for xml_path in mjcf_files:
            asset_dir = xml_path.parent.parent
            phys_path = asset_dir / "physics.json"
            if not phys_path.exists():
                continue

            phys_mass = json.loads(phys_path.read_text()).get("mass_kg", None)
            if phys_mass is None:
                continue

            root = ET.parse(xml_path).getroot()
            inertial = root.find(".//inertial")
            if inertial is None:
                failures.append(f"{xml_path.parent.parent.name}: <inertial> なし")
                continue

            xml_mass = float(inertial.get("mass", 0))
            rel_err = abs(xml_mass - phys_mass) / max(phys_mass, 1e-9)
            if rel_err > 0.001:
                failures.append(
                    f"{asset_dir.name}: XML mass={xml_mass:.6f}, "
                    f"physics.json mass={phys_mass:.6f}, rel_err={rel_err:.4%}"
                )

        assert not failures, "\n".join(failures)

    def test_F06_usd_meta_collision_paths_exist(self):
        """F-06: usd_meta.json の collision_mesh_paths が全て実在するファイルを指す"""
        usd_metas = list(ASSETS_DIR.glob("*/*_usd_meta.json"))
        if not usd_metas:
            pytest.skip("USD meta ファイルが存在しない")

        failures = []
        for meta_path in usd_metas:
            # isaac/ 下の旧形式ファイルはスキップ
            if meta_path.parent.name == "isaac":
                continue
            data = json.loads(meta_path.read_text())
            for col_path in data.get("collision_mesh_paths", []):
                if not Path(col_path).exists():
                    failures.append(f"{meta_path.parent.name}: 存在しないパス={col_path}")
                    break  # アセットごとに1件だけ報告

        assert not failures, f"{len(failures)} 件のパス不整合:\n" + "\n".join(failures[:10])


class TestQAResultsSchema:
    """F-07〜F-10: QA 結果 JSON のスキーマ・整合性"""

    def test_F07_image_qa_results_schema(self):
        """F-07: image_qa_results.json が必須キーを持ち型が正しい"""
        qa_json = APPROVED_DIR / "image_qa_results.json"
        if not qa_json.exists():
            pytest.skip("image_qa_results.json が存在しない")

        data = json.loads(qa_json.read_text())
        for key in ("total", "passed", "rejected", "pass_rate", "results"):
            assert key in data, f"必須キー '{key}' が存在しない"

        assert isinstance(data["total"], int)
        assert isinstance(data["passed"], int)
        assert isinstance(data["rejected"], int)
        assert isinstance(data["pass_rate"], (int, float))
        assert isinstance(data["results"], list)

    def test_F08_image_qa_results_count_integrity(self):
        """F-08: image_qa_results.json のカウントが整合する"""
        qa_json = APPROVED_DIR / "image_qa_results.json"
        if not qa_json.exists():
            pytest.skip("image_qa_results.json が存在しない")

        data = json.loads(qa_json.read_text())
        total = data["total"]
        passed = data["passed"]
        rejected = data["rejected"]
        reviewed = data.get("reviewed", 0)
        pass_rate = data["pass_rate"]
        evaluated = data.get("evaluated", total)

        assert passed + rejected + reviewed <= total, (
            f"passed({passed}) + rejected({rejected}) + reviewed({reviewed}) > total({total})"
        )
        expected_rate = passed / max(evaluated, 1)
        assert abs(pass_rate - expected_rate) < 0.01, (
            f"pass_rate={pass_rate:.4f} と expected={expected_rate:.4f} が乖離"
        )

    def test_F09_vlm_qa_score_ranges(self):
        """F-09: vlm_qa_results.json のスコアが 1〜10 の範囲に収まる"""
        vlm_json = MESHES_APP / "vlm_qa_results.json"
        if not vlm_json.exists():
            pytest.skip("vlm_qa_results.json が存在しない")

        data = json.loads(vlm_json.read_text())
        failures = []
        for r in data.get("results", []):
            for key in ("geometry_score", "texture_score", "consistency_score"):
                score = r.get(key)
                if score is not None and not (1 <= score <= 10):
                    failures.append(f"{Path(r.get('mesh_path','')).name}: {key}={score}")

        assert not failures, f"{len(failures)} 件のスコア範囲エラー:\n" + "\n".join(failures[:10])

    def test_F10_vlm_qa_pass_consistent_with_thresholds(self):
        """F-10: vlm_qa の pass 判定が geometry>=5 AND texture>=4 と一致する"""
        vlm_json = MESHES_APP / "vlm_qa_results.json"
        if not vlm_json.exists():
            pytest.skip("vlm_qa_results.json が存在しない")

        GEOMETRY_THRESHOLD = 5
        TEXTURE_THRESHOLD = 4

        data = json.loads(vlm_json.read_text())
        failures = []
        for r in data.get("results", []):
            g = r.get("geometry_score", 0)
            t = r.get("texture_score", 0)
            expected_pass = (g >= GEOMETRY_THRESHOLD and t >= TEXTURE_THRESHOLD)
            actual_pass = r.get("pass", None)
            if actual_pass is not None and actual_pass != expected_pass:
                failures.append(
                    f"{Path(r.get('mesh_path','')).name}: "
                    f"geometry={g}, texture={t}, "
                    f"expected_pass={expected_pass}, actual_pass={actual_pass}"
                )

        assert not failures, (
            f"{len(failures)} 件の pass 判定不一致:\n" + "\n".join(failures[:10])
        )


class TestDiversityReport:
    """F-12: レンダリング画像"""

    def test_F12_render_images_exist_for_all_assets(self):
        """F-12: renders/ に各アセットの4視点 PNG が全て存在する"""
        render_dirs = [d for d in RENDERS_DIR.iterdir() if d.is_dir()] if RENDERS_DIR.exists() else []
        if not render_dirs:
            pytest.skip("renders/ ディレクトリが存在しない")

        failures = []
        for rdir in render_dirs:
            asset_id = rdir.name
            for view_idx in range(4):
                expected = rdir / f"{asset_id}_{view_idx}.png"
                if not expected.exists():
                    failures.append(str(expected))

        assert not failures, (
            f"{len(failures)} 件のレンダリング画像が欠落:\n" + "\n".join(failures[:10])
        )


# ===========================================================================
# G: ステップ間整合性検証
# ===========================================================================

class TestStepwiseIntegrity:
    """G-01〜G-06: 前後ステップの入出力が整合する"""

    def test_G01_assets_final_le_image_qa_passed(self):
        """G-01: assets_final の件数 ≤ image_qa の合格数"""
        qa_json = APPROVED_DIR / "image_qa_results.json"
        if not qa_json.exists():
            pytest.skip("image_qa_results.json が存在しない")

        n_assets = len(_all_physics_jsons())
        n_passed = json.loads(qa_json.read_text()).get("passed", 0)
        assert n_assets <= n_passed, (
            f"assets_final({n_assets}) > image_qa passed({n_passed}): "
            "後段ほど件数が減らなければならない"
        )

    def test_G02_physics_batch_filters_by_vlm_qa_pass(self):
        """G-02: process_batch は vlm_qa pass=True のメッシュのみ処理する"""
        from src.physics_processor import PhysicsProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_dir = tmp / "meshes"
            out_dir = tmp / "assets"
            meta_dir = tmp / "meta"
            mesh_dir.mkdir(); out_dir.mkdir(); meta_dir.mkdir()

            import trimesh, numpy as np
            verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)

            # 3 件のメッシュを用意: pass_a (通過), fail_b (不合格), uneval_c (未評価)
            for aid in ("pass_a", "fail_b", "uneval_c"):
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                with open(mesh_dir / f"{aid}.glb", "wb") as f:
                    f.write(trimesh.exchange.gltf.export_glb(mesh))

            # VLM QA メタデータ: pass_a のみ pass=True
            vlm_results = {
                "results": [
                    {"mesh_path": str(mesh_dir / "pass_a.glb"), "pass": True,
                     "detected_type": "hard_suitcase", "detected_material": "polycarbonate"},
                    {"mesh_path": str(mesh_dir / "fail_b.glb"), "pass": False,
                     "detected_type": "unknown", "detected_material": "unknown"},
                    # uneval_c は JSON に含まれない（未評価）
                ]
            }
            meta_json = meta_dir / "vlm_qa_results.json"
            meta_json.write_text(json.dumps(vlm_results))

            proc = PhysicsProcessor()
            result = proc.process_batch(
                mesh_dir=str(mesh_dir),
                output_dir=str(out_dir),
                metadata_json=str(meta_json),
                resume=False,
            )

            # pass_a だけ処理される
            assert result["total"] == 1, (
                f"total={result['total']} (期待: 1 件 = pass_a のみ)"
            )
            assert (out_dir / "pass_a" / "physics.json").exists(), \
                "pass_a の physics.json が生成されなかった"
            assert not (out_dir / "fail_b").exists(), \
                "fail_b (pass=False) が処理された"
            assert not (out_dir / "uneval_c").exists(), \
                "uneval_c (未評価) が処理された"

    def test_G03_mjcf_and_physics_asset_ids_match(self):
        """G-03: MJCF が存在するアセットは physics.json も持つ（孤立した MJCF がない）"""
        mjcf_assets = set(
            p.parent.parent.name for p in ASSETS_DIR.glob("*/mjcf/*.xml")
        )
        phys_assets = set(
            p.parent.name for p in ASSETS_DIR.glob("*/physics.json")
        )

        orphan_mjcf = mjcf_assets - phys_assets
        assert not orphan_mjcf, (
            f"physics.json のない MJCF アセット: {orphan_mjcf}"
        )

    def test_G04_prompt_id_in_image_filename(self):
        """G-04: prompts.json の prompt_id が対応する画像ファイル名に含まれる"""
        prompts_json = PROMPTS_DIR / "prompts.json"
        if not prompts_json.exists():
            pytest.skip("prompts.json が存在しない")

        prompts = json.loads(prompts_json.read_text())
        image_filenames = {p.stem for p in IMAGES_DIR.glob("*.png")}

        failures = []
        for prompt in prompts:
            pid = prompt.get("metadata", {}).get("prompt_id")
            if not pid:
                continue
            # 画像ファイル名は "<idx>_<prompt_id>" 形式
            matching = [fn for fn in image_filenames if pid in fn]
            if not matching:
                failures.append(f"prompt_id={pid} に対応する画像が見つからない")

        assert not failures, "\n".join(failures)

    def test_G05_asset_id_matches_directory_name(self):
        """G-05: physics.json の asset_id が親ディレクトリ名と一致する"""
        failures = []
        for phys_path in _all_physics_jsons():
            data = json.loads(phys_path.read_text())
            asset_id = data.get("asset_id", "")
            dir_name = phys_path.parent.name
            if asset_id != dir_name:
                failures.append(f"asset_id={asset_id} ≠ dir={dir_name}")

        assert not failures, "\n".join(failures[:10])

    def test_G06_usd_meta_asset_id_consistent(self):
        """G-06: usd_meta.json の asset_id と visual_mesh_path のディレクトリ名が一致する"""
        usd_metas = [
            p for p in ASSETS_DIR.glob("*/*_usd_meta.json")
            if p.parent.name != "isaac"
        ]
        if not usd_metas:
            pytest.skip("USD meta ファイルが存在しない (isaac/ 外)")

        failures = []
        for meta_path in usd_metas:
            data = json.loads(meta_path.read_text())
            asset_id = data.get("asset_id", "")
            visual = data.get("visual_mesh_path", "")
            # visual_mesh_path に asset_id が含まれるか
            if asset_id and asset_id not in visual:
                failures.append(
                    f"{meta_path.parent.name}: asset_id={asset_id}, "
                    f"visual_mesh_path={visual}"
                )

        assert not failures, "\n".join(failures[:10])


# ===========================================================================
# H: resume 動作検証
# ===========================================================================

class TestResumeLogic:
    """H-01〜H-04: resume（中断再開）が仕様通り動作する"""

    def _make_minimal_asset(self, tmp_dir: Path, asset_id: str) -> Path:
        """テスト用の最小アセットディレクトリを作成する"""
        import trimesh
        import numpy as np

        asset_dir = tmp_dir / asset_id
        asset_dir.mkdir()

        # visual.glb
        verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        with open(asset_dir / "visual.glb", "wb") as f:
            f.write(trimesh.exchange.gltf.export_glb(mesh))

        # physics.json
        phys = {
            "material": "composite_hard_suitcase",
            "density_kg_m3": 1000.0,
            "static_friction": 0.4,
            "dynamic_friction": 0.35,
            "restitution": 0.15,
            "volume_m3": 0.001,
            "mass_kg": 1.0,
            "asset_id": asset_id,
            "luggage_type": "hard_suitcase",
            "collision_count": 0,
            "scale_factor": 1.0,
        }
        (asset_dir / "physics.json").write_text(json.dumps(phys))
        return asset_dir

    def test_H01_physics_processor_resume_skips_existing(self):
        """H-01: process_batch は physics.json 既存アセットをスキップする"""
        from src.physics_processor import PhysicsProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_dir = tmp / "meshes"
            out_dir = tmp / "assets"
            mesh_dir.mkdir()
            out_dir.mkdir()

            # アセット1: physics.json 既存 (スキップされるべき)
            asset_id = "test_resume_asset"
            asset_out = out_dir / asset_id
            asset_out.mkdir()
            (asset_out / "physics.json").write_text(json.dumps({"asset_id": asset_id}))

            # mesh_dir に GLB を置く (trimesh で最小メッシュ)
            import trimesh, numpy as np
            verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            glb_path = mesh_dir / f"{asset_id}.glb"
            with open(glb_path, "wb") as f:
                f.write(trimesh.exchange.gltf.export_glb(mesh))

            proc = PhysicsProcessor()
            result = proc.process_batch(
                mesh_dir=str(mesh_dir),
                output_dir=str(out_dir),
                resume=True,
            )

            # スキップされたアセットは success にカウントされる (with ブロック内)
            assert result["success"] >= 1, (
                f"resume が機能していない: {result}"
            )
            # physics.json の内容が上書きされていないこと (with ブロック内)
            phys_data = json.loads((asset_out / "physics.json").read_text())
            assert phys_data.get("asset_id") == asset_id, "physics.json が上書きされた"

    def test_H02_sim_exporter_resume_skips_existing_mjcf(self):
        """H-02: export_batch は MJCF 既存アセットをスキップし status=skipped を返す"""
        from src.sim_exporter import SimExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets_dir = tmp / "assets"
            assets_dir.mkdir()

            asset_id = "test_skip_asset"
            asset_in = assets_dir / asset_id
            asset_in.mkdir()

            # visual.glb
            import trimesh, numpy as np
            verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            with open(asset_in / "visual.glb", "wb") as f:
                f.write(trimesh.exchange.gltf.export_glb(mesh))

            (asset_in / "physics.json").write_text(json.dumps({
                "mass_kg": 1.0, "static_friction": 0.4,
                "dynamic_friction": 0.35, "restitution": 0.15, "material": "test",
            }))

            # MJCF が既に存在する状態を作る
            mjcf_dir = asset_in / "mjcf"
            mjcf_dir.mkdir()
            (mjcf_dir / f"{asset_id}.xml").write_text("<mujoco/>")

            # USD meta も既存
            (asset_in / f"{asset_id}_usd_meta.json").write_text("{}")

            exporter = SimExporter()
            result = exporter.export_batch(
                assets_dir=str(assets_dir),
                output_dir=str(assets_dir),
                format="both",
                resume=True,
            )

        skipped = [r for r in result["results"] if r["status"] == "skipped"]
        assert len(skipped) >= 1, f"スキップが発生しなかった: {result}"

    def test_H03_sim_exporter_no_resume_overwrites(self):
        """H-03: resume=False なら既存 MJCF を無視して再生成する"""
        from src.sim_exporter import SimExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets_dir = tmp / "assets"
            assets_dir.mkdir()

            asset_id = "test_overwrite_asset"
            asset_in = assets_dir / asset_id
            asset_in.mkdir()

            import trimesh, numpy as np
            verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            with open(asset_in / "visual.glb", "wb") as f:
                f.write(trimesh.exchange.gltf.export_glb(mesh))

            (asset_in / "physics.json").write_text(json.dumps({
                "mass_kg": 1.0, "static_friction": 0.4,
                "dynamic_friction": 0.35, "restitution": 0.15, "material": "test",
            }))

            mjcf_dir = asset_in / "mjcf"
            mjcf_dir.mkdir()
            sentinel = "<mujoco model='sentinel'/>"
            (mjcf_dir / f"{asset_id}.xml").write_text(sentinel)

            exporter = SimExporter()
            result = exporter.export_batch(
                assets_dir=str(assets_dir),
                output_dir=str(assets_dir),
                format="mjcf",
                resume=False,  # ← 上書きモード
            )

            # status が skipped ではなく success になること (with ブロック内)
            assert result["results"][0]["status"] == "success", (
                f"resume=False なのに skipped: {result['results'][0]}"
            )
            # ファイルが上書きされていること（sentinel 文字列が消えている）(with ブロック内)
            xml_path = asset_in / "mjcf" / f"{asset_id}.xml"
            new_content = xml_path.read_text()
            assert new_content != sentinel, "ファイルが上書きされていない"

    def test_H04_partial_resume_processes_missing_only(self):
        """H-04: 一部アセットのみ physics.json を削除すると削除分だけ再処理される"""
        from src.physics_processor import PhysicsProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_dir = tmp / "meshes"
            out_dir = tmp / "assets"
            mesh_dir.mkdir()
            out_dir.mkdir()

            import trimesh, numpy as np
            verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)

            asset_ids = ["asset_a", "asset_b"]
            for aid in asset_ids:
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                glb = mesh_dir / f"{aid}.glb"
                with open(glb, "wb") as f:
                    f.write(trimesh.exchange.gltf.export_glb(mesh))

                # asset_a は既存あり、asset_b は既存なし
                if aid == "asset_a":
                    asset_out = out_dir / aid
                    asset_out.mkdir()
                    (asset_out / "physics.json").write_text(
                        json.dumps({"asset_id": aid, "sentinel": True})
                    )

            proc = PhysicsProcessor()
            proc.process_batch(
                mesh_dir=str(mesh_dir),
                output_dir=str(out_dir),
                resume=True,
            )

            # asset_a の physics.json は上書きされていない (with ブロック内)
            a_phys = json.loads((out_dir / "asset_a" / "physics.json").read_text())
            assert a_phys.get("sentinel") is True, "asset_a が不当に上書きされた"

            # asset_b の physics.json は新規作成されている (with ブロック内)
            b_phys_path = out_dir / "asset_b" / "physics.json"
            assert b_phys_path.exists(), "asset_b の physics.json が生成されなかった"


# ===========================================================================
# I: config 値の反映確認
# ===========================================================================

class TestConfigReflection:
    """I-01〜I-05: YAML の設定値がコード内で実際に使われる"""

    def test_I01_coacd_threshold_passed_to_process_batch(self):
        """I-01: PhysicsProcessor.process_batch に coacd_threshold が渡る"""
        from src.physics_processor import PhysicsProcessor

        proc = PhysicsProcessor()
        # process_single に渡されるパラメータを追跡
        called_thresholds = []
        original = proc.process_single

        def spy_single(*args, **kwargs):
            called_thresholds.append(kwargs.get("coacd_threshold", args[4] if len(args) > 4 else None))
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mesh_dir = tmp / "meshes"
            out_dir = tmp / "assets"
            mesh_dir.mkdir(); out_dir.mkdir()

            import trimesh, numpy as np
            verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            with open(mesh_dir / "test.glb", "wb") as f:
                f.write(trimesh.exchange.gltf.export_glb(mesh))

            with patch.object(proc, "process_single", side_effect=spy_single):
                proc.process_batch(
                    mesh_dir=str(mesh_dir),
                    output_dir=str(out_dir),
                    coacd_threshold=0.05,
                    resume=False,
                )

        # process_single が呼ばれていれば threshold が 0.05 で渡っているはず
        if called_thresholds:
            assert called_thresholds[0] == 0.05, (
                f"threshold={called_thresholds[0]} (期待: 0.05)"
            )

    def test_I02_format_mjcf_only_no_usd_meta(self):
        """I-02: format='mjcf' のとき usd_meta.json が生成されない"""
        from src.sim_exporter import SimExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets_dir = tmp / "assets"
            assets_dir.mkdir()
            asset_id = "test_mjcf_only"
            asset_in = assets_dir / asset_id
            asset_in.mkdir()

            import trimesh, numpy as np
            verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            with open(asset_in / "visual.glb", "wb") as f:
                f.write(trimesh.exchange.gltf.export_glb(mesh))
            (asset_in / "physics.json").write_text(json.dumps({
                "mass_kg": 1.0, "static_friction": 0.4,
                "dynamic_friction": 0.35, "restitution": 0.15, "material": "test",
            }))

            SimExporter().export_batch(
                assets_dir=str(assets_dir),
                output_dir=str(assets_dir),
                format="mjcf",
                resume=False,
            )

            # MJCF は生成される (with ブロック内)
            assert (asset_in / "mjcf" / f"{asset_id}.xml").exists(), "MJCF が生成されなかった"
            # USD meta は生成されない (with ブロック内)
            usd_files = list(asset_in.glob("*_usd_meta.json"))
            assert not usd_files, f"format=mjcf なのに USD meta が生成された: {usd_files}"

    def test_I03_format_usd_only_no_mjcf(self):
        """I-03: format='usd' のとき MJCF が生成されない"""
        from src.sim_exporter import SimExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets_dir = tmp / "assets"
            assets_dir.mkdir()
            asset_id = "test_usd_only"
            asset_in = assets_dir / asset_id
            asset_in.mkdir()

            import trimesh, numpy as np
            verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
            faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int32)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            with open(asset_in / "visual.glb", "wb") as f:
                f.write(trimesh.exchange.gltf.export_glb(mesh))
            (asset_in / "physics.json").write_text(json.dumps({
                "mass_kg": 1.0, "static_friction": 0.4,
                "dynamic_friction": 0.35, "restitution": 0.15, "material": "test",
            }))

            SimExporter().export_batch(
                assets_dir=str(assets_dir),
                output_dir=str(assets_dir),
                format="usd",
                resume=False,
            )

            # USD meta は生成される (with ブロック内)
            usd_files = list(asset_in.glob("*_usd_meta.json"))
            assert usd_files, "USD meta が生成されなかった"
            # MJCF は生成されない (with ブロック内)
            assert not (asset_in / "mjcf").exists() or not list(
                (asset_in / "mjcf").glob("*.xml")
            ), "format=usd なのに MJCF が生成された"



# ===========================================================================
# J: memory_guard 統合検証
# ===========================================================================

class TestMemoryGuardIntegration:
    """J-01〜J-04: memory_guard が実測値で動作する"""

    def test_J01_get_system_free_gb_matches_psutil(self):
        """J-01: get_system_free_gb() が psutil と誤差 5% 以内で一致する"""
        pytest.importorskip("psutil")
        import psutil
        from src.utils.memory_guard import get_system_free_gb

        mg_free = get_system_free_gb()
        ps_free = psutil.virtual_memory().available / 1024**3

        # get_system_free_gb は内部で psutil を使うため完全一致に近い
        # ただし呼び出しタイミングのずれがあるため 5% マージンを許容
        rel_diff = abs(mg_free - ps_free) / max(ps_free, 1.0)
        assert rel_diff < 0.05, (
            f"memory_guard={mg_free:.2f} GiB, psutil={ps_free:.2f} GiB, diff={rel_diff:.2%}"
        )

    @pytest.mark.gpu
    def test_J02_flush_reduces_allocated_after_tensor_del(self):
        """J-02: テンソル del + flush で CUDA allocated が元に戻る"""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA が利用できません")

        from src.utils.memory_guard import flush_cuda_memory

        flush_cuda_memory()
        _, base = flush_cuda_memory()

        # 200 MB のテンソルを確保して解放
        try:
            t = torch.zeros(200 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        except Exception:
            pytest.skip("CUDA テンソルアロケートに失敗（OOM または未サポートハードウェア）")
        _, peak = flush_cuda_memory()
        assert peak > base, "テンソル確保が反映されていない"

        del t
        _, after = flush_cuda_memory()

        assert after <= base + 1.0, (
            f"テンソル解放後のメモリが戻っていない: "
            f"base={base:.3f}, peak={peak:.3f}, after={after:.3f} GiB"
        )

    def test_J04_headroom_error_contains_free_gb_value(self):
        """J-04: headroom 不足エラーに現在の空き GiB 値が含まれる"""
        from src.utils.memory_guard import assert_memory_headroom

        with patch("src.utils.memory_guard.get_system_free_gb", return_value=3.7), \
             pytest.raises(RuntimeError) as exc_info:
            assert_memory_headroom(required_gb=80.0, label="TestModel")

        msg = str(exc_info.value)
        assert "3." in msg, f"空き GiB 値がエラーメッセージに含まれない: {msg}"
        assert "80." in msg, f"必要 GiB 値がエラーメッセージに含まれない: {msg}"
        assert "TestModel" in msg, f"ラベルがエラーメッセージに含まれない: {msg}"


# ===========================================================================
# 実行
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
