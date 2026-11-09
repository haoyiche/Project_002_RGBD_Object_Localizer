"""CPU-only fail-closed contract tests for T1-06-A fixed-frame runner."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation import t1_06_a_fp_fixed_batch as batch


@pytest.fixture
def inputs(tmp_path):
    scene = tmp_path / "scene"
    models = tmp_path / "models"
    fp_root = tmp_path / "upstream"
    cache = tmp_path / "cache"
    for p in (scene / "rgb", scene / "depth", scene / "mask_visib", models, fp_root, cache):
        p.mkdir(parents=True, exist_ok=True)
    for im_id in (15, 25, 55):
        name = f"{im_id:06d}"
        (scene / "rgb" / f"{name}.png").write_bytes(f"rgb:{name}".encode())
        (scene / "depth" / f"{name}.png").write_bytes(f"depth:{name}".encode())
        (scene / "mask_visib" / f"{name}_000000.png").write_bytes(f"mask:{name}".encode())
    (scene / "scene_camera.json").write_text("{}", encoding="utf-8")
    (scene / "scene_gt.json").write_text("THIS_FILE_MUST_NEVER_BE_READ_OR_COPIED", encoding="utf-8")
    (models / "obj_000001.ply").write_bytes(b"model-ply")
    (models / "models_info.json").write_text("{}", encoding="utf-8")
    for rel in (
        "weights/2023-10-28-18-33-37/config.yml",
        "weights/2023-10-28-18-33-37/model_best.pth",
        "weights/2024-01-11-20-02-45/config.yml",
        "weights/2024-01-11-20-02-45/model_best.pth",
    ):
        p = fp_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(rel.encode())
    adapter = tmp_path / "adapter.py"
    posthoc = tmp_path / "posthoc.py"
    adapter.write_bytes(b"GT-blind-adapter-stub")
    posthoc.write_bytes(b"GT-blind-freeze-stub")
    args = SimpleNamespace(
        count=3,
        scene_dir=scene,
        models_dir=models,
        fp_root=fp_root,
        cache_root=cache,
        adapter=adapter,
        posthoc=posthoc,
        obj_id=1,
        gt_index=0,
        iteration=5,
        run_dir=tmp_path / "run",
    )
    return args


def test_selection_matches_b4_and_does_not_include_gt(inputs):
    plan = batch.make_plan(inputs)
    assert plan["frame_ids"] == [15, 25, 55]
    assert plan["count"] == 3
    assert plan["gt_read_during_prediction"] is False
    assert all(set(record) == {
        "rgb", "depth", "mask_visib", "scene_camera", "model_mesh", "models_info"
    } for record in plan["frame_input_sha256"].values())
    assert "scene_gt" not in str(plan)


def test_bundle_contains_exact_six_gt_blind_files(inputs):
    plan = batch.make_plan(inputs)
    base = inputs.run_dir / "input_bundles"
    base.mkdir(parents=True)
    bundle = batch.prepare_bundle(inputs, 55, inputs.run_dir, plan["frame_input_sha256"]["55"])
    files = [p for p in bundle.rglob("*") if p.is_file()]
    assert len(files) == 6
    assert not (bundle / "scene_gt.json").exists()


def test_hash_tampering_rejected(inputs):
    plan = batch.make_plan(inputs)
    (inputs.scene_dir / "rgb" / "000055.png").write_bytes(b"TAMPERED")
    with pytest.raises(RuntimeError, match="FROZEN_SELECTION_OR_SOURCE_CHANGED"):
        batch.verify_selection(plan, batch.make_plan(inputs))


def test_modified_bundle_rejected(inputs):
    plan = batch.make_plan(inputs)
    (inputs.run_dir / "input_bundles").mkdir(parents=True)
    hashes = plan["frame_input_sha256"]["55"]
    bundle = batch.prepare_bundle(inputs, 55, inputs.run_dir, hashes)
    (bundle / "rgb" / "000055.png").write_bytes(b"TAMPERED")
    with pytest.raises(ValueError, match="BUNDLE_HASH_MISMATCH"):
        batch.prepare_bundle(inputs, 55, inputs.run_dir, hashes)


def test_partial_frames_cannot_pass_audit(inputs):
    plan = batch.make_plan(inputs)
    inputs.run_dir.mkdir()
    with pytest.raises(RuntimeError, match="FRAME_NOT_FROZEN"):
        batch.collect(inputs, plan, allow_partial=False)
