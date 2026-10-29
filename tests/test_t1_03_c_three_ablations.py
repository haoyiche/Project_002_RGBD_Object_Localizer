"""T1-03-C numerical invariants on encoded-PNG synthetic fixtures."""
import json

import cv2
import numpy as np
import pytest

from src.evaluation.t1_02_bop_rgbd import make_manifest
from src.evaluation.t1_03_c_three_ablations import three_ablations, main


def _fixture(tmp_path):
    root = tmp_path / "000001"
    root.mkdir()
    K = [100, 0, 2, 0, 100, 2, 0, 0, 1]
    (root / "scene_camera.json").write_text(json.dumps({"0": {"cam_K": K, "depth_scale": .1}}))
    (root / "scene_gt.json").write_text(json.dumps({"0": [
        {"obj_id": 3, "cam_R_m2c": np.eye(3).ravel().tolist(), "cam_t_m2c": [10, 20, 500]}
    ]}))
    rgb = np.zeros((6, 6, 3), dtype=np.uint8)
    depth = (5000 + np.arange(36, dtype=np.uint16) * 10).reshape(6, 6)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:5, 1:5] = 255  # 16 visible pixels
    depth[1, 1] = 0  # 15 valid visible pixels
    for folder, name, arr in (
        ("rgb", "000000.png", rgb),
        ("depth", "000000.png", depth),
        ("mask_visib", "000000_000000.png", mask),
    ):
        target = root / folder / name
        target.parent.mkdir()
        assert cv2.imwrite(str(target), arr)
    return root, make_manifest(root, count=1)


def test_A_positive_depth_bias_changes_centroid_z_not_z_extent(tmp_path):
    root, manifest = _fixture(tmp_path)
    r = three_ablations(manifest, root)
    base, a = r["baseline"], r["A_depth_bias"]
    assert a["point_count"] == base["point_count"] == 15
    assert a["centroid_camera_m"][2] - base["centroid_camera_m"][2] == pytest.approx(.01, abs=1e-12)
    assert a["aabb_extent_camera_m"][2] == pytest.approx(base["aabb_extent_camera_m"][2], abs=1e-12)
    assert a["depth_min_m"] - base["depth_min_m"] == pytest.approx(.01)
    assert a["depth_max_m"] - base["depth_max_m"] == pytest.approx(.01)


def test_B_fx_multiplier_changes_only_x_geometry(tmp_path):
    root, manifest = _fixture(tmp_path)
    r = three_ablations(manifest, root)
    base, b = r["baseline"], r["B_fx_change"]
    assert b["point_count"] == base["point_count"]
    np.testing.assert_allclose(b["centroid_camera_m"][0], base["centroid_camera_m"][0] / 1.1, atol=1e-12)
    np.testing.assert_allclose(b["aabb_extent_camera_m"][0], base["aabb_extent_camera_m"][0] / 1.1, atol=1e-12)
    np.testing.assert_allclose(b["centroid_camera_m"][1:], base["centroid_camera_m"][1:], atol=1e-12)
    np.testing.assert_allclose(b["aabb_extent_camera_m"][1:], base["aabb_extent_camera_m"][1:], atol=1e-12)


def test_C_depth_dropout_reduces_count_and_never_expands_bbox(tmp_path):
    root, manifest = _fixture(tmp_path)
    r = three_ablations(manifest, root)
    base, c = r["baseline"], r["C_depth_dropout"]
    assert r["dropout_removed_count"] == 1
    assert c["point_count"] == base["point_count"] - 1
    assert np.all(np.asarray(c["aabb_extent_camera_m"]) <= np.asarray(base["aabb_extent_camera_m"]) + 1e-12)
    np.testing.assert_allclose(r["gt_model_origin_camera_m"], [.01, .02, .5])
    assert r["physical_accuracy_claimed"] is False
    assert r["pose_accuracy_claimed"] is False
    assert r["publishable"] is False and r["position_m"] is None


def test_invalid_perturbation_parameters_fail_closed(tmp_path):
    root, manifest = _fixture(tmp_path)
    for kwargs, code in [
        ({"depth_bias_m": 0}, "INVALID_DEPTH_BIAS"),
        ({"fx_multiplier": float("nan")}, "INVALID_FX_MULTIPLIER"),
        ({"dropout_stride": 1}, "INVALID_DROPOUT_STRIDE"),
    ]:
        with pytest.raises(ValueError, match=code):
            three_ablations(manifest, root, **kwargs)


def test_tampered_manifest_fails_before_perturbation(tmp_path):
    root, manifest = _fixture(tmp_path)
    manifest["samples"][0]["files"]["depth"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="MANIFEST_CONTENT_MISMATCH"):
        three_ablations(manifest, root)


def test_cli_outputs_diagnostic_only_report(tmp_path, capsys):
    root, manifest = _fixture(tmp_path)
    mp, output = tmp_path / "manifest.json", tmp_path / "out.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    main(["--scene-dir", str(root), "--manifest", str(mp), "--output", str(output)])
    r = json.loads(output.read_text(encoding="utf-8"))
    assert r["schema"] == "t1-03-c-geometry-perturbations-v1"
    assert r["baseline"]["point_count"] == 15
    assert r["publishable"] is False and r["position_m"] is None
    assert "PASS" in capsys.readouterr().out
