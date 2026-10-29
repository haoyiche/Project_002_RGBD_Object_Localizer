"""T1-03-B: encoded-PNG synthetic fixtures; real 50 frames verified on learner PC."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.evaluation import t1_02_bop_rgbd as t2
from src.evaluation import t1_03_b_real_geometry_batch as t3


def _png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), arr)


def _fixture(tmp_path):
    root = tmp_path / "000001"
    root.mkdir()
    cam, gt = {}, {}
    for im_id in (0, 1):
        cam[str(im_id)] = {"cam_K": [100, 0, 1, 0, 100, 1, 0, 0, 1], "depth_scale": .1}
        gt[str(im_id)] = [{"obj_id": 3, "cam_R_m2c": np.eye(3).ravel().tolist(),
                           "cam_t_m2c": [10, 20, 500]}]
        bgr = np.zeros((3, 4, 3), dtype=np.uint8)
        depth = np.zeros((3, 4), dtype=np.uint16)
        mask = np.zeros((3, 4), dtype=np.uint8)
        # Valid pixels: u=1,2 at v=1, with depths .5/.6 m on first image.
        depth[1, 1] = 5000 + im_id * 10
        depth[1, 2] = 6000 + im_id * 10
        mask[1, 1:3] = 255
        mask[0, 0] = 255  # visible but invalid depth
        _png(root / "rgb" / f"{im_id:06d}.png", bgr)
        _png(root / "depth" / f"{im_id:06d}.png", depth)
        _png(root / "mask_visib" / f"{im_id:06d}_000000.png", mask)
    (root / "scene_camera.json").write_text(json.dumps(cam), encoding="utf-8")
    (root / "scene_gt.json").write_text(json.dumps(gt), encoding="utf-8")
    return root, t2.make_manifest(root, count=2)


def test_two_frame_geometry_and_reference_semantics(tmp_path):
    root, manifest = _fixture(tmp_path)
    report = t3.analyze_manifest_geometry(manifest, root)
    assert report["frame_count"] == 2
    first = report["samples"][0]
    assert first["visible_mask_pixels"] == 3
    assert first["valid_visible_depth_pixels"] == 2
    assert first["invalid_visible_depth_pixels"] == 1
    assert first["valid_visible_depth_ratio"] == pytest.approx(2 / 3)
    np.testing.assert_allclose(first["centroid_camera_m"], [.003, 0, .55], atol=1e-12)
    np.testing.assert_allclose(first["bbox_extent_camera_m"], [.006, 0, .1], atol=1e-12)
    np.testing.assert_allclose(first["bbox_center_camera_m"], [.003, 0, .55], atol=1e-12)
    assert report["gt_translation_semantics"] == "MODEL_ORIGIN_NOT_VISIBLE_CLOUD_CENTROID"
    assert "pose_error_mm" not in first
    assert report["physical_accuracy_claimed"] is False
    assert report["pose_accuracy_claimed"] is False
    assert report["position_m"] is None and report["publishable"] is False


def test_frame_two_is_not_identical_to_frame_one(tmp_path):
    root, manifest = _fixture(tmp_path)
    a, b = t3.analyze_manifest_geometry(manifest, root)["samples"]
    assert a["im_id"] != b["im_id"]
    assert b["centroid_camera_m"][2] > a["centroid_camera_m"][2]
    np.testing.assert_allclose(b["centroid_camera_m"], [.003005, 0, .551], atol=1e-12)


def test_tampered_manifest_is_rejected(tmp_path):
    root, manifest = _fixture(tmp_path)
    manifest["samples"][0]["files"]["depth"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="MANIFEST_CONTENT_MISMATCH"):
        t3.analyze_manifest_geometry(manifest, root)


def test_changed_depth_file_is_rejected_before_geometry(tmp_path):
    root, manifest = _fixture(tmp_path)
    path = root / "depth" / "000000.png"
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth[1, 1] += 1
    _png(path, depth)
    with pytest.raises(ValueError, match="MANIFEST_CONTENT_MISMATCH"):
        t3.analyze_manifest_geometry(manifest, root)


def test_manifest_count_mismatch_rejects(tmp_path):
    root, manifest = _fixture(tmp_path)
    manifest["distinct_rgbd_frame_count"] = 99
    with pytest.raises(ValueError, match="MANIFEST_COUNT_MISMATCH"):
        t3.analyze_manifest_geometry(manifest, root)


def test_cli_writes_diagnostic_only_report(tmp_path, capsys):
    root, manifest = _fixture(tmp_path)
    mp = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    t3.main(["--scene-dir", str(root), "--manifest", str(mp), "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["frame_count"] == 2
    assert result["source_manifest_sha256"] == t2.sha256_file(mp)
    assert result["publishable"] is False and result["position_m"] is None
    assert "PASS" in capsys.readouterr().out
