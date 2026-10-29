"""T1-02 unit tests use genuine PNG encoding but SYNTHETIC image contents.

Passing them does NOT count as a real BOP dataset sample.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.evaluation import t1_02_bop_rgbd as t2


def _write_png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), arr)


def make_scene(tmp_path, *, frames=2):
    scene = tmp_path / "000001"
    scene.mkdir()
    cameras, gt = {}, {}
    for i in range(frames):
        cameras[str(i)] = {"cam_K": [100, 0, 1, 0, 100, 1, 0, 0, 1], "depth_scale": 0.1}
        gt[str(i)] = [
            {"obj_id": 3, "cam_R_m2c": np.eye(3).ravel().tolist(),
             "cam_t_m2c": [10, 20, 500]},
            {"obj_id": 8, "cam_R_m2c": np.eye(3).ravel().tolist(),
             "cam_t_m2c": [100, 0, 600]},
        ]
        rgb_bgr = np.zeros((3, 4, 3), dtype=np.uint8)
        rgb_bgr[1, 2] = [0, 12, 244]
        depth = np.zeros((3, 4), dtype=np.uint16)
        depth[1, 2] = 5000 + 10 * i
        mask0 = np.zeros((3, 4), dtype=np.uint8)
        mask0[1, 2] = 255
        mask1 = np.zeros((3, 4), dtype=np.uint8)
        mask1[0, 0] = 255
        _write_png(scene / "rgb" / f"{i:06d}.png", rgb_bgr)
        _write_png(scene / "depth" / f"{i:06d}.png", depth)
        _write_png(scene / "mask_visib" / f"{i:06d}_000000.png", mask0)
        _write_png(scene / "mask_visib" / f"{i:06d}_000001.png", mask1)
    (scene / "scene_camera.json").write_text(json.dumps(cameras), encoding="utf-8")
    (scene / "scene_gt.json").write_text(json.dumps(gt), encoding="utf-8")
    return scene


def test_png_is_decoded_with_real_dtype_and_rgb_order(tmp_path):
    scene = make_scene(tmp_path)
    c, rgb, depth_m, visible, usable = t2.read_instance(scene, im_id=0)
    assert rgb.dtype == np.uint8 and rgb.shape == (3, 4, 3)
    assert rgb[1, 2].tolist() == [244, 12, 0]  # BGR -> RGB
    assert depth_m.dtype == np.float64 and depth_m[1, 2] == pytest.approx(0.5)
    assert visible.sum() == 1 and usable.sum() == 1
    assert c.obj_id == 3


def test_manifest_two_unique_frames_and_hash_verification(tmp_path):
    scene = make_scene(tmp_path, frames=2)
    m = t2.make_manifest(scene, count=2)
    assert m["distinct_rgbd_frame_count"] == 2
    assert [s["im_id"] for s in m["samples"]] == [0, 1]
    assert m["samples"][0]["files"]["depth"]["sha256"] != m["samples"][1]["files"]["depth"]["sha256"]
    assert m["samples"][0]["valid_visible_depth_pixels"] == 1
    assert m["physical_accuracy_claimed"] is False
    assert t2.verify_manifest(m, scene)


def test_changed_png_is_detected_by_manifest(tmp_path):
    scene = make_scene(tmp_path)
    m = t2.make_manifest(scene)
    img_path = scene / "depth" / "000000.png"
    edited = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    edited[1, 2] += 1
    _write_png(img_path, edited)
    with pytest.raises(ValueError, match="MANIFEST_CONTENT_MISMATCH"):
        t2.verify_manifest(m, scene)


def test_8bit_depth_is_rejected(tmp_path):
    scene = make_scene(tmp_path)
    _write_png(scene / "depth" / "000000.png", np.full((3, 4), 255, dtype=np.uint8))
    with pytest.raises(ValueError, match="DEPTH_MUST_BE_UINT16_HW"):
        t2.read_instance(scene, im_id=0)


def test_rgb_depth_shape_mismatch_is_rejected(tmp_path):
    scene = make_scene(tmp_path)
    _write_png(scene / "rgb" / "000000.png", np.zeros((2, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="RGB_DEPTH_MASK_SHAPE_MISMATCH"):
        t2.read_instance(scene, im_id=0)


def test_wrong_mask_instance_without_depth_is_rejected(tmp_path):
    scene = make_scene(tmp_path)
    with pytest.raises(ValueError, match="NO_VALID_DEPTH_IN_VISIBLE_MASK"):
        t2.read_instance(scene, im_id=0, gt_index=1)


def test_corrupted_png_is_rejected(tmp_path):
    scene = make_scene(tmp_path)
    (scene / "depth" / "000000.png").write_bytes(b"not-a-png")
    with pytest.raises(ValueError, match="PNG_DECODE_FAILED:depth"):
        t2.read_instance(scene, im_id=0)


def test_manifest_rejects_insufficient_distinct_frames(tmp_path):
    scene = make_scene(tmp_path, frames=2)
    with pytest.raises(ValueError, match="INSUFFICIENT_DISTINCT_FRAMES"):
        t2.make_manifest(scene, count=50)


def test_scene_camera_gt_ids_must_match(tmp_path):
    scene = make_scene(tmp_path)
    path = scene / "scene_camera.json"
    data = json.loads(path.read_text())
    del data["1"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="CAMERA_GT_IMAGE_IDS_MISMATCH"):
        t2.make_manifest(scene)


def test_mask_must_be_binary_255(tmp_path):
    scene = make_scene(tmp_path)
    _write_png(scene / "mask_visib" / "000000_000000.png", np.full((3, 4), 1, dtype=np.uint8))
    with pytest.raises(ValueError, match="VISIBLE_MASK_NOT_BINARY_0_255"):
        t2.read_instance(scene, im_id=0)


def test_duplicate_content_under_distinct_frame_ids_is_rejected(tmp_path):
    scene = make_scene(tmp_path, frames=2)
    # Same RGB + depth bytes under a different im_id must not count as another frame.
    depth0 = scene / "depth" / "000000.png"
    (scene / "depth" / "000001.png").write_bytes(depth0.read_bytes())
    with pytest.raises(ValueError, match="DUPLICATE_RGBD_FRAME_CONTENT"):
        t2.make_manifest(scene, count=2)


def test_subset_archive_uses_present_pngs_not_all_scene_json_keys(tmp_path):
    scene = make_scene(tmp_path, frames=3)
    # Simulate BOP test_bop19: full scene camera/GT JSON, smaller image subset.
    for subdir, pattern in [("rgb", "000001.png"),
                            ("depth", "000001.png"),
                            ("mask_visib", "000001_000000.png"),
                            ("mask_visib", "000001_000001.png")]:
        (scene / subdir / pattern).unlink()
    manifest = t2.make_manifest(scene, count=2)
    assert [item["im_id"] for item in manifest["samples"]] == [0, 2]


def test_partial_png_pairs_rejected_instead_of_ignored(tmp_path):
    scene = make_scene(tmp_path, frames=2)
    (scene / "depth" / "000001.png").unlink()
    with pytest.raises(ValueError, match="RGB_DEPTH_IMAGE_IDS_MISMATCH"):
        t2.make_manifest(scene, count=1)
