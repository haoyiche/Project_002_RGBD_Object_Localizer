"""T1-01 BOP metadata, mm/m, GT instance and fail-closed schema tests."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest

from src.evaluation import t1_01_bop_contract as bop


def make_scene(tmp_path, *, scale=0.1, im_id=0, wrong_R=False):
    scene = tmp_path / 'scene'
    scene.mkdir()
    (scene / 'scene_camera.json').write_text(json.dumps({str(im_id): {
        'cam_K': [100, 0, 1, 0, 100, 1, 0, 0, 1], 'depth_scale': scale,
    }}), encoding='utf-8')
    rotation = [1, 0, 0, 0, 2 if wrong_R else 1, 0, 0, 0, 1]
    (scene / 'scene_gt.json').write_text(json.dumps({str(im_id): [
        {'obj_id': 3, 'cam_R_m2c': rotation, 'cam_t_m2c': [10, 20, 500]},
        {'obj_id': 8, 'cam_R_m2c': list(np.eye(3).reshape(-1)), 'cam_t_m2c': [100, 0, 600]},
    ]}), encoding='utf-8')
    for d, filename in [('rgb', f'{im_id:06d}.png'), ('depth', f'{im_id:06d}.png'),
                        ('mask_visib', f'{im_id:06d}_000000.png'),
                        ('mask_visib', f'{im_id:06d}_000001.png')]:
        directory = scene / d
        directory.mkdir(exist_ok=True)
        (directory / filename).write_bytes(b'fixture-placeholder')
    return scene


def test_bop_depth_is_scaled_twice_to_millimeters_then_meters(tmp_path):
    c = bop.load_bop_frame_contract(make_scene(tmp_path), im_id=0)
    depth, valid = bop.depth_raw_to_m(np.array([[5000, 0]], dtype=np.uint16), c)
    np.testing.assert_allclose(depth, [[.5, 0]])
    assert valid.tolist() == [[True, False]]
    np.testing.assert_allclose(bop.pixel_to_camera_m(u=2, v=1, raw_depth=5000, contract=c), [.005, 0, .5])


def test_gt_translation_is_mm_not_mask_centroid(tmp_path):
    c = bop.load_bop_frame_contract(make_scene(tmp_path), im_id=0)
    np.testing.assert_allclose(c.t_cam_from_model_m, [.01, .02, .5])
    assert c.obj_id == 3
    assert not np.allclose(c.t_cam_from_model_m, [.005, 0, .5])


def test_gt_index_matches_visible_mask_index_not_object_id(tmp_path):
    c = bop.load_bop_frame_contract(make_scene(tmp_path), im_id=0, gt_index=1)
    assert c.obj_id == 8
    assert c.mask_visib_path.name == '000000_000001.png'
    np.testing.assert_allclose(c.t_cam_from_model_m, [.1, 0, .6])


def test_missing_visible_mask_fails_closed(tmp_path):
    scene = make_scene(tmp_path)
    (scene / 'mask_visib' / '000000_000000.png').unlink()
    with pytest.raises(ValueError, match='MISSING_RGB_DEPTH_OR_VISIBLE_MASK'):
        bop.load_bop_frame_contract(scene, im_id=0)


def test_wrong_rotation_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='INVALID_ROTATION'):
        bop.load_bop_frame_contract(make_scene(tmp_path, wrong_R=True), im_id=0)


def test_bad_depth_scale_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='INVALID_DEPTH_SCALE'):
        bop.load_bop_frame_contract(make_scene(tmp_path, scale=0), im_id=0)


def test_unknown_image_or_instance_fails_closed(tmp_path):
    scene = make_scene(tmp_path)
    with pytest.raises(ValueError, match='IMAGE_ID_NOT_FOUND'):
        bop.load_bop_frame_contract(scene, im_id=5)
    with pytest.raises(ValueError, match='GT_INDEX_NOT_FOUND'):
        bop.load_bop_frame_contract(scene, im_id=0, gt_index=2)


def test_zero_depth_pixel_is_rejected(tmp_path):
    c = bop.load_bop_frame_contract(make_scene(tmp_path), im_id=0)
    with pytest.raises(ValueError, match='INVALID_DEPTH_SAMPLE'):
        bop.pixel_to_camera_m(u=1, v=1, raw_depth=0, contract=c)



def test_t1_01_active_three_ablations(tmp_path):
    scene = make_scene(tmp_path)
    base = bop.load_bop_frame_contract(scene, im_id=0)

    # A: 改变深度尺度
    a = replace(base, depth_scale_mm_per_unit=0.2)
    depth_a, _ = bop.depth_raw_to_m(
        np.array([[5000]], dtype=np.uint16), a
    )
    xyz_a = bop.pixel_to_camera_m(
        u=2, v=1, raw_depth=5000, contract=a
    )

    # B: 只改变 fx
    K_b = base.K.copy()
    K_b[0, 0] = 200.0
    b = replace(base, K=K_b)
    xyz_b = bop.pixel_to_camera_m(
        u=2, v=1, raw_depth=5000, contract=b
    )

    # C: 改变 GT 实例索引
    c = bop.load_bop_frame_contract(
        scene, im_id=0, gt_index=1
    )
    xyz_c = bop.pixel_to_camera_m(
        u=2, v=1, raw_depth=5000, contract=c
    )
    # A：depth_scale 改变
    np.testing.assert_allclose(depth_a, [[1.0]], atol=1e-12)
    np.testing.assert_allclose(xyz_a, [0.01, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        a.t_cam_from_model_m,
        base.t_cam_from_model_m,
        atol=1e-12,
    )

    # B：fx 改变
    np.testing.assert_allclose(xyz_b, [0.0025, 0.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(
        b.t_cam_from_model_m,
        base.t_cam_from_model_m,
        atol=1e-12,
    )

    # C：切换 GT 实例
    assert c.obj_id == 8
    assert c.mask_visib_path.name == "000000_000001.png"
    np.testing.assert_allclose(c.t_cam_from_model_m, [0.1, 0.0, 0.6], atol=1e-12)
    np.testing.assert_allclose(xyz_c, [0.005, 0.0, 0.5], atol=1e-12)

    print("A depth_m:", depth_a[0, 0])
    print("A xyz_camera_m:", xyz_a)
    print("B xyz_camera_m:", xyz_b)
    print("C obj_id:", c.obj_id)
    print("C mask:", c.mask_visib_path.name)
    print("C model_origin_camera_m:", c.t_cam_from_model_m)
    print("C pixel_xyz_camera_m:", xyz_c)

    # 根据你的运行前预测，在这里增加
    # np.testing.assert_allclose(...) 等断言