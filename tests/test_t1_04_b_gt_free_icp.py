"""T1-04-B1 GT-blind pose-hypothesis tests; PNGs are synthetic fixtures."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.evaluation import t1_04_b_gt_free_icp as m
from src.evaluation.t1_04_a_pose_metrics import axis_angle_rotation


def make_scene(tmp_path, *, missing_depth=False, depth_is_uint8=False,
               zero_mask=False):
    root = tmp_path / 'scene'
    for name in ('rgb', 'depth', 'mask_visib'):
        (root / name).mkdir(parents=True)
    # Deliberately NO scene_gt.json. Predictor must not use it.
    (root / 'scene_camera.json').write_text(json.dumps({
        '15': {'cam_K': [200, 0, 8, 0, 220, 7, 0, 0, 1], 'depth_scale': 1.0}
    }), encoding='utf-8')
    rows, cols = np.indices((16, 18))
    dep = (1000 + (rows*cols) // 7).astype(np.uint16)
    if depth_is_uint8:
        dep = dep.astype(np.uint8)
    mask = np.full(dep.shape, 255, dtype=np.uint8)
    if zero_mask:
        mask[:] = 0
    cv2.imwrite(str(root / 'rgb/000015.png'), np.full((16,18,3), 40, dtype=np.uint8))
    if not missing_depth:
        cv2.imwrite(str(root / 'depth/000015.png'), dep)
    cv2.imwrite(str(root / 'mask_visib/000015_000000.png'), mask)
    return root


def test_model_mm_to_m_contract():
    v = np.array([[x,y,z] for x in (0,80) for y in (0,70) for z in (0,90)], dtype=float)
    v = np.repeat(v, 5, axis=0)
    out = m.check_and_scale_model(v, {'size_x':80, 'size_y':70, 'size_z':90})
    np.testing.assert_allclose(np.ptp(out,axis=0), [0.080, 0.070, 0.090])


def test_model_extent_mismatch_rejected():
    x = np.arange(90, dtype=float).reshape(30,3)
    with pytest.raises(ValueError, match='MODEL_SIZE_MM_MISMATCH'):
        m.check_and_scale_model(x, {'size_x':999, 'size_y':999, 'size_z':999})


def test_prediction_observation_works_without_scene_gt(tmp_path):
    root = make_scene(tmp_path)
    assert not (root/'scene_gt.json').exists()
    points, info = m.read_oracle_observation(root, im_id=15, gt_index=0)
    assert points.shape == (16*18, 3)
    assert info['oracle_mask_used'] is True
    assert info['gt_pose_read_by_predictor'] is False
    assert len(info['sha256']['depth']) == 64
    assert np.all(points[:,2] > 0)


def test_prediction_full_pipeline_never_needs_gt(tmp_path, monkeypatch):
    root = make_scene(tmp_path)
    # Corrupted GT makes any accidental GT usage fail.
    (root/'scene_gt.json').write_text('THIS MUST NEVER BE READ', encoding='utf-8')
    points, _ = m.read_oracle_observation(root, im_id=15, gt_index=0)
    # Only model is replaced with synthetic fixture in this test.
    model = points - [0.0, 0.0, 1.0]
    monkeypatch.setattr(m, 'load_model_points', lambda models_dir, obj_id: (
        model, {'obj_id':obj_id, 'model_units_input':'mm','model_units_internal':'m'}
    ))
    report = m.make_prediction_report(
        scene_dir=root,models_dir=tmp_path,im_id=15,gt_index=0,obj_id=1)
    assert report['gt_pose_read_by_predictor'] is False
    assert report['gt_pose_used_to_construct_candidate'] is False
    assert report['oracle_supervision_used'] is True
    assert report['status'] == 'GEOMETRIC_CANDIDATE_UNVERIFIED'
    assert report['publishable'] is False
    assert report['position_m'] is None
    assert report['pose_accuracy_claimed'] is False
    assert len(report['pose_candidate_model_to_camera']['R_row_major']) == 9


def test_registration_recovers_rigid_full_overlap_synthetic():
    rng = np.random.default_rng(1202)
    model = rng.normal(size=(250,3)) * [0.022, 0.035, 0.012]
    R = axis_angle_rotation([0.25,0.33,1.0], 19)
    t = np.array([0.12,-0.04,1.04])
    obs = model @ R.T + t
    estimate, audit = m.fit_pca_multistart_icp(model, obs, max_iterations=12)
    np.testing.assert_allclose(estimate.t_m, t, atol=1e-5)
    np.testing.assert_allclose(estimate.R, R, atol=1e-5)
    assert audit['trimmed_rmse_m'] < 1e-6
    assert audit['support_fraction'] == pytest.approx(1)


def test_missing_depth_fails_closed(tmp_path):
    root = make_scene(tmp_path, missing_depth=True)
    with pytest.raises(ValueError, match='MISSING_RGB_DEPTH_OR_ORACLE_MASK'):
        m.read_oracle_observation(root, im_id=15, gt_index=0)


def test_wrong_depth_dtype_fails_closed(tmp_path):
    root = make_scene(tmp_path, depth_is_uint8=True)
    with pytest.raises(ValueError, match='DEPTH_MUST_BE_UINT16'):
        m.read_oracle_observation(root, im_id=15, gt_index=0)


def test_empty_mask_fails_closed(tmp_path):
    root = make_scene(tmp_path, zero_mask=True)
    with pytest.raises(ValueError, match='INSUFFICIENT_VISIBLE_DEPTH_POINTS'):
        m.read_oracle_observation(root, im_id=15, gt_index=0)


def test_invalid_icp_params_rejected():
    model = np.arange(120,dtype=float).reshape(40,3)
    with pytest.raises(ValueError, match='INVALID_ICP_PARAMETERS'):
        m.fit_pca_multistart_icp(model, model, max_iterations=0)


def test_correct_direction_pose_and_no_gt_synthetic():
    rng = np.random.default_rng(190)
    a = rng.normal(size=(60,3)) * [0.03, 0.01, 0.02]
    R = axis_angle_rotation([0,0,1], 7)
    b = a @ R.T + [0,0,1]
    pose, _ = m.fit_pca_multistart_icp(a,b, max_iterations=6)
    assert pose.source_frame == 'model'
    assert pose.target_frame == 'camera'
    assert not pose.R.flags.writeable


def test_cli_refuses_existing_output_before_estimation(tmp_path):
    old = tmp_path / 'old_prediction.json'
    old.write_text('{"old":true}', encoding='utf-8')
    with pytest.raises(FileExistsError, match='OUTPUT_ALREADY_EXISTS_USE_NEW_RUN_PATH'):
        m.main(['--scene-dir', str(tmp_path/'nonexistent_scene'),
                '--models-dir', str(tmp_path/'nonexistent_models'),
                '--im-id', '15', '--obj-id', '1', '--output', str(old)])
    assert old.read_text(encoding='utf-8') == '{"old":true}'
