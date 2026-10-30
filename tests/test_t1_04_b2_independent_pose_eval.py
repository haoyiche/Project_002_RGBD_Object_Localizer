"""T1-04-B2: fixed-candidate vs separate GT evaluator contract tests.

The synthetic pose in the fixture is NOT a real pose accuracy measurement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.t1_04_b2_independent_pose_eval import (
    evaluate_saved_candidate, main, sha256_file,
)
from src.evaluation.t1_04_a_pose_metrics import axis_angle_rotation


def fixture(tmp_path: Path):
    scene = tmp_path / 'scene'
    models = tmp_path / 'models'
    for folder in ('rgb', 'depth', 'mask_visib'):
        (scene / folder).mkdir(parents=True)
    models.mkdir()
    for rel, contents in {
        'scene_camera.json': b'{"15":{"cam_K":[200,0,8,0,200,8,0,0,1],"depth_scale":1}}',
        'rgb/000015.png': b'rgb-test-byte-fixture',
        'depth/000015.png': b'depth-test-byte-fixture',
        'mask_visib/000015_000000.png': b'mask-test-byte-fixture',
    }.items():
        (scene / rel).write_bytes(contents)
    (models / 'obj_000001.ply').write_bytes(b'ply-test-byte-fixture')
    (models / 'models_info.json').write_bytes(b'{"1":{"size_x":80,"size_y":70,"size_z":90}}')
    gt = [{'obj_id': 1, 'cam_R_m2c': np.eye(3).ravel().tolist(),
           'cam_t_m2c': [20, -10, 500]}]
    (scene / 'scene_gt.json').write_text(json.dumps({'15': gt}), encoding='utf-8')
    rot = axis_angle_rotation([0, 0, 1], 10).ravel().tolist()
    candidate = {
        'schema': 't1-04-b1-oracle-mask-gt-blind-icp-v1',
        'method': 'PCA_24_START_PARTIAL_OVERLAP_POINT_TO_POINT_ICP',
        'status': 'GEOMETRIC_CANDIDATE_UNVERIFIED',
        'estimation_performed': True,
        'gt_pose_read_by_predictor': False,
        'gt_pose_used_to_construct_candidate': False,
        'oracle_mask_used': True,
        'oracle_supervision_used': True,
        'publishable': False,
        'position_m': None,
        'pose_candidate_model_to_camera': {
            'R_row_major': rot, 't_m': [0.023, -0.006, 0.5],
        },
        'observation': {
            'im_id': 15, 'gt_index': 0,
            'oracle_mask_used': True,
            'gt_pose_read_by_predictor': False,
            'sha256': {
                'scene_camera': sha256_file(scene/'scene_camera.json'),
                'rgb': sha256_file(scene/'rgb/000015.png'),
                'depth': sha256_file(scene/'depth/000015.png'),
                'mask_visib': sha256_file(scene/'mask_visib/000015_000000.png'),
            },
        },
        'model': {
            'obj_id': 1, 'model_units_input': 'mm', 'model_units_internal': 'm',
            'mesh_sha256': sha256_file(models/'obj_000001.ply'),
            'models_info_sha256': sha256_file(models/'models_info.json'),
        },
    }
    path = tmp_path / 'fixed_candidate.json'
    path.write_text(json.dumps(candidate), encoding='utf-8')
    return scene, models, path, candidate


def evaluate(scene, models, candidate):
    return evaluate_saved_candidate(
        candidate_path=candidate, scene_dir=scene, models_dir=models,
    )


def test_fixed_candidate_uses_independent_scene_gt_to_measure_both_metrics(tmp_path):
    scene, models, path, _ = fixture(tmp_path)
    r = evaluate(scene, models, path)
    np.testing.assert_allclose(r['translation_error_mm'], 5.0, atol=1e-9)
    np.testing.assert_allclose(r['rotation_error_deg_raw'], 10.0, atol=1e-8)
    assert r['gt_pose_read_by_evaluator'] is True
    assert r['gt_pose_used_for_estimation'] is False
    assert r['symmetry_aware'] is False
    assert r['bop_benchmark_score_claimed'] is False
    assert r['physical_accuracy_claimed'] is False
    assert r['publishable'] is False and r['position_m'] is None
    assert r['candidate_sha256'] == sha256_file(path)


def test_evaluator_refuses_missing_gt(tmp_path):
    scene, models, path, _ = fixture(tmp_path)
    (scene/'scene_gt.json').unlink()
    with pytest.raises(ValueError, match='GT_NOT_AVAILABLE_FOR_EVALUATION'):
        evaluate(scene, models, path)


def test_evaluator_rejects_wrong_gt_object_id(tmp_path):
    scene, models, path, _ = fixture(tmp_path)
    (scene/'scene_gt.json').write_text(json.dumps({'15': [
        {'obj_id': 2, 'cam_R_m2c': np.eye(3).ravel().tolist(),
         'cam_t_m2c': [20, -10, 500]}
    ]}), encoding='utf-8')
    with pytest.raises(ValueError, match='GT_OBJECT_ID_MISMATCH'):
        evaluate(scene, models, path)


@pytest.mark.parametrize('field,relative', [
    ('depth','depth/000015.png'),
    ('rgb','rgb/000015.png'),
    ('mask_visib','mask_visib/000015_000000.png'),
    ('scene_camera','scene_camera.json'),
])
def test_changed_observation_bytes_rejected_before_gt(field, relative, tmp_path):
    scene, models, path, _ = fixture(tmp_path)
    (scene/relative).write_bytes(b'tampered')
    (scene/'scene_gt.json').unlink()
    with pytest.raises(ValueError, match='RECORDED_INPUT_HASH_MISMATCH:' + field):
        evaluate(scene, models, path)


def test_changed_model_bytes_rejected(tmp_path):
    scene, models, path, _ = fixture(tmp_path)
    (models/'obj_000001.ply').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='RECORDED_MODEL_HASH_MISMATCH:mesh'):
        evaluate(scene, models, path)


def test_tampered_candidate_hash_claim_rejected(tmp_path):
    scene, models, path, record = fixture(tmp_path)
    record['observation']['sha256']['depth'] = '0'*64
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='RECORDED_INPUT_HASH_MISMATCH:depth'):
        evaluate(scene, models, path)


def test_candidate_claiming_gt_leakage_is_rejected(tmp_path):
    scene, models, path, record = fixture(tmp_path)
    record['gt_pose_used_to_construct_candidate'] = True
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='GT_LEAKAGE_OR_UNVERIFIABLE_CANDIDATE'):
        evaluate(scene, models, path)


def test_invalid_estimate_rotation_is_rejected(tmp_path):
    scene, models, path, record = fixture(tmp_path)
    record['pose_candidate_model_to_camera']['R_row_major'] = [1,0,0,0,2,0,0,0,1]
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='INVALID_CANDIDATE_POSE'):
        evaluate(scene, models, path)


def test_bad_candidate_status_is_rejected(tmp_path):
    scene, models, path, record = fixture(tmp_path)
    record['status'] = 'PASS_POSE_ACCURACY'
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='INVALID_CANDIDATE_STATUS'):
        evaluate(scene, models, path)


def test_cli_refuses_to_overwrite_existing_report(tmp_path):
    scene, models, path, _ = fixture(tmp_path)
    out = tmp_path/'old_eval.json'
    out.write_text('OLD', encoding='utf-8')
    with pytest.raises(FileExistsError, match='OUTPUT_ALREADY_EXISTS_USE_NEW_RUN_PATH'):
        main(['--candidate', str(path), '--scene-dir', str(scene),
              '--models-dir', str(models), '--output', str(out)])
    assert out.read_text(encoding='utf-8') == 'OLD'


def test_cli_writes_safe_report(tmp_path):
    scene, models, path, _ = fixture(tmp_path)
    out = tmp_path/'fresh_eval.json'
    main(['--candidate', str(path), '--scene-dir', str(scene),
          '--models-dir', str(models), '--output', str(out)])
    r = json.loads(out.read_text(encoding='utf-8'))
    assert r['status'] == 'ANNOTATION_RELATIVE_EVALUATION_DIAGNOSTIC'
    assert r['translation_error_mm'] == pytest.approx(5.0)
    assert r['publishable'] is False and r['position_m'] is None
