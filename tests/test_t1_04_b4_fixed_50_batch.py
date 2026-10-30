"""B4 two-phase batch contract tests; synthetic file/JSON fixtures only.

These are NOT real LM 6D pose estimates or accuracy claims.
"""
import json

import numpy as np
import pytest

from src.evaluation import t1_04_b4_fixed_50_batch as m
from src.evaluation import t1_04_b_gt_free_icp as b1
from src.evaluation import t1_04_b2_independent_pose_eval as b2


def scene(tmp_path, ids=(15, 19, 40)):
    root = tmp_path / 'scene'
    for folder in ('rgb', 'depth', 'mask_visib'):
        (root / folder).mkdir(parents=True)
    for im_id in ids:
        for folder, filename in (('rgb', f'{im_id:06d}.png'),
                                 ('depth', f'{im_id:06d}.png'),
                                 ('mask_visib', f'{im_id:06d}_000000.png')):
            (root / folder / filename).write_bytes(f'{folder}-{im_id}'.encode())
    return root


def fake_prediction(*, scene_dir, models_dir, im_id, gt_index, obj_id):
    return {
        'schema': 't1-04-b1-oracle-mask-gt-blind-icp-v1',
        'status': 'GEOMETRIC_CANDIDATE_UNVERIFIED',
        'gt_pose_read_by_predictor': False,
        'gt_pose_used_to_construct_candidate': False,
        'oracle_mask_used': True, 'publishable': False, 'position_m': None,
        'observation': {'im_id': im_id, 'gt_index': gt_index},
        'model': {'obj_id': obj_id},
        'registration_diagnostic_only': {'trimmed_rmse_m': 0.003},
    }


def fake_eval(*, candidate_path, scene_dir, models_dir):
    candidate = json.loads(candidate_path.read_text(encoding='utf-8'))
    im_id = candidate['observation']['im_id']
    return {
        'candidate_sha256': b1._sha256(candidate_path),
        'sample': {'im_id': im_id, 'gt_index': 0, 'obj_id': 1},
        'gt_annotation_sha256': 'a'*64,
        'gt_pose_used_for_estimation': False,
        'gt_pose_read_by_evaluator': True,
        'symmetry_aware': False, 'bop_benchmark_score_claimed': False,
        'physical_accuracy_claimed': False,
        'publishable': False, 'position_m': None,
        'translation_error_mm': {15: 1., 19: 3., 40: 7.}[im_id],
        'rotation_error_deg_raw': {15: 10., 19: 20., 40: 30.}[im_id],
    }


def setup_frozen(tmp_path, monkeypatch, ids=(15, 19, 40)):
    root = scene(tmp_path, ids=ids)
    out = tmp_path / 'run'
    models = tmp_path / 'models'
    models.mkdir()
    monkeypatch.setattr(b1, 'make_prediction_report', fake_prediction)
    freeze = m.predict_batch(scene_dir=root, models_dir=models,
                             run_dir=out, count=len(ids))
    return root, models, out, freeze


def test_selection_is_sorted_and_does_not_require_gt(tmp_path):
    root = scene(tmp_path, ids=(40, 15, 19))
    (root / 'scene_gt.json').write_text('CORRUPTED GT', encoding='utf-8')
    assert m.select_rgbd_frames(root, count=2, gt_index=0) == [15, 19]


def test_missing_depth_partner_rejected_before_50_selection(tmp_path):
    root = scene(tmp_path)
    (root / 'depth/000040.png').unlink()
    with pytest.raises(ValueError, match='RGB_DEPTH_IMAGE_IDS_MISMATCH'):
        m.select_rgbd_frames(root, count=2, gt_index=0)


def test_missing_mask_rejected(tmp_path):
    root = scene(tmp_path)
    (root / 'mask_visib/000019_000000.png').unlink()
    with pytest.raises(ValueError, match='ORACLE_MASK_MISSING'):
        m.select_rgbd_frames(root, count=3, gt_index=0)


def test_insufficient_frame_count_rejected(tmp_path):
    root = scene(tmp_path)
    with pytest.raises(ValueError, match='INSUFFICIENT_DISTINCT_FRAMES'):
        m.select_rgbd_frames(root, count=50, gt_index=0)


def test_freeze_has_exact_files_and_candidate_hashes(tmp_path, monkeypatch):
    root, models, out, freeze = setup_frozen(tmp_path, monkeypatch)
    assert [x['im_id'] for x in freeze['candidates']] == [15, 19, 40]
    assert freeze['prediction_uses_scene_gt'] is False
    assert (out / 'frozen_predictions.json').exists()
    assert len(list(out.glob('candidate_*.json'))) == 3
    assert m._load_frozen(out) == freeze
    assert not (out / 'batch_summary.json').exists()


def test_run_directory_reuse_rejected(tmp_path, monkeypatch):
    root, models, out, _ = setup_frozen(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='RUN_DIRECTORY_ALREADY_EXISTS'):
        m.predict_batch(scene_dir=root, models_dir=models, run_dir=out, count=2)


def test_evaluation_refuses_incomplete_freeze(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    with pytest.raises(ValueError, match='INVALID_OR_MISSING_BATCH_JSON'):
        m.evaluate_batch(scene_dir=tmp_path, models_dir=tmp_path, run_dir=out)


def test_changed_candidate_is_rejected_before_gt_access(tmp_path, monkeypatch):
    root, models, out, _ = setup_frozen(tmp_path, monkeypatch)
    (out / 'candidate_000019.json').write_text('{"tampered":true}', encoding='utf-8')
    calls = []
    monkeypatch.setattr(b2, 'evaluate_saved_candidate', lambda **kw: calls.append(kw))
    with pytest.raises(ValueError, match='FROZEN_CANDIDATE_SHA256_MISMATCH'):
        m.evaluate_batch(scene_dir=root, models_dir=models, run_dir=out)
    assert calls == []
    assert not (out / 'batch_summary.json').exists()


def test_changed_code_sha_claim_rejected(tmp_path, monkeypatch):
    root, models, out, _ = setup_frozen(tmp_path, monkeypatch)
    p = out / 'frozen_predictions.json'
    data = json.loads(p.read_text(encoding='utf-8'))
    data['prediction_source_sha256'] = '0'*64
    p.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='PREDICTOR_CODE_CHANGED_SINCE_FREEZE'):
        m.evaluate_batch(scene_dir=root, models_dir=models, run_dir=out)


def test_full_scope_metrics_and_interpolation(tmp_path, monkeypatch):
    root, models, out, _ = setup_frozen(tmp_path, monkeypatch)
    monkeypatch.setattr(b2, 'evaluate_saved_candidate', fake_eval)
    summary = m.evaluate_batch(scene_dir=root, models_dir=models, run_dir=out)
    assert summary['frame_count_expected'] == summary['frame_count_evaluated'] == 3
    assert summary['failed_count'] == 0
    assert summary['is_50_frame_scope'] is False
    np.testing.assert_allclose(summary['translation_error_mm']['mean'], 11/3)
    np.testing.assert_allclose(summary['translation_error_mm']['median'], 3)
    np.testing.assert_allclose(summary['translation_error_mm']['p95_linear'], 6.6)
    assert summary['physical_accuracy_claimed'] is False
    assert summary['publishable'] is False and summary['position_m'] is None
    assert len(list(out.glob('evaluation_*.json'))) == 3
    assert (out / 'batch_summary.json').exists()


def test_evaluation_failure_records_error_not_partial_p95(tmp_path, monkeypatch):
    root, models, out, _ = setup_frozen(tmp_path, monkeypatch)
    def fail_second(**kw):
        if '000019' in str(kw['candidate_path']):
            raise ValueError('GT_OBJECT_ID_MISMATCH')
        return fake_eval(**kw)
    monkeypatch.setattr(b2, 'evaluate_saved_candidate', fail_second)
    with pytest.raises(ValueError, match='GT_OBJECT_ID_MISMATCH'):
        m.evaluate_batch(scene_dir=root, models_dir=models, run_dir=out)
    err = json.loads((out / 'batch_failure.json').read_text(encoding='utf-8'))
    assert err['status'] == 'INCOMPLETE_BATCH_NOT_REPORTABLE'
    assert err['failed_frame_id'] == 19
    assert err['completed_before_failure'] == 1
    assert not (out / 'batch_summary.json').exists()
    with pytest.raises(ValueError, match='INCOMPLETE_BATCH_FAILED'):
        m.evaluate_batch(scene_dir=root, models_dir=models, run_dir=out)


def test_prediction_failure_records_no_freeze(tmp_path, monkeypatch):
    root = scene(tmp_path)
    models = tmp_path / 'models'
    out = tmp_path / 'run'
    def fail_first(**kw):
        raise ValueError('MODEL_SIZE_MM_MISMATCH')
    monkeypatch.setattr(b1, 'make_prediction_report', fail_first)
    with pytest.raises(ValueError, match='MODEL_SIZE_MM_MISMATCH'):
        m.predict_batch(scene_dir=root, models_dir=models, run_dir=out, count=3)
    assert not (out / 'frozen_predictions.json').exists()
    assert not (out / 'batch_summary.json').exists()
    assert json.loads((out / 'batch_failure.json').read_text())['phase'] == 'PREDICTION'


def test_summary_must_not_overwrite(tmp_path, monkeypatch):
    root, models, out, _ = setup_frozen(tmp_path, monkeypatch)
    (out / 'batch_summary.json').write_text('OLD', encoding='utf-8')
    monkeypatch.setattr(b2, 'evaluate_saved_candidate', fake_eval)
    with pytest.raises(ValueError, match='OUTPUT_ALREADY_EXISTS'):
        m.evaluate_batch(scene_dir=root, models_dir=models, run_dir=out)
    assert (out / 'batch_summary.json').read_text(encoding='utf-8') == 'OLD'


def test_invalid_error_metric_fails_closed(tmp_path, monkeypatch):
    root, models, out, _ = setup_frozen(tmp_path, monkeypatch)
    def bad_metric(**kw):
        ev = fake_eval(**kw)
        ev['translation_error_mm'] = float('nan')
        return ev
    monkeypatch.setattr(b2, 'evaluate_saved_candidate', bad_metric)
    with pytest.raises(ValueError, match='INVALID_ERROR_METRICS'):
        m.evaluate_batch(scene_dir=root, models_dir=models, run_dir=out)
    assert not (out/'batch_summary.json').exists()


def test_stats_rejects_negative():
    with pytest.raises(ValueError, match='INVALID_ERROR_METRICS'):
        m._stats([1.0, -1.0])


def test_exact_50_scope_linear_p95_requires_all_50(tmp_path, monkeypatch):
    ids = tuple(range(100, 150))
    root, models, out, freeze = setup_frozen(tmp_path, monkeypatch, ids=ids)
    def eval_50(*, candidate_path, scene_dir, models_dir):
        c = json.loads(candidate_path.read_text(encoding='utf-8'))
        im_id = c['observation']['im_id']
        return {
            'candidate_sha256': b1._sha256(candidate_path),
            'sample': {'im_id': im_id, 'gt_index': 0, 'obj_id': 1},
            'gt_annotation_sha256': 'b'*64,
            'gt_pose_used_for_estimation': False,
            'gt_pose_read_by_evaluator': True,
            'symmetry_aware': False, 'bop_benchmark_score_claimed': False,
            'physical_accuracy_claimed': False,
            'publishable': False, 'position_m': None,
            'translation_error_mm': float(im_id - 100),
            'rotation_error_deg_raw': float(im_id - 100),
        }
    monkeypatch.setattr(b2, 'evaluate_saved_candidate', eval_50)
    summary = m.evaluate_batch(scene_dir=root, models_dir=models, run_dir=out)
    assert summary['is_50_frame_scope'] is True
    assert summary['frame_count_expected'] == summary['frame_count_evaluated'] == 50
    assert len(summary['samples']) == 50
    assert summary['translation_error_mm']['p95_linear'] == pytest.approx(46.55)
    assert summary['rotation_error_deg_raw']['max'] == 49.0
    assert summary['status'] == 'COMPLETE_SCOPE_ANNOTATION_RELATIVE_DIAGNOSTIC_ONLY'
