"""T1-05-A tests: a heuristic is never mistaken for independent pose accuracy."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import t1_04_b_gt_free_icp as b1
from src.evaluation import t1_04_b4_fixed_50_batch as b4
from src.evaluation.t1_05_a_gt_free_quality_gate import (
    GATE_SCHEMA, GateThresholds, PASS, REVIEW,
    inspect_candidate, inspect_frozen_batch,
)


def candidate(im_id=15, *, support=0.83, rmse=0.0035, support_threshold=0.008):
    return {
        'schema': 't1-04-b1-oracle-mask-gt-blind-icp-v1',
        'status': 'GEOMETRIC_CANDIDATE_UNVERIFIED',
        'gt_pose_read_by_predictor': False,
        'gt_pose_used_to_construct_candidate': False,
        'oracle_mask_used': True,
        'publishable': False, 'position_m': None,
        'pose_candidate_model_to_camera': {
            'R_row_major': np.eye(3).ravel().tolist(),
            't_m': [0, 0, 1.0],
        },
        'registration_diagnostic_only': {
            'support_fraction': support,
            'trimmed_rmse_m': rmse,
            'support_threshold_m': support_threshold,
        },
        'observation': {'im_id': im_id, 'gt_index': 0},
        'model': {'obj_id': 1},
    }


def test_pass_is_heuristic_only_never_accuracy():
    x = inspect_candidate(candidate(), GateThresholds())
    assert x['decision'] == PASS
    assert x['pose_accuracy_verified'] is False
    assert x['gt_pose_read_by_gate'] is False


def test_low_support_flagged_even_when_rmse_small():
    x = inspect_candidate(candidate(support=0.5, rmse=0.0001), GateThresholds())
    assert x['decision'] == REVIEW
    assert x['reasons'] == ['SUPPORT_FRACTION_BELOW_THRESHOLD']


def test_high_rmse_flagged_even_with_high_support():
    x = inspect_candidate(candidate(support=0.95, rmse=0.009), GateThresholds())
    assert x['reasons'] == ['TRIMMED_RMSE_ABOVE_THRESHOLD']


def test_both_failures_are_recorded():
    x = inspect_candidate(candidate(support=0.2, rmse=0.01), GateThresholds())
    assert len(x['reasons']) == 2


def test_stricter_min_support_cannot_turn_review_into_pass():
    c = candidate(support=0.83)
    assert inspect_candidate(c, GateThresholds(min_support_fraction=0.8))['decision'] == PASS
    assert inspect_candidate(c, GateThresholds(min_support_fraction=0.9))['decision'] == REVIEW


def test_stricter_rmse_cannot_turn_review_into_pass():
    c = candidate(rmse=0.0035)
    assert inspect_candidate(c, GateThresholds(max_trimmed_rmse_m=0.004))['decision'] == PASS
    assert inspect_candidate(c, GateThresholds(max_trimmed_rmse_m=0.003))['decision'] == REVIEW


def test_threshold_mismatch_is_rejected_not_recomputed():
    with pytest.raises(ValueError, match='SUPPORT_METRIC_THRESHOLD_MISMATCH'):
        inspect_candidate(candidate(), GateThresholds(support_threshold_m=0.004))


@pytest.mark.parametrize('field,bad', [
    ('support_fraction', float('nan')),
    ('support_fraction', 1.2),
    ('trimmed_rmse_m', -0.1),
    ('trimmed_rmse_m', float('inf')),
])
def test_bad_metric_fails_closed(field, bad):
    c = candidate()
    c['registration_diagnostic_only'][field] = bad
    with pytest.raises(ValueError):
        inspect_candidate(c, GateThresholds())


def test_illegal_so3_rejected():
    c = candidate()
    c['pose_candidate_model_to_camera']['R_row_major'] = (-np.eye(3)).ravel().tolist()
    with pytest.raises(ValueError):
        inspect_candidate(c, GateThresholds())


def test_gt_leakage_flag_rejected():
    c = candidate()
    c['gt_pose_read_by_predictor'] = True
    with pytest.raises(ValueError, match='CANDIDATE_SAFETY_CONTRACT_VIOLATION'):
        inspect_candidate(c, GateThresholds())


def test_hidden_gt_field_rejected():
    c = candidate()
    c['translation_error_mm'] = 0.0
    with pytest.raises(ValueError, match='GT_FIELD_IN_PREDICTION_DOCUMENT'):
        inspect_candidate(c, GateThresholds())


def test_invalid_gate_thresholds_rejected():
    for args in [(-0.1, 0.004, 0.008), (0.8, 0, 0.008),
                 (0.8, 0.004, -0.01), (float('nan'), 0.004, 0.008)]:
        with pytest.raises(ValueError):
            GateThresholds(*args)


def _frozen_fixture(tmp_path):
    root = tmp_path / 'old_b4_run'
    root.mkdir()
    records = []
    for im_id, s in ((15, 0.83), (19, 0.70)):
        p = root / f'candidate_{im_id:06d}.json'
        p.write_text(json.dumps(candidate(im_id, support=s)), encoding='utf-8')
        records.append({'im_id': im_id, 'candidate_file': p.name,
                        'candidate_sha256': b1._sha256(p)})
    freeze = {
        'schema': b4.FREEZE_SCHEMA,
        'status': 'ALL_CANDIDATES_FROZEN_BEFORE_GT_EVALUATION',
        'frame_count': 2, 'obj_id': 1, 'gt_index': 0,
        'prediction_source_sha256': b1._sha256(Path(b1.__file__)),
        'prediction_uses_scene_gt': False, 'oracle_mask_used': True,
        'publishable': False, 'position_m': None,
        'candidates': records,
    }
    (root / 'frozen_predictions.json').write_text(json.dumps(freeze), encoding='utf-8')
    return root


def test_gate_frozen_batch_counts_all_and_writes_safe_report(tmp_path):
    root = _frozen_fixture(tmp_path)
    out = tmp_path / 'result.json'
    report = inspect_frozen_batch(frozen_dir=root, thresholds=GateThresholds(), output=out)
    assert report['schema'] == GATE_SCHEMA
    assert report['frame_count'] == 2
    assert report['pass_heuristic_count'] == 1
    assert report['review_count'] == 1
    assert report['pass_heuristic_fraction'] == .5
    assert report['gt_pose_read_by_gate'] is False
    assert report['publishable'] is False and report['position_m'] is None
    assert out.is_file()
    with pytest.raises(FileExistsError, match='OUTPUT_ALREADY_EXISTS'):
        inspect_frozen_batch(frozen_dir=root, thresholds=GateThresholds(), output=out)


def test_modified_candidate_hash_fail_closed(tmp_path):
    root = _frozen_fixture(tmp_path)
    p = root / 'candidate_000015.json'
    p.write_text(p.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='FROZEN_CANDIDATE_SHA256_MISMATCH'):
        inspect_frozen_batch(frozen_dir=root, thresholds=GateThresholds(),
                             output=tmp_path / 'result.json')


def test_refuses_placing_output_inside_frozen_directory(tmp_path):
    root = _frozen_fixture(tmp_path)
    with pytest.raises(ValueError, match='OUTPUT_MUST_BE_OUTSIDE_FROZEN_DIRECTORY'):
        inspect_frozen_batch(frozen_dir=root, thresholds=GateThresholds(),
                             output=root / 'gate.json')


def test_gate_does_not_look_at_fake_gt_eval_file(tmp_path):
    root = _frozen_fixture(tmp_path)
    # Even if the directory contains an evaluation report, only frozen candidates
    # are read; neither accuracy claims nor gate outcomes depend on this file.
    (root / 'batch_summary.json').write_text('{ BAD GT EVALUATION JSON', encoding='utf-8')
    out = tmp_path / 'review.json'
    report = inspect_frozen_batch(frozen_dir=root, thresholds=GateThresholds(), output=out)
    assert report['frame_count'] == 2
    assert report['gt_pose_read_by_gate'] is False
