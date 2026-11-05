"""T1-05-A posthoc audit tests; synthetic records, no BOP data required."""
import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.t1_05_a_posthoc_gate_audit import audit, main, CONFIGS, PASS, REVIEW

SAMPLES = [
    (15, .95, .0025,   5.,  2.),
    (19, .84, .0028, 160., 20.),
    (47, .91, .0035, 100., 15.),
    (55, .8384401114206128, .003635847068801166, 176.214, 21.724),
    (240,.75, .0032, 120., 12.),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return sha(path)


def fixture(tmp_path, samples=SAMPLES):
    b4_dir = tmp_path / 'b4'
    gate_dir = tmp_path / 'gate'
    b4_dir.mkdir()
    gate_dir.mkdir()
    candidates, summary_rows, objs = [], [], {}
    for im_id, sf, rmse, rot, trans in samples:
        obs = {'im_id': im_id, 'gt_index': 0}
        candidate = {
            'observation': obs, 'model': {'obj_id': 1},
            'registration_diagnostic_only': {
                'support_fraction': sf, 'trimmed_rmse_m': rmse,
                'support_threshold_m': .008},
            'gt_pose_read_by_predictor': False,
            'gt_pose_used_to_construct_candidate': False,
            'publishable': False,
        }
        cfile = f'candidate_{im_id:06d}.json'
        csha = save(b4_dir / cfile, candidate)
        candidates.append({'im_id': im_id, 'candidate_file': cfile,
                           'candidate_sha256': csha})
        ev = {
            'candidate_sha256': csha, 'sample': {'im_id': im_id, 'gt_index': 0,
                                                'obj_id': 1},
            'gt_pose_read_by_evaluator': True,
            'gt_pose_used_for_estimation': False,
            'symmetry_aware': False, 'publishable': False,
            'rotation_error_deg_raw': rot, 'translation_error_mm': trans,
        }
        esha = save(b4_dir / f'evaluation_{im_id:06d}.json', ev)
        summary_rows.append({'im_id': im_id, 'candidate_sha256': csha,
                             'evaluation_sha256': esha,
                             'rotation_error_deg_raw': rot,
                             'translation_error_mm': trans})
        objs[im_id] = (sf, rmse, csha)
    freeze = {
        'status': 'ALL_CANDIDATES_FROZEN_BEFORE_GT_EVALUATION',
        'prediction_uses_scene_gt': False, 'oracle_mask_used': True,
        'frame_count': len(samples), 'gt_index': 0, 'obj_id': 1,
        'candidates': candidates,
    }
    fhash = save(b4_dir / 'frozen_predictions.json', freeze)
    summary = {'frame_count_expected': len(samples),
               'frame_count_evaluated': len(samples), 'failed_count': 0,
               'frozen_predictions_sha256': fhash,
               'dataset_gt_used_only_in_evaluation': True,
               'raw_rotation_not_symmetry_aware': True,
               'publishable': False, 'samples': summary_rows}
    save(b4_dir / 'batch_summary.json', summary)
    for name, (min_support, max_rmse) in CONFIGS.items():
        decisions = []
        for im_id, (sf, rmse, csha) in objs.items():
            reasons = []
            if sf < min_support:
                reasons.append('SUPPORT_FRACTION_BELOW_THRESHOLD')
            if rmse > max_rmse:
                reasons.append('TRIMMED_RMSE_ABOVE_THRESHOLD')
            decisions.append({
                'im_id': im_id, 'candidate_sha256': csha,
                'decision': REVIEW if reasons else PASS, 'reasons': reasons,
                'support_fraction': sf, 'trimmed_rmse_m': rmse,
                'support_threshold_m': .008, 'pose_accuracy_verified': False,
                'gt_pose_read_by_gate': False,
            })
        passed = sum(x['decision'] == PASS for x in decisions)
        report = {
            'schema': 't1-05-a-gt-free-heuristic-review-v1',
            'frozen_predictions_sha256': fhash,
            'frame_count': len(samples), 'gt_pose_read_by_gate': False,
            'prediction_correctness_claimed': False, 'publishable': False,
            'position_m': None,
            'thresholds': {'min_support_fraction': min_support,
                           'max_trimmed_rmse_m': max_rmse,
                           'support_threshold_m': .008},
            'decisions': decisions,
            'pass_heuristic_count': passed,
            'review_count': len(samples) - passed,
            'pass_heuristic_fraction': passed / len(samples),
        }
        save(gate_dir / f'{name}.json', report)
    return b4_dir, gate_dir


def test_join_and_metrics_on_previously_frozen_gate_reports(tmp_path):
    b4, gate = fixture(tmp_path)
    r = audit(b4_dir=b4, gate_dir=gate, expected_frames=5)
    assert r['all_frames_raw_rotation_gt_90_count'] == 4
    assert r['all_frames_raw_rotation_gt_150_count'] == 2
    assert r['gates']['baseline']['pass_count'] == 4
    assert r['gates']['A_support_090']['pass_count'] == 2
    assert r['gates']['B_rmse_003']['pass_count'] == 2
    assert r['gates']['baseline']['passed_with_raw_rotation_gt_90_count'] == 3
    assert r['gates']['baseline']['raw_rotation_gt_90_fraction_among_pass'] == .75
    assert 55 in r['gates']['baseline']['passed_high_rotation_gt_150_image_ids']
    assert 55 not in r['gates']['A_support_090']['passed_image_ids']
    assert 55 not in r['gates']['B_rmse_003']['passed_image_ids']
    assert r['unbiased_model_selection_claimed'] is False
    assert r['gt_pose_used_by_gate'] is False
    assert r['gt_reference_used_in_posthoc_audit'] is True


def test_candidate_modification_detected(tmp_path):
    b4, gate = fixture(tmp_path)
    p = b4 / 'candidate_000055.json'
    data = json.loads(p.read_text())
    data['registration_diagnostic_only']['support_fraction'] = .99
    save(p, data)
    with pytest.raises(ValueError, match='FROZEN_CANDIDATE_HASH_MISMATCH'):
        audit(b4_dir=b4, gate_dir=gate, expected_frames=5)


def test_evaluation_modification_detected(tmp_path):
    b4, gate = fixture(tmp_path)
    p = b4 / 'evaluation_000055.json'
    data = json.loads(p.read_text())
    data['rotation_error_deg_raw'] = 0.
    save(p, data)
    with pytest.raises(ValueError, match='B4_EVALUATION_HASH_MISMATCH'):
        audit(b4_dir=b4, gate_dir=gate, expected_frames=5)


def test_gate_decision_tampering_detected(tmp_path):
    b4, gate = fixture(tmp_path)
    p = gate / 'baseline.json'
    data = json.loads(p.read_text())
    data['decisions'][0]['decision'] = REVIEW
    save(p, data)
    with pytest.raises(ValueError, match='GATE_DECISION_DOES_NOT_MATCH_FROZEN_DIAGNOSTIC'):
        audit(b4_dir=b4, gate_dir=gate, expected_frames=5)


def test_wrong_support_unit_and_threshold_rejected(tmp_path):
    b4, gate = fixture(tmp_path)
    p = gate / 'A_support_090.json'
    data = json.loads(p.read_text())
    data['thresholds']['support_threshold_m'] = .004
    save(p, data)
    with pytest.raises(ValueError, match='GATE_SUPPORT_DISTANCE_THRESHOLD_MISMATCH'):
        audit(b4_dir=b4, gate_dir=gate, expected_frames=5)


def test_wrong_count_rejected(tmp_path):
    b4, gate = fixture(tmp_path)
    with pytest.raises(ValueError, match='FROZEN_SAMPLE_COUNT_MISMATCH'):
        audit(b4_dir=b4, gate_dir=gate, expected_frames=50)


def test_zero_pass_has_null_fraction_not_divide_by_zero(tmp_path):
    samples = [(i, .10, .01, 140., 20.) for i in (15,19,47,55,240)]
    b4, gate = fixture(tmp_path, samples)
    r = audit(b4_dir=b4, gate_dir=gate, expected_frames=5)
    for v in r['gates'].values():
        assert v['pass_count'] == 0
        assert v['raw_rotation_gt_90_fraction_among_pass'] is None


def test_cli_writes_once_and_never_reads_scene_gt(tmp_path):
    b4, gate = fixture(tmp_path)
    # An invalid fake GT annotation is deliberately present, but never opened.
    (b4 / 'scene_gt.json').write_text('NOT JSON', encoding='utf-8')
    out = tmp_path / 'audit' / 'posthoc.json'
    args = ['--b4-dir', str(b4), '--gate-dir', str(gate),
            '--expected-frames', '5', '--output', str(out)]
    main(args)
    assert json.loads(out.read_text())['frame_count'] == 5
    with pytest.raises(FileExistsError, match='OUTPUT_ALREADY_EXISTS'):
        main(args)
