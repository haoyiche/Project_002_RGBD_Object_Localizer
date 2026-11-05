"""T1-05-A / D-E: retrospective audit of three frozen, GT-free gate reports.

NEVER generates/changes poses, never calls ICP, and never opens scene_gt.json.
Uses the existing, separately generated B4 GT-evaluation JSON files *only*
after validating frozen candidate, evaluation and gate-report provenance.

This is GT-exposed, post-hoc failure analysis, NOT unbiased parameter validation,
symmetry-aware BOP scoring, an accuracy certificate, or a deployment approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CONFIGS = {
    'baseline': (0.80, 0.004),
    'A_support_090': (0.90, 0.004),
    'B_rmse_003': (0.80, 0.003),
}
PASS = 'PASS_HEURISTIC_NOT_POSE_ACCURACY'
REVIEW = 'REVIEW_HEURISTIC_NOT_REJECTED_BY_GT'


def _require(ok: bool, code: str):
    if not ok:
        raise ValueError(code)


def _sha256(path: Path):
    hash_ = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for buf in iter(lambda: fh.read(1 << 20), b''):
            hash_.update(buf)
    return hash_.hexdigest()


def _read(path: Path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    _require(isinstance(obj, dict), 'JSON_NOT_OBJECT')
    return obj


def _number(value, code):
    _require(type(value) in (int, float) and math.isfinite(value), code)
    return float(value)


def _same(a, b, code, tol=1e-12):
    _require(math.isclose(_number(a, code), _number(b, code),
                          abs_tol=tol, rel_tol=0), code)


def audit(*, b4_dir: Path, gate_dir: Path, expected_frames: int = 50) -> dict:
    """Verify on-disk lineage, then join fixed GT results to fixed gate decisions."""
    b4_dir, gate_dir = Path(b4_dir), Path(gate_dir)
    _require(type(expected_frames) is int and expected_frames >= 1,
             'INVALID_EXPECTED_FRAME_COUNT')
    freeze_file = b4_dir / 'frozen_predictions.json'
    freeze = _read(freeze_file)
    summary = _read(b4_dir / 'batch_summary.json')
    freeze_hash = _sha256(freeze_file)
    _require(freeze.get('status') == 'ALL_CANDIDATES_FROZEN_BEFORE_GT_EVALUATION'
             and freeze.get('prediction_uses_scene_gt') is False
             and freeze.get('oracle_mask_used') is True,
             'UNVERIFIABLE_FROZEN_PREDICTION_SOURCE')
    _require(freeze.get('frame_count') == expected_frames
             and isinstance(freeze.get('candidates'), list)
             and len(freeze['candidates']) == expected_frames,
             'FROZEN_SAMPLE_COUNT_MISMATCH')
    _require(summary.get('frame_count_expected') == expected_frames
             and summary.get('frame_count_evaluated') == expected_frames
             and summary.get('failed_count') == 0
             and summary.get('frozen_predictions_sha256') == freeze_hash
             and summary.get('dataset_gt_used_only_in_evaluation') is True
             and summary.get('raw_rotation_not_symmetry_aware') is True
             and summary.get('publishable') is False,
             'B4_INDEPENDENT_EVALUATION_CONTRACT_MISMATCH')

    frozen_by_id = {}
    candidate_by_id = {}
    for row in freeze['candidates']:
        im_id = row.get('im_id')
        _require(type(im_id) is int and im_id >= 0 and im_id not in frozen_by_id,
                 'FROZEN_DUPLICATE_OR_INVALID_IMAGE_ID')
        file_name = f'candidate_{im_id:06d}.json'
        _require(row.get('candidate_file') == file_name,
                 'FROZEN_CANDIDATE_FILE_NAME_MISMATCH')
        expected_hash = row.get('candidate_sha256')
        _require(isinstance(expected_hash, str) and len(expected_hash) == 64
                 and _sha256(b4_dir / file_name) == expected_hash,
                 'FROZEN_CANDIDATE_HASH_MISMATCH')
        candidate = _read(b4_dir / file_name)
        _require(candidate.get('observation', {}).get('im_id') == im_id
                 and candidate.get('observation', {}).get('gt_index') == freeze.get('gt_index')
                 and candidate.get('model', {}).get('obj_id') == freeze.get('obj_id')
                 and candidate.get('gt_pose_read_by_predictor') is False
                 and candidate.get('gt_pose_used_to_construct_candidate') is False
                 and candidate.get('publishable') is False,
                 'FROZEN_CANDIDATE_CONTRACT_MISMATCH')
        diag = candidate['registration_diagnostic_only']
        _same(diag.get('support_threshold_m'), 0.008,
              'FROZEN_SUPPORT_DISTANCE_THRESHOLD_UNEXPECTED')
        sf = _number(diag.get('support_fraction'), 'INVALID_FROZEN_SUPPORT')
        rmse = _number(diag.get('trimmed_rmse_m'), 'INVALID_FROZEN_RMSE')
        _require(0 <= sf <= 1 and rmse >= 0, 'INVALID_FROZEN_DIAGNOSTIC_RANGE')
        frozen_by_id[im_id] = expected_hash
        candidate_by_id[im_id] = (sf, rmse)

    rows = summary.get('samples')
    _require(isinstance(rows, list) and len(rows) == expected_frames,
             'B4_EVALUATION_SAMPLE_COUNT_MISMATCH')
    gt_by_id = {}
    for row in rows:
        im_id = row.get('im_id')
        _require(type(im_id) is int and im_id in frozen_by_id
                 and im_id not in gt_by_id
                 and row.get('candidate_sha256') == frozen_by_id[im_id],
                 'B4_EVALUATION_SAMPLE_LINEAGE_MISMATCH')
        eval_path = b4_dir / f'evaluation_{im_id:06d}.json'
        _require(_sha256(eval_path) == row.get('evaluation_sha256'),
                 'B4_EVALUATION_HASH_MISMATCH')
        evaluated = _read(eval_path)
        _require(evaluated.get('candidate_sha256') == frozen_by_id[im_id]
                 and evaluated.get('gt_pose_read_by_evaluator') is True
                 and evaluated.get('gt_pose_used_for_estimation') is False
                 and evaluated.get('symmetry_aware') is False
                 and evaluated.get('publishable') is False,
                 'B4_EVALUATION_FLAGS_MISMATCH')
        _require(evaluated.get('sample') == {
            'im_id': im_id, 'gt_index': freeze['gt_index'], 'obj_id': freeze['obj_id']},
            'B4_EVALUATION_IDENTITY_MISMATCH')
        rot = _number(row.get('rotation_error_deg_raw'), 'INVALID_RAW_ROTATION_ERROR')
        trn = _number(row.get('translation_error_mm'), 'INVALID_TRANSLATION_ERROR')
        _require(0 <= rot <= 180.00001 and trn >= 0,
                 'INVALID_POSE_ERROR_RANGE')
        _same(rot, evaluated.get('rotation_error_deg_raw'), 'B4_ROTATION_ERROR_MISMATCH')
        _same(trn, evaluated.get('translation_error_mm'), 'B4_TRANSLATION_ERROR_MISMATCH')
        gt_by_id[im_id] = (rot, trn)
    _require(set(gt_by_id) == set(frozen_by_id), 'EVALUATION_SCOPE_MISMATCH')

    results = {}
    pass_sets = {}
    all_ids = set(frozen_by_id)
    for config, (min_support, max_rmse) in CONFIGS.items():
        gate = _read(gate_dir / f'{config}.json')
        _require(gate.get('schema') == 't1-05-a-gt-free-heuristic-review-v1'
                 and gate.get('frozen_predictions_sha256') == freeze_hash
                 and gate.get('frame_count') == expected_frames
                 and gate.get('gt_pose_read_by_gate') is False
                 and gate.get('prediction_correctness_claimed') is False
                 and gate.get('publishable') is False
                 and gate.get('position_m') is None,
                 'GATE_REPORT_PROVENANCE_MISMATCH')
        thresholds = gate.get('thresholds', {})
        _same(thresholds.get('min_support_fraction'), min_support,
              'GATE_MIN_SUPPORT_MISMATCH')
        _same(thresholds.get('max_trimmed_rmse_m'), max_rmse,
              'GATE_MAX_RMSE_MISMATCH')
        _same(thresholds.get('support_threshold_m'), 0.008,
              'GATE_SUPPORT_DISTANCE_THRESHOLD_MISMATCH')
        decisions = gate.get('decisions')
        _require(isinstance(decisions, list) and len(decisions) == expected_frames,
                 'GATE_DECISION_COUNT_MISMATCH')
        passed = set()
        seen = set()
        for decision in decisions:
            im_id = decision.get('im_id')
            _require(type(im_id) is int and im_id in all_ids and im_id not in seen
                     and decision.get('candidate_sha256') == frozen_by_id[im_id],
                     'GATE_DECISION_IDENTITY_OR_HASH_MISMATCH')
            seen.add(im_id)
            sf, rmse = candidate_by_id[im_id]
            _same(sf, decision.get('support_fraction'), 'GATE_SUPPORT_DATA_TAMPERED')
            _same(rmse, decision.get('trimmed_rmse_m'), 'GATE_RMSE_DATA_TAMPERED')
            _same(decision.get('support_threshold_m'), 0.008,
                  'GATE_PER_CANDIDATE_SUPPORT_DISTANCE_MISMATCH')
            reasons = []
            if sf < min_support:
                reasons.append('SUPPORT_FRACTION_BELOW_THRESHOLD')
            if rmse > max_rmse:
                reasons.append('TRIMMED_RMSE_ABOVE_THRESHOLD')
            expectation = REVIEW if reasons else PASS
            _require(decision.get('decision') == expectation
                     and decision.get('reasons') == reasons
                     and decision.get('gt_pose_read_by_gate') is False
                     and decision.get('pose_accuracy_verified') is False,
                     'GATE_DECISION_DOES_NOT_MATCH_FROZEN_DIAGNOSTIC')
            if expectation == PASS:
                passed.add(im_id)
        _require(seen == all_ids, 'GATE_SCOPE_MISMATCH')
        _require(gate.get('pass_heuristic_count') == len(passed)
                 and gate.get('review_count') == expected_frames - len(passed),
                 'GATE_PASS_REVIEW_COUNT_MISMATCH')
        _same(gate.get('pass_heuristic_fraction'), len(passed)/expected_frames,
              'GATE_PASS_FRACTION_MISMATCH')
        high90 = sorted(i for i in passed if gt_by_id[i][0] > 90)
        high150 = sorted(i for i in passed if gt_by_id[i][0] > 150)
        trans10 = sorted(i for i in passed if gt_by_id[i][1] > 10)
        results[config] = {
            'pass_count': len(passed),
            'review_count': expected_frames - len(passed),
            'coverage_fraction': len(passed)/expected_frames,
            'passed_with_raw_rotation_gt_90_count': len(high90),
            'passed_with_raw_rotation_gt_150_count': len(high150),
            'passed_with_translation_gt_10mm_count': len(trans10),
            'raw_rotation_gt_90_fraction_among_pass': (
                len(high90)/len(passed) if passed else None),
            'raw_rotation_gt_150_fraction_among_pass': (
                len(high150)/len(passed) if passed else None),
            'passed_image_ids': sorted(passed),
            'passed_high_rotation_gt_90_image_ids': high90,
            'passed_high_rotation_gt_150_image_ids': high150,
        }
        pass_sets[config] = passed
    _require(pass_sets['A_support_090'] <= pass_sets['baseline']
             and pass_sets['B_rmse_003'] <= pass_sets['baseline'],
             'GATE_MONOTONICITY_VIOLATED')
    high90_total = sum(rot > 90 for rot, _ in gt_by_id.values())
    high150_total = sum(rot > 150 for rot, _ in gt_by_id.values())
    return {
        'schema': 't1-05-a-posthoc-gt-audit-v1',
        'status': 'POSTHOC_GT_EXPOSED_DIAGNOSTIC_ONLY',
        'source_frozen_predictions_sha256': freeze_hash,
        'source_b4_summary_sha256': _sha256(b4_dir / 'batch_summary.json'),
        'source_gate_report_sha256': {
            name: _sha256(gate_dir / f'{name}.json') for name in CONFIGS},
        'frame_count': expected_frames,
        'all_frames_raw_rotation_gt_90_count': high90_total,
        'all_frames_raw_rotation_gt_150_count': high150_total,
        'raw_rotation_not_symmetry_aware': True,
        'oracle_instance_mask_used': True,
        'gt_pose_used_by_gate': False,
        'gt_reference_used_in_posthoc_audit': True,
        'thresholds_previously_exposed_to_gt_data': True,
        'unbiased_model_selection_claimed': False,
        'bop_official_scoring_claimed': False,
        'physical_accuracy_claimed': False,
        'publishable': False,
        'position_m': None,
        'gates': results,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b4-dir', type=Path, required=True)
    p.add_argument('--gate-dir', type=Path, required=True)
    p.add_argument('--expected-frames', type=int, default=50)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('OUTPUT_ALREADY_EXISTS')
    result = audit(b4_dir=args.b4_dir, gate_dir=args.gate_dir,
                   expected_frames=args.expected_frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write('\n')
    print('POSTHOC ONLY: not independent held-out validation')
    print('B4 raw rotation >90 deg:', result['all_frames_raw_rotation_gt_90_count'])
    print('B4 raw rotation >150 deg:', result['all_frames_raw_rotation_gt_150_count'])
    for name, r in result['gates'].items():
        print(f"{name}: PASS={r['pass_count']} REVIEW={r['review_count']} "
              f"coverage={r['coverage_fraction']:.1%} "
              f">90deg among PASS={r['passed_with_raw_rotation_gt_90_count']} "
              f"({r['raw_rotation_gt_90_fraction_among_pass']}) "
              f">150deg among PASS={r['passed_with_raw_rotation_gt_150_count']}")
    print('Audit saved:', out)


if __name__ == '__main__':
    main()
