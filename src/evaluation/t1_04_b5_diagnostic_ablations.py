"""T1-04-B5: three one-factor ICP diagnostics on GT-identified failure cases.

Important: cases (47,55,240) were SELECTED USING PREVIOUS B4 GT metrics.
Consequently this is diagnostic / development analysis, NEVER an unbiased test
or BOP score. Prediction phase uses depth + Oracle mask + model only. All
candidates are persisted and hashed BEFORE any evaluation phase reads GT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation import t1_04_b_gt_free_icp as b1
from src.evaluation import t1_04_b2_independent_pose_eval as b2

SCHEMA = 't1-04-b5-gt-informed-case-diagnostics-v1'
CASES = (47, 55, 240)
# Each variant differs from baseline in exactly ONE parameter.
VARIANTS = {
    'baseline': dict(max_iterations=18, trim_fraction=0.75, support_threshold_m=0.008),
    'trim_060': dict(max_iterations=18, trim_fraction=0.60, support_threshold_m=0.008),
    'iter_036': dict(max_iterations=36, trim_fraction=0.75, support_threshold_m=0.008),
    'support_004': dict(max_iterations=18, trim_fraction=0.75, support_threshold_m=0.004),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('INVALID_OR_MISSING_REPORT') from exc
    require(isinstance(value, dict), 'INVALID_JSON_OBJECT')
    return value


def save_new(path, payload):
    # Exclusive creation. Never overwrite a previously generated experiment.
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def filename(im_id, variant):
    return f'candidate_{im_id:06d}_{variant}.json'


def make_report(im_id, variant, scene_dir, models_dir, obj_id, gt_index, *,
                observed=None, obs_info=None, model=None, model_info=None):
    params = VARIANTS[variant]
    if observed is None or obs_info is None:
        observed, obs_info = b1.read_oracle_observation(
            scene_dir, im_id=im_id, gt_index=gt_index)
    if model is None or model_info is None:
        model, model_info = b1.load_model_points(models_dir, obj_id)
    pose, diag = b1.fit_pca_multistart_icp(model, observed, **params)
    return {
        'schema': b2.CANDIDATE_SCHEMA,
        'method': 'PCA_24_START_PARTIAL_OVERLAP_POINT_TO_POINT_ICP',
        'status': 'GEOMETRIC_CANDIDATE_UNVERIFIED',
        'estimation_performed': True,
        'gt_pose_read_by_predictor': False,
        'gt_pose_used_to_construct_candidate': False,
        'oracle_mask_used': True,
        'oracle_supervision_used': True,
        'pose_accuracy_claimed': False,
        'physical_accuracy_claimed': False,
        'symmetry_aware': False,
        'publishable': False,
        'position_m': None,
        'pose_candidate_model_to_camera': {
            'R_row_major': pose.R.ravel().tolist(),
            't_m': pose.t_m.tolist(),
        },
        'registration_diagnostic_only': diag,
        'observation': obs_info,
        'model': model_info,
        'diagnostic_ablation': {
            'name': variant,
            'parameters': params,
            'frames_selected_from_prior_gt_results': True,
            'new_GT_used_for_pose_prediction': False,
            'not_unbiased_performance': True,
        },
        'note': 'GT-informed failure-case selection, GT-blind prediction; diagnostic only.',
    }


def predict_cases(*, scene_dir, models_dir, run_dir, cases=CASES,
                  obj_id=1, gt_index=0):
    scene_dir, models_dir, run_dir = Path(scene_dir), Path(models_dir), Path(run_dir)
    cases = tuple(cases)
    require(len(cases) > 0 and len(set(cases)) == len(cases) and
            all(type(x) is int and x >= 0 for x in cases), 'INVALID_CASES')
    require(type(obj_id) is int and obj_id > 0 and
            type(gt_index) is int and gt_index >= 0, 'INVALID_INSTANCE_SELECTORS')
    require(not run_dir.exists(), 'RUN_DIRECTORY_ALREADY_EXISTS')
    run_dir.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        model, model_info = b1.load_model_points(models_dir, obj_id)
        for im_id in cases:
            observed, obs_info = b1.read_oracle_observation(
                scene_dir, im_id=im_id, gt_index=gt_index)
            for variant in VARIANTS:
                report = make_report(im_id, variant, scene_dir, models_dir,
                                     obj_id, gt_index, observed=observed,
                                     obs_info=obs_info, model=model,
                                     model_info=model_info)
                out = run_dir / filename(im_id, variant)
                save_new(out, report)
                records.append({'im_id': im_id, 'variant': variant,
                                'filename': out.name, 'sha256': b1._sha256(out)})
        frozen = {
            'schema': SCHEMA, 'status': 'PREDICTIONS_FROZEN',
            'case_selection': 'Prior B4 GT-informed diagnostic cases; not held out',
            'cases': list(cases), 'obj_id': obj_id, 'gt_index': gt_index,
            'variants': VARIANTS,
            'prediction_gt_pose_access': False, 'oracle_mask_used': True,
            'source_sha256': b1._sha256(Path(__file__)),
            'b1_source_sha256': b1._sha256(Path(b1.__file__)),
            'records': records,
            'publishable': False, 'position_m': None,
        }
        save_new(run_dir / 'frozen_ablations.json', frozen)
        return frozen
    except Exception as exc:
        save_new(run_dir / 'prediction_failure.json', {
            'stage': 'PREDICTION', 'completed': len(records),
            'reason': type(exc).__name__, 'message': str(exc),
        })
        raise


def audit_frozen(run_dir):
    root = Path(run_dir)
    require(not (root / 'prediction_failure.json').exists() and
            not (root / 'evaluation_failure.json').exists(), 'FAILED_RUN_NOT_EVALUABLE')
    frozen = read_json(root / 'frozen_ablations.json')
    require(frozen.get('schema') == SCHEMA and
            frozen.get('status') == 'PREDICTIONS_FROZEN', 'INVALID_FREEZE_SCHEMA')
    cases = frozen.get('cases')
    require(isinstance(cases, list) and cases and
            len(set(cases)) == len(cases) and
            all(type(x) is int and x >= 0 for x in cases), 'INVALID_FROZEN_CASES')
    require(frozen.get('variants') == VARIANTS and
            frozen.get('prediction_gt_pose_access') is False and
            frozen.get('oracle_mask_used') is True and
            frozen.get('publishable') is False and
            frozen.get('position_m') is None, 'INVALID_FREEZE_SEMANTICS')
    require(frozen.get('source_sha256') == b1._sha256(Path(__file__)) and
            frozen.get('b1_source_sha256') == b1._sha256(Path(b1.__file__)),
            'PREDICTOR_CODE_CHANGED')
    records = frozen.get('records')
    require(isinstance(records, list) and len(records) == len(cases) * len(VARIANTS),
            'INCOMPLETE_FROZEN_CANDIDATES')
    expected = {(im_id, variant) for im_id in cases for variant in VARIANTS}
    observed = set()
    for record in records:
        require(isinstance(record, dict), 'INVALID_FROZEN_ENTRY')
        key = (record.get('im_id'), record.get('variant'))
        require(key in expected and key not in observed and
                record.get('filename') == filename(*key), 'INVALID_FROZEN_ENTRY')
        observed.add(key)
        path = root / record['filename']
        require(path.is_file() and b1._sha256(path) == record.get('sha256'),
                'FROZEN_CANDIDATE_SHA256_MISMATCH')
    require(observed == expected and
            {p.name for p in root.glob('candidate_*.json')} ==
            {filename(*key) for key in expected}, 'CANDIDATE_SET_CHANGED')
    return frozen


def evaluate_cases(*, scene_dir, models_dir, run_dir):
    root = Path(run_dir)
    require(not (root / 'diagnostic_summary.json').exists() and
            not list(root.glob('evaluation_*.json')), 'EVALUATION_ALREADY_EXISTS')
    frozen = audit_frozen(root)  # ALL candidate hashes checked BEFORE any GT read.
    results = []
    try:
        for record in frozen['records']:
            candidate_path = root / record['filename']
            result = b2.evaluate_saved_candidate(
                candidate_path=candidate_path, scene_dir=scene_dir,
                models_dir=models_dir)
            require(result.get('candidate_sha256') == record['sha256'] and
                    result.get('gt_pose_read_by_evaluator') is True and
                    result.get('gt_pose_used_for_estimation') is False and
                    result.get('publishable') is False and
                    result.get('position_m') is None,
                    'INVALID_EVALUATION_PROVENANCE')
            candidate = read_json(candidate_path)
            data = {
                'im_id': record['im_id'], 'variant': record['variant'],
                'candidate_sha256': record['sha256'],
                'translation_error_mm': result['translation_error_mm'],
                'rotation_error_deg_raw': result['rotation_error_deg_raw'],
                'trimmed_icp_rmse_m': candidate['registration_diagnostic_only']['trimmed_rmse_m'],
                'support_fraction': candidate['registration_diagnostic_only']['support_fraction'],
            }
            save_new(root / f"evaluation_{record['im_id']:06d}_{record['variant']}.json", result)
            results.append(data)
        # C is diagnostic-only: its changing threshold MUST NOT affect R/t/RMSE.
        for im_id in frozen['cases']:
            base = read_json(root / filename(im_id, 'baseline'))
            c = read_json(root / filename(im_id, 'support_004'))
            p, q = base['pose_candidate_model_to_camera'], c['pose_candidate_model_to_camera']
            require(p == q, 'SUPPORT_ONLY_CHANGED_POSE')
            d0, d1 = base['registration_diagnostic_only'], c['registration_diagnostic_only']
            require(d0['trimmed_rmse_m'] == d1['trimmed_rmse_m'] and
                    d1['support_fraction'] <= d0['support_fraction'],
                    'SUPPORT_THRESHOLD_INVARIANCE_FAILED')
        summary = {
            'schema': SCHEMA, 'status': 'GT_INFORMED_CASE_DIAGNOSTIC_ONLY',
            'case_count': len(frozen['cases']), 'variant_count': len(VARIANTS),
            'record_count': len(results), 'frozen_sha256': b1._sha256(root/'frozen_ablations.json'),
            'evaluator_source_sha256': b1._sha256(Path(b2.__file__)),
            'case_selection_used_prior_gt': True,
            'new_gt_used_only_for_independent_evaluation': True,
            'model_and_oracle_mask_used': True,
            'not_held_out': True, 'bop_benchmark_score_claimed': False,
            'physical_accuracy_claimed': False, 'publishable': False,
            'position_m': None, 'samples': results,
        }
        save_new(root / 'diagnostic_summary.json', summary)
        return summary
    except Exception as exc:
        save_new(root / 'evaluation_failure.json', {
            'stage': 'EVALUATION', 'completed': len(results),
            'reason': type(exc).__name__, 'message': str(exc),
        })
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('predict', 'evaluate'))
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == 'predict':
        report = predict_cases(scene_dir=args.scene_dir, models_dir=args.models_dir,
                               run_dir=args.run_dir)
        print('GT-free diagnostic candidates frozen:', len(report['records']))
        print('GT-informed case selection, NOT unbiased benchmark:', report['cases'])
        print('Frozen record:', args.run_dir / 'frozen_ablations.json')
    else:
        summary = evaluate_cases(scene_dir=args.scene_dir, models_dir=args.models_dir,
                                 run_dir=args.run_dir)
        print('GT-independent candidate evaluation count:', summary['record_count'])
        for frame in summary['samples']:
            print(f"im_id={frame['im_id']:06d} variant={frame['variant']} "
                  f"t_mm={frame['translation_error_mm']:.3f} "
                  f"rot_deg={frame['rotation_error_deg_raw']:.3f} "
                  f"rmse_mm={1000*frame['trimmed_icp_rmse_m']:.3f} "
                  f"support={frame['support_fraction']:.3f}")
        print('B5 GT-informed case diagnostic: PASS (NOT benchmark accuracy)')
        print('Diagnostic report:', args.run_dir / 'diagnostic_summary.json')


if __name__ == '__main__':
    main()
