"""T1-04-B4: freeze GT-pose-blind PCA+ICP predictions, THEN evaluate fixed poses.

Prediction phase never reads scene_gt.json, T1-02 Manifest or BOP pose fields.
Oracle object ID and instance mask index are explicit supervised inputs.
This reports annotation-relative raw rotation and translation, NOT physical
accuracy, symmetry-aware BOP scores, or end-to-end pose estimation.

B4 is intentionally all-or-nothing: if any selected frame fails, there is no
aggregate accuracy report. Partial per-frame files are NOT a passed batch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src.evaluation import t1_04_b_gt_free_icp as b1

FREEZE_SCHEMA = 't1-04-b4-fixed-predictions-freeze-v1'
SUMMARY_SCHEMA = 't1-04-b4-annotation-relative-batch-v1'


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _save_new_json(path: Path, data):
    """No overwrite; incomplete/broken runs must not appear freshly successful."""
    path = Path(path)
    if path.exists():
        raise FileExistsError('OUTPUT_ALREADY_EXISTS')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                prefix='.t104b4_', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise FileExistsError('OUTPUT_ALREADY_EXISTS')
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def _read_json(path: Path):
    try:
        result = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('INVALID_OR_MISSING_BATCH_JSON') from exc
    _require(isinstance(result, dict), 'INVALID_BATCH_JSON_OBJECT')
    return result


def _image_ids(folder: Path):
    _require(folder.is_dir(), 'RGB_OR_DEPTH_DIRECTORY_MISSING')
    images = list(folder.glob('*.png'))
    ids = []
    for p in images:
        if len(p.stem) != 6 or not p.stem.isascii() or not p.stem.isdecimal():
            raise ValueError('INVALID_PNG_FRAME_NAME')
        ids.append(int(p.stem))
    _require(len(set(ids)) == len(ids), 'DUPLICATE_FRAME_ID')
    return set(ids)


def select_rgbd_frames(scene_dir, *, count: int, gt_index: int):
    """Deliberately uses only PNG names, NOT scene_gt or GT-containing Manifest."""
    if type(count) is not int or count < 1:
        raise ValueError('COUNT_MUST_BE_POSITIVE')
    if type(gt_index) is not int or gt_index < 0:
        raise ValueError('INVALID_GT_INDEX')
    root = Path(scene_dir)
    rgb, depth = _image_ids(root / 'rgb'), _image_ids(root / 'depth')
    _require(rgb == depth, 'RGB_DEPTH_IMAGE_IDS_MISMATCH')
    ids = sorted(rgb)
    _require(len(ids) >= count,
             f'INSUFFICIENT_DISTINCT_FRAMES:available={len(ids)},required={count}')
    chosen = ids[:count]
    for im_id in chosen:
        mask = root / 'mask_visib' / f'{im_id:06d}_{gt_index:06d}.png'
        _require(mask.is_file(), f'ORACLE_MASK_MISSING:{im_id:06d}')
    return chosen


def _failed(run_dir: Path, *, phase: str, im_id, exc: Exception, completed: int):
    """Failure metadata explicitly disclaims completed batch; no misleading metrics."""
    path = run_dir / 'batch_failure.json'
    if not path.exists():
        _save_new_json(path, {
            'status': 'INCOMPLETE_BATCH_NOT_REPORTABLE', 'phase': phase,
            'failed_frame_id': im_id, 'completed_before_failure': completed,
            'error_type': type(exc).__name__, 'reason': str(exc),
            'publishable': False, 'position_m': None,
        })


def predict_batch(*, scene_dir, models_dir, run_dir, count=50, obj_id=1,
                  gt_index=0):
    """Phase 1: freeze ALL predictions first. This function never reads pose GT."""
    root, models, out = Path(scene_dir), Path(models_dir), Path(run_dir)
    _require(type(obj_id) is int and obj_id > 0, 'INVALID_OBJ_ID')
    _require(not out.exists(), 'RUN_DIRECTORY_ALREADY_EXISTS')
    ids = select_rgbd_frames(root, count=count, gt_index=gt_index)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for im_id in ids:
        path = out / f'candidate_{im_id:06d}.json'
        try:
            result = b1.make_prediction_report(
                scene_dir=root, models_dir=models, im_id=im_id,
                obj_id=obj_id, gt_index=gt_index,
            )
            _require(result.get('gt_pose_read_by_predictor') is False and
                     result.get('gt_pose_used_to_construct_candidate') is False,
                     'GT_LEAKAGE_FLAG')
            _require(result.get('oracle_mask_used') is True and
                     result.get('status') == 'GEOMETRIC_CANDIDATE_UNVERIFIED',
                     'INVALID_CANDIDATE_SEMANTICS')
            _require(result.get('publishable') is False and
                     result.get('position_m') is None,
                     'UNSAFE_CANDIDATE_PUBLICATION_STATE')
            _require(result['observation']['im_id'] == im_id and
                     result['observation']['gt_index'] == gt_index and
                     result['model']['obj_id'] == obj_id,
                     'CANDIDATE_IDENTITY_MISMATCH')
            _save_new_json(path, result)
            records.append({'im_id': im_id, 'candidate_file': path.name,
                            'candidate_sha256': b1._sha256(path)})
        except Exception as exc:
            _failed(out, phase='PREDICTION', im_id=im_id,
                    exc=exc, completed=len(records))
            raise
    freeze = {
        'schema': FREEZE_SCHEMA, 'status': 'ALL_CANDIDATES_FROZEN_BEFORE_GT_EVALUATION',
        'dataset': 'lm', 'scene_dir_hint': root.as_posix(),
        'model_dir_hint': models.as_posix(),
        'frame_count': count, 'obj_id': obj_id, 'gt_index': gt_index,
        'prediction_source_sha256': b1._sha256(Path(b1.__file__)),
        'prediction_uses_scene_gt': False, 'oracle_mask_used': True,
        'candidates': records,
        'publishable': False, 'position_m': None,
    }
    _save_new_json(out / 'frozen_predictions.json', freeze)
    return freeze


def _load_frozen(run_dir: Path):
    run_dir = Path(run_dir)
    _require(not (run_dir / 'batch_failure.json').exists(), 'INCOMPLETE_BATCH_FAILED')
    freeze = _read_json(run_dir / 'frozen_predictions.json')
    _require(freeze.get('schema') == FREEZE_SCHEMA and
             freeze.get('status') == 'ALL_CANDIDATES_FROZEN_BEFORE_GT_EVALUATION',
             'INVALID_FREEZE_SCHEMA_OR_STATUS')
    records = freeze.get('candidates')
    _require(isinstance(records, list) and
             type(freeze.get('frame_count')) is int and freeze['frame_count'] >= 1 and
             len(records) == freeze['frame_count'], 'INCOMPLETE_FROZEN_PREDICTIONS')
    ids = [row.get('im_id') for row in records]
    _require(all(type(im_id) is int and im_id >= 0 for im_id in ids) and
             ids == sorted(set(ids)), 'INVALID_FRAME_ORDER_OR_DUPLICATE')
    _require(freeze.get('prediction_uses_scene_gt') is False and
             freeze.get('oracle_mask_used') is True and
             freeze.get('publishable') is False and
             freeze.get('position_m') is None, 'INVALID_FREEZE_SAFETY_FLAGS')
    _require(b1._sha256(Path(b1.__file__)) == freeze.get('prediction_source_sha256'),
             'PREDICTOR_CODE_CHANGED_SINCE_FREEZE')
    actual = {p.name for p in run_dir.glob('candidate_*.json')}
    expected = {f'candidate_{i:06d}.json' for i in ids}
    _require(actual == expected, 'CANDIDATE_FILE_SET_CHANGED')
    for row, im_id in zip(records, ids):
        _require(row.get('candidate_file') == f'candidate_{im_id:06d}.json',
                 'INVALID_CANDIDATE_PATH')
        _require(b1._sha256(run_dir / row['candidate_file']) == row.get('candidate_sha256'),
                 f'FROZEN_CANDIDATE_SHA256_MISMATCH:{im_id:06d}')
    return freeze


def _stats(values):
    values = np.asarray(values, dtype=float)
    _require(values.ndim == 1 and len(values) > 0 and
             np.isfinite(values).all() and (values >= 0).all(),
             'INVALID_ERROR_METRICS')
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'p95_linear': float(np.percentile(values, 95, method='linear')),
            'max': float(values.max())}


def evaluate_batch(*, scene_dir, models_dir, run_dir):
    """Phase 2: never refit or select candidates by GT; all N must evaluate."""
    from src.evaluation import t1_04_b2_independent_pose_eval as b2

    root, models, out = Path(scene_dir), Path(models_dir), Path(run_dir)
    _require(not (out / 'batch_summary.json').exists(), 'OUTPUT_ALREADY_EXISTS')
    # Verify ALL candidate hashes before touching scene_gt.json.
    freeze = _load_frozen(out)
    _require(not list(out.glob('evaluation_*.json')), 'EVALUATION_FILES_ALREADY_EXIST')
    measured = []
    expected_gt_hash = None
    for row in freeze['candidates']:
        im_id = row['im_id']
        path = out / row['candidate_file']
        try:
            ev = b2.evaluate_saved_candidate(
                candidate_path=path, scene_dir=root, models_dir=models)
            _require(ev.get('candidate_sha256') == row['candidate_sha256'],
                     'EVAL_CANDIDATE_HASH_MISMATCH')
            _require(ev.get('sample') == {'im_id': im_id,
                                         'gt_index': freeze['gt_index'],
                                         'obj_id': freeze['obj_id']},
                     'EVAL_SAMPLE_ID_MISMATCH')
            _require(ev.get('gt_pose_used_for_estimation') is False and
                     ev.get('gt_pose_read_by_evaluator') is True and
                     ev.get('symmetry_aware') is False and
                     ev.get('bop_benchmark_score_claimed') is False and
                     ev.get('physical_accuracy_claimed') is False and
                     ev.get('publishable') is False and ev.get('position_m') is None,
                     'EVAL_SAFETY_FLAGS_INVALID')
            if expected_gt_hash is None:
                expected_gt_hash = ev['gt_annotation_sha256']
            _require(expected_gt_hash == ev.get('gt_annotation_sha256'),
                     'GT_ANNOTATIONS_CHANGED_DURING_BATCH')
            t, a = ev['translation_error_mm'], ev['rotation_error_deg_raw']
            _require(all(np.isfinite(x) for x in (t, a)) and
                     t >= 0 and 0 <= a <= 180.00001, 'INVALID_ERROR_METRICS')
            p = out / f'evaluation_{im_id:06d}.json'
            _save_new_json(p, ev)
            candidate = _read_json(path)
            measured.append({
                'im_id': im_id, 'candidate_sha256': row['candidate_sha256'],
                'evaluation_sha256': b1._sha256(p),
                'translation_error_mm': t, 'rotation_error_deg_raw': a,
                'trimmed_icp_rmse_m': candidate['registration_diagnostic_only']['trimmed_rmse_m'],
            })
        except Exception as exc:
            _failed(out, phase='EVALUATION', im_id=im_id,
                    exc=exc, completed=len(measured))
            raise
    summary = {
        'schema': SUMMARY_SCHEMA,
        'status': 'COMPLETE_SCOPE_ANNOTATION_RELATIVE_DIAGNOSTIC_ONLY',
        'frame_count_expected': freeze['frame_count'],
        'frame_count_evaluated': len(measured), 'failed_count': 0,
        'is_50_frame_scope': freeze['frame_count'] == 50,
        'dataset': freeze['dataset'], 'obj_id': freeze['obj_id'],
        'gt_index': freeze['gt_index'],
        'frozen_predictions_sha256': b1._sha256(out/'frozen_predictions.json'),
        'gt_annotation_sha256': expected_gt_hash,
        'evaluator_source_sha256': b1._sha256(Path(b2.__file__)),
        'translation_error_mm': _stats([e['translation_error_mm'] for e in measured]),
        'rotation_error_deg_raw': _stats([e['rotation_error_deg_raw'] for e in measured]),
        'p95_definition': 'np.percentile(x,95,method=linear), full selected scope only',
        'raw_rotation_not_symmetry_aware': True,
        'oracle_mask_used': True,
        'dataset_gt_used_only_in_evaluation': True,
        'bop_benchmark_score_claimed': False,
        'physical_accuracy_claimed': False,
        'publishable': False, 'position_m': None,
        'samples': measured,
    }
    _save_new_json(out / 'batch_summary.json', summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', required=True, choices=('predict', 'evaluate'))
    ap.add_argument('--scene-dir', required=True, type=Path)
    ap.add_argument('--models-dir', required=True, type=Path)
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--count', type=int, default=50)
    ap.add_argument('--obj-id', type=int, default=1)
    ap.add_argument('--gt-index', type=int, default=0)
    args = ap.parse_args(argv)
    if args.phase == 'predict':
        freeze = predict_batch(scene_dir=args.scene_dir, models_dir=args.models_dir,
                               run_dir=args.run_dir, count=args.count,
                               obj_id=args.obj_id, gt_index=args.gt_index)
        print('Frozen GT-pose-blind candidates:', freeze['frame_count'])
        print('Phase 1: PASS; GT pose never read; Oracle mask=True')
        print('Frozen record:', args.run_dir / 'frozen_predictions.json')
    else:
        # count/obj_id/gt_index are loaded from frozen record, not CLI defaults.
        summary = evaluate_batch(scene_dir=args.scene_dir, models_dir=args.models_dir,
                                 run_dir=args.run_dir)
        print('Evaluated fixed candidates:', summary['frame_count_evaluated'])
        print('Translation (mm):', summary['translation_error_mm'])
        print('Raw rotation (deg):', summary['rotation_error_deg_raw'])
        print('Failure count:', summary['failed_count'])
        print('No symmetry-aware / physical / BOP benchmark claim.')
        print('Phase 2: PASS; summary:', args.run_dir / 'batch_summary.json')


if __name__ == '__main__':
    main()
