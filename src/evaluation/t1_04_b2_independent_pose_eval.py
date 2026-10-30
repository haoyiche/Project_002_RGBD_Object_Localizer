"""T1-04-B2: evaluate a previously saved GT-pose-blind 6D pose candidate.

This module NEVER runs registration or chooses candidates using GT. It first
validates the saved B1 hypothesis, current image/model hashes and identity,
then reads scene_gt.json solely to measure annotation-relative pose errors.

Not a symmetry-aware BOP benchmark, independently measured physical accuracy,
or a robot command. The oracle instance mask and object ID are supervised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src.evaluation.t1_04_a_pose_metrics import PoseM2C, compare_pose_m2c


CANDIDATE_SCHEMA = 't1-04-b1-oracle-mask-gt-blind-icp-v1'
EVAL_SCHEMA = 't1-04-b2-independent-bop-gt-evaluation-v1'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, reason: str):
    if not condition:
        raise ValueError(reason)


def _read_candidate(path: Path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('CANDIDATE_FILE_MISSING')
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('INVALID_CANDIDATE_JSON') from exc
    _require(isinstance(record, dict), 'INVALID_CANDIDATE_RECORD')
    _require(record.get('schema') == CANDIDATE_SCHEMA, 'INVALID_CANDIDATE_SCHEMA')
    _require(record.get('status') == 'GEOMETRIC_CANDIDATE_UNVERIFIED',
             'INVALID_CANDIDATE_STATUS')
    _require(record.get('estimation_performed') is True, 'NOT_A_POSE_ESTIMATE')
    _require(record.get('gt_pose_read_by_predictor') is False
             and record.get('gt_pose_used_to_construct_candidate') is False,
             'GT_LEAKAGE_OR_UNVERIFIABLE_CANDIDATE')
    _require(record.get('oracle_mask_used') is True
             and record.get('oracle_supervision_used') is True,
             'ORACLE_INPUT_CONTRACT_MISMATCH')
    _require(record.get('publishable') is False and record.get('position_m') is None,
             'UNSAFE_CANDIDATE_PUBLICATION_STATE')
    return record


def _check_identity_and_files(candidate, scene_dir: Path, models_dir: Path):
    try:
        obs = candidate['observation']
        model = candidate['model']
        im_id = obs['im_id']
        gt_index = obs['gt_index']
        obj_id = model['obj_id']
        hashes = obs['sha256']
        model_hash = model['mesh_sha256']
        model_info_hash = model['models_info_sha256']
    except (KeyError, TypeError) as exc:
        raise ValueError('CANDIDATE_AUDIT_MISSING') from exc
    _require(all(type(value) is int and value >= 0 for value in
                 (im_id, gt_index)) and type(obj_id) is int and obj_id > 0,
             'INVALID_CANDIDATE_IDENTITY')
    _require(isinstance(hashes, dict), 'CANDIDATE_AUDIT_MISSING')
    _require(obs.get('oracle_mask_used') is True
             and obs.get('gt_pose_read_by_predictor') is False,
             'CANDIDATE_AUDIT_FLAGS_INVALID')
    _require(model.get('model_units_input') == 'mm'
             and model.get('model_units_internal') == 'm', 'MODEL_UNITS_NOT_VERIFIED')

    image_stem = f'{im_id:06d}'
    paths = {
        'scene_camera': scene_dir / 'scene_camera.json',
        'rgb': scene_dir / 'rgb' / f'{image_stem}.png',
        'depth': scene_dir / 'depth' / f'{image_stem}.png',
        'mask_visib': scene_dir / 'mask_visib' / f'{image_stem}_{gt_index:06d}.png',
    }
    for name, path in paths.items():
        _require(path.is_file(), f'RECORDED_INPUT_MISSING:{name}')
        _require(hashes.get(name) == sha256_file(path), f'RECORDED_INPUT_HASH_MISMATCH:{name}')

    model_paths = {
        'mesh': models_dir / f'obj_{obj_id:06d}.ply',
        'models_info': models_dir / 'models_info.json',
    }
    for name, path in model_paths.items():
        _require(path.is_file(), f'RECORDED_MODEL_MISSING:{name}')
        expected = model_hash if name == 'mesh' else model_info_hash
        _require(expected == sha256_file(path), f'RECORDED_MODEL_HASH_MISMATCH:{name}')
    return im_id, gt_index, obj_id


def evaluate_saved_candidate(*, candidate_path, scene_dir, models_dir) -> dict:
    """Verify fixed prediction, THEN read pose GT; never refit or select by GT."""
    candidate_path = Path(candidate_path)
    scene_dir = Path(scene_dir)
    models_dir = Path(models_dir)
    record = _read_candidate(candidate_path)
    im_id, gt_index, obj_id = _check_identity_and_files(record, scene_dir, models_dir)

    # A valid SO(3)/SE(3) estimate must be stored, not invented by the evaluator.
    try:
        pose_data = record['pose_candidate_model_to_camera']
        _require(isinstance(pose_data, dict), 'INVALID_CANDIDATE_POSE')
        R_est = np.asarray(pose_data['R_row_major'], dtype=np.float64).reshape(3, 3)
        t_est_m = np.asarray(pose_data['t_m'], dtype=np.float64)
        estimate = PoseM2C(R_est, t_est_m)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('INVALID_CANDIDATE_POSE') from exc

    # Strict separation: only the EVALUATOR accesses scene_gt.json, and only
    # after verifying the persisted candidate and its observation/model hashes.
    gt_path = scene_dir / 'scene_gt.json'
    _require(gt_path.is_file(), 'GT_NOT_AVAILABLE_FOR_EVALUATION')
    try:
        gt_json = json.loads(gt_path.read_text(encoding='utf-8'))
        entries = gt_json[str(im_id)]
        _require(isinstance(entries, list) and gt_index < len(entries),
                 'GT_INSTANCE_NOT_FOUND')
        gt_entry = entries[gt_index]
        _require(type(gt_entry['obj_id']) is int and gt_entry['obj_id'] == obj_id,
                 'GT_OBJECT_ID_MISMATCH')
        gt = PoseM2C.from_bop(
            np.asarray(gt_entry['cam_R_m2c'], dtype=np.float64).reshape(3, 3),
            gt_entry['cam_t_m2c'],
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc) in (
                'GT_INSTANCE_NOT_FOUND', 'GT_OBJECT_ID_MISMATCH'):
            raise
        raise ValueError('INVALID_GT_POSE_OR_SCENE_GT') from exc

    metrics = compare_pose_m2c(estimate, gt, symmetry_policy='raw')
    return {
        'schema': EVAL_SCHEMA,
        'status': 'ANNOTATION_RELATIVE_EVALUATION_DIAGNOSTIC',
        'candidate_sha256': sha256_file(candidate_path),
        'gt_annotation_sha256': sha256_file(gt_path),
        'sample': {'im_id': im_id, 'gt_index': gt_index, 'obj_id': obj_id},
        'candidate_estimation_performed': True,
        'gt_pose_used_for_estimation': False,  # B1 provenance claim; no access in B1 test
        'gt_pose_read_by_evaluator': True,
        'oracle_mask_used': True,
        'translation_error_mm': metrics['translation_error_mm'],
        'rotation_error_deg_raw': metrics['rotation_error_deg_raw'],
        'symmetry_aware': False,
        'bop_benchmark_score_claimed': False,
        'physical_accuracy_claimed': False,
        'publishable': False,
        'position_m': None,
        'reference_kind': 'BOP_POSE_ANNOTATION_NOT_INDEPENDENT_PHYSICAL_GT',
        'note': ('A single fixed oracle-mask PCA+ICP candidate compared to dataset GT; '
                 'raw rotation angle is not symmetry-aware and this is not a BOP score.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('OUTPUT_ALREADY_EXISTS_USE_NEW_RUN_PATH')
    report = evaluate_saved_candidate(
        candidate_path=args.candidate, scene_dir=args.scene_dir,
        models_dir=args.models_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent,
                                prefix='.t1_04_b2_', suffix='.tmp', delete=False) as out:
            tmp = Path(out.name)
            json.dump(report, out, indent=2, ensure_ascii=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        if args.output.exists():
            raise FileExistsError('OUTPUT_ALREADY_EXISTS_USE_NEW_RUN_PATH')
        os.replace(tmp, args.output)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    print('Fixed candidate evaluated:', args.candidate)
    print('GT pose read by evaluator:', report['gt_pose_read_by_evaluator'])
    print('Translation error to dataset GT (mm):', report['translation_error_mm'])
    print('Raw rotation error to dataset GT (deg):', report['rotation_error_deg_raw'])
    print('Symmetry-aware:', report['symmetry_aware'])
    print('BOP benchmark score claimed:', report['bop_benchmark_score_claimed'])
    print('Physical accuracy claimed:', report['physical_accuracy_claimed'])
    print('publishable:', report['publishable'], 'position_m:', report['position_m'])
    print('T1-04-B2 annotation-relative evaluation: PASS (not physical accuracy)')
    print('Evaluation report:', args.output)


if __name__ == '__main__':
    main()
