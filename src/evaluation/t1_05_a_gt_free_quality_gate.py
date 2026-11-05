"""T1-05-A: GT-pose-blind *heuristic review* of frozen B4 ICP hypotheses.

Reads ONLY frozen_predictions.json and its exact candidate_*.json files.
Does not read scene_gt.json, B4 batch_summary.json or any B2 evaluation.
Passing a heuristic gate NEVER certifies 6D pose accuracy, geometry validity,
industrial safety, or BOP benchmark success. Oracle instance masks remain used.

The support fraction is meaningful only with the SAME distance threshold that
was used to calculate it at prediction time; this module DOES NOT recompute
support from camera/model point sets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np

from src.evaluation.t1_04_a_pose_metrics import PoseM2C
from src.evaluation import t1_04_b4_fixed_50_batch as b4
from src.evaluation import t1_04_b_gt_free_icp as b1

GATE_SCHEMA = 't1-05-a-gt-free-heuristic-review-v1'
PASS = 'PASS_HEURISTIC_NOT_POSE_ACCURACY'
REVIEW = 'REVIEW_HEURISTIC_NOT_REJECTED_BY_GT'


def _finite_number(v, *, name: str) -> float:
    if type(v) not in (int, float) or not math.isfinite(float(v)):
        raise ValueError(f'INVALID_{name}')
    return float(v)


@dataclass(frozen=True)
class GateThresholds:
    min_support_fraction: float = 0.80
    max_trimmed_rmse_m: float = 0.004
    support_threshold_m: float = 0.008

    def __post_init__(self):
        support = _finite_number(self.min_support_fraction, name='MIN_SUPPORT_FRACTION')
        rmse = _finite_number(self.max_trimmed_rmse_m, name='MAX_TRIMMED_RMSE')
        threshold = _finite_number(self.support_threshold_m, name='SUPPORT_THRESHOLD')
        if not 0 <= support <= 1:
            raise ValueError('INVALID_MIN_SUPPORT_FRACTION')
        if not rmse > 0:
            raise ValueError('INVALID_MAX_TRIMMED_RMSE')
        if not threshold > 0:
            raise ValueError('INVALID_SUPPORT_THRESHOLD')
        object.__setattr__(self, 'min_support_fraction', support)
        object.__setattr__(self, 'max_trimmed_rmse_m', rmse)
        object.__setattr__(self, 'support_threshold_m', threshold)


def inspect_candidate(candidate: dict, thresholds: GateThresholds) -> dict:
    """Non-GT-only decision on fixed pose metrics. Fail closed on bad metadata.

    PASS means this candidate met two declared diagnostic thresholds only.
    It is NOT a probability, precision result, or deployment approval.
    """
    if not isinstance(candidate, dict):
        raise ValueError('INVALID_CANDIDATE_DOCUMENT')
    if (candidate.get('schema') != 't1-04-b1-oracle-mask-gt-blind-icp-v1'
            or candidate.get('status') != 'GEOMETRIC_CANDIDATE_UNVERIFIED'
            or candidate.get('gt_pose_read_by_predictor') is not False
            or candidate.get('gt_pose_used_to_construct_candidate') is not False
            or candidate.get('oracle_mask_used') is not True
            or candidate.get('publishable') is not False
            or candidate.get('position_m') is not None):
        raise ValueError('CANDIDATE_SAFETY_CONTRACT_VIOLATION')
    if any(k in candidate for k in ('ground_truth', 'gt_pose', 'translation_error_mm',
                                    'rotation_error_deg_raw', 'scene_gt')):
        raise ValueError('GT_FIELD_IN_PREDICTION_DOCUMENT')
    try:
        raw_pose = candidate['pose_candidate_model_to_camera']
        R = np.asarray(raw_pose['R_row_major'], dtype=np.float64).reshape(3, 3)
        t = np.asarray(raw_pose['t_m'], dtype=np.float64)
        # Reuse the strict SO(3) and frame convention contract.
        PoseM2C(R=R, t_m=t)
        metrics = candidate['registration_diagnostic_only']
        fraction = _finite_number(metrics['support_fraction'], name='SUPPORT_FRACTION')
        rmse = _finite_number(metrics['trimmed_rmse_m'], name='TRIMMED_RMSE')
        actual_threshold = _finite_number(metrics['support_threshold_m'],
                                          name='OBSERVED_SUPPORT_THRESHOLD')
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc) in (
                'INVALID_SUPPORT_FRACTION', 'INVALID_TRIMMED_RMSE',
                'INVALID_OBSERVED_SUPPORT_THRESHOLD', 'INVALID_ROTATION_MATRIX',
                'NONFINITE_ROTATION', 'INVALID_TRANSLATION_SHAPE',
                'NONFINITE_TRANSLATION'):
            raise
        raise ValueError('INVALID_CANDIDATE_GEOMETRY_OR_DIAGNOSTIC') from exc
    if not (0 <= fraction <= 1) or rmse < 0:
        raise ValueError('INVALID_DIAGNOSTIC_RANGE')
    if not math.isclose(actual_threshold, thresholds.support_threshold_m,
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError('SUPPORT_METRIC_THRESHOLD_MISMATCH')
    reasons = []
    if fraction < thresholds.min_support_fraction:
        reasons.append('SUPPORT_FRACTION_BELOW_THRESHOLD')
    if rmse > thresholds.max_trimmed_rmse_m:
        reasons.append('TRIMMED_RMSE_ABOVE_THRESHOLD')
    return {
        'decision': PASS if not reasons else REVIEW,
        'reasons': reasons,
        'support_fraction': fraction,
        'trimmed_rmse_m': rmse,
        'support_threshold_m': actual_threshold,
        'pose_accuracy_verified': False,
        'gt_pose_read_by_gate': False,
    }


def inspect_frozen_batch(*, frozen_dir: Path, thresholds: GateThresholds,
                         output: Path) -> dict:
    """All frames counted; verify B4 frozen hashes without loading GT reports."""
    frozen_dir, output = Path(frozen_dir), Path(output)
    if output.exists():
        raise FileExistsError('OUTPUT_ALREADY_EXISTS')
    if output.resolve().is_relative_to(frozen_dir.resolve()):
        raise ValueError('OUTPUT_MUST_BE_OUTSIDE_FROZEN_DIRECTORY')
    freeze = b4._load_frozen(frozen_dir)  # validates every candidate SHA/code SHA
    rows = []
    for record in freeze['candidates']:
        candidate = b4._read_json(frozen_dir / record['candidate_file'])
        if (candidate.get('observation', {}).get('im_id') != record['im_id']
                or candidate.get('observation', {}).get('gt_index') != freeze['gt_index']
                or candidate.get('model', {}).get('obj_id') != freeze['obj_id']):
            raise ValueError('CANDIDATE_IDENTITY_MISMATCH')
        result = inspect_candidate(candidate, thresholds)
        rows.append({'im_id': record['im_id'], 'candidate_sha256': record['candidate_sha256'],
                     **result})
    count = len(rows)
    passed = sum(r['decision'] == PASS for r in rows)
    report = {
        'schema': GATE_SCHEMA,
        'status': 'HEURISTIC_DIAGNOSTIC_ONLY_NOT_ACCURACY',
        'frozen_predictions_sha256': b1._sha256(frozen_dir / 'frozen_predictions.json'),
        'gate_source_sha256': b1._sha256(Path(__file__)),
        'frame_count': count,
        'pass_heuristic_count': passed,
        'review_count': count-passed,
        'pass_heuristic_fraction': passed / count,
        'thresholds': asdict(thresholds),
        'oracle_mask_used': True,
        'gt_pose_read_by_gate': False,
        'prior_gt_exposure_disclaimer': (
            'This LM 50-frame set was previously inspected using GT; thresholds '
            'and comparisons are an exploratory diagnostic, NOT unbiased model selection.'),
        'prediction_correctness_claimed': False,
        'bop_benchmark_score_claimed': False,
        'publishable': False,
        'position_m': None,
        'decisions': rows,
    }
    b4._save_new_json(output, report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-dir', type=Path, required=True)
    p.add_argument('--min-support', type=float, default=0.80)
    p.add_argument('--max-rmse-mm', type=float, default=4.0)
    p.add_argument('--support-threshold-mm', type=float, default=8.0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    thresholds = GateThresholds(args.min_support, args.max_rmse_mm / 1000.0,
                                args.support_threshold_mm / 1000.0)
    report = inspect_frozen_batch(frozen_dir=args.frozen_dir, thresholds=thresholds,
                                  output=args.output)
    print('Frozen candidates inspected:', report['frame_count'])
    print('PASS_HEURISTIC (not pose accuracy):', report['pass_heuristic_count'])
    print('REVIEW_HEURISTIC:', report['review_count'])
    print('PASS_HEURISTIC fraction:', report['pass_heuristic_fraction'])
    print('GT pose read by gate:', report['gt_pose_read_by_gate'])
    print('No GT-based accuracy or deployment claim; see report disclaimer.')
    print('Report:', args.output)


if __name__ == '__main__':
    main()
