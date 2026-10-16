"""EXP034-C: validated, per-cluster 3D localization output contract.

All lengths are meters. A gate passing means 'accepted by this configured
policy', not physically correct localization or safety certification.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalizationGates:
    """Illustrative defaults from EXP034-B2, NOT calibrated production limits."""

    min_points: int = 5
    max_points: int = 20
    max_extent_m: tuple = (0.04, 0.04, 0.02)
    max_centroid_bbox_gap_m: float = 0.02

    def __post_init__(self):
        for name in ('min_points', 'max_points'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'{name} must be an integer (not bool)')
        if self.min_points < 1 or self.max_points < self.min_points:
            raise ValueError('point-count limits must satisfy 1 <= min_points <= max_points')
        try:
            extent = np.asarray(self.max_extent_m, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('max_extent_m must contain numeric values') from exc
        if extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent <= 0):
            raise ValueError('max_extent_m must be three finite, positive lengths in meters')
        try:
            gap = float(self.max_centroid_bbox_gap_m)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('max_centroid_bbox_gap_m must be numeric') from exc
        if (isinstance(self.max_centroid_bbox_gap_m, (bool, np.bool_))
                or not np.isfinite(gap) or gap < 0):
            raise ValueError('max_centroid_bbox_gap_m must be finite and nonnegative')


def _validate_inputs(points, labels, frame_id):
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError('frame_id must be a nonempty string')
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.dtype.kind not in 'fiu':
        raise ValueError('points must be numeric with shape (N,3)')
    pts = pts.astype(np.float64)
    if not np.isfinite(pts).all():
        raise ValueError('points must be finite')
    labs = np.asarray(labels)
    if labs.shape != (len(pts),) or labs.dtype.kind not in 'iu':
        raise ValueError('labels must be (N,) integer values')
    if np.any(labs < -1):
        raise ValueError('labels must be -1 or nonnegative cluster IDs')
    return pts, labs


def _distance_3d(a, b):
    """Use hypot rather than squaring, avoiding avoidable overflow/underflow."""
    delta = a - b
    return float(np.hypot(np.hypot(delta[0], delta[1]), delta[2]))


def validate_cluster_localization(points, labels, *, frame_id='camera_optical_frame', gates=None):
    """Localize each nonnegative label independently, with explicit publication gate.

    Invalid API input raises ValueError. A geometrically rejected, well-formed
    cluster gets INVALID_CLUSTER and position_m=None, plus diagnostics.
    All noise (or valid empty (0,3) data) returns NO_OBJECTS, not an exception.
    """
    if gates is None:
        gates = LocalizationGates()
    if not isinstance(gates, LocalizationGates):
        raise ValueError('gates must be a LocalizationGates instance')
    pts, labs = _validate_inputs(points, labels, frame_id)
    objects = []
    valid_count = 0
    for cluster_id in np.unique(labs[labs >= 0]):
        cluster = pts[labs == cluster_id]
        centroid = cluster.mean(axis=0)
        lower = cluster.min(axis=0)
        upper = cluster.max(axis=0)
        extent = upper - lower
        bbox_center = lower + 0.5 * extent
        gap_m = _distance_3d(centroid, bbox_center)
        if not (np.isfinite(centroid).all() and np.isfinite(extent).all()
                and np.isfinite(bbox_center).all() and np.isfinite(gap_m)):
            raise ValueError('geometry is not representable by finite float64 values')

        reasons = []
        if not gates.min_points <= len(cluster) <= gates.max_points:
            reasons.append('POINT_COUNT_OUT_OF_RANGE')
        if np.any(extent > np.asarray(gates.max_extent_m)):
            reasons.append('EXTENT_EXCEEDED')
        if gap_m > gates.max_centroid_bbox_gap_m:
            reasons.append('CENTROID_BBOX_GAP')
        accepted = not reasons
        if accepted:
            valid_count += 1
        objects.append({
            'cluster_label': int(cluster_id),
            'frame_id': frame_id,
            'status': 'VALID' if accepted else 'INVALID_CLUSTER',
            'publishable': accepted,
            'position_m': centroid.tolist() if accepted else None,
            'reasons': reasons,
            'diagnostics': {
                'point_count': int(len(cluster)),
                'centroid_m': centroid.tolist(),
                'bbox_min_m': lower.tolist(),
                'bbox_max_m': upper.tolist(),
                'bbox_center_m': bbox_center.tolist(),
                'extent_m': extent.tolist(),
                'centroid_bbox_distance_m': gap_m,
            },
        })
    invalid_count = len(objects) - valid_count
    if not objects:
        status = 'NO_OBJECTS'
    elif not invalid_count:
        status = 'OK'
    elif not valid_count:
        status = 'ALL_INVALID'
    else:
        status = 'PARTIAL_INVALID'
    return {
        'frame_id': frame_id,
        'status': status,
        'objects': objects,
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'noise_count': int(np.count_nonzero(labs == -1)),
    }


def extract_publishable_positions(report):
    """Downstream helper: exclude invalid-cluster diagnostics from position stream."""
    published = []
    for item in report['objects']:
        if item['status'] == 'VALID' and item['publishable'] and item['position_m'] is not None:
            published.append({
                'cluster_label': item['cluster_label'],
                'frame_id': item['frame_id'],
                'position_m': item['position_m'],
            })
    return published
