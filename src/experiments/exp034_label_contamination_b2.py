"""EXP034-B2: one deliberately incorrect cluster label contaminates localization.

Controlled independent variable: change label of existing point P11 from -1 to 0.
Keep all XYZ, B1 duplicate sampling points, and every other label fixed.
This tests downstream localization robustness, NOT whether DBSCAN would make this error.
"""
import json
from pathlib import Path

import numpy as np

from src.experiments.exp034_cluster_localization_minimal import (
    compare_with_open3d,
    localize_clusters,
)
from src.experiments.exp034_density_bias_b1 import make_density_biased_scene


P11_INDEX = 11
P11_XYZ = np.array([0.90, 0.50, 1.200], dtype=np.float64)


def make_label_contaminated_scene():
    """Return B1 points unchanged, with ONLY P11's label changed -1 -> 0."""
    points, labels = make_density_biased_scene()
    new_labels = labels.copy()
    assert labels[P11_INDEX] == -1
    np.testing.assert_array_equal(points[P11_INDEX], P11_XYZ)
    new_labels[P11_INDEX] = 0
    return points, new_labels


def demo_gates(nominal_object, candidate_object):
    """Illustrative gates, NOT calibrated production acceptance criteria."""
    nominal_extent = np.asarray(nominal_object['extent_m'], dtype=np.float64)
    extent = np.asarray(candidate_object['extent_m'], dtype=np.float64)
    centroid = np.asarray(candidate_object['centroid_m'], dtype=np.float64)
    bbox_center = np.asarray(candidate_object['bbox_center_m'], dtype=np.float64)
    distance = float(np.linalg.norm(centroid - bbox_center))
    return {
        'point_count_gate_5_to_20_pass': bool(5 <= candidate_object['point_count'] <= 20),
        'extent_gate_max_twice_B1_pass': bool(np.all(extent <= 2 * nominal_extent)),
        'centroid_bbox_distance_gate_0p02m_pass': bool(distance <= 0.02),
        'centroid_to_bbox_center_distance_m': distance,
        'disclaimer': 'Illustrative thresholds only; real thresholds require validation on known objects and failure cases.',
    }


def build_report():
    before_points, before_labels = make_density_biased_scene()
    after_points, after_labels = make_label_contaminated_scene()

    # Strong controlled-variable audit: no coordinates or other labels changed.
    np.testing.assert_array_equal(after_points, before_points)
    changed_indices = np.flatnonzero(after_labels != before_labels)
    np.testing.assert_array_equal(changed_indices, [P11_INDEX])
    assert (before_labels[P11_INDEX], after_labels[P11_INDEX]) == (-1, 0)
    print('Single-label intervention (P11 only): PASS')

    before = {o['cluster_label']: o for o in localize_clusters(before_points, before_labels)}
    after = {o['cluster_label']: o for o in localize_clusters(after_points, after_labels)}
    assert set(before) == set(after) == {0, 1}
    assert [after[i]['point_count'] for i in (0, 1)] == [10, 5]
    assert int(np.count_nonzero(after_labels == -1)) == 1
    assert after_labels[10] == -1
    print('Label, cluster and noise counts: PASS')

    # Predicted from hand sums for B1 A: (0.09,0.09,10.40); add P11: (0.90,0.50,1.20).
    expected_A_center = np.array([0.99 / 10, 0.59 / 10, 11.60 / 10])
    expected_min = np.array([0.00, 0.00, 1.150])
    expected_max = np.array([0.90, 0.50, 1.200])
    expected_bbox_center = np.array([0.45, 0.25, 1.175])
    expected_extent = np.array([0.90, 0.50, 0.050])
    for key, oracle in (
        ('centroid_m', expected_A_center),
        ('bbox_min_m', expected_min),
        ('bbox_max_m', expected_max),
        ('bbox_center_m', expected_bbox_center),
        ('extent_m', expected_extent),
    ):
        np.testing.assert_allclose(after[0][key], oracle, rtol=0, atol=1e-12)
    print('Independent hand-computed localization oracle: PASS')

    # Object B remains invariant: a strong negative control.
    for key in ('centroid_m', 'bbox_min_m', 'bbox_max_m', 'bbox_center_m', 'extent_m'):
        np.testing.assert_allclose(after[1][key], before[1][key], rtol=0, atol=1e-12)
    assert after[1]['point_count'] == before[1]['point_count'] == 5
    print('Control object B remains unchanged: PASS')

    centroid_shift_mm = (
        np.asarray(after[0]['centroid_m']) - np.asarray(before[0]['centroid_m'])
    ) * 1000.0
    np.testing.assert_allclose(centroid_shift_mm, [89.0, 49.0, 40 / 9], rtol=0, atol=1e-9)
    extent_factor = np.asarray(after[0]['extent_m']) / np.asarray(before[0]['extent_m'])
    np.testing.assert_allclose(extent_factor, [45.0, 25.0, 5.0], rtol=0, atol=1e-9)
    print(f'Centroid shift: PASS | (X,Y,Z) = {centroid_shift_mm.round(6).tolist()} mm')
    print(f'AABB extent inflation: PASS | (X,Y,Z) = {extent_factor.round(6).tolist()} times')

    nominal_gates = demo_gates(before[0], before[0])
    contaminated_gates = demo_gates(before[0], after[0])
    assert nominal_gates['point_count_gate_5_to_20_pass']
    assert contaminated_gates['point_count_gate_5_to_20_pass']
    assert nominal_gates['extent_gate_max_twice_B1_pass']
    assert nominal_gates['centroid_bbox_distance_gate_0p02m_pass']
    assert not contaminated_gates['extent_gate_max_twice_B1_pass']
    assert not contaminated_gates['centroid_bbox_distance_gate_0p02m_pass']
    np.testing.assert_allclose(
        contaminated_gates['centroid_to_bbox_center_distance_m'],
        np.sqrt(0.351**2 + 0.191**2 + 0.015**2), rtol=0, atol=1e-12,
    )
    print('Example gates: PASS | count misses pollution; geometry gates reject it')

    report = {
        'experiment': 'EXP034-B2',
        'design': 'B1 point set unchanged; ONLY P11 label -1 -> 0 (synthetic upstream label error).',
        'caution': 'Not a claim that DBSCAN would output this label; not a real-object-center accuracy test.',
        'objects_before': [before[0], before[1]],
        'objects_after': [after[0], after[1]],
        'noise_count_before': int(np.count_nonzero(before_labels == -1)),
        'noise_count_after': int(np.count_nonzero(after_labels == -1)),
        'A_centroid_shift_mm': centroid_shift_mm.tolist(),
        'A_extent_inflation_factors': extent_factor.tolist(),
        'gates_before': nominal_gates,
        'gates_after': contaminated_gates,
    }
    return report, after_points, after_labels


def main():
    report, points, labels = build_report()
    compare_with_open3d(points, labels, report['objects_after'])
    print('\n=== EXP034-B2 OBJECT LOCALIZATION ===')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    out = Path('results/exp034/exp034_B2_label_contamination.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Saved:', out)
    print('EXP034-B2 VERIFIED. EXP034 UNIT COMPLETION: PENDING')


if __name__ == '__main__':
    main()
