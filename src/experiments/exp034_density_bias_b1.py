"""EXP034-B1: change only the sampling density of cluster A.

The cluster labels are deliberately held fixed. This experiment is about
localization geometry, not about rerunning DBSCAN or changing its parameters.
"""
import json
from pathlib import Path

import numpy as np

from src.experiments.exp034_cluster_localization_minimal import (
    compare_with_open3d,
    localize_clusters,
    make_scene,
)


EXTRA_POINT = np.array([0.01, 0.01, 1.160], dtype=np.float64)
EXTRA_COUNT = 4


def make_density_biased_scene():
    """Return original 12 points plus four identical A points (label 0)."""
    points, labels = make_scene()
    extra_points = np.repeat(EXTRA_POINT[np.newaxis, :], EXTRA_COUNT, axis=0)
    extra_labels = np.zeros(EXTRA_COUNT, dtype=labels.dtype)
    return np.vstack([points, extra_points]), np.concatenate([labels, extra_labels])


def verify_prediction():
    """Check hand-derived oracle, untouched control B, and Open3D geometry."""
    original_points, original_labels = make_scene()
    points, labels = make_density_biased_scene()

    # Controlled-variable check: no original point or label was modified.
    np.testing.assert_array_equal(points[:12], original_points)
    np.testing.assert_array_equal(labels[:12], original_labels)
    np.testing.assert_array_equal(points[12:], np.tile(EXTRA_POINT, (EXTRA_COUNT, 1)))
    np.testing.assert_array_equal(labels[12:], np.zeros(EXTRA_COUNT, dtype=np.int64))

    original = {o['cluster_label']: o for o in localize_clusters(original_points, original_labels)}
    modified = {o['cluster_label']: o for o in localize_clusters(points, labels)}
    assert set(original) == set(modified) == {0, 1}
    assert [modified[i]['point_count'] for i in (0, 1)] == [9, 5]
    assert int(np.count_nonzero(labels == -1)) == 2
    print('Fixed-label controlled modification: PASS')

    # Independent hand-computed oracle: 4 points at z=1.150, 5 at z=1.160.
    expected_z = (4 * 1.150 + 5 * 1.160) / 9
    expected_A = np.array([0.01, 0.01, expected_z])
    np.testing.assert_allclose(modified[0]['centroid_m'], expected_A, rtol=0, atol=1e-12)
    np.testing.assert_allclose(modified[1]['centroid_m'], [0.51, 0.01, 1.152], rtol=0, atol=1e-12)
    print('Independent hand-computed oracle: PASS')

    # Sampling density cannot change extrema when all added points are inside AABB.
    for field in ('bbox_min_m', 'bbox_max_m', 'bbox_center_m', 'extent_m'):
        np.testing.assert_allclose(modified[0][field], original[0][field], rtol=0, atol=1e-12)
        np.testing.assert_allclose(modified[1][field], original[1][field], rtol=0, atol=1e-12)
    np.testing.assert_allclose(modified[1]['centroid_m'], original[1]['centroid_m'], rtol=0, atol=1e-12)
    print('AABB invariance and unchanged object B: PASS')

    old_z = original[0]['centroid_m'][2]
    new_z = modified[0]['centroid_m'][2]
    bbox_z = modified[0]['bbox_center_m'][2]
    shift_mm = (new_z - old_z) * 1000
    assert old_z < bbox_z < new_z
    np.testing.assert_allclose(shift_mm, ((4 * 1.150 + 5 * 1.160) / 9 - 1.152) * 1000,
                               rtol=0, atol=1e-9)
    print(f'Centroid crossing: PASS | Z shift = {shift_mm:.6f} mm')
    print(f'Centroid Z relative to AABB center = {(new_z - bbox_z) * 1000:.6f} mm')

    # Check independently implemented Open3D methods against NumPy output.
    compare_with_open3d(points, labels, [modified[0], modified[1]])

    payload = {
        'experiment': 'EXP034-B1',
        'note': 'Fixed DBSCAN labels; only four repeated A points were added. Ground-truth solid center is unknown.',
        'objects_before': [original[0], original[1]],
        'objects_after': [modified[0], modified[1]],
        'noise_count_before': int(np.count_nonzero(original_labels == -1)),
        'noise_count_after': int(np.count_nonzero(labels == -1)),
        'A_centroid_shift_mm': [
            float((np.array(modified[0]['centroid_m']) - np.array(original[0]['centroid_m']))[i] * 1000)
            for i in range(3)
        ],
    }
    return payload


def main():
    payload = verify_prediction()
    print('\n=== EXP034-B1 OBJECT LOCALIZATION ===')
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    out = Path('results/exp034/exp034_B1_density_bias.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Saved:', out)
    print('EXP034-B1 VERIFIED. EXP034 UNIT COMPLETION: PENDING')


if __name__ == '__main__':
    main()
