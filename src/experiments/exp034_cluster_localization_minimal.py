"""EXP034-A: predict -> calculate -> compare NumPy against Open3D."""
import json
from pathlib import Path

import numpy as np

FRAME_ID = "camera_optical_frame"


def make_scene():
    points = np.array([
        [0.00, 0.00, 1.150], [0.02, 0.00, 1.150],
        [0.00, 0.02, 1.150], [0.02, 0.02, 1.150],
        [0.01, 0.01, 1.160],
        [0.50, 0.00, 1.150], [0.52, 0.00, 1.150],
        [0.50, 0.02, 1.150], [0.52, 0.02, 1.150],
        [0.51, 0.01, 1.160],
        [0.25, 0.25, 1.150], [0.90, 0.50, 1.200],
    ], dtype=np.float64)
    labels = np.array([0] * 5 + [1] * 5 + [-1] * 2, dtype=np.int64)
    return points, labels


def localize_clusters(points, labels):
    pts = np.asarray(points, dtype=np.float64)
    labs = np.asarray(labels)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if not np.isfinite(pts).all():
        raise ValueError("points must be finite")
    if labs.shape != (len(pts),) or labs.dtype.kind not in "iu":
        raise ValueError("labels must be (N,) integers")
    if np.any(labs < -1):
        raise ValueError("labels may only contain -1 or nonnegative IDs")

    objects = []
    for cluster_id in np.unique(labs[labs >= 0]):
        cluster = pts[labs == cluster_id]
        lower = cluster.min(axis=0)
        upper = cluster.max(axis=0)
        objects.append({
            "cluster_label": int(cluster_id),
            "frame_id": FRAME_ID,
            "point_count": int(len(cluster)),
            "centroid_m": cluster.mean(axis=0).tolist(),
            "bbox_min_m": lower.tolist(),
            "bbox_max_m": upper.tolist(),
            "bbox_center_m": ((lower + upper) / 2).tolist(),
            "extent_m": (upper - lower).tolist(),
        })
    return objects


def compare_with_open3d(points, labels, objects):
    import open3d as o3d

    print("Open3D:", o3d.__version__)
    for obj in objects:
        cloud_points = points[labels == obj["cluster_label"]]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_points)
        bbox = pcd.get_axis_aligned_bounding_box()
        np.testing.assert_allclose(obj["centroid_m"], pcd.get_center(), atol=1e-12)
        np.testing.assert_allclose(obj["bbox_min_m"], bbox.get_min_bound(), atol=1e-12)
        np.testing.assert_allclose(obj["bbox_max_m"], bbox.get_max_bound(), atol=1e-12)
        np.testing.assert_allclose(obj["bbox_center_m"], bbox.get_center(), atol=1e-12)
        np.testing.assert_allclose(obj["extent_m"], bbox.get_extent(), atol=1e-12)
    print("Open3D comparison: PASS")


def main():
    points, labels = make_scene()
    objects = localize_clusters(points, labels)
    assert len(objects) == 2
    assert [o["point_count"] for o in objects] == [5, 5]
    np.testing.assert_array_equal(np.flatnonzero(labels == -1), [10, 11])

    expected_centroids = [[0.01, 0.01, 1.152], [0.51, 0.01, 1.152]]
    expected_bounds_min = [[0.0, 0.0, 1.15], [0.5, 0.0, 1.15]]
    expected_bounds_max = [[0.02, 0.02, 1.16], [0.52, 0.02, 1.16]]
    for i, obj in enumerate(objects):
        np.testing.assert_allclose(obj["centroid_m"], expected_centroids[i], atol=1e-12)
        np.testing.assert_allclose(obj["bbox_min_m"], expected_bounds_min[i], atol=1e-12)
        np.testing.assert_allclose(obj["bbox_max_m"], expected_bounds_max[i], atol=1e-12)
        np.testing.assert_allclose(obj["extent_m"], [0.02, 0.02, 0.01], atol=1e-12)
        np.testing.assert_allclose(obj["bbox_center_m"][2], 1.155, atol=1e-12)
        assert abs(obj["centroid_m"][2] - obj["bbox_center_m"][2]) > 0.002
    print("Independent hand-computed oracle: PASS")

    compare_with_open3d(points, labels, objects)

    polluted_A = np.vstack((points[labels == 0], points[labels == -1]))
    polluted_center = polluted_A.mean(axis=0)
    polluted_extent = np.ptp(polluted_A, axis=0)
    np.testing.assert_allclose(polluted_center, [1.20/7, 0.80/7, 8.11/7], atol=1e-12)
    np.testing.assert_allclose(polluted_extent, [0.90, 0.50, 0.05], atol=1e-12)
    print("Noise-contamination counterexample: PASS")

    print("\n=== OBJECT LOCALIZATION ===")
    print(json.dumps({"objects": objects, "noise_count": 2}, ensure_ascii=False, indent=2))
    out = Path("results/exp034/exp034_A_localization.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"objects": objects, "noise_count": 2}, indent=2), encoding="utf-8")
    print("Saved:", out)
    print("EXP034-A VERIFIED. Unit completion: PENDING")


if __name__ == "__main__":
    main()
