"""OS-01-D2: segment a dominant BACKGROUND plane from a full RGB-D scene.

Synthetic scene reused from EXP028, not a calibrated physical tabletop.
All XYZ and RANSAC thresholds are in meters; output is diagnostics only.
"""

import numpy as np
import open3d as o3d

from src.experiments.exp028_open3d_target_visualization import create_scene
from src.geometry.mask_to_pointcloud import mask_to_pointcloud


INTRINSICS = dict(fx=100.0, fy=100.0, cx=49.5, cy=39.5)


def analyze_scene(*, scene=None):
    """Analyze supplied (depth_m, rgb, mask) or the legacy demo scene.

    A supplied scene allows downstream stages to consume the very same
    capture that D2 validated. The old no-argument API is preserved.
    """
    if scene is None:
        depth_m, rgb, target_mask = create_scene()
    else:
        if not isinstance(scene, (tuple, list)) or len(scene) != 3:
            raise ValueError("scene must be (depth_m, rgb, target_mask)")
        depth_m, rgb, target_mask = scene

    # Full-scene cloud: 8000 pixels, of which 100 have invalid depth.
    scene_points, scene_colors, valid_scene = mask_to_pointcloud(
        depth_m, rgb, np.ones_like(target_mask, dtype=bool), **INTRINSICS
    )

    # Target-only mask is a separate SYNTHETIC comparison label.
    # It is neither independent physical ground truth nor used to guide RANSAC.
    target_points, target_colors, valid_target = mask_to_pointcloud(
        depth_m, rgb, target_mask, **INTRINSICS
    )

    scene_cloud = o3d.geometry.PointCloud()
    scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
    scene_cloud.colors = o3d.utility.Vector3dVector(scene_colors)

    plane, inlier_indices = scene_cloud.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=300,
    )

    # Convert Open3D's inlier index list back to the full cloud's row order.
    is_plane = np.zeros(len(scene_points), dtype=bool)
    is_plane[np.asarray(inlier_indices, dtype=np.int64)] = True

    remaining_points = scene_points[~is_plane]
    remaining_colors = scene_colors[~is_plane]

    # The full-scene selected mask maps each cloud point to an RGB-D pixel.
    expected_target_membership = target_mask[valid_scene]
    membership_matches = np.array_equal(~is_plane, expected_target_membership)

    # Exact equality is expected for this deterministic, noiseless scene.
    points_match = np.array_equal(remaining_points, target_points)
    colors_match = np.array_equal(remaining_colors, target_colors)

    # The background occupies the plane Z=2.0 m in camera coordinates.
    normal = np.asarray(plane[:3], dtype=np.float64)
    d = float(plane[3])
    normal_norm = float(np.linalg.norm(normal))
    reference_plane_error_m = float(
        abs(normal @ np.array([0.0, 0.0, 2.0]) + d) / normal_norm
    )

    accepted = (
        len(scene_points) == 7900
        and len(target_points) == 1500
        and np.count_nonzero(is_plane) == 6400
        and membership_matches
        and points_match
        and colors_match
        and reference_plane_error_m < 1e-8
    )

    return {
        "status": "DIAGNOSTIC_ONLY" if accepted else "SEGMENTATION_MISMATCH",
        "scene_point_count": int(len(scene_points)),
        "plane_inlier_count": int(np.count_nonzero(is_plane)),
        "remaining_point_count": int(len(remaining_points)),
        "mask_target_point_count": int(len(target_points)),
        "invalid_depth_count": int(depth_m.size - np.count_nonzero(valid_scene)),
        "target_membership_matches": bool(membership_matches),
        "target_xyz_matches": bool(points_match),
        "target_rgb_matches": bool(colors_match),
        "reference_plane_error_m": reference_plane_error_m,
        "publishable": False,
        "position_m": None,
    }


def main():
    result = analyze_scene()
    print("Open3D:", o3d.__version__)
    for key, value in result.items():
        print(f"{key}: {value}")
    if result["status"] != "DIAGNOSTIC_ONLY":
        raise SystemExit("D2 synthetic scene segmentation verification FAILED")
    print("D2 background plane verification: PASS")


if __name__ == "__main__":
    main()
