import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from src.geometry.depth_to_pointcloud_vectorized import (
    depth_to_pointcloud_vectorized,
)


OUTPUT = Path("results/exp027/synthetic_rgbd_colored.ply")


def create_synthetic_scene():
    height, width = 120, 160

    # Background plane: Z = 2.0 m
    depth_m = np.full(
        (height, width),
        2.0,
        dtype=np.float64,
    )

    rgb = np.empty(
        (height, width, 3),
        dtype=np.uint8,
    )
    rgb[:] = [150, 160, 175]

    # Closer orange target: Z = 1.2 m
    target_mask = np.zeros(
        (height, width),
        dtype=bool,
    )
    target_mask[40:80, 60:100] = True

    depth_m[target_mask] = 1.2
    rgb[target_mask] = [245, 135, 50]

    # Invalid region: 100 pixels
    depth_m[10:20, 20:30] = 0.0

    return depth_m, rgb, target_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the interactive 3D window.",
    )
    args = parser.parse_args()

    print("=== EXP027 Open3D Visualization ===")

    depth_m, rgb, target_mask = create_synthetic_scene()

    height, width = depth_m.shape

    params = dict(
        fx=140.0,
        fy=140.0,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
    )

    valid = np.isfinite(depth_m) & (depth_m > 0)

    points = depth_to_pointcloud_vectorized(
        depth_m=depth_m,
        **params,
    )

    # Identical mask and row-major order
    colors = rgb[valid].astype(np.float64) / 255.0

    # Which output rows belong to the target?
    target_rows = target_mask[valid]
    background_rows = ~target_rows

    # ====================================
    # Geometry and alignment checks
    # ====================================

    assert depth_m.shape == (120, 160)
    assert rgb.shape == (120, 160, 3)
    assert np.count_nonzero(valid) == 19100

    assert points.shape == (19100, 3)
    assert colors.shape == (19100, 3)

    assert np.count_nonzero(target_rows) == 1600
    assert np.count_nonzero(background_rows) == 17500

    np.testing.assert_allclose(
        points[target_rows, 2],
        1.2,
        atol=1e-12,
        rtol=0,
    )

    np.testing.assert_allclose(
        points[background_rows, 2],
        2.0,
        atol=1e-12,
        rtol=0,
    )

    assert np.all(
        rgb[valid][target_rows] == [245, 135, 50]
    )
    assert np.all(
        rgb[valid][background_rows] == [150, 160, 175]
    )

    assert np.isfinite(points).all()
    assert colors.min() >= 0.0
    assert colors.max() <= 1.0

    print("Depth shape:", depth_m.shape)
    print("RGB shape:", rgb.shape)
    print("Points shape:", points.shape)
    print("Colors shape:", colors.shape)
    print("Target points:", np.count_nonzero(target_rows))
    print("Background points:", np.count_nonzero(background_rows))
    print("Target Z:", np.unique(points[target_rows, 2]))
    print("Background Z:", np.unique(points[background_rows, 2]))
    print("Geometry and color checks: PASS")

    # ====================================
    # Open3D PointCloud
    # ====================================

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    assert len(pcd.points) == 19100
    assert len(pcd.colors) == 19100
    assert pcd.has_points()
    assert pcd.has_colors()

    print("Open3D PointCloud: PASS")

    # ====================================
    # Export
    # ====================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = o3d.io.write_point_cloud(
        str(OUTPUT),
        pcd,
    )

    assert saved, "Failed to write PLY file."
    assert OUTPUT.is_file()

    print("Saved:", OUTPUT)
    print("EXP027 Open3D checks PASSED.")

    # ====================================
    # Optional visualization
    # ====================================

    if args.show:
        coordinate_frame = (
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.3,
                origin=[0.0, 0.0, 0.0],
            )
        )

        print("Opening interactive 3D window...")

        o3d.visualization.draw_geometries(
            [pcd, coordinate_frame],
            window_name="EXP027 RGB-D Point Cloud",
            width=1100,
            height=750,
        )


if __name__ == "__main__":
    main()
