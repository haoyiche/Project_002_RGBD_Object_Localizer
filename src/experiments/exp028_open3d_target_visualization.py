import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from src.geometry.mask_to_pointcloud import mask_to_pointcloud


OUTPUT = Path("results/exp028/target_pointcloud_colored.ply")


def create_scene():
    height, width = 80, 100

    # Gray background: Z = 2.0 m
    depth = np.full(
        (height, width), 2.0, dtype=np.float64
    )

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = [150, 160, 175]

    # Orange target: 40 x 40 pixels
    target_mask = np.zeros(
        (height, width), dtype=bool
    )
    target_mask[20:60, 30:70] = True

    depth[target_mask] = 1.2
    rgb[target_mask] = [245, 135, 50]

    # Invalid depth: 10 x 10 pixels
    depth[30:40, 40:50] = 0.0

    return depth, rgb, target_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    print("=== EXP028 Open3D Target Visualization ===")

    depth, rgb, target_mask = create_scene()

    params = dict(
        fx=100.0,
        fy=100.0,
        cx=49.5,
        cy=39.5,
    )

    # Reuse EXP028 production API
    points, colors, selected = mask_to_pointcloud(
        depth_m=depth,
        rgb=rgb,
        target_mask=target_mask,
        **params,
    )

    # ======================================
    # 1. Shape and Mask Verification
    # ======================================

    assert depth.shape == (80, 100)
    assert rgb.shape == (80, 100, 3)

    assert np.count_nonzero(target_mask) == 1600
    assert np.count_nonzero(selected) == 1500

    assert points.shape == (1500, 3)
    assert colors.shape == (1500, 3)
    assert selected.shape == (80, 100)

    # Invalid target region must be excluded
    assert not selected[30:40, 40:50].any()

    print("Depth shape:", depth.shape)
    print("Target mask pixels:", target_mask.sum())
    print("Selected points:", selected.sum())
    print("Points shape:", points.shape)
    print("Colors shape:", colors.shape)
    print("Shape and Mask: PASS")

    # ======================================
    # 2. Geometry Verification
    # ======================================

    # All output points belong to target Z=1.2m
    np.testing.assert_allclose(
        points[:, 2],
        1.2,
        rtol=0,
        atol=1e-12,
    )

    # Independent reference for first valid pixel:
    # u=30, v=20, Z=1.2
    # X=(30-49.5)*1.2/100 = -0.234
    # Y=(20-39.5)*1.2/100 = -0.234
    np.testing.assert_allclose(
        points[0],
        [-0.234, -0.234, 1.2],
        rtol=0,
        atol=1e-12,
    )

    assert np.isfinite(points).all()

    print("First target point:", points[0])
    print("Target Z:", np.unique(points[:, 2]))
    print("Geometry: PASS")

    # ======================================
    # 3. RGB Alignment Verification
    # ======================================

    expected_color = (
        np.array([245, 135, 50], dtype=np.float64)
        / 255.0
    )

    np.testing.assert_allclose(
        colors,
        np.tile(expected_color, (1500, 1)),
        rtol=0,
        atol=1e-12,
    )

    assert colors.min() >= 0.0
    assert colors.max() <= 1.0

    print("RGB alignment: PASS")

    # ======================================
    # 4. Open3D PointCloud
    # ======================================

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    assert pcd.has_points()
    assert pcd.has_colors()
    assert len(pcd.points) == 1500
    assert len(pcd.colors) == 1500

    print("Open3D PointCloud: PASS")

    # ======================================
    # 5. Export and Reload PLY
    # ======================================

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    saved = o3d.io.write_point_cloud(
        str(OUTPUT), pcd
    )

    assert saved
    assert OUTPUT.is_file()

    loaded = o3d.io.read_point_cloud(str(OUTPUT))

    assert len(loaded.points) == 1500
    assert len(loaded.colors) == 1500

    print("PLY saved:", OUTPUT)
    print("PLY reload: PASS")
    print("EXP028 Open3D checks PASSED.")

    # ======================================
    # 6. Optional Interactive Window
    # ======================================

    if args.show:
        frame = (
            o3d.geometry.TriangleMesh
            .create_coordinate_frame(
                size=0.3,
                origin=[0.0, 0.0, 0.0],
            )
        )

        print("Opening Open3D window...")

        o3d.visualization.draw_geometries(
            [pcd, frame],
            window_name="EXP028 Target Point Cloud",
            width=1100,
            height=750,
        )


if __name__ == "__main__":
    main()
