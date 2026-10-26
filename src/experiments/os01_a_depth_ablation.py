from pathlib import Path

import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud


def main():
    sample_path = Path(
        "data/raw/synthetic/os01_sample_000.npz"
    )

    with np.load(sample_path, allow_pickle=False) as sample:
        rgb = sample["rgb"]
        depth_m = sample["depth_m"]
        mask = sample["target_mask"]

        fx = sample["fx"].item()
        fy = sample["fy"].item()
        cx = sample["cx"].item()
        cy = sample["cy"].item()

    # Baseline
    points_before, _, selected_before = mask_to_pointcloud(
        depth_m, rgb, mask, fx, fy, cx, cy
    )

    # A2: add 0.2 m to valid target depths only
    depth_modified = depth_m.copy()

    valid_target = (
        mask
        & np.isfinite(depth_m)
        & (depth_m > 0)
    )

    depth_modified[valid_target] += 0.2

    # Modified result
    points_after, _, selected_after = mask_to_pointcloud(
        depth_modified, rgb, mask, fx, fy, cx, cy
    )

    centroid_before = points_before.mean(axis=0)
    centroid_after = points_after.mean(axis=0)

    delta_mm = (
        centroid_after - centroid_before
    ) * 1000.0

    print("Valid points before:", len(points_before))
    print("Valid points after:", len(points_after))
    print(
        "Selected mask unchanged:",
        np.array_equal(selected_before, selected_after)
    )

    print("Centroid before (m):", centroid_before)
    print("Centroid after (m):", centroid_after)
    print("Centroid delta (mm):", delta_mm)

    print(
        "Invalid depth preserved:",
        depth_modified[2, 3] == 0.0
    )


if __name__ == "__main__":
    main()