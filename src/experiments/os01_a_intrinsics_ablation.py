from pathlib import Path

import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud


def main():
    sample_path = Path("data/raw/synthetic/os01_sample_000.npz")

    with np.load(sample_path, allow_pickle=False) as sample:
        rgb = sample["rgb"]
        depth_m = sample["depth_m"]
        mask = sample["target_mask"]

        fx = sample["fx"].item()
        fy = sample["fy"].item()
        cx = sample["cx"].item()
        cy = sample["cy"].item()

    # Baseline: original camera intrinsics
    points_before, _, selected_before = mask_to_pointcloud(
        depth_m, rgb, mask, fx, fy, cx, cy
    )

    # A1: change fx only
    fx_modified = fx * 2.0

    points_after, _, selected_after = mask_to_pointcloud(
        depth_m, rgb, mask,
        fx_modified, fy, cx, cy
    )

    centroid_before = points_before.mean(axis=0)
    centroid_after = points_after.mean(axis=0)

    delta_mm = (centroid_after - centroid_before) * 1000.0

    print("fx before:", fx)
    print("fx after:", fx_modified)

    print("Points before:", points_before.shape)
    print("Points after:", points_after.shape)

    print("Selected mask unchanged:",
          np.array_equal(selected_before, selected_after))

    print("Centroid before (m):", centroid_before)
    print("Centroid after (m):", centroid_after)

    print("Centroid delta (mm):", delta_mm)


if __name__ == "__main__":
    main()