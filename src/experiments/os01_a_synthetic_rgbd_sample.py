from pathlib import Path

import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud


def main():
    # 1. Synthetic RGB image: (H, W, 3)
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[2:4, 3:6] = [200, 60, 40]

    # 2. Synthetic depth map: (H, W), meters
    depth_m = np.zeros((6, 8), dtype=np.float64)
    depth_m[2:4, 3:6] = 1.0

    # Inject one invalid depth inside the target
    depth_m[2, 3] = 0.0

    # 3. Target mask: (H, W), bool
    target_mask = np.zeros((6, 8), dtype=bool)
    target_mask[2:4, 3:6] = True

    # 4. Camera intrinsics, pixel units
    fx, fy = 100.0, 100.0
    cx, cy = 3.5, 2.5

    # 5. Reuse EXP028 production function
    points, colors, selected = mask_to_pointcloud(
        depth_m,
        rgb,
        target_mask,
        fx, fy, cx, cy,
    )

    # 6. Save reproducible input
    output = Path("data/raw/synthetic/os01_sample_000.npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        rgb=rgb,
        depth_m=depth_m,
        target_mask=target_mask,
        fx=fx, fy=fy, cx=cx, cy=cy,
        frame_id="camera_optical_frame",
        reference_kind="synthetic_input",
    )

    print("RGB shape:", rgb.shape)
    print("Depth shape:", depth_m.shape)
    print("Mask pixels:", np.count_nonzero(target_mask))
    print("Valid selected pixels:", np.count_nonzero(selected))
    print("Points shape:", points.shape)
    print("Colors shape:", colors.shape)
    print("Centroid (m):", points.mean(axis=0))
    print("Saved:", output)


if __name__ == "__main__":
    main()