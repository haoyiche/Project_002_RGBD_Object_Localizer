import numpy as np


def mask_to_pointcloud(
    depth_m, rgb, target_mask, fx, fy, cx, cy
):
    # 1. Convert input arrays
    depth = np.asarray(depth_m, dtype=np.float64)
    image = np.asarray(rgb)
    mask = np.asarray(target_mask)

    # 2. Validate shapes and types
    if depth.ndim != 2 or 0 in depth.shape:
        raise ValueError("depth_m must be non-empty 2D")

    h, w = depth.shape

    if image.shape != (h, w, 3):
        raise ValueError("rgb must have shape (H,W,3)")

    if image.dtype != np.uint8:
        raise ValueError("rgb must be uint8")

    if mask.shape != (h, w) or mask.dtype != np.bool_:
        raise ValueError("target_mask must be bool [H,W]")

    # 3. Validate camera intrinsics
    intrinsics = np.asarray(
        [fx, fy, cx, cy], dtype=np.float64
    )

    if not np.isfinite(intrinsics).all():
        raise ValueError("intrinsics must be finite")

    fx, fy, cx, cy = intrinsics

    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive")

    # 4. Select valid target pixels
    valid = np.isfinite(depth) & (depth > 0)
    select_mask = mask & valid

    # 5. Extract pixel coordinates and depth
    v, u = np.nonzero(select_mask)
    z = depth[select_mask]

    # 6. Back-project to 3D
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy

    points = np.column_stack((x, y, z))

    # 7. Extract corresponding RGB colors
    colors = image[select_mask].astype(np.float64) / 255.0

    return points, colors, select_mask
