import numpy as np


def depth_to_pointcloud_vectorized(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Convert a Z-depth map to an Nx3 camera point cloud.

    - Input depth is in meters.
    - Output XYZ is in meters.
    - Camera convention: X right, Y down, Z forward.
    - Invalid depths are filtered.
    - Valid points preserve row-major pixel order.
    """

    depth_m = np.asarray(depth_m, dtype=np.float64)

    # 1. Validate input shape
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array.")

    if depth_m.size == 0:
        raise ValueError("depth_m must not be empty.")

    # 2. Validate camera intrinsics
    if not np.isfinite([fx, fy, cx, cy]).all():
        raise ValueError("Camera intrinsics must be finite.")

    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive.")

    # 3. Pixel coordinate grids
    v_grid, u_grid = np.indices(depth_m.shape)

    # 4. Valid depth mask
    valid = (
        np.isfinite(depth_m)
        & (depth_m > 0)
    )

    # 5. Extract valid pixels in row-major order
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    z = depth_m[valid]

    # 6. Vectorized back-projection
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 7. Return Nx3
    points = np.column_stack((x, y, z))

    return points
