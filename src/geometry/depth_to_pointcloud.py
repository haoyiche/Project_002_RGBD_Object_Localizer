import numpy as np

from src.geometry.pixel_to_camera import pixel_to_camera


def depth_to_pointcloud(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Convert a Z-depth map into an Nx3 camera point cloud.

    Contract:
    - depth_m: 2D depth map in meters.
    - fx, fy: camera focal lengths in pixels.
    - cx, cy: principal point coordinates in pixels.
    - output: Nx3 array of XYZ points in meters.

    Camera coordinates:
    X right, Y down, Z forward.

    Invalid depths (<=0, NaN, Inf) are skipped.
    Valid points are returned in row-major pixel order.
    """

    depth_m = np.asarray(depth_m, dtype=np.float64)

    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array.")

    if depth_m.size == 0:
        raise ValueError("depth_m must not be empty.")

    if not np.isfinite([fx, fy, cx, cy]).all():
        raise ValueError("Camera intrinsics must be finite.")

    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive.")

    height, width = depth_m.shape

    points = []

    for v in range(height):
        for u in range(width):

            z = float(depth_m[v, u])

            if not np.isfinite(z) or z <= 0:
                continue

            point = pixel_to_camera(
                u=float(u),
                v=float(v),
                depth_m=z,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )

            points.append(point)

    return np.asarray(
        points,
        dtype=np.float64,
    ).reshape(-1, 3)