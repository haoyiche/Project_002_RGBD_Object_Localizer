def pixel_to_camera(
    u: float,
    v: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    """
    Back-project one image pixel into camera coordinates.

    Contract
    --------
    - depth_m is camera Z-depth in meters.
    - fx, fy are focal lengths in pixel units.
    - u, v, cx, cy are pixel coordinates.
    - output X, Y, Z are in meters.

    This function performs pure geometric back-projection.
    It does not:
    - convert sensor-specific depth units,
    - validate image bounds,
    - infer depth semantics,
    - apply sensor-specific range limits.
    """

    if fx <= 0:
        raise ValueError("fx must be positive.")

    if fy <= 0:
        raise ValueError("fy must be positive.")

    if depth_m <= 0:
        raise ValueError("depth_m must be positive.")

    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m

    return x, y, z
