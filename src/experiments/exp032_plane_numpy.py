import numpy as np


def plane_from_three_points(p0, p1, p2):
    """Return normalized plane [a,b,c,d] through 3 non-collinear points."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    for name, p in (("p0", p0), ("p1", p1), ("p2", p2)):
        if p.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {p.shape}")
        if not np.isfinite(p).all():
            raise ValueError(f"{name} must contain finite values")

    u = p1 - p0
    v = p2 - p0
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise ValueError("edge vectors must be finite")

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if not (np.isfinite(norm_u) and np.isfinite(norm_v)):
        raise ValueError("edge lengths must be finite")
    if norm_u == 0.0 or norm_v == 0.0:
        raise ValueError("duplicate points or zero-length edge")

    # Angle-based degeneracy, insensitive to uniform coordinate scaling.
    # Normalizing edges first reduces overflow/underflow in their cross product.
    unit_cross = np.cross(u / norm_u, v / norm_v)
    sin_theta = np.linalg.norm(unit_cross)
    if not np.isfinite(sin_theta) or sin_theta < 1e-8:
        raise ValueError("three points are nearly collinear or degenerate")

    n_hat = unit_cross / sin_theta
    d = -np.dot(n_hat, p0)
    if not np.isfinite(d):
        raise ValueError("plane offset must be finite")
    return np.array([n_hat[0], n_hat[1], n_hat[2], d], dtype=np.float64)


def point_plane_distances(points, plane):
    """Return perpendicular distances from (N,3) points to a plane (4,)."""
    pts = np.asarray(points, dtype=np.float64)
    pl = np.asarray(plane, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {pts.shape}")
    if pl.shape != (4,):
        raise ValueError(f"plane must be (4,), got {pl.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("points must be finite")
    if not np.isfinite(pl).all():
        raise ValueError("plane must be finite")

    a, b, c, d = pl
    n_norm = np.sqrt(a*a + b*b + c*c)
    if n_norm < 1e-12:
        raise ValueError("plane normal is degenerate")
    a, b, c, d = a/n_norm, b/n_norm, c/n_norm, d/n_norm
    signed = pts[:, 0]*a + pts[:, 1]*b + pts[:, 2]*c + d
    return np.abs(signed)
