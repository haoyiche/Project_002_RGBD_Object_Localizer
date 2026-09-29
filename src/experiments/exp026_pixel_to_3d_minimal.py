from src.geometry.pixel_to_camera import pixel_to_camera


FX = 600.0
FY = 600.0
CX = 320.0
CY = 240.0


def show_case(name, u, v, depth_m, fx=FX, fy=FY):
    point = pixel_to_camera(
        u=u,
        v=v,
        depth_m=depth_m,
        fx=fx,
        fy=fy,
        cx=CX,
        cy=CY,
    )

    print(
        f"{name}: "
        f"pixel=({u:.1f}, {v:.1f}), "
        f"depth_m={depth_m:.3f} "
        f"-> camera=({point[0]:.6f}, "
        f"{point[1]:.6f}, "
        f"{point[2]:.6f})"
    )

    return point


def main():
    print("=== EXP026 Minimal Pixel-to-3D Experiment ===")

    # Q1: principal point
    p1 = show_case(
        "Q1 principal point",
        u=320,
        v=240,
        depth_m=1.2,
    )

    # Q2: 60 px right of principal point
    p2 = show_case(
        "Q2 right offset",
        u=380,
        v=240,
        depth_m=1.2,
    )

    # Q3: double depth
    p3 = show_case(
        "Q3 double depth",
        u=380,
        v=240,
        depth_m=2.4,
    )

    # Q4: double fx
    p4 = show_case(
        "Q4 double fx",
        u=380,
        v=240,
        depth_m=1.2,
        fx=1200,
    )

    # Extra: pixel left of principal point
    p5 = show_case(
        "Extra left offset",
        u=260,
        v=240,
        depth_m=1.2,
    )

    print()
    print("=== Automatic Checks ===")

    assert abs(p1[0] - 0.0) < 1e-9
    assert abs(p1[1] - 0.0) < 1e-9
    assert abs(p1[2] - 1.2) < 1e-9

    assert abs(p2[0] - 0.12) < 1e-9
    assert abs(p2[1] - 0.0) < 1e-9

    assert abs(p3[0] - 0.24) < 1e-9

    assert abs(p4[0] - 0.06) < 1e-9

    assert p5[0] < 0

    print("All EXP026 minimal checks PASSED.")


if __name__ == "__main__":
    main()
