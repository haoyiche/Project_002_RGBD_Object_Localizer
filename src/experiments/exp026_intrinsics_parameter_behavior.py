from src.geometry.pixel_to_camera import pixel_to_camera


FX = 600.0
FY = 600.0
CX = 320.0
CY = 240.0


def print_point(name, point):
    x, y, z = point

    print(
        f"{name:<28} "
        f"X={x: .6f}, "
        f"Y={y: .6f}, "
        f"Z={z: .6f}, "
        f"X/Z={x / z: .6f}, "
        f"Y/Z={y / z: .6f}"
    )


def main():
    print("=== EXP026 Intrinsics Parameter Behavior ===")
    print()

    # --------------------------------------------------
    # Case A
    # Change v only
    # Prediction:
    # X = 0.0
    # Y = 0.12
    # --------------------------------------------------

    case_a = pixel_to_camera(
        u=320,
        v=300,
        depth_m=1.2,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    )

    print_point("Case A: change v", case_a)

    assert abs(case_a[0] - 0.0) < 1e-9
    assert abs(case_a[1] - 0.12) < 1e-9


    # --------------------------------------------------
    # Case B
    # Change cx only
    # Prediction:
    # X: 0.12 -> 0.06
    # --------------------------------------------------

    case_b_original = pixel_to_camera(
        u=380,
        v=240,
        depth_m=1.2,
        fx=FX,
        fy=FY,
        cx=320,
        cy=CY,
    )

    case_b_modified = pixel_to_camera(
        u=380,
        v=240,
        depth_m=1.2,
        fx=FX,
        fy=FY,
        cx=350,
        cy=CY,
    )

    print()
    print_point("Case B: cx=320", case_b_original)
    print_point("Case B: cx=350", case_b_modified)

    assert abs(case_b_original[0] - 0.12) < 1e-9
    assert abs(case_b_modified[0] - 0.06) < 1e-9


    # --------------------------------------------------
    # Case C
    # Change fy only
    # Prediction:
    # fy: 600 -> 300
    # Y: 0.12 -> 0.24
    # X unchanged
    # --------------------------------------------------

    case_c_600 = pixel_to_camera(
        u=320,
        v=300,
        depth_m=1.2,
        fx=FX,
        fy=600,
        cx=CX,
        cy=CY,
    )

    case_c_300 = pixel_to_camera(
        u=320,
        v=300,
        depth_m=1.2,
        fx=FX,
        fy=300,
        cx=CX,
        cy=CY,
    )

    print()
    print_point("Case C: fy=600", case_c_600)
    print_point("Case C: fy=300", case_c_300)

    assert abs(case_c_600[1] - 0.12) < 1e-9
    assert abs(case_c_300[1] - 0.24) < 1e-9
    assert abs(case_c_600[0] - case_c_300[0]) < 1e-9


    # --------------------------------------------------
    # Case D
    # Same pixel, different depth
    #
    # Prediction:
    # Z: 1.2 -> 0.6
    # X: 0.12 -> 0.06
    # Y: 0.12 -> 0.06
    #
    # X/Z and Y/Z unchanged
    # --------------------------------------------------

    case_d_far = pixel_to_camera(
        u=380,
        v=300,
        depth_m=1.2,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    )

    case_d_near = pixel_to_camera(
        u=380,
        v=300,
        depth_m=0.6,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    )

    print()
    print_point("Case D: Z=1.2", case_d_far)
    print_point("Case D: Z=0.6", case_d_near)

    assert abs(case_d_far[0] - 0.12) < 1e-9
    assert abs(case_d_far[1] - 0.12) < 1e-9

    assert abs(case_d_near[0] - 0.06) < 1e-9
    assert abs(case_d_near[1] - 0.06) < 1e-9

    far_x_over_z = case_d_far[0] / case_d_far[2]
    far_y_over_z = case_d_far[1] / case_d_far[2]

    near_x_over_z = case_d_near[0] / case_d_near[2]
    near_y_over_z = case_d_near[1] / case_d_near[2]

    assert abs(far_x_over_z - near_x_over_z) < 1e-9
    assert abs(far_y_over_z - near_y_over_z) < 1e-9


    print()
    print("=== Summary ===")

    print(
        "Changing v affects Y but not X "
        "when other parameters are fixed."
    )

    print(
        "Increasing cx reduces (u - cx), "
        "therefore reducing X."
    )

    print(
        "Reducing fy increases Y because "
        "Y is inversely proportional to fy."
    )

    print(
        "Changing depth scales X, Y, Z together "
        "while X/Z and Y/Z remain constant."
    )

    print()
    print("All EXP026 parameter behavior checks PASSED.")


if __name__ == "__main__":
    main()
