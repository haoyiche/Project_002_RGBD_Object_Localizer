from math import sqrt

from src.geometry.pixel_to_camera import pixel_to_camera


FX = 600.0
FY = 600.0
CX = 320.0
CY = 240.0

U = 380.0
V = 300.0


def point_norm(point):
    x, y, z = point
    return sqrt(x * x + y * y + z * z)


def main():
    print("=== EXP026 Depth Unit Debug Experiment ===")
    print()

    # Correct interpretation:
    # 1200 mm = 1.2 m
    correct_depth_m = 1.2

    correct_point = pixel_to_camera(
        u=U,
        v=V,
        depth_m=correct_depth_m,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    )

    # Incorrect interpretation:
    # raw value 1200 mm is accidentally treated as 1200 m
    wrong_depth_m = 1200.0

    wrong_point = pixel_to_camera(
        u=U,
        v=V,
        depth_m=wrong_depth_m,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
    )

    print("Correct interpretation:")
    print(
        f"  depth = {correct_depth_m:.3f} m"
    )
    print(
        f"  point = "
        f"({correct_point[0]:.6f}, "
        f"{correct_point[1]:.6f}, "
        f"{correct_point[2]:.6f}) m"
    )

    print()

    print("Wrong interpretation:")
    print(
        f"  depth = {wrong_depth_m:.3f} m"
    )
    print(
        f"  point = "
        f"({wrong_point[0]:.6f}, "
        f"{wrong_point[1]:.6f}, "
        f"{wrong_point[2]:.6f}) m"
    )

    print()

    scale_x = wrong_point[0] / correct_point[0]
    scale_y = wrong_point[1] / correct_point[1]
    scale_z = wrong_point[2] / correct_point[2]

    correct_norm = point_norm(correct_point)
    wrong_norm = point_norm(wrong_point)
    scale_norm = wrong_norm / correct_norm

    print("Scale error:")
    print(f"  X scale = {scale_x:.1f}x")
    print(f"  Y scale = {scale_y:.1f}x")
    print(f"  Z scale = {scale_z:.1f}x")
    print(f"  Euclidean distance scale = {scale_norm:.1f}x")

    print()

    assert abs(scale_x - 1000.0) < 1e-9
    assert abs(scale_y - 1000.0) < 1e-9
    assert abs(scale_z - 1000.0) < 1e-9
    assert abs(scale_norm - 1000.0) < 1e-9

    print("Prediction verified:")
    print(
        "A 1000x depth-unit error produces "
        "a 1000x 3D scale error."
    )

    print()
    print("EXP026 depth unit debug check PASSED.")


if __name__ == "__main__":
    main()
