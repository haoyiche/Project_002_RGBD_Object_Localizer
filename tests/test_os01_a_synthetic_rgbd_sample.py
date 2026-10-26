import numpy as np

from src.experiments.os01_a_synthetic_rgbd_sample import main
from src.geometry.mask_to_pointcloud import mask_to_pointcloud


def test_synthetic_rgbd_sample_roundtrip(tmp_path, monkeypatch):
    # 1. Run generator in an isolated temporary directory
    monkeypatch.chdir(tmp_path)
    main()

    # 2. Verify archive exists
    sample_path = (
        tmp_path / "data/raw/synthetic/os01_sample_000.npz"
    )
    assert sample_path.is_file()

    # 3. Reload saved data
    with np.load(sample_path, allow_pickle=False) as sample:
        rgb = sample["rgb"]
        depth_m = sample["depth_m"]
        mask = sample["target_mask"]

        fx = sample["fx"].item()
        fy = sample["fy"].item()
        cx = sample["cx"].item()
        cy = sample["cy"].item()

        frame_id = sample["frame_id"].item()
        reference_kind = sample["reference_kind"].item()

    # 4. Verify input contract
    assert rgb.shape == (6, 8, 3)
    assert rgb.dtype == np.uint8
    assert depth_m.shape == (6, 8)
    assert mask.shape == (6, 8)
    assert mask.dtype == np.bool_

    assert frame_id == "camera_optical_frame"
    assert reference_kind == "synthetic_input"

    # 5. Recompute geometry from reloaded data
    points, colors, selected = mask_to_pointcloud(
        depth_m, rgb, mask, fx, fy, cx, cy
    )

    assert np.count_nonzero(mask) == 6
    assert np.count_nonzero(selected) == 5
    assert points.shape == (5, 3)
    assert colors.shape == (5, 3)

    # 6. Independent expected XYZ coordinates in meters
    expected_points = np.array([
        [ 0.005, -0.005, 1.0],
        [ 0.015, -0.005, 1.0],
        [-0.005,  0.005, 1.0],
        [ 0.005,  0.005, 1.0],
        [ 0.015,  0.005, 1.0],
    ])

    np.testing.assert_allclose(
        points, expected_points, rtol=0, atol=1e-12
    )