import numpy as np
import open3d as o3d

from src.experiments.os01_a_synthetic_rgbd_sample import (
    main as generate_sample,
)
from src.pipeline.os01_d_open3d_compare import (
    main as run_comparison,
)


def test_d1_full_pipeline(tmp_path, monkeypatch, capsys):
    # 测试自己生成数据，不依赖本地已有 .npz
    monkeypatch.chdir(tmp_path)

    generate_sample()
    run_comparison()

    output = capsys.readouterr().out

    assert "Open3D comparison: PASS" in output
    assert "Original points: (5, 3)" in output
    assert "NumPy labels: [0, 0, 0, 0, 0]" in output
    assert "Open3D labels: [0, 0, 0, 0, 0]" in output
    assert "Coarse count: 3" in output


def test_d1_voxel_centroid_shift():
    points = np.array([
        [ 0.005, -0.005, 1.0],
        [ 0.015, -0.005, 1.0],
        [-0.005,  0.005, 1.0],
        [ 0.005,  0.005, 1.0],
        [ 0.015,  0.005, 1.0],
    ], dtype=np.float64)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    coarse = np.asarray(
        cloud.voxel_down_sample(0.020).points
    )

    original_centroid = points.mean(axis=0)
    coarse_centroid = coarse.mean(axis=0)

    assert coarse.shape == (3, 3)

    np.testing.assert_allclose(
        original_centroid,
        [0.007, 0.001, 1.0],
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        coarse_centroid,
        [0.005, 0.005 / 3, 1.0],
        rtol=0,
        atol=1e-12,
    )

    shift_mm = np.linalg.norm(
        coarse_centroid - original_centroid
    ) * 1000

    np.testing.assert_allclose(
        shift_mm,
        np.sqrt(4 + 4 / 9),
        rtol=0,
        atol=1e-9,
    )