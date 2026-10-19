"""EXP035 Boss C/D 专项测试。"""

import copy
import numpy as np
import pytest

from src.experiments.exp035_boss_student import audit_chain_and_compute

try:
    from src.geometry.rigid_transform import RigidTransform
except ImportError:
    RigidTransform = None


# =========================================================
# 工具：跨版本构造 RigidTransform
# =========================================================

def _make_transform(R, t, source, target, verified=False):
    return RigidTransform(
        R=np.asarray(R, dtype=np.float64),
        t_m=np.asarray(t, dtype=np.float64),
        source_frame_id=source,
        target_frame_id=target,
        calibration_verified=verified,
    )


# =========================================================
# Fixtures：每次测试拿到独立实例，不共享可变对象
# =========================================================

@pytest.fixture
def report_valid_only():
    return {
        "frame_id": "camera_optical_frame",
        "status": "OK",
        "valid_count": 1,
        "invalid_count": 0,
        "noise_count": 0,
        "objects": [
            {
                "cluster_label": 1,
                "frame_id": "camera_optical_frame",
                "status": "VALID",
                "publishable": True,
                "position_m": [0.51, 0.01, 1.152],
            }
        ],
        "published_positions": [
            {
                "cluster_label": 1,
                "frame_id": "camera_optical_frame",
                "position_m": [0.51, 0.01, 1.152],
            }
        ],
    }


@pytest.fixture
def report_b2():
    return {
        "frame_id": "camera_optical_frame",
        "status": "PARTIAL_INVALID",
        "valid_count": 1,
        "invalid_count": 1,
        "noise_count": 1,
        "objects": [
            {
                "cluster_label": 0,
                "frame_id": "camera_optical_frame",
                "status": "INVALID_CLUSTER",
                "publishable": False,
                "position_m": None,
                "reasons": ["EXTENT_EXCEEDED", "CENTROID_BBOX_GAP"],
                "diagnostics": {
                    "point_count": 10,
                    "centroid_m": [0.099, 0.059, 1.16],
                },
            },
            {
                "cluster_label": 1,
                "frame_id": "camera_optical_frame",
                "status": "VALID",
                "publishable": True,
                "position_m": [0.51, 0.01, 1.152],
            },
        ],
        "published_positions": [
            {
                "cluster_label": 1,
                "frame_id": "camera_optical_frame",
                "position_m": [0.51, 0.01, 1.152],
            }
        ],
    }


@pytest.fixture
def tf_base_camera():
    return _make_transform(
        R=np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]], dtype=np.float64),
        t=np.array([0.30, -0.10, 0.20]),
        source="camera_optical_frame",
        target="robot_base",
    )


# =========================================================
# 测试 1：正常数值链，只输出诊断
# =========================================================

def test_normal_chain_diagnostic_only(report_valid_only, tf_base_camera):
    result = audit_chain_and_compute(report_valid_only, [tf_base_camera])

    assert result["publishable"] is False
    assert result["published_positions"] == []
    assert len(result["diagnostics"]) == 1

    diag = result["diagnostics"][0]
    assert diag["cluster_label"] == 1
    assert diag["source_frame_id"] == "camera_optical_frame"
    assert diag["final_frame_id"] == "robot_base"

    np.testing.assert_allclose(
        diag["final_position_m"],
        [0.29, 0.41, 1.352],
        rtol=0, atol=1e-12,
    )

    # 不修改原始 report
    assert report_valid_only["published_positions"][0]["position_m"] == [
        0.51, 0.01, 1.152
    ]


# =========================================================
# 测试 2：无效簇隔离
# =========================================================

def test_invalid_cluster_not_transformed(report_b2, tf_base_camera):
    result = audit_chain_and_compute(report_b2, [tf_base_camera])
    labels = [d["cluster_label"] for d in result["diagnostics"]]
    assert 0 not in labels
    assert 1 in labels
    assert result["publishable"] is False


# =========================================================
# 测试 3：缓存污染被拒绝
# =========================================================

def test_cache_contamination_rejected(report_b2, tf_base_camera):
    bad = copy.deepcopy(report_b2)
    bad["published_positions"].append({
        "cluster_label": 0,
        "frame_id": "camera_optical_frame",
        "position_m": [0.099, 0.059, 1.16],
    })
    with pytest.raises(ValueError):
        audit_chain_and_compute(bad, [tf_base_camera])


# =========================================================
# 测试 4：坐标系不匹配
# =========================================================

def test_object_frame_mismatch_rejected(report_valid_only, tf_base_camera):
    bad = copy.deepcopy(report_valid_only)
    bad["objects"][0]["frame_id"] = "world"
    with pytest.raises(ValueError):
        audit_chain_and_compute(bad, [tf_base_camera])


def test_report_frame_mismatch_rejected(report_valid_only, tf_base_camera):
    bad = copy.deepcopy(report_valid_only)
    bad["frame_id"] = "world"
    with pytest.raises(ValueError):
        audit_chain_and_compute(bad, [tf_base_camera])


def test_transform_source_mismatch_rejected(report_valid_only):
    bad_tf = _make_transform(
        R=np.eye(3),
        t=np.zeros(3),
        source="world",
        target="robot_base",
    )
    with pytest.raises(ValueError):
        audit_chain_and_compute(report_valid_only, [bad_tf])


# =========================================================
# 测试 5：标定布尔值伪造不能发布
# =========================================================

def test_calibration_flag_forgery_still_not_publishable(report_valid_only, tf_base_camera):
    fake = _make_transform(
        R=tf_base_camera.R,
        t=tf_base_camera.t_m,
        source=tf_base_camera.source_frame_id,
        target=tf_base_camera.target_frame_id,
        verified=True,
    )
    assert fake.calibration_verified is True
    result = audit_chain_and_compute(report_valid_only, [fake])
    assert result["publishable"] is False
    assert result["published_positions"] == []


def test_invalid_cluster_frame_mismatch_rejected(report_b2, tf_base_camera):
    bad = copy.deepcopy(report_b2)
    bad["objects"][0]["frame_id"] = "world"
    with pytest.raises(ValueError, match="frame_id"):
        audit_chain_and_compute(bad, [tf_base_camera])


def test_cache_subtle_drift_rejected(report_valid_only, tf_base_camera):
    bad = copy.deepcopy(report_valid_only)
    bad["published_positions"][0]["position_m"][0] += 1e-7
    with pytest.raises(ValueError, match="position_m mismatch"):
        audit_chain_and_compute(bad, [tf_base_camera])


def test_reject_impostor_transform(report_valid_only):
    class Impostor:
        R = np.diag([1, 1, -1]).astype(float)
        t_m = np.zeros(3)
        source_frame_id = "camera_optical_frame"
        target_frame_id = "robot_base"
        def transform_point(self, point_m, *, frame_id):
            return np.array([point_m[0], point_m[1], -point_m[2]])
    with pytest.raises(ValueError, match="RigidTransform"):
        audit_chain_and_compute(report_valid_only, [Impostor()])


# =========================================================
# 测试 6：旧 unsafe_pipeline 放行 vs 新版拒绝
# =========================================================

def unsafe_pipeline(report, transforms):
    """Boss D 旧实现：故意保留漏洞，用于对照。"""
    positions = report.get("published_positions", [])
    output = []
    for item in positions:
        p = item["position_m"]
        for transform in transforms:
            p = transform.transform_point(
                p,
                frame_id=transform.source_frame_id,
            )
        output.append({
            "cluster_label": item["cluster_label"],
            "position_m": p.tolist() if hasattr(p, "tolist") else list(p),
        })
    return {
        "publishable": all(
            getattr(t, "calibration_verified", False) for t in transforms
        ),
        "published_positions": output,
    }


def test_unsafe_accepts_contamination_new_rejects(
    report_b2, tf_base_camera
):
    bad = copy.deepcopy(report_b2)
    bad["published_positions"].append({
        "cluster_label": 0,
        "frame_id": "camera_optical_frame",
        "position_m": [0.099, 0.059, 1.16],
    })

    old = unsafe_pipeline(bad, [tf_base_camera])
    assert len(old["published_positions"]) == 2   # 旧实现放行

    with pytest.raises(ValueError):
        audit_chain_and_compute(bad, [tf_base_camera])  # 新版拒绝


def test_unsafe_flag_true_and_bad_cache_old_publishes_new_rejects(report_b2, tf_base_camera):
    bad = copy.deepcopy(report_b2)
    bad["published_positions"].append({
        "cluster_label": 0,
        "frame_id": "camera_optical_frame",
        "position_m": [0.099, 0.059, 1.16],
    })
    fake = _make_transform(
        R=tf_base_camera.R, t=tf_base_camera.t_m,
        source="camera_optical_frame", target="robot_base", verified=True,
    )
    old = unsafe_pipeline(bad, [fake])
    assert old["publishable"] is True
    assert {p["cluster_label"] for p in old["published_positions"]} == {0, 1}
    with pytest.raises(ValueError):
        audit_chain_and_compute(bad, [fake])


def test_unsafe_accepts_wrong_frame_new_rejects(
    report_valid_only, tf_base_camera
):
    bad = copy.deepcopy(report_valid_only)
    bad["objects"][0]["frame_id"] = "world"
    bad["published_positions"][0]["frame_id"] = "world"

    old = unsafe_pipeline(bad, [tf_base_camera])
    assert len(old["published_positions"]) == 1   # 旧实现静默返回

    with pytest.raises(ValueError):
        audit_chain_and_compute(bad, [tf_base_camera])  # 新版拒绝