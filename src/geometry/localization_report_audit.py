"""EXP034 Boss C: student-authored report consistency audit.

Note: validates local report contract; not global calibration/safety checks.
"""

import math


_REQUIRED_REPORT_FIELDS = ("objects", "status", "valid_count", "invalid_count")
_VALID_STATUSES         = {"OK", "PARTIAL_INVALID", "ALL_INVALID", "NO_OBJECTS"}


def _is_non_negative_int(v):
    if isinstance(v, bool):
        return False
    if not isinstance(v, int):
        return False
    return v >= 0


def _is_finite_number(v):
    # 显式拒绝 bool，因为 bool 是 int 子类
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(v)


def _is_finite_3d(v):
    if v is None:
        return False
    if not isinstance(v, (list, tuple)):
        return False
    if len(v) != 3:
        return False
    return all(_is_finite_number(x) for x in v)


def _is_valid_frame_id(fid):
    if not isinstance(fid, str):
        return False
    return fid.strip() != ""


def _is_valid_cluster_label(label):
    if isinstance(label, bool):
        return False
    if not isinstance(label, int):
        return False
    return label >= 0


def audit_and_extract(report):
    """
    审核定位报告一致性，返回可发布位置列表。

    Raises:
        ValueError: 契约矛盾（缺失字段、类型错误、状态不一致等）
    """
    # ---------- 0. 顶层结构 ----------
    if not isinstance(report, dict):
        raise ValueError("report must be a dict")

    for field in _REQUIRED_REPORT_FIELDS:
        if field not in report:
            raise ValueError(f"report missing required field: {field}")

    objects = report["objects"]
    if not isinstance(objects, list):
        raise ValueError("report['objects'] must be a list")

    if not _is_non_negative_int(report["valid_count"]):
        raise ValueError("report['valid_count'] must be non-negative int")
    if not _is_non_negative_int(report["invalid_count"]):
        raise ValueError("report['invalid_count'] must be non-negative int")

    if report["status"] not in _VALID_STATUSES:
        raise ValueError(f"invalid report status: {report['status']!r}")

    # ---------- 1. 逐簇审计 ----------
    recomputed_valid   = 0
    recomputed_invalid = 0
    seen_labels        = set()
    result             = []

    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError("each object must be a dict")

        if "cluster_label" not in obj:
            raise ValueError("object missing cluster_label")
        label = obj["cluster_label"]
        if not _is_valid_cluster_label(label):
            raise ValueError(
                f"cluster_label must be non-negative int, got {label!r}"
            )
        if label in seen_labels:
            raise ValueError(f"duplicate cluster_label: {label}")
        seen_labels.add(label)

        status      = obj.get("status")
        publishable = obj.get("publishable")
        pos         = obj.get("position_m")
        frame_id    = obj.get("frame_id")

        if status == "VALID":
            recomputed_valid += 1
            if publishable is not True:
                raise ValueError(
                    f"VALID object {label} must have publishable=True"
                )
            if not _is_finite_3d(pos):
                raise ValueError(
                    f"VALID object {label} must have finite 3D position_m"
                )
            if not _is_valid_frame_id(frame_id):
                raise ValueError(
                    f"VALID object {label} must have non-empty frame_id"
                )
            result.append({
                "cluster_label": label,
                "frame_id":      frame_id,
                "position_m":    [float(v) for v in pos],
            })

        elif status == "INVALID_CLUSTER":
            recomputed_invalid += 1
            if publishable is not False:
                raise ValueError(
                    f"INVALID object {label} must have publishable=False"
                )
            if pos is not None:
                raise ValueError(
                    f"INVALID object {label} must have position_m=None"
                )

        else:
            raise ValueError(f"object {label} has invalid status: {status!r}")

    # ---------- 2. 整帧计数核对 ----------
    if report["valid_count"] != recomputed_valid:
        raise ValueError(
            f"valid_count mismatch: report={report['valid_count']}, "
            f"recomputed={recomputed_valid}"
        )
    if report["invalid_count"] != recomputed_invalid:
        raise ValueError(
            f"invalid_count mismatch: report={report['invalid_count']}, "
            f"recomputed={recomputed_invalid}"
        )

    # ---------- 3. 整帧状态核对 ----------
    n_total = recomputed_valid + recomputed_invalid
    if n_total == 0:
        expected_status = "NO_OBJECTS"
    elif recomputed_valid == n_total:
        expected_status = "OK"
    elif recomputed_valid == 0:
        expected_status = "ALL_INVALID"
    else:
        expected_status = "PARTIAL_INVALID"

    if report["status"] != expected_status:
        raise ValueError(
            f"status mismatch: report={report['status']}, "
            f"expected={expected_status}"
        )

    return result
