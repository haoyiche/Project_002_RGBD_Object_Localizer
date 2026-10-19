"""
EXP035 Boss C/D 修正版实现。

覆盖清单：
1. report.frame_id 与所有 object.frame_id 一致性检查
2. 缓存坐标比较 rtol=0
3. 严格要求每一级变换是 RigidTransform
4. 使用 transform_point() 逐级计算诊断坐标
5. 不修改输入 report / transform
6. 发布权限始终 fail-closed
"""

import math
import numpy as np

from src.geometry.rigid_transform import RigidTransform

from src.geometry.localization_report_audit import audit_and_extract


# =========================================================
# 基础校验
# =========================================================

def _is_finite_number(v):
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(v)


def _is_finite_3d(v):
    if v is None or not isinstance(v, (list, tuple)) or len(v) != 3:
        return False
    return all(_is_finite_number(x) for x in v)


def _as_vec3(v, name):
    if not _is_finite_3d(v):
        raise ValueError(f"{name} must be a finite 3D vector")
    return np.asarray(v, dtype=np.float64)


def _is_rigid_transform(tr):
    return isinstance(tr, RigidTransform)


def _check_transform_chain(transforms):
    if not isinstance(transforms, (list, tuple)) or len(transforms) == 0:
        raise ValueError("transforms must be a non-empty list")

    for idx, tr in enumerate(transforms):
        if not _is_rigid_transform(tr):
            raise ValueError(
                f"transform[{idx}] must be a RigidTransform instance, "
                f"got {type(tr).__name__}"
            )

        R = np.asarray(tr.R, dtype=np.float64)
        t = np.asarray(tr.t_m, dtype=np.float64)

        if R.shape != (3, 3):
            raise ValueError(f"transform[{idx}].R must be (3, 3)")
        if t.shape != (3,):
            raise ValueError(f"transform[{idx}].t_m must be (3,)")
        if not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError(f"transform[{idx}] contains non-finite values")

        if (not isinstance(tr.source_frame_id, str)
                or tr.source_frame_id.strip() == ""):
            raise ValueError(f"transform[{idx}] invalid source_frame_id")
        if (not isinstance(tr.target_frame_id, str)
                or tr.target_frame_id.strip() == ""):
            raise ValueError(f"transform[{idx}] invalid target_frame_id")

        if idx > 0:
            prev_target = transforms[idx - 1].target_frame_id
            if prev_target != tr.source_frame_id:
                raise ValueError(
                    f"transform chain mismatch at index {idx}: "
                    f"{prev_target} != {tr.source_frame_id}"
                )


def _check_report_frame_id(report):
    """顶层 frame_id 必须与所有对象（含无效簇）一致。"""
    report_frame_id = report.get("frame_id")
    if not isinstance(report_frame_id, str) or report_frame_id.strip() == "":
        raise ValueError("report must have a non-empty frame_id")

    for obj in report["objects"]:
        if obj.get("frame_id") != report_frame_id:
            raise ValueError(
                f"object {obj.get('cluster_label')} frame_id "
                f"({obj.get('frame_id')}) does not match "
                f"report frame_id ({report_frame_id})"
            )
    return report_frame_id


def _compare_positions(report_positions, audited_positions):
    """清单 2：缓存比较使用 rtol=0，严格数值匹配。"""
    if not isinstance(report_positions, list):
        raise ValueError("published_positions must be a list")

    audited_map = {p["cluster_label"]: p for p in audited_positions}
    if len(report_positions) != len(audited_map):
        raise ValueError(
            f"published_positions has {len(report_positions)} items, "
            f"but audit produced {len(audited_map)} valid objects"
        )

    seen = set()
    for item in report_positions:
        if not isinstance(item, dict):
            raise ValueError("published_positions item must be a dict")

        label = item.get("cluster_label")
        if label not in audited_map:
            raise ValueError(
                f"published_positions contains invalid label: {label}"
            )
        if label in seen:
            raise ValueError(
                f"duplicate cluster_label in published_positions: {label}"
            )
        seen.add(label)

        expected = audited_map[label]

        if item.get("frame_id") != expected["frame_id"]:
            raise ValueError(
                f"frame_id mismatch for label {label}: "
                f"{item.get('frame_id')} != {expected['frame_id']}"
            )

        if not _is_finite_3d(item.get("position_m")):
            raise ValueError(f"position_m invalid for label {label}")

        if not np.allclose(
            np.asarray(item["position_m"], dtype=np.float64),
            np.asarray(expected["position_m"], dtype=np.float64),
            rtol=0,            # 清单 2：禁止相对误差
            atol=1e-12,
        ):
            raise ValueError(f"position_m mismatch for label {label}")


# =========================================================
# 主函数
# =========================================================

def audit_chain_and_compute(report, transforms):
    """
    输入：
        EXP034 定位报告
        非空的 RigidTransform 变换列表
    输出：
        仅供诊断的变换结果报告

    安全要求：
        任何情况下不产生可发布的机器人位置。
    """
    # ---------- 1. 基础结构检查 ----------
    if not isinstance(report, dict):
        raise ValueError("report must be a dict")

    _check_transform_chain(transforms)

    # ---------- 2. 从 objects 重新审核 VALID 位置 ----------
    audited_positions = audit_and_extract(report)

    # ---------- 3. report.frame_id 与 object.frame_id 一致性 ----------
    report_frame_id = _check_report_frame_id(report)

    # ---------- 4. 缓存核对（若存在） ----------
    if "published_positions" in report:
        _compare_positions(report["published_positions"], audited_positions)

    # ---------- 5. 链起点与 report.frame_id 一致 ----------
    chain_source = transforms[0].source_frame_id
    if report_frame_id != chain_source:
        raise ValueError(
            f"report frame_id ({report_frame_id}) does not match "
            f"transform chain source ({chain_source})"
        )

    # ---------- 6. 逐级使用 transform_point() 计算诊断坐标 ----------
    # 清单 4：不手写 R @ p + t，而是调用 RigidTransform 的方法
    diagnostics = []
    for pos in audited_positions:
        label = pos["cluster_label"]
        current_p = _as_vec3(pos["position_m"], f"position[{label}]")
        current_frame = pos["frame_id"]
        steps = []

        for idx, tr in enumerate(transforms):
            if current_frame != tr.source_frame_id:
                raise ValueError(
                    f"frame mismatch at step {idx} for label {label}: "
                    f"current={current_frame}, "
                    f"expected source={tr.source_frame_id}"
                )

            next_p = tr.transform_point(
                current_p, frame_id=current_frame
            )
            next_p = np.asarray(next_p, dtype=np.float64)

            if next_p.shape != (3,) or not np.isfinite(next_p).all():
                raise ValueError(
                    f"transform_point produced invalid output at "
                    f"step {idx} for label {label}"
                )

            steps.append({
                "step": idx,
                "source_frame_id": tr.source_frame_id,
                "target_frame_id": tr.target_frame_id,
                "position_m": [float(x) for x in next_p],
            })

            current_p = next_p
            current_frame = tr.target_frame_id

        diagnostics.append({
            "cluster_label": label,
            "source_frame_id": pos["frame_id"],
            "final_frame_id": transforms[-1].target_frame_id,
            "final_position_m": [float(x) for x in current_p],
            "steps": steps,
        })

    # ---------- 7. 发布权限始终 fail-closed ----------
    result = {
        "publishable": False,
        "published_positions": [],
        "diagnostics": diagnostics,
        "reasons": ["NO_TRUSTED_GLOBAL_VALIDATION_SYSTEM"],
    }

    assert result["publishable"] is False
    assert result["published_positions"] == []
    return result