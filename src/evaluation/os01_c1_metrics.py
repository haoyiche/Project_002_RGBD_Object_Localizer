import numpy as np


def evaluate_case(
    *,
    sample_id,
    report,
    reference_position_m,
    reference_frame_id,
):
    result = {
        "sample_id": sample_id,
        "eligible": False,
        "reason": None,
        "error_mm": None,
    }

    # 1. Pipeline did not produce an accepted diagnostic result.
    if report["status"] != "DIAGNOSTIC_ONLY":
        result["reason"] = "PIPELINE_REJECTED"
        return result

    # 2. Only one accepted cluster can be matched unambiguously
    # in this initial single-target evaluator.
    objects = report["diagnostics"]["geometry_gate"]["objects"]

    valid_objects = [
        obj for obj in objects
        if obj["local_gate_passed"]
    ]

    if len(valid_objects) != 1:
        result["reason"] = "NO_UNIQUE_VALID_CLUSTER"
        return result

    # 3. Reference must exist.
    if reference_position_m is None:
        result["reason"] = "MISSING_REFERENCE"
        return result

    # 4. Coordinates must use the same frame.
    if reference_frame_id != report["frame_id"]:
        result["reason"] = "FRAME_MISMATCH"
        return result

    # 5. Extract the diagnostic centroid, not top-level position_m.
    estimated = np.asarray(
        valid_objects[0]["centroid_camera_m"],
        dtype=np.float64,
    )

    reference = np.asarray(
        reference_position_m,
        dtype=np.float64,
    )

    if estimated.shape != (3,) or reference.shape != (3,):
        raise ValueError("positions must have shape (3,)")

    if not (
        np.isfinite(estimated).all()
        and np.isfinite(reference).all()
    ):
        raise ValueError("positions must contain finite values")

    # 6. Euclidean error: m -> mm.
    error_m = np.linalg.norm(estimated - reference)

    result["eligible"] = True
    result["error_mm"] = float(error_m * 1000.0)

    return result