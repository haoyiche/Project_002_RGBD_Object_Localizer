
"""EXP037-A: Minimal extrinsic rotation validation.

Synthetic camera-to-world transformation.
All coordinates are in meters.
"""

import numpy as np


# 1. Camera-frame point
p_camera = np.array([0.0, 0.10, 1.0], dtype=np.float64)

# 2. True extrinsic parameters
R_true = np.eye(3, dtype=np.float64)
t_true = np.zeros(3, dtype=np.float64)

# 3. Incorrect extrinsic parameters
R_used = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ],
    dtype=np.float64,
)
t_used = np.zeros(3, dtype=np.float64)

# 4. Coordinate transformations
p_world_true = R_true @ p_camera + t_true
p_world_used = R_used @ p_camera + t_used

# 5. Localization error
error_vector_m = p_world_used - p_world_true
error_vector_mm = error_vector_m * 1000.0
error_norm_mm = np.linalg.norm(error_vector_mm)

# 6. Results
print("P_camera:", p_camera)
print("P_world_true:", p_world_true)
print("P_world_used:", p_world_used)
print("Error vector (mm):", error_vector_mm)
print("Error norm (mm):", error_norm_mm)

print("Point shape:", p_camera.shape)
print("Rotation shape:", R_used.shape)
print("Point dtype:", p_camera.dtype)
