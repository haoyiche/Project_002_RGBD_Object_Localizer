# Project_002_RGBD_Object_Localizer

## 1. Project

**Level 2 — 3D Vision & Geometry**

主项目：**RGB-D Object Localizer**（P02）。

目标是将 RGB、深度及目标 Mask 转换为相机坐标系下的三维几何诊断结果，并逐步建立定位评估与可信坐标变换能力。

> **当前安全边界：**OS-01-A/B/C/D 均基于确定性合成 RGB-D 数据验证输入契约、诊断 Pipeline、批量评估、Open3D/SE(3) 和同帧一致性。尚无可信的真实相机物理精度与外参验证；所有集成报告均保持 `publishable=False`、`position_m=None`。

---

## 2. Current Status

```text
Level 1: COMPLETE
Level 2: CONTINUE

P02 / OS-01 — RGB-D Object Localizer
  OS-01-A: COMPLETE — RGB-D input contract (e35ff69)
  OS-01-B: COMPLETE — single-sample diagnostic pipeline (bc7e3ff)
  OS-01-C: COMPLETE — 20-sample traceable synthetic evaluation (41a69c5)
  OS-01-D: ENGINEERING + BOSS PASS; Git/GitHub closure pending
           D1 Open3D / NumPy / Voxel
           D2 Background Plane and Target Membership
           D3 Synthetic Camera → World Demo Transform
           D4 Same-Capture Validation Contract

Current Learning Unit:
  OS-01-D — Learning note, final regression, Git/GitHub closure

Next:
  Level 2 gap review: real data / calibration / physical validation
```

**验收边界：** OS-01-D 已完成运行前预测、主动参数修改、故障注入、综合 Boss 和全仓回归，**尚未 Commit/Push**，不得提前标记 `COMPLETE`。真实机器人定位和独立物理精度未通过验收，整个 Level 2 仍为 `CONTINUE`。

---

## 3. Level 2 Goal

```text
RGB + Depth
→ Detection / Segmentation
→ Mask × Depth
→ Back Projection
→ Target Point Cloud
→ Filtering / Plane Removal / Clustering
→ 3D Position
→ Coordinate Transform
→ Pose Estimation
→ Error Evaluation
```

以上为 **Level 2 总体路线**，不是 OS-01-D 已经完成的功能清单。例如，真实相机标定、真实坐标变换链、物理精度和机器人发布尚未通过验收；Plane Removal 也不应在仅有五点共面的目标点云上无条件执行。

---

## 4. Core Topics

- Camera Intrinsics / Extrinsics
- Projection / Back Projection
- RGB-D Alignment and Input Validation
- Mask-to-Pointcloud
- Radius Outlier Filtering
- 3D Centroid and Axis-Aligned Bounding Box (AABB)
- DBSCAN Clustering (Core / Border / Noise)
- Geometry Gates and Explicit Status Contracts
- Coordinate Frames and Transformation Validation
- Synthetic Reference Discrepancy / Error Analysis
- Traceable Evaluation, SHA-256, CSV / JSON Reporting
- Open3D Voxel, RANSAC Background Plane, Mask Membership Verification
- SE(3) Transform Direction, Frame-Bound Capture Snapshot and Fail-Closed
- ICP / Global Registration and 3D Position / 6D Pose (Level 2 broader topics)

---

## 5. Level 2 BOSS

Level 2 最终目标仍包括：

- RGB-D Object Localizer：稳定产生可解释的三维目标结果。
- 至少 20 个样本的误差或一致性统计，包含 Mean / Median / P95 / Max。
- 能解释内参、外参、深度、反投影、坐标变换的数学和代码。
- 点云处理代码与检测 / 分割模型代码解耦。
- 至少一次坐标方向错误注入与故障定位验证。
- 可复现实验、测试、学习笔记以及 Git commit / GitHub push。

**注意：**OS-01-C 已完成合成样本上的 20 样本评估，OS-01-D 已完成 Open3D 与合成坐标链诊断，但不代表所有 Level 2 Boss 条件均已完成；真实物理定位精度与坐标链验证必须单独验收。

---

## 6. Learning Progress

| Learning Unit | 内容 | 状态 / 证据 |
| --- | --- | --- |
| EXP026 | Camera Intrinsics & Pixel to 3D | COMPLETE（此前学习记录） |
| OS-01-A | RGB-D Input Contract | COMPLETE · `e35ff69` |
| OS-01-B1 | Back Projection, Radius Filter, Geometry Statistics | PASS |
| OS-01-B2 | DBSCAN, EXP034 Gates, Fail-Closed Pipeline | PASS |
| OS-01-B | 单样本工程集成、综合 Boss、笔记及 GitHub | **COMPLETE · `bc7e3ff`** |
| OS-01-C1 | 单样本误差评估、资格判定、异常与真实 Pipeline 集成 | PASS · 10 tests |
| OS-01-C2 | 20 个不同合成输入、Manifest、批量评估与分组审计 | PASS · 4 tests |
| OS-01-C | 综合 Boss、笔记、全仓回归与 GitHub | **COMPLETE · `41a69c5`** |
| OS-01-D1 | NumPy/Open3D 对照、Voxel 主动修改、质心偏移 | PASS · 2 tests |
| OS-01-D2 | RANSAC 背景平面、Mask 身份检查与等量身份交换 | PASS · 2 tests |
| OS-01-D3 | 合成 Camera → World 变换、反向外参/上游阻断 | PASS · 4 tests |
| OS-01-D4 | 同帧快照、UID/哈希/有效期、故障注入 | PASS · 7 tests |
| OS-01-D | 综合 Boss、全仓回归，笔记与 GitHub 闭环 | **Engineering/Boss PASS；Git pending** |
| Level 2 最终 Boss | Open3D / 坐标链 / 真实数据迁移等剩余验收 | **CONTINUE** |

---

## 7. OS-01-A/B — Diagnostic RGB-D Pipeline

### 7.1 Pipeline

```text
RGB + Depth (m) + Target Mask
    ↓
Input Validation / Mask ∩ Valid Depth
    ↓
Camera Back Projection → XYZ Camera Point Cloud (m)
    ↓
Radius Outlier Filter
    ↓
Point Cloud Statistics (Centroid / AABB)
    ↓
DBSCAN on Filtered Points
    ↓
EXP034 Per-Cluster Geometry Gates
    ↓
Diagnostic Report (Fail-Closed)
```

**关键数据契约：**

- `rgb`: `(H, W, 3)`, `uint8`；`depth_m`: `(H, W)`, 米；`target_mask`: `(H, W)`, `bool`。
- `points` / `filtered_points`: `(N, 3)` / `(M, 3)`，相机坐标系，米。
- `labels`: `(M,)`，与 `filtered_points` 同源同序；`-1` 表示 DBSCAN Noise，不能作为簇 ID。
- Radius Filter 的邻居计数**不包含自身**；DBSCAN 的 `min_points` 邻域计数**包含自身**。

### 7.2 Pipeline Status Contract

| Condition | Top-Level Status |
| --- | --- |
| Empty target Mask | `NO_OBJECTS` |
| No valid depth under Mask | `NO_VALID_DEPTH` |
| No points survive Radius Filter | `NO_POINTS_AFTER_FILTER` |
| DBSCAN finds no clusters | `NO_CLUSTERS` |
| Every DBSCAN cluster fails EXP034 gates | `GEOMETRY_GATE_REJECTED` |
| At least one cluster passes geometry gates | `DIAGNOSTIC_ONLY` |

部分簇通过、部分失败时，具体门禁状态 `PARTIAL_INVALID` 保留在 `report["diagnostics"]["geometry_gate"]["status"]`。顶层的 `DIAGNOSTIC_ONLY` **不表示坐标可以下发给机器人**。

### 7.3 Safety Contract

```python
report["position_m"] = None
report["publishable"] = False
```

`EXP034` 的局部 `publishable=True` 仅代表相应教学几何门禁通过，**不是**可信外参、真实物理准确性、运动规划或机器人安全授权。当前正式 Pipeline 始终保持 Fail-closed。

### 7.4 Implementation and Notes

- `src/geometry/mask_to_pointcloud.py`
- `src/geometry/pointcloud_filter.py`
- `src/geometry/pointcloud_statistics.py`
- `src/geometry/dbscan.py`
- `src/geometry/cluster_localization.py`
- `src/pipeline/os01_single_sample.py`
- `experiments/OS01_B_rgbd_pipeline.md`

---

## 8. OS-01-C — Traceable Multi-Sample Evaluation

### 8.1 Evaluation Flow

```text
Deterministic Synthetic Sample Generator
    ↓
20 distinct RGB-D inputs + Sample Manifest
    ↓
SHA-256 / Schema / Input Integrity Validation
    ↓
OS-01-B localize_single_sample()
    ↓
OS-01-C1 evaluate_case()
    ↓
Per-Sample Status / Exclusion Reason / Error Eligibility
    ↓
CSV Results + JSON Summary
    ↓
Failure Breakdown / Synthetic Discrepancy Statistics
```

**C1 单目标限制：**仅当 Pipeline 诊断有效、恰有一个通过门禁的簇、参考值存在且合法、目标与 `frame_id` 相符时计算位置差异。合法但不可评估时返回 `eligible=False` 与 `reason`；非法坐标在实际检查到时抛出 `ValueError`。`None` 不得当作零误差。

### 8.2 Scenario Design

| Scenario Group | Count | Purpose |
| --- | ---: | --- |
| `normal_geometry` | 6 | 不同深度与像素平移，评估固定缺失像素下的解析偏差 |
| `mask_perturbation` | 4 | Mask 缺失或空目标 |
| `depth_anomaly` | 4 | 零深度、NaN、Inf、全部无效深度 |
| `cluster_gate_pressure` | 4 | 过滤后空点云、全 Noise、门禁拒绝等 |
| `missing_reference` | 2 | 几何通过但参考值缺失时正确排除 |
| **Total** | **20** | 20 份不同的 RGB-D 输入内容 |

Manifest 位于 `data/manifests/os01_c_samples.json`，包括样本身份、生成参数、参考来源、输入文件路径及哈希。可重新生成的 `.npz` 输入位于 `data/raw/synthetic/os01_c/`，默认受 `.gitignore` 排除，不必提交二进制数据。

### 8.3 Observed Batch Results (2026-10-27)

| Metric | Observed |
| --- | ---: |
| Total synthetic samples | 20 |
| Geometry passed | 8 / 20 (**40%**) |
| Geometry not passed | 12 / 20 (**60%**) |
| Evaluable samples | 6 / 20 (**30%**) |
| Excluded from error stats | 14 / 20 (**70%**) |
| Missing references | 2 |
| Mean (eligible only) | **2.2361 mm** |
| Median (eligible only) | **2.2361 mm** |
| P95 (eligible only) | **2.4820 mm** |
| Max (eligible only) | **2.5044 mm** |

**统计指标名：**`synthetic_nominal_reference_discrepancy_mm`。当前数据使用合成名义参考位置，而非独立物理真值；`physical_accuracy_claimed=false`。上述 Mean / Median / P95 / Max 的分母均是 **6 个可评估样本**，而不是 20 个。

| Pipeline Status | Count |
| --- | ---: |
| `DIAGNOSTIC_ONLY` | 8 |
| `GEOMETRY_GATE_REJECTED` | 8 |
| `NO_CLUSTERS` | 1 |
| `NO_OBJECTS` | 1 |
| `NO_POINTS_AFTER_FILTER` | 1 |
| `NO_VALID_DEPTH` | 1 |

排除原因共 14 个：`PIPELINE_REJECTED` 12 个、`MISSING_REFERENCE` 2 个。故障注入场景中的门禁通过率不应被当作真实场景自然成功率。

### 8.4 Depth-Dependent Synthetic Centroid Discrepancy

当前 `normal_geometry` 场景的完整目标包含 6 个像素，但左上角一个深度无效，仅有 5 个有效点。固定像素相对布局、固定内参和相同目标深度时：

```text
Nominal mean pixel:   (u, v) = (4.0, 2.5)
Effective mean pixel: (u, v) = (4.2, 2.6)
Pixel offset:         (du, dv) = (+0.2, +0.1) px
fx = fy = 100 px
```

由反投影关系得：

```text
dX = 0.002 * Z   (m)
dY = 0.001 * Z   (m)
Synthetic discrepancy_mm = sqrt(5) * Z
```

其中 `Z` 以米计。六个深度值 `0.88, 0.92, 0.96, 1.04, 1.08, 1.12 m` 的预测差异与实际 CSV 逐项一致。该规律是**确定性输入构造下的结构性偏差**，不能外推为真实相机的物理定位精度。

### 8.5 Reproducibility and Files

| Asset | Path |
| --- | --- |
| Single-case evaluator | `src/evaluation/os01_c1_metrics.py` |
| 20-sample generator / validator | `src/evaluation/os01_c2_generate.py` |
| Batch evaluator | `src/evaluation/os01_c2_batch_evaluate.py` |
| C1 tests | `tests/test_os01_c1_metrics.py` |
| Generator tests | `tests/test_os01_c2_generation.py` |
| Batch tests | `tests/test_os01_c2_batch_evaluate.py` |
| Manifest | `data/manifests/os01_c_samples.json` |
| Per-sample CSV | `outputs/os01_c/evaluation_results.csv` |
| Summary JSON | `outputs/os01_c/evaluation_summary.json` |
| Learning note | `experiments/OS01_C_multisample_evaluation.md` |

报告记录 Manifest / 输入哈希与关键代码的 SHA-256，以便检测非预期变动。**哈希本身不能证明参考位置是独立物理真值**；若 Manifest 和输入文件同时被改写，仍需可信的外部版本记录或签名作验证锚点。

---

## 9. OS-01-D — Open3D / Coordinate Chain / Same-Capture Contract

### 9.1 D1 — Open3D Baseline and Voxel Ablation

从 OS-01-A 五点目标点云出发，NumPy 与 Open3D DBSCAN 在 `eps=0.011 m, min_points=3` 下均得到 **1 簇、0 Noise**。`voxel_size=0.005 m` 保留五点；改为 `0.020 m` 后得到三个体素代表点。

| 比较项 | 数值 |
| --- | --- |
| 原始质心 | `[0.007,0.001,1.0] m` |
| 20 mm 体素代表点等权质心 | `[0.005,0.0016667,1.0] m` |
| 质心位移 | 约 `2.1082 mm` |

**含义：** Voxel 改变采样密度与统计权重。若在 DBSCAN 之前下采样，邻居数可能从 `[3,3,2,4,3]` 变为 `[2,1,2]`，导致全 Noise。

### 9.2 D2 — Full-Scene RANSAC and Target Identity

复用 EXP028 的固定合成场景：80×100 像素、100 个无效深度、7900 个有效 3D 点；背景 `Z=2.0 m`，目标 `Z=1.2 m`。RANSAC 找到 6400 个背景内点并保留 1500 个目标点，逐点身份/XYZ/RGB 均匹配。

故障注入将一个目标深度与一个背景深度交换，**6400/1500 的数量维持不变**，但身份/XYZ/RGB 均不匹配，返回 `SEGMENTATION_MISMATCH`、`publishable=False`。

D2 的更新接口允许 `analyze_scene(scene=(depth_m,rgb,target_mask))`，以便后续验证同一采集快照；默认仍可运行固定合成演示。

### 9.3 D3 — Synthetic Camera to World-Demo

合成旋转 `Rz(+90°)`，平移 `[0.10,-0.20,0.30] m`：

```text
centroid_camera_m     = [0.004, 0.004, 1.2]
centroid_world_demo_m = [0.096,-0.196, 1.5]
```

Round-trip 误差约 `5.06e-15 mm`，**只能证明数值自洽**。误用逆向外参返回 `FRAME_MISMATCH`；D2 拒绝则返回 `UPSTREAM_REJECTED`；人为填写 `calibration_verified=True` 仍不允许发布。

### 9.4 D4 — Same-Capture Consistency

`Capture` 携带 `frame_uid`、`frame_id`、`capture_timestamp_ns` 和 RGB/Depth/Mask；`validate_capture()` 拷贝快照、调用 D2、记录 SHA-256 与时效信息；`analyze_validated_capture()` 在校验通过后，仅使用该 D2 验证过的快照进行数值变换。

七个专项测试覆盖：正常同帧、跨帧 UID、同 UID 数据修改、过期、D2 上游失败、逆向外参以及伪造标定声明。**7 passed**。此外 D1/D2/D3 专项分别为 **2/2/4 passed**。

**限制：** 本接口是同进程的教学一致性检查，不是带可信签名的生产票据；`now_ns` 来自调用方，不能替代可信时间同步；`@dataclass(frozen=True)` 并不让数组/字典的内部内容自动不可变。

### 9.5 Files and Diagnostics

- `src/pipeline/os01_d_open3d_compare.py`
- `src/pipeline/os01_d2_plane_background.py`
- `src/pipeline/os01_d3_coordinate_chain.py`
- `src/pipeline/os01_d4_same_capture.py`
- `tests/test_os01_d1_open3d_compare.py`
- `tests/test_os01_d2_plane_background.py`
- `tests/test_os01_d3_coordinate_chain.py`
- `tests/test_os01_d4_same_capture.py`
- `experiments/OS01_D_open3d_coordinate_chain.md`

**最终安全出口：** `position_m=None`、`publishable=False`，不把合成 `world_demo` 坐标冒充已认证机器人 `base_link` 坐标。

---

## 10. How to Reproduce

在仓库根目录启动已配置的 Python 环境（本轮用户环境：Python 3.11，conda `ai_3d`）：

```powershell
# OS-01-A: regenerate the minimal sample when needed
python -m src.experiments.os01_a_synthetic_rgbd_sample

# OS-01-B: single-sample diagnostic Pipeline
python -m src.pipeline.os01_single_sample

# OS-01-C1: single-case evaluator and real-Pipeline integration
python -m pytest tests/test_os01_c1_metrics.py -v

# OS-01-C2: generate and validate 20 synthetic inputs
python -m pytest tests/test_os01_c2_generation.py -v
python -m src.evaluation.os01_c2_generate

# OS-01-C2: evaluate the batch, write CSV / JSON
python -m pytest tests/test_os01_c2_batch_evaluate.py -v
python -m src.evaluation.os01_c2_batch_evaluate
Get-Content .\outputs\os01_c\evaluation_summary.json -Encoding UTF8

# OS-01-D: compare, segment, transform and bind the same capture
python -m src.pipeline.os01_d_open3d_compare
python -m src.pipeline.os01_d2_plane_background
python -m src.pipeline.os01_d3_coordinate_chain
python -m src.pipeline.os01_d4_same_capture

python -m pytest tests/test_os01_d1_open3d_compare.py -v
python -m pytest tests/test_os01_d2_plane_background.py -v
python -m pytest tests/test_os01_d3_coordinate_chain.py -v
python -m pytest tests/test_os01_d4_same_capture.py -v

# Full project regression
python -m pytest -q
```

**最近一次完整回归记录（2026-10-29，OS-01-D4 之后）：**

```text
732 passed, 106 subtests passed in 6.41s
```

以上为一次实测快照，不代表未来修改后仍自动通过。

---

## 11. Known Limitations and Next Steps

- **Synthetic only:** 尚未完成真实 RGB-D 采集、相机内外参可信标定、工作空间覆盖、时间同步与独立物理真值测量。
- **No robot output:** `position_m=None`、`publishable=False`；算法评估成功不构成机器人执行许可。
- **Single-target metric:** C1 要求唯一通过门禁簇；多目标场景需建立显式身份匹配，不能挑最近簇掩盖匹配错误。
- **Small sample and biased distribution:** 6 个可评估样本的 P95 仅为描述性统计；故障注入场景不能代表真实故障率。
- **Computation:** 当前 NumPy DBSCAN 采用两两距离矩阵，存储量 O(N²)，大规模点云需要优化邻域搜索或限制规模。
- **Broader Level 2:** 已完成 Open3D 与合成 SE(3) 单元集成；下一阶段重点是实际数据、可信内外参、真实物理参考值、空间覆盖和时间同步的差距盘点。FoundationPose / 6D Pose 依主路线进阶。
- **D4 trust boundary:** 防御性快照与 SHA-256 不能单独防止整个验证票据及数据被联合伪造，且票据内部的数组/字典不具备严格不可变保证。
- **Git closure:** OS-01-C 已于 `41a69c5` 完成 GitHub 闭环；OS-01-D 当前仅完成功能、Boss 与回归，待文档入库、逐文件暂存、Commit/Push、`HEAD == origin/master` 验证。

---

## 12. Documentation and Git History

- **AI Lab learning protocol:** `AI_Lab_SOP.md`（AI Lab Project 核心规范文件，可能位于仓库外）。
- **OS-01-B learning note:** `experiments/OS01_B_rgbd_pipeline.md`。
- **OS-01-C learning note:** `experiments/OS01_C_multisample_evaluation.md`。
- **OS-01-D learning note:** `experiments/OS01_D_open3d_coordinate_chain.md`。
- **OS-01-A reference commit:** `e35ff69`。
- **OS-01-B completed commit:** `bc7e3ff`。
- **OS-01-C completed commit:** `41a69c5`。
- **OS-01-D commit:** pending；完成 Push 并核对远端 SHA 后再标记 COMPLETE。

**Level 2: CONTINUE**。代码测试通过并非整个 Level 或 Learning Unit 的最终通关证据；还需要 SOP 要求的理解、预测、主动实验、笔记、Boss 和 Git/GitHub 记录。
