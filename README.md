# Project_002_RGBD_Object_Localizer

## 1. Project

**Level 2 — 3D Vision & Geometry**

主项目：**RGB-D Object Localizer**（P02）。

目标是将 RGB、深度及目标 Mask 转换为相机坐标系下的三维几何诊断结果，并逐步建立定位评估与可信坐标变换能力。

> **当前安全边界：**OS-01-A/B/C 使用确定性合成 RGB-D 数据验证输入契约、单样本 Pipeline 和多样本评估；尚无真实相机独立物理真值验证，也不输出可供机器人执行的坐标。

---

## 2. Current Status

```text
Level 1: COMPLETE
Level 2: CONTINUE

P02 / OS-01 — RGB-D Object Localizer
  OS-01-A: COMPLETE — RGB-D input contract
  OS-01-B: COMPLETE — single-sample diagnostic pipeline
           GitHub commit: bc7e3ff
  OS-01-C: TESTS + BOSS PASS — 20-sample synthetic evaluation
           Learning notes saved; Git commit / GitHub push pending

Current Learning Unit:
  OS-01-C — Multi-Sample Traceable Evaluation (closure)

Next after closure:
  Level 2 remaining verification: Open3D integration,
  coordinate-chain / direction-error injection, and physical validation planning
```

**状态说明：**OS-01-C 已完成代码、预测、20 样本评估、综合 Boss 和全仓回归；直到本关的 Git commit / GitHub push 与远端 SHA 对齐，才将其标为 `COMPLETE`。整个 Level 2 的最终 Boss 仍未完成。

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

以上为 **Level 2 总体路线**，不是 OS-01-C 已经完成的功能清单。例如，真实相机标定、真实坐标变换链、物理精度和机器人发布尚未通过验收；Plane Removal 也不应在仅有五点共面的目标点云上无条件执行。

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

**注意：**OS-01-C 已完成合成样本上的 20 样本评估，不代表所有 Level 2 Boss 条件均已完成；真实物理定位精度与坐标链验证必须单独验收。

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
| OS-01-C | 综合 Boss、笔记、全仓回归 | **Tests/Boss PASS；Git/GitHub pending** |
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

## 9. How to Reproduce

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

# Full project regression
python -m pytest -q
```

**最近一次完整回归记录（2026-10-27）：**

```text
717 passed, 106 subtests passed in 3.05s
```

以上为一次实测快照，不代表未来修改后仍自动通过。

---

## 10. Known Limitations and Next Steps

- **Synthetic only:** 尚未完成真实 RGB-D 采集、相机内外参可信标定、工作空间覆盖、时间同步与独立物理真值测量。
- **No robot output:** `position_m=None`、`publishable=False`；算法评估成功不构成机器人执行许可。
- **Single-target metric:** C1 要求唯一通过门禁簇；多目标场景需建立显式身份匹配，不能挑最近簇掩盖匹配错误。
- **Small sample and biased distribution:** 6 个可评估样本的 P95 仅为描述性统计；故障注入场景不能代表真实故障率。
- **Computation:** 当前 NumPy DBSCAN 采用两两距离矩阵，存储量 O(N²)，大规模点云需要优化邻域搜索或限制规模。
- **Broader Level 2:** 继续核对 Open3D 实际调用、坐标方向错误注入、变换链及真实物理精度验收，并根据主路线安排 FoundationPose / 6D Pose 进阶。
- **Git closure:** OS-01-C 本轮尚需审查拟提交的代码、Manifest、结果报告和笔记，完成 Git commit、GitHub push，并确认 `HEAD == origin/master` 后更新状态为 `COMPLETE`。

---

## 11. Documentation and Git History

- **AI Lab learning protocol:** `AI_Lab_SOP.md`（AI Lab Project 核心规范文件，可能位于仓库外）。
- **OS-01-B learning note:** `experiments/OS01_B_rgbd_pipeline.md`。
- **OS-01-C learning note:** `experiments/OS01_C_multisample_evaluation.md`。
- **OS-01-A reference commit:** `e35ff69`。
- **OS-01-B completed commit:** `bc7e3ff`。
- **OS-01-C commit:** pending；不得在成功 Push 前填写或宣称 COMPLETE。

**Level 2: CONTINUE**。代码测试通过并非整个 Level 或 Learning Unit 的最终通关证据；还需要 SOP 要求的理解、预测、主动实验、笔记、Boss 和 Git/GitHub 记录。
