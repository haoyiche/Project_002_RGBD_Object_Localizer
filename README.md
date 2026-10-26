# Project_002_RGBD_Object_Localizer

## 1. Project

**Level 2 — 3D Vision & Geometry**

主项目：**RGB-D Object Localizer**

目标是把二维视觉结果转换成真实三维空间中的位置与姿态信息。

> 目前 OS-01-B 仅完成合成 RGB-D 数据的诊断流水线，尚不能输出可供机器人执行的真实坐标。

---

## 2. Current Status

```text
Level 1: COMPLETE
Level 2: CONTINUE

OS-01-A: COMPLETE
OS-01-B: Engineering Tests + Boss PASS
         Learning Notes Ready; Git/GitHub Closure Pending

Current Learning Unit:
OS-01-B — RGB-D Object Localizer Pipeline
```

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

以上是 Level 2 总体目标，不表示每一项均已在 OS-01-B 中完成。目前尚未验证真实相机标定、物理精度和机器人发布。

---

## 4. Core Topics

- Camera Intrinsics
- Camera Extrinsics
- Projection / Back Projection
- RGB-D
- Point Cloud
- Coordinate Transform
- ICP / Global Registration
- 3D Position / 6D Pose
- Error Analysis

---

## 5. Level 2 BOSS

- RGB-D Object Localizer
- 稳定输出目标 3D 位置
- 至少 20 个样本误差 / 一致性统计
- Mean / Median / P95 / Max
- 能解释内参、外参、深度、反投影、坐标变换
- 点云代码与模型代码解耦
- 至少一次坐标方向错误注入实验
- 可复现实验代码与笔记
- Git commit + GitHub push

> Level 2 BOSS 尚未整体完成；OS-01-B 的小单元综合 Boss 已通过。

---

## 6. Learning Progress

```text
EXP026 — Camera Intrinsics & Pixel to 3D: COMPLETE
OS-01-A — RGB-D Input Contract: COMPLETE
OS-01-B — RGB-D Pipeline: Tests/Boss PASS; Git Closure Pending
```

---

## 7. OS-01 RGB-D Object Localizer

### 7.1 Integrated Pipeline

```text
RGB + Depth + Target Mask
    ↓
3D Back Projection
    ↓
Radius Outlier Filter
    ↓
Point Cloud Statistics
    ↓
DBSCAN Clustering
    ↓
EXP034 Geometry Gates
    ↓
Diagnostic Report (Fail-Closed)
```

### 7.2 Implemented Components

- **OS-01-A**：RGB-D 输入契约、可复现合成样本。
- **OS-01-B1**：Mask 反投影、Radius Filter、几何统计与异常状态。
- **OS-01-B2**：DBSCAN 聚类、EXP034 簇级门禁、状态映射与 TDD 测试。

### 7.3 Pipeline Status Contract

| Condition | Top-Level Status |
| --- | --- |
| Empty target mask | `NO_OBJECTS` |
| No valid depth under mask | `NO_VALID_DEPTH` |
| No points survive radius filtering | `NO_POINTS_AFTER_FILTER` |
| DBSCAN produces no clusters | `NO_CLUSTERS` |
| All clusters fail geometry gates | `GEOMETRY_GATE_REJECTED` |
| At least one cluster passes geometry gates | `DIAGNOSTIC_ONLY` |

Detailed partial failures remain visible under `diagnostics.geometry_gate.status` as `PARTIAL_INVALID`.

### 7.4 Safety Contract

当前仅使用合成数据验证 Pipeline 的算法逻辑和接口行为。无论簇是否通过局部门禁，最终安全出口都必须保持：

```python
report["position_m"] = None
report["publishable"] = False
```

EXP034 的局部几何门禁通过，不代表完成可信标定、独立物理真值验证、时间同步、坐标变换和机器人安全审批。

### 7.5 How to Reproduce

在项目根目录、已激活相应 Python 环境时运行：

```powershell
python -m src.pipeline.os01_single_sample
python -m pytest tests/test_os01_b1_pipeline.py tests/test_os01_b2_geometry_gate.py -q
python -m pytest -q
git diff --check
```

OS-01-B 全量回归记录（2026-10-26）：

```text
703 passed, 106 subtests passed in 2.51s
```

该结果为当时的实测记录，不代表后续变更后自动持续通过。

### 7.6 Documentation

- `experiments/OS01_B_rgbd_pipeline.md`

### 7.7 Known Limitations and Next Step

- 目前没有真实相机采集与可信外参验证。
- 未完成独立物理精度和工作空间覆盖评估。
- EXP034 默认门禁属于教学配置，不是经过标定的生产阈值。
- 当前 NumPy DBSCAN 使用点对距离矩阵，空间开销为 O(N²)，不适合未经限流的大规模点云。
- OS-01-C：进入可追溯的多样本 RGB-D 评估，覆盖至少 20 个样本及误差统计。
- **Level 2 仍为 CONTINUE；OS-01-B 只有完成 Git commit / GitHub push 后才能标记 COMPLETE。**
