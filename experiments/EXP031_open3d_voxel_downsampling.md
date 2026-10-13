# EXP031｜Open3D Voxel Downsampling（体素下采样）

> **归属**：Level 2 / T1 / OS-01 / P02 RGB-D Object Localizer
> **日期**：2026-10-13
> **状态**：`CONTINUE`（理论、实验、Boss Test 已通过；本笔记仍需本人审核，Git Commit / GitHub Push 待完成）
> **环境**：Windows / Python 3.11.17 / NumPy 2.2.6 / Open3D 0.20.0 / pytest 9.1.1

## 1. 学习目标、知识地图和前置知识

- 前置：EXP028 mask→点云、EXP029 质心/AABB、EXP030 Radius Outlier Removal。
- 本次目标：解释体素网格索引；预测体素大小、边界、采样密度、颜色变化对输出的影响；独立使用 NumPy 实现体素聚合并与 Open3D 对照；能辨析定位质心偏差。
- 下游：RANSAC、DBSCAN、OBB、RGB-D 定位误差评测，以及坐标变换/6D Pose。

## 2. 原理和输入输出

### 2.1 定义

体素下采样将 3D 空间分为立方体网格，每个**非空体素**聚合成一个代表点。该操作减小点数，但不是以邻居阈值识别离群点。孤立点也可能独自占据一个体素并保留。

传统 Open3D `PointCloud.voxel_down_sample` 的本次使用约定（0.20.0）：

\[
\mathbf{o}=\min_i(\mathbf{p}_i)-\frac{s}{2},\qquad
\mathbf{k}_i=\left\lfloor\frac{\mathbf{p}_i-\mathbf{o}}{s}\right\rfloor
\]

其中 `min` 按 X/Y/Z 分量分别计算，`s` 为体素边长；本实验 XYZ 与 `s` 单位均为米。网格起点**不保证位于世界坐标零点**。

第 j 个非空体素的输出位置、颜色、原始成员数分别是：

\[
\bar{\mathbf p}_j=\frac{1}{n_j}\sum_{i\in V_j}\mathbf p_i,\qquad
\bar{\mathbf c}_j=\frac{1}{n_j}\sum_{i\in V_j}\mathbf c_i,\qquad
n_j=|V_j|.
\]

### 2.2 Shape 与参数

| 名称 | Shape / 类型 | 单位 / 解释 |
|---|---|---|
| `points` | `(N,3)` float64 | XYZ，本实验为 m |
| `colors` | `(N,3)` float64 | RGB，惯例 0～1 |
| `voxel_size` | 正有限实数 | 体素立方体边长，m |
| `voxel_keys` | `(M,3)` int64 | 非空体素索引（NumPy 参考实现） |
| `xyz_out`, `rgb_out` | 各 `(M,3)` | 体素内 XYZ / RGB 平均 |
| `counts` | `(M,)` int64 | 每个非空体素的原始成员数量 |

`s` 增大通常减少点数、提高处理效率，但改变局部几何分辨率、每组平均颜色和下游质心统计；它**不能替代离群点去除**。

## 3. 实验与预测记录

### 3.1 基线最小实验（5 点）

输入 X：`[0, 0.01, 0.08, 0.09, 1]`；颜色：红、绿、蓝、白、紫；Y=0，Z=1。

- `s=0.05`：输出 3 点，X=`[0.005, 0.085, 1]`，RGB=`[[.5,.5,0],[.5,.5,1],[1,0,1]]`。
- `s=0.20`：输出 2 点，X=`[0.045, 1]`，RGB=`[[.5,.5,.5],[1,0,1]]`。
- **初始认知误区**：最初假设网格从 `X=0` 开始。虽然预测点数恰巧正确，但推导前提错误；后续重新学习了 `origin = min - s/2`。

### 3.2 EXP031-A：边界敏感性

**唯一修改**：B 的 X 从 `0.01` 改为 `0.04`；XYZ 其他维度和颜色不变。

- **原预测**：`s=0.05` 仍输出 3 个点；错误依据是把网格当作 `[0,.05)` 起始。
- **修正后的预测（运行前）**：`origin_x=-0.025`，A/B/C/D/E 索引分别为 `[0,1,2,2,20]`；故 `s=0.05` 输出 4 点，X=`[0,.04,.085,1]`；`s=0.20` 输出 2 点，X=`[.0525,1]`。
- **实测**：完全一致；0.05 时 A、B 颜色分别保留红、绿，C/D 平均为 `[.5,.5,1]`。
- **结论**：两点距离小于体素边长不意味着处于同一体素，必须检查网格起点和体素索引。

### 3.3 EXP031-B：采样密度与质心权重

**唯一因素**：在 A 的五点修改版本上额外增加两份 C 点及同色观测，总计 7 点。

- **运行前预测**：`s=0.05` 输出 4 点，C/D/C₂/C₃ 聚合后 X=`0.0825`，RGB=`[.25,.25,1]`；`s=0.20` 输出 2 点。
- **实测**：如预测。原始七点 X 质心 `1.37/7 ≈ 0.195714`；0.05 等权体素质心 `0.280625`；0.20 等权体素质心 `≈0.530833`。
- **相对修改前五点版本**：原始点质心 `0.242 → 0.195714`；0.05 体素质心 `0.28125 → 0.280625`；0.20 体素质心 `0.52625 → 0.530833`。
- **结论**：体素等权质心会丢失原始点密度权重，可能与原始点均值产生显著偏差。

加权恢复原始**观测点**算术质心（并不自动等于物体真实几何中心）：

\[
\mathbf C=\frac{\sum_j n_j\bar{\mathbf p}_j}{\sum_j n_j}.
\]

### 3.4 EXP031-C：只改颜色

**唯一修改**：D 点 RGB `[1,1,1] → [1,0,0]`，所有 XYZ、其余颜色、体素边长不变。

- **运行前预测**：`s=0.05` 4 点，C/D/C₂/C₃ 的 RGB 为 `[.25,0,.75]`；`s=0.20` 2 点，六点聚合颜色为 `[1/3,1/6,1/2]`；XYZ 和质心不变。
- **实测**：全部一致。
- **结论**：空间分组取决于 XYZ，RGB 在分组内部聚合；下采样后不可以用 EXP030 的原始 `kept_indices` 配对 RGB。

### 3.5 EXP031-D：NumPy 参考实现

核心实现：

```python
origin = points.min(axis=0) - voxel_size / 2
keys = np.floor((points - origin) / voxel_size).astype(np.int64)
voxel_keys, inverse, counts = np.unique(
    keys, axis=0, return_inverse=True, return_counts=True
)
xyz_sums = np.zeros((len(voxel_keys), 3), dtype=np.float64)
rgb_sums = np.zeros((len(voxel_keys), 3), dtype=np.float64)
np.add.at(xyz_sums, inverse, points)
np.add.at(rgb_sums, inverse, colors)
xyz_out = xyz_sums / counts[:, None]
rgb_out = rgb_sums / counts[:, None]
```

`inverse[i]` 是原始点 i 所属的非空体素分组编号，连接原始点与分组聚合；代码需要额外的输入验证、空数组与极端索引防护，见正式文件。

项目文件：

- `src/experiments/exp031_open3d_voxel_minimal.py`：Open3D 最小实验；
- `src/experiments/exp031_voxel_numpy.py`：独立 NumPy 参考实现；
- `tests/test_pointcloud_voxel.py`：专项测试。

## 4. 自动化测试与工程证据

用户在 Windows `ai_3d` 环境中的实际测试记录（2026-10-13）：

```text
Python 3.11.17
NumPy 2.2.6
pytest 9.1.1
Open3D 0.20.0

pytest tests/test_pointcloud_voxel.py -v -rs
31 passed in 0.39s

pytest -q
95 passed, 106 subtests passed in 0.97s
```

已覆盖：手算真值；两种体素边长；Open3D 对照（XYZ/RGB 使用同一个空间排序索引）；空点云；体素大小非法值；形状与有限性；单点体素；输入顺序扰动；网格边界；颜色独立性；极端体素索引拒绝。

**一个真实排错**：原诊断 `zip("ABCDE", voxel_indices)` 只打印五个标签，七点实验遗漏 C₂/C₃ 日志。加入完整标签列表与 `strict=True` 后解决。另修正过测试中错误的 counts 排序预期和 XYZ/RGB 各自排序的问题。

## 5. 独立 Boss Test：96/100（教学评分）

- Q1 概念与属性聚合：20/20。
- Q2 等权与加权质心：20/20。例：点 `[.1,.4,.9]`、权重 `[10,3,1]`，等权质心 `≈.4667`，加权质心 `3.1/14≈.2214`。
- Q3 改变最小坐标对网格的影响：20/20。加入 `X=-.01` 后原有点索引恰未改变，但一般情况下网格移动可能改变别的点分组。
- Q4 排序与属性配对 Debug：19/20。XYZ 与 RGB 必须用同一个索引排序；重复空间坐标情况下，独立匹配需进一步处理，优先按唯一体素键。
- Q5 Mask + Depth 3D 定位迁移：17/20。稀疏背景体素可使等权质心偏移；先做背景/主簇去除，再对同一真值集评测。加权恢复的只是原始观测点均值，**不能保证恢复物体真实几何中心**。

**性质边界**：点云采样、遮挡、背景误分割会影响所有质心估计；RANSAC 用于几何模型（如平面）分割，DBSCAN 或半径/主簇策略用于密度与连通性筛选，不能把它们当成相同算法。

## 6. 常见误区及验证方式

| 原误解 / 风险 | 正确认知与验证 |
|---|---|
| 假设体素网格从 0 开始 | 用 `origin = min(points)-s/2` 打印实际索引，专门检查边界点 |
| 距离 `< voxel_size` 必然同体素 | 计算完整 3D 体素键，检查是否相同 |
| 下采样总能剔除孤立点 | 单独体素仍产生代表点，需另用离群点检测 |
| 下采样不影响质心 | 计算 raw mean、等权 voxel mean、按 counts 加权 mean 三者 |
| 保留原点索引即可取颜色 | 体素代表点及其颜色是聚合值，必须按相同成员分组 |
| 分别排序 XYZ/RGB 再比较 | 使用同一空间排序索引，或按体素键对齐属性 |
| 所有点都打印了 | 标签数量与点数一致、使用 `zip(..., strict=True)` |
| 测试通过就代表学完 | 需要独立解释/迁移、笔记、Git 与 GitHub 闭环 |

## 7. 运行与复现

PowerShell（在 P02 根目录）：

```powershell
$py = "D:\conda_envs\ai_3d\python.exe"
& $py src/experiments/exp031_open3d_voxel_minimal.py
& $py -m pytest tests/test_pointcloud_voxel.py -v -rs
& $py -m pytest -q
```

当前 Git 工作树曾报告有 3 份 `ScreenCamera*.json`、3 张 `ScreenCapture*.png`、两个 `.bak_*` 文件未跟踪。**不要使用 `git add .`**；只暂存本关已确认的 `.py`、测试与这份 `.md`。

## 8. 迁移到 P02 的下一步

在相同 Mask + Depth 数据上对比：

1. 原始观测点算术质心；
2. 下采样体素等权质心；
3. 按 counts 加权体素质心；
4. 经过合理的深度筛选、离群/背景过滤和主簇选择后的中心。

优先用有可信真值的有效样本，记录位置误差、P95、有效率、失败类型与计算时间。**真实物体中心的定义**（物体几何中心、可见表面中心、包围盒中心等）必须先确定，否则误差无法严谨比较。

## 9. 当前验收与 Git 状态

- [x] 知识地图、理论、Shape / 单位、算法拆解
- [x] 最小实验；A/B/C 三组运行前预测和实测分析
- [x] NumPy 参考实现与 Open3D 比较
- [x] 31/31 EXP031 专项测试与 95 项回归 + 106 子测试
- [x] 独立 Boss Test（96/100）
- [ ] **本人审核本笔记，尤其是“我自己的理解”和遗留问题**
- [ ] 将笔记保存到 P02 的 `experiments/` 并复测
- [ ] Git Commit
- [ ] GitHub Push，核对 local/remote SHA

**状态：`CONTINUE`，不是 `LEVEL COMPLETE`。**

### 我自己的理解（依据本次个人作答整理，提交前本人确认）

> 体素下采样是空间合并，不是离群删除。点数由非空体素决定，体素归属取决于网格起点和三维索引。聚合时 XYZ 与 RGB 跟随同一分组；点云密度变化会影响各体素内部均值，但如果对代表点等权求质心，会丢失原始点数的统计权重。要把这个知识迁移到 RGB-D 定位，必须有明确的目标中心定义和可追溯的真值评测。
