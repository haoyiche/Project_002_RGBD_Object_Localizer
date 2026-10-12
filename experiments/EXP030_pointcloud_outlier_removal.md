# EXP030｜Point Cloud Outlier Removal（半径离群点滤波）

> - 项目：Project_002_RGBD_Object_Localizer
> - 当前阶段：Level 2 / EXP030
> - 记录日期：2026-10-12
> - 状态：**技术实验及独立测试通过；笔记待本人复核，Git/GitHub 尚未验收，Learning Unit 为 CONTINUE**
> - 工作单位约定：三维 XYZ 使用 **米（m）**。

## 0. 学习目标、知识地图与通关要求

### 0.1 本关目标

- 能解释 Radius Outlier Removal（ROR）的原理、输入、输出和应用局限。
- 能推导半径 `radius`、最少邻居数 `min_neighbors` 的变化对计数与保留集合的影响。
- 能独立写出 NumPy 版滤波，识别边界、重复坐标与自身索引的区别。
- 能以 KD-Tree 完成生产版邻域查询，并用独立参考实现验算。
- 能保证 EXP028 的 XYZ 与 RGB 使用**相同的原始点索引**过滤。
- 能把过滤结果送入 EXP029 统计，区分空目标、无有效深度、全被过滤与低置信度。
- 能评估性能、排查静默 Bug，并能迁移到工业 RGB-D 视觉场景。

### 0.2 知识地图

```text
EXP026 单像素反投影
  ↓
EXP027 深度图 → 三维点云
  ↓
EXP028 目标 Mask → 带 RGB 的点云
  ↓
EXP030 Radius Outlier Removal（当前）
  ↓
EXP029 质心 / AABB / Median 统计
  ↓
目标三维定位 / 抓取或测量的可靠性判断
```

EXP029 的学习先于 EXP030，但在实际流水线中，**滤波要发生在几何统计之前**。

### 0.3 验收标准与目前证据

| 项目 | 证据 | 结论 |
|---|---|---|
| 最小理论与实验 | 五点云最小实现、Shape/公式/参数拆解 | PASS |
| 主动修改 A/B/C | 先预测后实测，全部通过 | PASS |
| 生产版 API | KD-Tree Smoke Test、参数/空输入/颜色检查 | PASS |
| 正式滤波单元测试 | 15/15 | PASS |
| 正式集成测试 | 6/6 | PASS |
| 全项目回归 | 最近一次 64/64 | PASS（最终提交前仍需重跑） |
| 性能预测与 Benchmark | F1～F4 与固定随机种子实测 | PASS |
| 独立通关题 | Q1～Q6，Q4 修改后独立验证 | PASS |
| 笔记 | 本文档已拟定，待本人审阅和本地保存 | PENDING |
| Git Commit / GitHub Push / 远程 SHA | 尚未提供本关验收证据 | PENDING |

---

# 第一部分｜原理（对齐 SOP `01_原理.md`）

## 1. 它是什么、解决什么问题、为什么需要

Radius Outlier Removal（半径离群点滤波）利用三维空间中的**局部邻域密度**筛掉孤立的点。对每个点寻找一定半径内的**其他索引点**，邻居数量低于阈值则删除该点。

它在 RGB-D 工业视觉中的作用，是在计算目标质心或 AABB 前，降低孤立深度噪声对统计量的干扰。它不是目标分割算法，也**不能仅凭局部密度判断一个点是否属于真实物体**。

## 2. 输入、输出、Shape、数据类型和单位

生产版接口：`src/geometry/pointcloud_filter.py` 的 `radius_outlier_filter(points, radius, min_neighbors)`。

| 名称 | Shape / 类型 | 说明 |
|---|---|---|
| `points` | `(N,3)`，可转换为有限 `float64` | XYZ，单位 m；允许 `(0,3)` |
| `radius` | 有限正实数；不接受 bool | 搜索半径，与 XYZ 同单位，m |
| `min_neighbors` | 正整数；不接受 bool | 阈值，单位为“其他点个数” |
| `neighbor_counts` | `(N,)` 整数 | 每个原始点的邻居数 |
| `keep_mask` | `(N,)` bool | 每个原始点是否保留 |
| `kept_indices` | `(M,)` 整数 | 按原始顺序排列的保留索引 |
| `filtered_points` | `(M,3)` float64 | 过滤后的点云，可为空 |

RGB 不是用另一套独立规则过滤，而是：

```python
filtered_colors = colors[kept_indices]
# 等价地（若 shape 一致）：colors[keep_mask]
```

## 3. 核心公式：符号、单位、运算与变化

设原始点云为 `P = {p_0, ..., p_(N-1)}`，`p_i=(x_i,y_i,z_i)`（单位 m）。

**欧氏距离：**

\[
d_{ij}=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2+(z_i-z_j)^2}\quad[\mathrm{m}]
\]

**邻居集合：**

\[
\mathcal N_i(r)=\{j\mid j\ne i,\; d_{ij}\le r\}
\]

**邻居数量与保留条件：**

\[
c_i=\lvert\mathcal N_i(r)\rvert,\qquad
k_i=\mathbf 1[c_i\ge K]
\]

其中 `r = radius`（m）、`K = min_neighbors`（整数）、`c_i`（个数）、`k_i`（布尔）。

**运算层：** 两点 XYZ 相减 → 三维欧氏距离 → 包含 `<= r` 的边界 → 按**索引**排除 `i==j` → 数量与阈值比较 → 用同一个掩码筛选 XYZ 与 RGB。

**变化层：**

- 固定点云、`K` 时，`r` 增大，`c_i` 不减，保留集合不缩小。
- 固定点云、`r` 时，`K` 增大，`c_i` 不变，保留集合不扩大。
- 自身索引 `i==j` 不计数；不同索引即使 XYZ 完全相同，也可以互为邻居。
- 一次性过滤对**原始**点云计算全部 `c_i`；不会在本次调用中反复删除。
- `<=` 是包含数学边界的契约，不保证浮点误差在所有情况下都消失。

## 4. 为什么这样设计及其局限

- `EXP030` 接收空点云并返回空结果；`EXP029` 的均值与坐标极值要求非空，所以业务层必须判断后再调用。
- 单个保留点传入 `EXP029`，数学上可以得到 centroid 等于该点、AABB extent 为 `[0,0,0]`；**这不代表真实物体尺寸为零**。
- 稀疏的真实尖角、细线、薄片可能被误删；密集噪声簇可能全部通过；半径滤波只是一种局部密度规则。
- 在坐标单位为米的模块中，`50 mm = 0.05 m`，不是 `radius=50`。

---

# 第二部分｜核心参数与代码阅读（对齐 SOP `02_核心参数.md`）

## 5. `radius`

| 维度 | 记录 |
|---|---|
| 定义 | 邻居搜索球的半径 |
| 单位 | m（必须与 XYZ 一致） |
| 类型与 Shape | 有限正实数标量；bool 禁止 |
| 来源 | 依点云采样密度、深度分辨率、目标尺寸和真值验证选定 |
| 增大 | 邻居计数不减；更易保留孤立噪声、密集伪影 |
| 减小 | 邻居计数不增；真实稀疏细节更易被误删 |
| 代码位置 | `query_ball_point` 候选半径及最终 `distances <= r` |
| 常见错误 | 单位 mm/m 混淆，边界 `<` 与 `<=` 混淆，浮点边界误差 |

我自己的理解：半径不是“越大越保险”，而是决定哪些三维点被视为同一局部邻域。它必须与相机分辨率、目标结构尺寸和 XYZ 单位配合。

## 6. `min_neighbors`

| 维度 | 记录 |
|---|---|
| 定义 | 保留一个点所需的其他邻居数下限 |
| 单位 | 点数，无物理量纲 |
| 类型与 Shape | 正整数标量；bool 禁止 |
| 来源 | 噪声密度与真实结构采样密度的验证实验 |
| 增大 | 保留集合不扩大；可能全部过滤 |
| 减小 | 保留集合不缩小；可能保留更多离群点 |
| 代码位置 | `counts >= k` |
| 常见错误 | 错把自身算作邻居；`K=0`；把“初始邻域过滤”误当成迭代过滤 |

我自己的理解：`min_neighbors` 衡量局部的采样支持程度，而非真实物体存在的概率。过高会误删细小的真实结构。

## 7. 最小版 NumPy 广播与 KD-Tree 核心代码拆解

### 7.1 最小版：`src/experiments/exp030_radius_minimal.py`

```python
differences = points[:, None, :] - points[None, :, :]
# (N,1,3) - (1,N,3) -> (N,N,3)
distance_matrix = np.linalg.norm(differences, axis=2)  # (N,N), m
within_radius = distance_matrix <= radius              # (N,N), bool
np.fill_diagonal(within_radius, False)                  # 排除 i == j
neighbor_counts = within_radius.sum(axis=1)             # (N,)
keep_mask = neighbor_counts >= min_neighbors            # (N,)
filtered_points = points[keep_mask]                      # (M,3)
```

特点：教学逻辑直观；主要数组内存规模为 `O(N²)`。

### 7.2 生产版：`src/geometry/pointcloud_filter.py`

```python
tree = cKDTree(xyz)
query_radius = np.nextafter(r, np.inf)

for i in range(n):
    candidates = tree.query_ball_point(
        xyz[i], query_radius, p=2, eps=0
    )
    other_indices = np.asarray(candidates, dtype=np.intp)
    other_indices = other_indices[other_indices != i]
    distances = np.linalg.norm(xyz[other_indices] - xyz[i], axis=1)
    counts[i] = np.count_nonzero(distances <= r)

keep_mask = counts >= k
kept_indices = np.flatnonzero(keep_mask)
filtered_points = xyz[keep_mask].copy()
```

关键含义：空间索引用于找候选点；**排除自身看索引，不是 `distance > 0`**；距离复算维持 `<= r` 契约；保留索引顺序不依赖 KD-Tree 候选输出顺序。

注意：候选查询使用 `nextafter`，但所有浮点极端边界是否完全一致仍需针对具体平台验证。现有证据是定向边界测试与随机对照通过，而非数学上的全情形证明。

---

# 第三部分｜实验记录（对齐 SOP `03_实验记录.md`）

## 8. 实验环境与复现方法

- OS：Windows / PowerShell（用户本机日志）。
- Python 路径：`D:\conda_envs\ai_3d\python.exe`。
- 项目根目录：`D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer`。
- 核心依赖：NumPy、SciPy `cKDTree`。
- Python / NumPy / SciPy 实际版本：**待从本机命令补录，不能猜测**。
- 计时方法：`time.perf_counter`，预热后每种方法 5 次，取中位数；随机种子 `20261012`。

复现命令（从仓库根目录执行）：

```powershell
$python = 'D:\conda_envs\ai_3d\python.exe'
& $python -m src.experiments.exp030_active_modification_a
& $python -m src.experiments.exp030_active_modification_b
& $python -m src.experiments.exp030_active_modification_c
& $python -m src.experiments.exp030_production_smoke
& $python -m src.experiments.exp030_radius_benchmark
& $python -m src.experiments.exp030_rgbd_filter_integration
& $python -m unittest discover -s tests -p 'test_*.py' -v
& $python -c 'import sys,numpy,scipy; print(sys.version); print(numpy.__version__); print(scipy.__version__)'
```

> 此处是**复现清单**，不是宣称这些命令在本文档生成之后已重新运行。

## 9. 最小五点实验

输入：

```text
A=(0,0,1), B=(0.1,0,1), C=(0,0.1,1), D=(0.1,0.1,1), E=(1,1,5)
radius=0.11, min_neighbors=2
```

预测与实测一致：

```text
neighbor_counts=[2,2,2,2,0]
keep_mask=[T,T,T,T,F]
filtered_points.shape=(4,3)
```

过滤前 centroid `[0.24,0.24,1.8]`，AABB extent `[1,1,4]`；过滤后 centroid `[0.05,0.05,1.0]`，AABB extent `[0.1,0.1,0]`。

注意：这里 E 的 Y=1，与后面的 RGB-D 集成例子 E 的 Y=0.5 **不同**，因此两次实验的过滤前 centroid/AABB 也不同。

## 10. 三次主动修改：预测 → 实测 → 原因

| 修改 | 为什么改 | 预测 | 实测 | 原因 |
|---|---|---|---|---|
| A：`radius 0.11 → 0.15`，`K=2` | 检查半径影响 | 4 个密集点邻居数由 2 变 3，保留集合不变 | PASS | 半径扩大增加对角线邻居，但 4 点本来已过阈值 |
| B：`radius=0.15`，`K=2/3/4` | 检查阈值影响 | K=2/3 留 4 点；K=4 全部删除 | PASS | 密集点的最大邻居数只有 3 |
| C：边界 `<`/`<=`、是否排除自身 | 检查静默 Bug | 正确 `[2,1,1,0]`；错误边界 `[0,0,0,0]`；自计数 `[3,2,2,1]` | PASS | 边界与索引语义决定精确计数 |

C 的四点独立场景为 `A=(0,0,1), B=(1,0,1), C=(0,1,1), D=(3,3,1)`，`radius=1.0, K=2`。正确情况下只保留 A；使用 `<` 会清空；计入自身会保留 A/B/C。

## 11. 生产版 API 与单元测试

- `src/geometry/pointcloud_filter.py`：`radius_outlier_filter`，KD-Tree 邻域候选与索引排除。
- `src/experiments/exp030_production_smoke.py`：不同半径与阈值、边界、空输入、重复坐标、非法值、RGB 对齐。
- `tests/test_pointcloud_filter.py`：**15 项 PASS**，包括独立显式循环 oracle、固定种子随机点云、单调性、非迭代与不修改输入。
- 全项目原有 43 项 + EXP030 的 15 项 + 新增集成 6 项 = **64/64 PASS**（按提交给导师的最近一次终端日志）。

## 12. 性能预测 F1～F4 与实测

NumPy 最小实现主要数组的理论内存（十进制 MB）：

| N | `(N,N,3)` float64 差向量 | `(N,N)` float64 距离 | `(N,N)` bool | 合计 |
|---:|---:|---:|---:|---:|
| 1,000 | 24 MB | 8 MB | 1 MB | 33 MB |
| 5,000 | 600 MB | 200 MB | 25 MB | 825 MB |
| 100,000 | 240 GB | 80 GB | 10 GB | 330 GB |

这些是**主要数组的理论大小相加**，不等于进程真实峰值 RSS；真实内存与数组生命周期、中间临时数组和分配器有关。

固定种子 `20261012`、每方法预热、5 次计时取中位数后的本机实测：

| 场景 | 平均邻居数 | NumPy 中位数 | KD-Tree 中位数 | NumPy/KD-Tree |
|---|---:|---:|---:|---:|
| 稀疏 N=200 | 3.11 | 2.710 ms | 6.593 ms | 0.41x |
| 稀疏 N=300 | 3.57 | 4.142 ms | 8.738 ms | 0.47x |
| 稀疏 N=500 | 3.47 | 9.754 ms | 12.789 ms | 0.76x |
| 稀疏 N=1,000 | 3.75 | 37.683 ms | 26.093 ms | 1.44x |
| 密集 N=300 | 299.00 | 4.094 ms | 14.273 ms | 0.29x |

分析：小 N 时 NumPy 向量化的常数开销更小；稀疏 N=1,000 时 KD-Tree 开始占优；密集 N=300 时，树查询仍需处理大量候选点且有逐点 Python 循环。**当前测得的拐点仅对本机、本实现、本参数有效**。

## 13. RGB-D 三模块集成 G1～G5

测试数据：深度图 `[[1,1,0],[1,1,5]]`，Mask `[[T,T,F],[T,T,T]]`；`fx=fy=10, cx=cy=0`。

相机反投影：`X=(u-cx)*Z/fx; Y=(v-cy)*Z/fy`。

五个有效 XYZ（行优先）：

```text
[0.0,0.0,1.0]  红
[0.1,0.0,1.0]  绿
[0.0,0.1,1.0]  蓝
[0.1,0.1,1.0]  黄
[1.0,0.5,5.0]  青（孤立异常点）
```

`radius=0.11, min_neighbors=2`，实测：

```text
neighbor_counts=[2,2,2,2,0]
keep_mask=[T,T,T,T,F]
kept_indices=[0,1,2,3]
```

| 统计 | 滤波前 | 滤波后 |
|---|---|---|
| Point Count | 5 | 4 |
| Centroid（m） | `[0.24,0.14,1.8]` | `[0.05,0.05,1.0]` |
| AABB Extent（m） | `[1.0,0.5,4.0]` | `[0.1,0.1,0.0]` |

**非连续索引的专项回归：** 异常点放在点云中间后，正确 `kept_indices=[0,1,3,4]`，RGB 应为**红、绿、黄、青**。此测试可以识别错误的 `colors[:M]` 操作。

### 13.1 实验原型中的业务状态（非生产级安全认证）

| 状态 | 含义 | 是否调用 EXP029 |
|---|---|---|
| `EMPTY_MASK` | 目标 Mask 无选中像素 | 否 |
| `NO_VALID_DEPTH` | Mask 非空，目标有效深度点数为 0 | 否 |
| `ALL_FILTERED` | 初始点云非空，但滤波后为空 | 否 |
| `LOW_CONFIDENCE` | 演示阈值下仅保留 1 点 | 会计算退化统计，但不可视作可靠定位 |
| `SUCCESS` | 演示规则下剩余点数至少为 2 | 会统计；不等于工业可靠性已被证明 |

这些状态目前由实验原型 `src/experiments/exp030_rgbd_filter_integration.py` 管理，不应宣称整个工业定位业务模块已正式投产。

## 14. 独立通关 Q1～Q6

| 验收 | 内容 | 状态 |
|---|---|---|
| Q1 | 解释滤波原理、参数单调性、误删与性能 | PASS |
| Q2 | 六点数据，重复坐标、非连续保留索引、质心与 AABB | PASS |
| Q3 | `distances > 0` 错排除重复坐标邻居的静默 Bug | PASS |
| Q4 | 只用 NumPy 双重循环独立实现及输入契约 | PASS（修正后已实测） |
| Q5 | 排序损坏原始顺序、`colors[:M]` 错位、`[-0:]` 陷阱 | PASS（已更正颜色映射细节） |
| Q6 | 单位错误、稀疏真实结构、密集伪影及任务级验收 | PASS |

Q2 的六点数据正确 `neighbor_counts=[3,1,2,1,2,3]`，`K=3` 保留 `[0,5]`，两点 XYZ 均为 `[0,0,1]`，所以统计结果退化为单一位置与零 extent。

Q4 本机修复实测输出：`Q2 Independent Implementation: PASS`、`Invalid Radius Contract: PASS`、`Empty Input Contract: PASS`、`Duplicate Coordinate Semantics: PASS`、`All Filtered Contract: PASS`、`EXP030 Independent Verification: PASS`。

---

# 第四部分｜常见问题、排错记录与工程迁移（对齐 SOP `04_常见问题与误区.md`）

## 15. 常见问题及验证

### 15.1 将 `<= radius` 写成 `< radius`

- 原先误解：边界等号不重要。
- 正确理解：恰好等于半径的点必须计入；精确整数距离的边界样例可证明这个差异。
- 如何验证：四点边界测试正确 `[2,1,1,0]`，错误 `<` 为 `[0,0,0,0]`。

### 15.2 忘记排除自身索引

- 原先误解：每个点到自身的距离是 0，也应算邻居。
- 正确理解：自身索引不可算，但不同索引的重复 XYZ 应互为邻居。
- 如何验证：无排除的 `[3,2,2,1]`；正确 `[2,1,1,0]`。

### 15.3 用 `distances > 0` 排除自身

- 错误根因：把“坐标重合”误当成“索引相同”。
- 如何验证：六点样例正确 `[3,1,2,1,2,3]`，错误 `[2,1,2,1,2,2]`。
- 补充：当 `min_neighbors=2` 时，两种计数可能产生**相同的** `keep_mask`，因此要独立断言 `neighbor_counts`。

### 15.4 对保留点重新排序，或使用 `colors[:M]`

- 错误根因：混淆按邻居数排序、阈值筛选、原始顺序和颜色索引。
- 正确修复：`kept_indices=np.flatnonzero(counts>=K)`，XYZ/RGB 同时用这个索引。
- 测试：非连续索引 `[0,1,3,4]`；若 `M=0`，注意切片 `[-0:]` 等于 `[0:]`，不是空切片。

### 15.5 将通过 Shape 检查当成正确性

- 原先误解：`neighbor_counts.shape==(N,)` 即代表正确。
- 正确理解：错误的邻居数组可以拥有完全相同的 Shape。
- 如何验证：和**独立手算数值**比较，而非只检查 Shape 或 `filtered_points` 数量。

### 15.6 空输入、空输出、单点退化

- `EXP030` 可以输入/输出 `(0,3)`；`EXP029` 不接受空点云。
- 若保留 1 点，`EXP029` 可以计算，但位置与尺寸可靠性不足。
- 必须在业务层记录 `EMPTY_MASK/NO_VALID_DEPTH/ALL_FILTERED/LOW_CONFIDENCE/SUCCESS`，不应把 `ALL_FILTERED` 解释为目标肯定不存在。

### 15.7 独立实现的参数漏洞与文件缺失

**问题 A：** `radius=True` 通过了天真的 `np.isfinite(radius)` 和 `radius > 0` 检查；`radius="0.1"` 可能产生非预期 `TypeError`。

**根因：** Python `bool` 可作为整数参与数值运算；没有先限定实数标量类型。

**修复：** 用 `numbers.Real` 验证标量类型、拒绝 `bool/np.bool_`，再验证有限且大于零；独立实测已通过非法参数检查。

**问题 B：** 曾遇到 `No such file or directory: 'src\\experiments\\exp030_independent_implementation.py'`。

**根因：** 在聊天里写了代码，却没有保存为本地 `.py` 文件。

**修复与验证：** 在正确项目目录中建立文件，再 `py_compile`、导入并执行独立断言，输出全部 PASS。

## 16. 工业视觉迁移：三类失效模式

| 场景 | 风险 | 对策 | 代价或限制 |
|---|---|---|---|
| XYZ 为米，`radius=50` 被误认成 50 mm | 搜索半径实际为 50 m，几乎所有点互为邻居 | 统一单位并转换：`radius_m=radius_mm/1000`；检查合理范围 | 需要维护单位契约与设备标定 |
| 细长金属结构采样稀疏，K 太大 | 删除真实结构，AABB 虽缩小但尺寸可能更错 | 调参；按结构分区；视情况换统计滤波 | 会保留更多噪声，对参数敏感 |
| 20 个点构成密集伪影 | 伪影簇可通过 ROR 甚至 DBSCAN | 结合 ROI、物体先验、几何一致性、多帧信息 | 需要额外计算、标定和验证 |

**工业验收必须基于真值和任务误差，而不仅是 AABB 变小。** 可在标定标准件上分别测有滤波/无滤波的质心误差、尺寸误差、多帧稳定性、伪影误检率和误抓取风险，并事先明确可接受阈值。

**拒绝抓取条件示例（阈值须通过实际标定验证，不能照搬）：** 深度有效点极少、过滤比例异常、剩余几何退化、尺寸/位置违反安全工作区约束、多帧估计不一致、无法区分目标与伪影簇。此时应重新采集、变换视角或交由人工处理。

## 17. 当前仍不稳定 / Known Issues

1. 没有测量真实进程峰值 RSS，Benchmark 报告的是计时中位数与理论主要数组内存。
2. 生产版 KD-Tree 使用逐点 Python 循环；大规模或极密邻域还有性能优化空间。
3. 半径恰好落在浮点边界时，仍需更系统的数值鲁棒性验证。
4. 重复坐标可能人为增加局部密度；在传感器重复采样场景需要评估去重策略。
5. RGB-D 集成中的置信度规则仅是教学原型，不等于已认证的工业安全逻辑。
6. 还未在真实标准件数据上建立误差真值、噪声模型、误删率和拒绝抓取指标。
7. 实际 Python / NumPy / SciPy 版本及 Git 提交元信息待补录。

## 18. 本人的复述与复盘（请复核后保留或改写）

> 我理解 Radius Filter 是用局部空间密度决定是否保留点，不是判断目标真实与否。`radius` 用米控制邻域，`min_neighbors` 用点数控制阈值；邻居必须排除自身**索引**，不能排除所有零距离点。即使三维统计函数可以算出单点的 centroid 和零 extent，也不能据此认为物体位置与尺寸已经可靠。只有固定真值、比较定位误差，并保持 RGB/XYZ 原始索引一致，才能在工程上评价滤波是否有帮助。

**本次最有价值的 Debug 经验：** 一份程序可以无异常地运行、Shape 全部正确，但邻居数量或颜色对应关系仍然是错的；要使用手算独立参考和非连续索引样例找出静默 Bug。

---

# 第五部分｜文件清单、Git 与最后验收

## 19. 本关预计新增的文件（提交前以 `git status` 为准）

```text
src/geometry/pointcloud_filter.py
src/experiments/exp030_radius_minimal.py
src/experiments/exp030_active_modification_a.py
src/experiments/exp030_active_modification_b.py
src/experiments/exp030_active_modification_c.py
src/experiments/exp030_production_smoke.py
src/experiments/exp030_radius_benchmark.py
src/experiments/exp030_rgbd_filter_integration.py
src/experiments/exp030_independent_implementation.py
tests/test_pointcloud_filter.py
tests/test_rgbd_filter_integration.py
experiments/EXP030_pointcloud_outlier_removal.md
```

目录或个别文件与当前工作区不同时，应以实际存在文件和 `git status` 为准，不得靠猜测强行提交。

## 20. 最终复现及 Git 验收清单

- [x] 概念、Shape、单位、公式与参数变化解释。
- [x] 最小实验与三次主动修改；预测与实测对照。
- [x] 生产版 KD-Tree、独立 NumPy 实现及 Debug。
- [x] EXP030 滤波 15 项单元测试。
- [x] EXP030 集成 6 项测试、全项目最近一次 64/64。
- [x] 性能测试及局限性解释。
- [x] Q1～Q6 独立通关能力验收。
- [ ] 在本地保存本文档，逐项复核，补录依赖版本。
- [ ] 提交前重新运行完整 64 项测试。
- [ ] 检查 `git status`、`git diff` 与 `git diff --cached`。
- [ ] **只暂存本关相关文件**，避免根目录其他 `ScreenCamera_*`、`ScreenCapture_*` 等无关文件。
- [ ] 提交有意义的 Git Commit，记录完整 SHA。
- [ ] GitHub Push 后核对本地 `HEAD` 与远程 `origin/master` 的 SHA。
- [ ] 根据所有证据，将 Learning Unit 状态由 `CONTINUE` 更新为 `LEVEL COMPLETE`。

建议 Commit message：

```text
Complete EXP030 radius outlier filtering and RGB-D integration
```

## 21. 下一步

1. 审阅本文档，核对所有数值与本机实际脚本，补录依赖版本及个人复述。
2. 本地保存到 `experiments/EXP030_pointcloud_outlier_removal.md`。
3. 最后一次全项目回归，审查仅本关相关 Git 变更。
4. Commit、Push，核对远程 SHA 后再宣布 EXP030 COMPLETE。

---

**Exit Protocol（当前）**：本次已完成独立实现 Q4 最终实测及笔记草稿；核心原理与实践能力证据齐备；仍需笔记本人复核、最终全量回归、Git 和 GitHub 验收。**Level 状态：CONTINUE。**

## 最终本机验收记录（2026-10-12）

- Python：3.11.17
- NumPy：2.2.6
- SciPy：1.17.1
- EXP030 单元测试：15/15 PASS
- EXP030 集成测试：6/6 PASS
- 全项目最终回归：64/64 PASS，0.071 秒
- 独立通关 Q1～Q6：PASS
- Git Commit / Push：待验证
- 未解决问题：极端浮点边界、真实点云性能、工业场景置信度标定
