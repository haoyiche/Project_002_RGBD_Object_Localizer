# EXP029｜Target Point Cloud → 3D Center & Bounding Box

> **AI Lab / Level 2 / Project_002_RGBD_Object_Localizer**\
> 学习主题：目标点云几何统计（Centroid、AABB、Median Center）\
> 当前状态：**理论、预测、主动修改、单元测试、集成回归、独立知识验收完成；笔记待本地复核，Git commit / GitHub push 尚待验收**\
> 最后整理：2026-10-10\
> 依据：`AI_Lab_SOP.md` 的“目标 → 知识地图 → 最小理论 → 最小实验 → 公式与代码拆解 → 主动修改 → 先预测后验证 → 笔记 → 通关测试 → Git/GitHub”闭环。

---

## 01｜学习目标与验收标准

### 1.1 本关目标

本关要解决的不是“能否调用 `np.mean()`”，而是**把 EXP028 从 RGB-D 与目标 Mask 提取的三维点云，转换为可验证的目标位置与空间范围估计**。完成后应能：

1. 区分点云质心、AABB 中心与各轴中位数中心，并解释各自的局限。
2. 从 `[N, 3]` 的三维点数组，计算六项统计量。
3. 解释 `axis=0`、Shape、dtype、单位与行优先采样顺序。
4. 对输入执行 Shape、空输入、数值转换及 NaN/Inf 检查。
5. 对增加正常点、添加异常点、刚体平移作出预测并用实验验证。
6. 连接 EXP028 的 Mask → Point Cloud 与 EXP029 的 Point Cloud → Statistics。
7. 能独立手算、实现、发现静默 Bug，并迁移到旋转+平移的坐标系变换。
8. 完成笔记、可重复测试、Git commit、GitHub push 后才正式标记 COMPLETE。

### 1.2 最终工程产物

- `src/geometry/pointcloud_statistics.py`：生产版六项几何统计 API。
- `src/experiments/exp029_center_bbox_minimal.py`：最小实验。
- `src/experiments/exp029_active_modification_a.py`：增加正常点。
- `src/experiments/exp029_active_modification_b.py`：添加远离表面的深度异常点。
- `src/experiments/exp029_active_modification_c.py`：整体平移。
- `src/experiments/exp029_production_smoke.py`：生产 API 的初步契约实验。
- `src/experiments/exp029_rgbd_to_stats_integration.py`：EXP028→EXP029 端到端实验。
- `tests/test_pointcloud_statistics.py`：10 项生产统计函数单元测试。
- `tests/test_rgbd_stats_integration.py`：3 项集成测试。
- `experiments/EXP029_pointcloud_statistics.md`：本笔记，需保存到本地项目。

以上文件名及测试情况基于本轮终端日志；本笔记不是对用户本地文件系统的实时扫描。Git 状态也必须以本地命令实际输出为准。

### 1.3 通关定义

**运行通过 ≠ 学习通关**。必须同时证明：理论理解、独立预测、主动修改、代码解释、异常处理、独立计算/实现/调试/迁移、实验记录、完整笔记，以及 Git/GitHub 闭环。

---

## 02｜知识地图：本关在 Project_002 中的位置

```text
EXP026：单像素 + 深度 + 相机内参 → 相机坐标系 XYZ
    ↓
EXP027：完整深度图 → 三维点云 [N,3]
    ↓
EXP028：RGB-D + Target Mask → 目标 points [N,3]、colors [N,3]
    ↓
EXP029：目标 points [N,3] → Centroid、AABB、Median Center
    ↓
后续：对象定位状态、坐标系转换、鲁棒点云处理、OBB/姿态、工业视觉应用
```

**上游依赖**：正确的深度单位、相机内参、目标掩码、像素与 RGB 对齐。\
**当前模块边界**：只分析传入的有限非空 XYZ 点，**不负责分割、投影、滤噪或判定物体是否存在**。\
**下游用途**：目标位置估计、包围盒展示、尺寸粗估、不同视角下的几何比较。

---

## 03｜核心问题：三个“中心”不是一回事

给定 `points` 为 `N×3` 个三维坐标，每一行 `P_i=(X_i,Y_i,Z_i)`：

### 3.1 Centroid，采样点质心

\[
C=\frac1N\sum_{i=1}^{N}P_i
\]

- 使用所有采样点；大量点集中在某处，会把均值拉向该处。
- 若观测面不完整或点密度不均匀，**点云质心未必是实体物体的质量中心、体积中心或真实几何中心**。
- 单个异常点会影响均值。

### 3.2 AABB，轴对齐包围盒

\[
L=(\min X_i,\min Y_i,\min Z_i),\qquad
U=(\max X_i,\max Y_i,\max Z_i)
\]

\[
B=\frac{L+U}{2},\qquad E=U-L
\]

- `min_bound=L`、`max_bound=U`、`bbox_center=B`、`bbox_extent=E`。
- 各轴独立取极值；极值坐标不必来自同一个真实点。
- 只要极值没改变，中间增加多少个点都可能不影响 AABB。
- 对离群值特别敏感；与坐标轴对齐，所以旋转可能改变 Extent。

### 3.3 Median Center，各轴中位数中心

\[
M=(\operatorname{median}(X),\operatorname{median}(Y),\operatorname{median}(Z))
\]

- 三个坐标轴分别排序并取中位值。
- 在少量极端值出现时，通常比均值更稳定，但**不代表总能定位真实物体中心，也不是完整的异常值去除算法**。
- `median_center` 未必恰好等于任何一个输入三维点。

### 3.4 概念比较

| 量 | 依赖哪些点 | 单个远端离群点 | 对坐标轴方向敏感 | 等于实体真实中心？ |
|---|---|---|---|---|
| Centroid | 所有点的均值 | 会发生偏移 | 作为三维向量按刚体变换 | 不保证 |
| AABB Center | 每轴最小/最大值 | 通常非常敏感 | 是 | 不保证 |
| Median Center | 每轴有序值的中位位置 | 在少量离群点下通常稳定 | 各轴统计依赖坐标系 | 不保证 |

---

## 04｜输入、输出、单位、Shape 与 dtype

### 4.1 输入契约

```python
points: numpy.ndarray 或可转为数值的 Python 嵌套序列
shape:  (N, 3), N >= 1
dtype:  内部统一 float64
value:  每个坐标必须 finite，不接受 NaN、+Inf、-Inf
unit:   由上游决定；本项目 EXP028 使用米（m）
order:  每一行是一个 XYZ 点；N 行为 N 个观测点
```

### 4.2 输出契约

```python
{
    "centroid":      np.ndarray,  # (3,), float64, m
    "min_bound":     np.ndarray,  # (3,), float64, m
    "max_bound":     np.ndarray,  # (3,), float64, m
    "bbox_center":   np.ndarray,  # (3,), float64, m
    "bbox_extent":   np.ndarray,  # (3,), float64, m
    "median_center": np.ndarray,  # (3,), float64, m
}
```

**单位注意**：上述单位成立的前提是输入使用米。若输入毫米，输出自然也是毫米；本模块**不会自动把毫米转成米**。`bbox_extent` 是观测点云的轴向范围，不是经过物体补全后的真实三维尺寸。

### 4.3 `axis=0` 的物理含义

```python
points = np.array([
    [0., 0., 1.],
    [2., 0., 1.],
    [0., 2., 1.],
])  # shape = (3, 3)

np.mean(points, axis=0)  # [2/3, 2/3, 1]，对列求均值，即 XYZ 三坐标
np.mean(points, axis=1)  # [1/3, 1, 1]，对每个点内部的 XYZ 求均值，语义错误
```

`N==3` 恰好时两个输出都是 `(3,)`，所以 Shape 断言可能放过错误。要同时用**物理语义 + 独立预期值**检测。

---

## 05｜六项统计量的最小实现与代码拆解

**生产入口**：`src/geometry/pointcloud_statistics.py` 中的 `compute_pointcloud_statistics(points)`。

```python
import numpy as np


def compute_pointcloud_statistics(points):
    try:
        xyz = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("points must contain numeric XYZ coordinates") from exc

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if xyz.shape[0] == 0:
        raise ValueError("points must not be empty")

    if not np.isfinite(xyz).all():
        raise ValueError("points must contain only finite values")

    centroid = np.mean(xyz, axis=0)
    min_bound = np.min(xyz, axis=0)
    max_bound = np.max(xyz, axis=0)
    bbox_center = (min_bound + max_bound) / 2.0
    bbox_extent = max_bound - min_bound
    median_center = np.median(xyz, axis=0)

    return {
        "centroid": centroid,
        "min_bound": min_bound,
        "max_bound": max_bound,
        "bbox_center": bbox_center,
        "bbox_extent": bbox_extent,
        "median_center": median_center,
    }
```

关键代码逐行理解：

| 语句 | 输入/输出 | 为什么需要 | 典型错法 |
|---|---|---|---|
| `np.asarray(..., dtype=np.float64)` | 嵌套序列→数值矩阵 | 统一数值精度，接受整数列表 | 让字符串或 object 数组进入运算 |
| `xyz.ndim != 2 or xyz.shape[1] != 3` | Shape 校验 | 每行必须是 XYZ | 接受 `(N,2)` 并臆造 Z |
| `xyz.shape[0] == 0` | N=0 拒绝 | 空数组无合法 AABB | 返回 `[0,0,0]` 伪造目标 |
| `np.isfinite(xyz).all()` | 检查全体元素 | 拦截 NaN、Inf | 静默传播 NaN |
| `np.mean(..., axis=0)` | `[N,3]→[3]` | 点云算术均值 | 写成 `axis=1` |
| `np.min/np.max(..., axis=0)` | `[N,3]→[3]` | XYZ 各轴极值 | 当作单个真实边界点 |
| `(min+max)/2` | `[3]` | 包围盒几何中心 | 误用 `mean(points)` |
| `max-min` | `[3]` | XYZ 轴向范围 | 错写 `max+min` |
| `np.median(..., axis=0)` | `[N,3]→[3]` | 各轴鲁棒中心 | 误认为离群点已经被删除 |

**性能直觉**：读取全体 N 个点求均值、极值需要遍历数据；Median 往往需要排序或选择相关计算，具体性能取决于 NumPy 实现。对于当前实验规模，优先保证正确性与可测试性。

---

## 06｜最小实验：三个点就能让 Centroid 与 BBox Center 不同

输入：

```text
P1 = (0,0,1)
P2 = (2,0,1)
P3 = (0,2,1)
```

先预测，再运行 `src/experiments/exp029_center_bbox_minimal.py`。

| 量 | 独立手算结果 |
|---|---|
| `centroid` | `[2/3, 2/3, 1]` |
| `min_bound` | `[0,0,1]` |
| `max_bound` | `[2,2,1]` |
| `bbox_center` | `[1,1,1]` |
| `bbox_extent` | `[2,2,0]` |
| `median_center` | `[0,0,1]` |

解释：三点构成平面上的直角三角形，点的均值不等于覆盖它的轴对齐矩形的中心。Z Extent 为 0 只说明**当前可见点集**在 Z 方向上没有范围，不代表真实物体必然零厚度。

最小实验预测、实测及语法检查已在本次学习中通过；若在新机器上复现，必须重新运行而不是引用旧日志当作本次证据。

---

## 07｜主动修改 A：增加一个正常点，极值不变

### 7.1 修改原因

验证“点数量变多，Centroid 与 AABB 是否一定同时变化”。

原始三点，再加入 `P4=(2,2,1)`：

```text
原 centroid    = [2/3, 2/3, 1]
新 centroid    = [1, 1, 1]
新 bbox_center = [1, 1, 1]
新 bbox_extent = [2, 2, 0]
```

### 7.2 预测与实测

- 预测：第四个点使采样分布更对称，质心变为 `[1,1,1]`。
- 预测：未突破原有各轴 min/max，AABB Center 和 Extent 不变。
- 实测：`exp029_active_modification_a.py` PASS；预测题 A1～A5 5/5 PASS。
- 额外验证：`np.mean(points, axis=1)` 对四点得到 `[1/3,1,1,5/3]`，与三维质心毫无关系。

### 7.3 结论

AABB 对点云**数量与内部密度不敏感**，只关心每轴范围；Centroid 关心所有点的位置。

---

## 08｜主动修改 B：一个深度异常点怎样拉偏中心

### 8.1 输入与修改

原始四个角点：

```text
[0,0,1], [2,0,1], [0,2,1], [2,2,1]
```

修改：添加 `outlier=[1,1,10]`，点数由 4 变为 5。

### 8.2 先预测后运行的参考结果

| 量 | 原始四点 | 加入异常点后 | 变化 |
|---|---|---|---|
| Centroid | `[1,1,1]` | `[1,1,2.8]` | `[0,0,1.8]` |
| AABB Min | `[0,0,1]` | `[0,0,1]` | 不变 |
| AABB Max | `[2,2,1]` | `[2,2,10]` | Z +9 |
| AABB Center | `[1,1,1]` | `[1,1,5.5]` | `[0,0,4.5]` |
| AABB Extent | `[2,2,0]` | `[2,2,9]` | `[0,0,9]` |
| Median Center | `[1,1,1]` | `[1,1,1]` | 不变（仅本例） |

### 8.3 结果与解释

- `exp029_active_modification_b.py` 实测 PASS，预测 B1～B5 5/5 PASS。
- 单个 Z=10 的异常点把 AABB 最大 Z 拉至 10，而均值 Z 只移动至 2.8。
- 本例中 Median Z 仍为 1，但不意味着 Median 永远等于真实深度。
- **工程含义**：测量噪声、背景泄漏、错误 Mask 和无效深度处理都会影响目标尺寸估计。先分清“统计模块正确”与“上游输入质量可靠”。

---

## 09｜主动修改 C：纯平移与不变量

### 9.1 输入

```python
points = np.array([
    [0., 0., 1.], [2., 0., 1.],
    [0., 2., 1.], [2., 2., 1.],
])
t = np.array([3., -2., 0.5])
modified = points + t
```

### 9.2 预测的全部新点

```text
[3,-2,1.5]
[5,-2,1.5]
[3, 0,1.5]
[5, 0,1.5]
```

### 9.3 统计量变化

```text
old centroid     = [1,1,1]
new centroid     = [4,-1,1.5]
new min_bound    = [3,-2,1.5]
new max_bound    = [5, 0,1.5]
new bbox_center  = [4,-1,1.5]
old/new extent   = [2,2,0]
```

推导：

\[
\mathrm{mean}(P+t)=\mathrm{mean}(P)+t
\]
\[
\min(P+t)=\min(P)+t,\quad \max(P+t)=\max(P)+t
\]
\[
E'=\bigl(U+t\bigr)-\bigl(L+t\bigr)=E
\]
\[
\|(P_i+t)-(P_j+t)\|=\|P_i-P_j\|
\]

### 9.4 结果

- 预测 C1～C5 5/5 PASS。
- `exp029_active_modification_c.py` 实测 PASS、语法检查 PASS。
- 完整点对距离矩阵比较 PASS。
- **结论只直接适用于纯平移；旋转可能改变轴对齐包围盒的 Extent。**

---

## 10｜API Contract：空点、单点、错误 Shape、非有限值

| 输入 | 是否接受 | 理由/结果 |
|---|---|---|
| `(N,3)`，N≥1，全部 finite | 是 | 正常计算 |
| Python 整数嵌套列表 | 是 | 转为 `float64` |
| `(0,3)` | 否，`ValueError` | 无有效三维点，无法定义统计包围盒 |
| `(4,2)` 或其他错误 Shape | 否，`ValueError` | 每点必须有 XYZ |
| 含 NaN | 否，`ValueError` | 避免污染结果 |
| 含 `+Inf/-Inf` | 否，`ValueError` | 极值/均值失去物理意义 |
| 无法数值转换 | 否，`ValueError` | 统一调用者可理解的异常契约 |
| `(1,3)` 单个有限点 | 是 | 各中心/边界等于该点；`extent=[0,0,0]` |

**为什么空点与单点不同？**

- 空点：`np.mean` 可能警告并返回 NaN，`np.min/max` 会因零长度报错。业务上可能是未检测到目标、深度缺失或提取失败，不应制造一个 `[0,0,0]` 的伪位置。
- 单点：至少有一个有效三维观测；位置统计合法，但无法推断物体的空间扩展。零 Extent 是“本次观察退化”，**不是物体真实长宽高为零**。

**工程加强项**：`np.asarray(..., dtype=np.float64)` 的异常捕获建议同时包含 `OverflowError`；超大 Python 整数不一定抛 `TypeError/ValueError`。

契约预测 D1～D5 已独立回答，5/5 PASS。

---

## 11｜生产版 API 与 Smoke Test 验证

接口：

```python
from src.geometry.pointcloud_statistics import compute_pointcloud_statistics

stats = compute_pointcloud_statistics(points)
```

`exp029_production_smoke.py` 在本次学习中的实际日志包含：

```text
Geometry: PASS
Single point: PASS
empty : correctly rejected
wrong_shape : correctly rejected
nan : correctly rejected
inf : correctly rejected
nonnumeric : correctly rejected
API Contract: PASS
EXP029 Production Smoke Test: PASS
EXP029 Production API Verification: PASS
```

Smoke Test 不能替代正式单元测试；单次输出正确也不代表所有边界情形都被覆盖。

---

## 12｜正式单元测试：10 项、独立预期与不变量

文件：`tests/test_pointcloud_statistics.py`。

| 测试 | 重点 |
|---|---|
| `test_independent_geometry_reference` | 三点数据的六项手算参考及 Shape/dtype |
| `test_single_point` | 单点退化契约 |
| `test_accept_integer_list` | 整数列表转换为 float64 |
| `test_independent_axis_extrema` | 各轴 min/max 来自不同点仍正确 |
| `test_outlier_sensitivity` | 远端异常点影响三种中心的方式 |
| `test_translation_invariance` | 中心/边界随 t 平移，Extent 不变 |
| `test_reject_empty_cloud` | 空点云拒绝 |
| `test_reject_wrong_shapes` | 错误维度与列数拒绝 |
| `test_reject_nonfinite_coordinates` | NaN、正负 Inf 拒绝 |
| `test_reject_nonnumeric_and_preserve_input` | 无效数值与输入不被修改 |

**实测**：EXP029 专项 10/10 PASS；此前 EXP026～EXP028 的 30 项测试不受破坏。

### 12.1 新的独立数值参考（Q2）

```text
points:
[-2,0,1], [2,0,1], [2,4,3], [8,0,3]

centroid      = [2.5,1,2]
min_bound     = [-2,0,1]
max_bound     = [8,4,3]
bbox_center   = [3,2,2]
bbox_extent   = [10,4,2]
median_center = [2,0,2]
```

这些值可用于发现“返回数组形状正常但公式写错”的静默错误。

---

## 13｜EXP028 → EXP029 集成：预测与实际链路

### 13.1 合成 RGB-D 输入

```python
depth_m = np.array([
    [1.0, 0.0, 2.0],
    [2.0, 3.0, 4.0],
])
target_mask = np.array([
    [True,  True,  False],
    [True,  False, True ],
])
fx, fy, cx, cy = 2., 2., 1., 1.
```

RGB：

```python
rgb = np.array([
    [[255,0,0],   [0,255,0],   [0,0,255]],
    [[255,255,0], [255,0,255], [0,255,255]],
], dtype=np.uint8)
```

### 13.2 目标有效像素筛选

有效深度条件：`np.isfinite(depth_m) & (depth_m > 0)`。与布尔 Mask 相与后：

```text
selected =
[[ True, False, False],
 [ True, False,  True]]
```

行优先选中 `(u,v,Z)=(0,0,1)、(0,1,2)、(2,1,4)`。

### 13.3 像素到 XYZ 的公式与单位

\[
X=\frac{(u-c_x)Z}{f_x},\qquad Y=\frac{(v-c_y)Z}{f_y},\qquad Z=\mathrm{depth}(u,v)
\]

`u,v,cx,cy` 为像素坐标；`fx,fy` 为像素单位的焦距；`Z` 为米；求得 `X,Y,Z` 全为米。

独立结果：

```text
points =
[[-0.5,-0.5,1],
 [-1.0, 0.0,2],
 [ 2.0, 0.0,4]]

colors =
[[1,0,0],  # 红
 [1,1,0],  # 黄
 [0,1,1]]  # 青
```

`points.shape = (3,3)`、`colors.shape = (3,3)`。颜色归一化是从 `uint8` RGB 的 `[0,255]` 变为浮点 `[0,1]`；几何统计不直接使用颜色，但顺序对齐证明上游点云可靠。

### 13.4 EXP029 六项统计量

```text
centroid      = [ 1/6, -1/6, 7/3]
min_bound     = [-1,   -0.5, 1]
max_bound     = [ 2,    0,   4]
bbox_center   = [ 0.5, -0.25, 2.5]
bbox_extent   = [ 3,    0.5, 3]
median_center = [-0.5,  0,   2]
```

### 13.5 预测与实际对照

- E1：筛选与两个数组 Shape，预测正确。
- E2：XYZ、红黄青点色对应顺序，预测正确。
- E3：六项数值与独立手算结果一致。
- E4：空结果的跨模块契约与业务层跳过逻辑正确。
- 集成脚本 `exp029_rgbd_to_stats_integration.py` 输出 `EXP029 RGB-D Integration: PASS`，语法检查 PASS。

---

## 14｜空目标的业务层策略与异常传播

### 14.1 契约分层

```text
EXP028 mask_to_pointcloud(...)
    → points [0,3] 可以是合法提取结果
    ↓
业务/调度层检查 points.shape[0]
    ├── 0：报告无有效三维目标点；跳过本帧统计 / 重试 / 告警
    └── >0：调用 EXP029 compute_pointcloud_statistics(points)
              → 对空点直接抛 ValueError（防御性校验）
```

### 14.2 不要混淆三种含义

- **没有有效三维目标点**：可以由提取结果确定。
- **没有目标 Mask 像素**：Mask 本身为空，可进一步确认。
- **真实世界中不存在目标**：仅靠空点云**不能**推定；还需要检测器、深度传感器和图像质量等上下文。

### 14.3 三项自动化集成测试

文件：`tests/test_rgbd_stats_integration.py`。

1. `test_rgbd_to_statistics_reference`：完整数据链，独立 XYZ/RGB/统计量断言。
2. `test_empty_target_mask`：目标 Mask 全 False，空数组契约。
3. `test_all_target_depth_invalid`：有 Mask 但深度全无效，仍然无法生成三维点。

**实测**：集成测试 3/3 PASS；全项目回归 43/43 PASS，耗时约 0.168 秒（以本轮用户终端日志为准）。

---

## 15｜独立通关测试：解释、计算、实现、排错、迁移

### Q1 Explain：三种中心

结果：PASS。能够解释均值、轴向极值中点和逐轴中位数；明确三者都不保证是真实物体中心。

### Q2 Calculate：独立四点参考

结果：PASS。参考值见第 12 节；全部输出为 `(3,)`，单位继承输入米。

### Q3 Debug：错误 Extent 加法

```python
bbox_extent = max_bound + min_bound  # 错
bbox_extent = max_bound - min_bound  # 对
```

在 Q2 数据上：错误值 `[6,4,4]`，正确值 `[10,4,2]`。Shape 与类型都可能正确，独立值断言才能可靠揭示静默语义错误。

### Q4 Implement：独立实现

结果：核心契约与六项计算正确，PASS。独立实现 `estimate_object_geometry(points)`，包含类型转换、形状/非空/有限性验证、六项统计及返回字典。工程增强建议：捕获 `(TypeError, ValueError, OverflowError)`，统一转换为 `ValueError`。

### Q5 Debug：错误求均值轴

```text
points: shape=(3,3)
mean(axis=1) 错误值 = [1/3,1,1]
mean(axis=0) 正确值 = [2/3,2/3,1]
```

N=3 时两者 Shape 恰好都是 `(3,)`，形状检查存在盲区；应使用独立数值断言。Q5 通过。

### Q6 Transfer：绕 Z 轴转 90° 后平移

原四点：`[0,0,1]、[4,0,1]、[0,2,1]、[4,2,1]`。

```python
R = np.array([[0,-1,0], [1,0,0], [0,0,1]])
t = np.array([1,2,0])
transformed = points @ R.T + t
```

```text
transformed =
[[ 1,2,1],
 [ 1,6,1],
 [-1,2,1],
 [-1,6,1]]
new centroid    = [0,4,1]
original extent = [4,2,0]
new extent      = [2,4,0]
```

结论：`C' = RC + t`，不必重新遍历点云就能更新质心；刚体变换保持两点之间的欧氏距离；AABB 因与坐标轴绑定，旋转后轴向尺寸可能改变。

**重要补充**：旧 AABB 的八个角点旋转后再包围，可以提供一个包围范围，但对任意稀疏点云，未必是旋转后实际点云的最紧 AABB；计算精确新 AABB 一般需检查变换后的点云或掌握更多几何信息。

---

## 16｜Debug 记录：两类典型工程事故

### 16.1 实际遇到的工作目录/环境错误

**现象**：

```text
Set-Content : 未能找到路径 C:\Users\...\src\experiments\...
ModuleNotFoundError: No module named 'src'
解释器：D:\anaconda\python.exe
提示符：(base) PS C:\Users\...>
```

**根因**：在用户主目录与 base Conda 环境执行了项目相对路径命令；不是 EXP029 几何算法错误。

**处理**：

```powershell
Set-Location "D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer"
conda activate D:\conda_envs\ai_3d
Get-Location
python -c "import sys; print(sys.executable)"
```

**修复验证**：再次运行主动修改 C，得到实验及语法检查 PASS。

**预防**：执行前核对当前目录、Python 解释器和目标文件存在性；不要通过重新安装一堆包处理路径错误。

### 16.2 会悄悄给出错误结果的数学 Bug

| 症状 | 根因 | 最小检查 |
|---|---|---|
| Centroid shape 正确但数值错误 | `axis=1` 而非 `axis=0` | 用 4 个非对称点和独立参考值 |
| Extent 数值错误但不报错 | `max+min` 而非 `max-min` | 参考结果、平移不变性 |
| Bounding Box 被拉得特别大 | 极端离群点进入点云 | 检查上游 Mask / 深度 / min/max |
| Median Center 看似合理但与真实点不一致 | 各轴独立求中位数 | 查看真实点集合及定义 |
| 没有目标却得到虚假位置 | 返回全零中心掩盖空点云 | 输入空结果应被明确处理 |
| XYZ 与颜色错位 | 选点顺序/Mask 索引不一致 | 按像素行列写独立参考 |

---

## 17｜几何不变量、边界性质与测试思想

除了人工算出的标准答案，也要利用数学生成更一般的验证条件：

1. `min_bound <= max_bound` 按各轴恒成立。
2. `bbox_extent >= 0` 按各轴恒成立。
3. `bbox_center` 各分量在对应的 min/max 之间。
4. Centroid 各分量位于相应 min/max 区间内。
5. 单点时 `min=max=centroid=bbox_center=median_center`、`extent=0`。
6. 所有点加同一个 `t`：各中心、min/max 同步加 `t`，Extent 不变。
7. 刚体旋转和平移：两两欧氏距离不变；质心服从 `C'=RC+t`。
8. 改变原数据中某些非极值点，AABB 可以不变，但 Centroid 可能变化。

**性质测试不是银弹**：例如错误写成 `max+min` 时，某些特别对称的输入可能恰好没暴露；最终仍需独立的数值样例、多类数据和边界检查相结合。

---

## 18｜如何复现：环境、命令与预期结果

### 18.1 本项目使用的工作目录与解释器

```powershell
Set-Location "D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer"
$python = "D:\conda_envs\ai_3d\python.exe"
& $python -c "import sys; print(sys.executable)"
```

### 18.2 主动修改与生产演示

```powershell
& $python -m src.experiments.exp029_center_bbox_minimal
& $python -m src.experiments.exp029_active_modification_a
& $python -m src.experiments.exp029_active_modification_b
& $python -m src.experiments.exp029_active_modification_c
& $python -m src.experiments.exp029_production_smoke
& $python -m src.experiments.exp029_rgbd_to_stats_integration
```

### 18.3 只运行 EXP029 专项单测

```powershell
& $python -m unittest discover -s tests -p "test_pointcloud_statistics.py" -v
```

本轮日志：**10/10 PASS**。

### 18.4 只运行集成测试

```powershell
& $python -m unittest discover -s tests -p "test_rgbd_stats_integration.py" -v
```

本轮日志：**3/3 PASS**。

### 18.5 全量回归测试

```powershell
& $python -m unittest discover -s tests -p "test_*.py" -v
```

本轮日志：**43/43 PASS**，其中之前 EXP026～EXP028 为 30 项，EXP029 统计模块 10 项，新增集成测试 3 项。

### 18.6 Python 语法检查

```powershell
& $python -m py_compile `
    src\geometry\pointcloud_statistics.py `
    tests\test_pointcloud_statistics.py `
    tests\test_rgbd_stats_integration.py
```

**注意**：测试成功是本轮实际日志记录，不代表今后修改后仍自动保持 PASS；每次合并、提交前重新运行。

---

## 19｜工程局限：接下来必须知道的边界

### 必须掌握

- 当前中心是**观测点云的统计结果**，不是完整实体的真值位置。
- Depth 值单位错误会按比例污染 XYZ 和 Extent。
- 相机内参错误会产生错误 XYZ，本模块不能补救上游投影错误。
- 遮挡、视角、噪声、错误 Mask 会改变统计结果。
- AABB 与坐标轴对齐，旋转敏感；单点或平面点云的零 Extent 不等于实体实际零厚度。
- 只有端到端独立参考和自动回归才能证明模块连接语义未破坏。

### 建议理解

- 深度离群点对 AABB 的破坏通常比对均值更严重。
- 坐标逐轴 Median 鲁棒，但可能没有真实对应点。
- 比较不同物体或不同相机帧的 AABB 前，要先确认它们所用坐标系一致。

### 知道存在即可（本关不展开）

- OBB（Oriented Bounding Box）与 PCA 主轴方向。
- RANSAC、统计离群点去除、半径离群点去除。
- 三维聚类、体素降采样、占据栅格、体积重建。
- 多相机标定与世界坐标系下的时序跟踪。

---

## 20｜我的理解、验收闭环与 Git/GitHub 待办

### 20.1 用自己的话总结

> 我之前会从深度图获得三维点，也会用 Mask 提取感兴趣区域。EXP029 让我把这组点变成能够描述位置和范围的数据：平均位置用 Centroid，XYZ 范围用 AABB，各轴中位位置用 Median。不同统计量回答的是不同问题；没有一种天然等同真实物体中心。更重要的是，算法数值正确、输入可靠、上下游契约正确是三件需要分别验收的事。

**建议本地笔记复核时进一步补写**：自己实际最容易弄错的一行代码、为什么错、以后如何检测，以及下一次看到 RGB-D 目标点云时会优先检查的三个参数。

### 20.2 闭环检查清单

- [x] 已理解目标、上游 EXP028 与下游 3D Object Localizer 的关系
- [x] 能独立解释 Centroid、AABB、Median
- [x] 能说明每个数据 Shape / dtype / 单位
- [x] 已完成最小理论和数值实验
- [x] 主动修改 A：预测与实测通过
- [x] 主动修改 B：预测与实测通过
- [x] 主动修改 C：预测与实测通过
- [x] API 输入契约预测与 Smoke Test 通过
- [x] EXP029 正式单元测试 10/10 PASS
- [x] RGB-D → 点云 → 统计集成实验通过
- [x] 集成单测 3/3 PASS
- [x] 全项目测试 43/43 PASS
- [x] 独立知识通关 Q1～Q6 通过
- [ ] 笔记已保存到 `experiments/EXP029_pointcloud_statistics.md`，并由本人复核/补充
- [ ] 确认项目 `git status`，筛选本关文件，避免误提交截图/大型数据
- [ ] 重新运行全量回归并检查 Git diff
- [ ] Git commit 完成并记录 SHA
- [ ] GitHub push 完成并验证远端 SHA 与本地一致
- [ ] 正式将 EXP029 标记为 COMPLETE

### 20.3 Git/GitHub 工作流（尚未执行，不能提前打勾）

```powershell
Set-Location "D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer"
git status --short
git diff --check

# 复核文件后逐一 git add；不要直接 git add .，避免无关截屏/JSON 等资产进入提交。
# git add src/geometry/pointcloud_statistics.py ...

# 暂存后检查：
# git diff --cached --stat
# git diff --cached --check
# git commit -m "Complete EXP029 point cloud statistics and RGB-D integration"
# git push origin master
# git rev-parse HEAD
# git ls-remote origin refs/heads/master
```

**最后验收标准**：远端 `master` 的 SHA 与本地 `HEAD` 一致，且本关核心代码、测试与笔记均已纳入提交。此时才按 SOP 宣布 EXP029 COMPLETE。

### 20.4 下一个知识连接点（此处仅记录，不提前开关）

继续把统计中心放进统一坐标系，并在后续实验中研究更真实的物体包围盒、遮挡和深度噪声条件下的三维定位可靠性。进入下一关前，先关闭本关笔记与 Git/GitHub 待办。
