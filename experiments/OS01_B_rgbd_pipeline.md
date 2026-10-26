# OS-01-B｜RGB-D Object Localizer：单样本工程流水线、聚类门禁与安全状态

> 项目：`Project_002_RGBD_Object_Localizer`  
> 路线：Level 2 · 3D Vision / OS-01 RGB-D Object Localizer  
> Learning Unit：OS-01-B（B1 单样本流水线 + B2 DBSCAN／EXP034 集成）  
> 整理日期：2026-10-26  
> 数据边界：本单元仅使用 **6×8 确定性合成 RGB-D 样本**，未使用真实相机或独立物理真值。  
> 当前验收：**核心算法、实验预测、状态 Debug、综合 Boss 与全项目回归 PASS；笔记已整理；Git commit / GitHub push 尚未完成。**  
> 注意：以下“实测”均为本轮学习中用户提供的 Windows PowerShell 输出；没有记录的验证不写成已通过。

## 0. Entry Protocol：目标、前置知识、产物、验收标准

### 0.1 为什么学习 OS-01-B

OS-01-A 已完成 RGB-D 输入契约、合成数据重载、反投影及输入扰动；EXP026～EXP037 训练了三维反投影、Mask 到点云、半径过滤、几何统计、DBSCAN、几何门禁以及坐标可信性审计。

OS-01-B 的问题不再是“某个算法能不能运行”，而是：**能否把这些已验证的模块组成可重复测试、状态语义明确、禁止未经授权发布机器人坐标的单样本工程流水线？**

### 0.2 前置知识与知识地图

```text
OS-01-A：RGB / Depth / Mask / K / frame_id 输入契约
    ↓
EXP026 / EXP028：像素反投影、Mask ∩ 有效深度
    ↓
EXP030 等：Radius Outlier Filter，排除孤立点
    ↓
EXP029：点云质心与 Axis-Aligned Bounding Box（AABB）
    ↓
EXP033：DBSCAN，区分 Core / Border / Noise
    ↓
EXP034：逐簇几何门禁与诊断报告
    ↓
OS-01-B：正式 Pipeline、状态映射、Fail-closed 安全出口
    ↓
OS-01-C：更多可追溯 RGB-D 样本及数据集级评测
    ↓
真实传感器、坐标系可信变换、物理精度与安全验证（未来工作）
```

**必须掌握**：NumPy Shape、布尔索引、像素与相机坐标、m/mm 单位、基本 Pytest。  
**建议掌握**：EXP033 DBSCAN、EXP034 `LocalizationGates`、坐标系和安全审计。  
**本单元不验收**：真实相机标定、基座坐标变换、物理毫米精度、机器人运动安全认证。

### 0.3 最终产物

- `src/pipeline/__init__.py`：Pipeline 包入口。
- `src/pipeline/os01_single_sample.py`：正式单样本 Pipeline 与合成样本命令行演示。
- `src/geometry/dbscan.py`：从 EXP033 迁入的可复用 DBSCAN 实现。
- `src/experiments/exp033_dbscan_numpy.py`：保留旧测试入口的兼容导出。
- `tests/test_os01_b1_pipeline.py`：B1 输入、过滤参数及空分支测试。
- `tests/test_os01_b2_geometry_gate.py`：B2 门禁拒绝与全 Noise 测试。
- `experiments/OS01_B_rgbd_pipeline.md`：本学习笔记。
- 待完成的工程收尾：README 核对、指定路径暂存、Git commit、GitHub push。

### 0.4 本单元验收标准

能够解释每一层的输入／输出／Shape／单位；主动预测参数变化；复现三条关键路径（通过／拒绝／无簇）；证明早退状态与安全出口；完成至少一次 RED→GREEN；完成综合 Boss；通过全仓回归；提交笔记与代码至 Git/GitHub。**程序运行成功仅是其中一项。**

---

## 1. 输入输出与数据契约（Shape First）

| 项目 | 类型 / Shape | 单位 | 语义 |
|---|---|---|---|
| `rgb` | `uint8`，`(H,W,3)` | 像素通道 0～255 | 与 Depth／Mask 已对齐的 RGB 图像 |
| `depth_m` | 浮点，`(H,W)` | m | 正且有限的光轴深度 Z 才有效 |
| `target_mask` | `bool`，`(H,W)` | 无 | 待定位目标的候选像素 |
| `fx, fy, cx, cy` | 有限数值标量 | px | 相机内参，`fx,fy>0` |
| `frame_id` | 非空字符串 | 无 | 当前为 `camera_optical_frame`，仅是坐标系标识 |
| `selected` | `bool`，`(H,W)` | 无 | `target_mask & valid_depth` |
| `points` | `float64`，`(N,3)` | m | 反投影后的相机系 XYZ |
| `colors` | `float64`，`(N,3)` | 0～1 | 与点云逐行对齐的 RGB 颜色 |
| `neighbor_counts` | 整数，`(N,)` | 个 | Radius Filter 的邻居数，**不含自身** |
| `kept_indices` | 整数，`(M,)` | 无 | 被保留点在过滤前点云中的行索引 |
| `filtered_points` | `float64`，`(M,3)` | m | 半径过滤后的 XYZ，`M≤N` |
| `labels` | `int64`，`(M,)` | 无 | DBSCAN 输出；`-1` 为 Noise，非负整数为簇 ID |
| `centroid_camera_m` | 三元素浮点列表 | m | **过滤后整片点云**的采样质心；不一定是有效簇定位 |
| `bbox_extent_m` | 三元素浮点列表 | m | 过滤后整片点云的 AABB 尺寸 |
| `geometry_gate` | 字典或 `None` | 混合 | EXP034 逐簇验证的诊断性摘要 |
| 顶层 `position_m` | `None` | m（若未来存在） | 本单元故意不提供机器人可执行位置 |
| 顶层 `publishable` | `False` | 布尔 | 本单元永久处于诊断限定状态 |

**Shape 契约：**

```python
points.shape == (N, 3)
filtered_points.shape == (M, 3)
labels.shape == (M,)
# labels[i] 对应 filtered_points[i]，而非最初 points[i]
```

不能仅检查数组长度，还必须检查**数据来源和行顺序**。即使两组点数量相同，只要重新排序而标签未同步排序，几何统计仍可能错误。

### 1.1 相机反投影

已校正且已对齐的针孔相机模型：

\[
X=\frac{(u-c_x)Z}{f_x},\qquad
Y=\frac{(v-c_y)Z}{f_y},\qquad
Z=\operatorname{depth\_m}[v,u]
\]

- `u,v,cx,cy,fx,fy`：像素尺度；`u` 为列，`v` 为行。
- `(u-cx)/fx` 和 `(v-cy)/fy`：无量纲方向比值。
- `X,Y,Z`：m；转成 mm 必须乘 `1000`。
- `Z` 是光轴方向深度，不是沿视线射线的欧氏距离。
- 当前坐标约定为相机光学系：X 向右、Y 向下、Z 向前。

有效像素：

```python
valid_depth = np.isfinite(depth_m) & (depth_m > 0)
selected = target_mask & valid_depth
```

### 1.2 点云统计

对非空的 `(M,3)` 点云：

\[
\mathbf C=\frac{1}{M}\sum_{i=1}^{M}\mathbf P_i,\qquad
\mathbf E=\max_i\mathbf P_i-\min_i\mathbf P_i
\]

- `C`：点云采样质心 `(3,)`，m。
- `E`：轴对齐包围盒 AABB 的 extent `(3,)`，m。
- `B=(min+max)/2`：AABB 中心 `(3,)`，m。
- `gap=||C-B||₂`：质心与 AABB 中心距离，m。
- 空点云不能调用要求非空输入的 `compute_pointcloud_statistics()`，必须由 Pipeline 先处理。

---

## 2. Pipeline 六层处理链与代码拆解

```text
RGB / Depth / Mask + Intrinsics + frame_id
       ↓
① mask_to_pointcloud：Mask ∩ 有效深度 → N×3 点云
       ↓
② radius_outlier_filter：邻域计数 → M×3 过滤点云
       ↓
③ compute_pointcloud_statistics：整体质心与 AABB（M>0）
       ↓
④ dbscan_numpy：过滤点云 → M 个聚类标签
       ↓
⑤ validate_cluster_localization：逐簇几何门禁
       ↓
⑥ Pipeline 状态映射与安全隔离：结构化诊断报告
```

核心函数位置：

| API | 位置 | 输入重点 | 输出重点 |
|---|---|---|---|
| `mask_to_pointcloud()` | `src/geometry/mask_to_pointcloud.py` | Depth、RGB、Mask、K | `points, colors, selected` |
| `radius_outlier_filter()` | `src/geometry/pointcloud_filter.py` | 点云、`radius`、`min_neighbors` | 过滤点、计数、保留索引 |
| `compute_pointcloud_statistics()` | `src/geometry/pointcloud_statistics.py` | 非空 `(M,3)` | 质心、AABB 等 |
| `dbscan_numpy()` | `src/geometry/dbscan.py` | `(M,3)`、`eps`、`min_points` | `(M,)` 标签 |
| `validate_cluster_localization()` | `src/geometry/cluster_localization.py` | `(M,3)`、`labels`、`LocalizationGates` | `status`、`objects`、有效/无效簇统计 |
| `localize_single_sample()` | `src/pipeline/os01_single_sample.py` | 对齐 RGB-D 与参数 | 顶层状态、安全字段及 `diagnostics` |

**接口伪代码（用于理解调用关系，不替代源码）：**

```python
points, colors, selected = mask_to_pointcloud(
    depth_m, rgb, target_mask, fx, fy, cx, cy
)
filtered = radius_outlier_filter(
    points, radius=radius_m, min_neighbors=min_neighbors
)
filtered_points = filtered["filtered_points"]

if len(filtered_points) == 0:
    # 由 Pipeline 按 Mask、有效深度、过滤情况返回相应状态
    ...

stats = compute_pointcloud_statistics(filtered_points)
labels = dbscan_numpy(
    filtered_points,
    eps=dbscan_eps_m,
    min_points=dbscan_min_points,
)
local_report = validate_cluster_localization(
    filtered_points,
    labels,
    frame_id=frame_id,
    gates=gates,
)

# 只将门禁诊断结果写入 diagnostics；不要传递局部发布许可
report["position_m"] = None
report["publishable"] = False
```

### 2.1 为什么 EXP033 要迁移到 `src/geometry/dbscan.py`

EXP033 原实现通过了旧测试，但正式 Pipeline 不应长期依赖 `src/experiments`。本轮将算法迁移至 `src/geometry/dbscan.py`，原 `src/experiments/exp033_dbscan_numpy.py` 保留兼容导出，**同一个算法只维护一份实现**。

已观察回归：`python -m pytest tests/test_exp033_dbscan_numpy.py -q` → **29 passed in 0.34s**。

### 2.2 不盲目加入 RANSAC 去平面

当前五个目标点恰好都在 `Z=1.0 m` 的平面上。若在已选择的目标点云上直接运行“去平面”，可能把整个目标点云误删。桌面背景分割与目标点云分割是不同阶段，不能仅因为项目学过 RANSAC 就强行拼接。

---

## 3. Radius Filter 与 DBSCAN：公式和关键参数

### 3.1 Radius Outlier Filter

对点 `Pᵢ`，半径内**其他索引**的点数为：

\[
n_i(r)=\#\{j\neq i:\;\|P_j-P_i\|_2\le r\}
\]

保留规则：`n_i(radius_m) >= min_neighbors`。计算使用原始输入点集的一次性邻居计数，过滤后保持原索引顺序。

### 3.2 DBSCAN

DBSCAN 的 `eps` 邻域**包含自身**：

\[
N_\varepsilon(P_i)=\{P_j:\|P_j-P_i\|_2\le\varepsilon\}
\]

- **Core**：`|N_eps(Pᵢ)| >= min_points`。
- **Border**：自身不是 Core，但位于某个 Core 的邻域内；加入簇但不继续扩展。
- **Noise**：未归属于任何密度可达簇，标签为 `-1`。
- **簇标签**：`0,1,...` 仅为编号，不代表真实物体类别；不同实现可能赋予不同编号，应比较簇分区而非武断依赖 ID。

### 3.3 参数字典

| 参数 | 类型 / 单位 | 改大以后（其他条件固定） | 代码位置及风险 |
|---|---|---|---|
| `radius_m` | 正实数 / m | Radius Filter 邻居通常增多，保留点通常不减少 | `radius_outlier_filter(..., radius=...)` |
| `min_neighbors` | 正整数 / 个，**不含自身** | 保留条件更严格，点数通常减少 | Radius Filter，过高可删掉真实稀疏目标 |
| `dbscan_eps_m` | 正实数 / m | DBSCAN 邻域增大，可能连通原本分开的簇 | `dbscan_numpy(..., eps=...)`，过大可能合并目标 |
| `dbscan_min_points` | 正整数 / 个，**含自身** | Core 更难形成，Noise 可能增加 | DBSCAN，过高导致 `NO_CLUSTERS` |
| `gates.min_points` | 正整数 / 每簇点数 | 几何门禁更严格，可能拒绝稀疏簇 | `LocalizationGates`，**不同于 DBSCAN min_points** |
| `gates.max_points` | 正整数 / 每簇点数 | 放宽过大点簇的限制 | 教学默认 20，不能照搬真实大点云 |
| `gates.max_extent_m` | 3 个正数 / m | 放宽每轴 AABB 尺寸限制 | 默认 `(0.04,0.04,0.02)` m |
| `gates.max_centroid_bbox_gap_m` | 非负实数 / m | 放宽中心偏移限制 | 默认 `0.02` m |

**重要区别：**`min_neighbors=2`（不含自身）与 `dbscan_min_points=3`（含自身）不是同一个参数，尽管在某些固定邻域条件下看似相差 1。

EXP034 的默认点数门禁为 `[5,20]`，是**教学演示阈值**。真实传感器的点数会受分辨率、深度、Mask、距离和采样密度影响，不能直接用于工业合格判定。

---

## 4. 可复现的合成样本与基准结果

### 4.1 样本定义

```python
rgb = np.zeros((6, 8, 3), dtype=np.uint8)
rgb[2:4, 3:6] = [200, 60, 40]

depth_m = np.zeros((6, 8), dtype=np.float64)
depth_m[2:4, 3:6] = 1.0
depth_m[2, 3] = 0.0                 # Mask 内特意注入无效深度

target_mask = np.zeros((6, 8), dtype=bool)
target_mask[2:4, 3:6] = True

fx, fy, cx, cy = 100.0, 100.0, 3.5, 2.5
frame_id = "camera_optical_frame"
```

Mask 的 True 像素数为 6；其中一个深度为 0，故有效三维点数为 5。

| 点 | 像素 `(v,u)` | 相机系 `(X,Y,Z)`，m |
|---|---|---|
| P0 | `(2,4)` | `[ 0.005,-0.005,1.0]` |
| P1 | `(2,5)` | `[ 0.015,-0.005,1.0]` |
| P2 | `(3,3)` | `[-0.005, 0.005,1.0]` |
| P3 | `(3,4)` | `[ 0.005, 0.005,1.0]` |
| P4 | `(3,5)` | `[ 0.015, 0.005,1.0]` |

基准计算：

```text
mask_pixel_count       = 6
valid_point_count      = 5
filtered_point_count   = 5
neighbor_counts        = [2, 2, 1, 3, 2]  # Radius，不含自身
centroid_camera_m      = [0.007, 0.001, 1.0] m
bbox_extent_m          = [0.020, 0.010, 0.0] m
```

AABB 中心为 `[0.005,0,1.0] m`，质心与 AABB 中心差为 `[0.002,0.001,0] m`：

\[
\mathrm{gap}=\sqrt{0.002^2+0.001^2}\approx 0.002236\ \mathrm m=2.236\ \mathrm{mm}
\]

**浮点输出**出现 `0.007000000000000001` 是正常表示误差，不能直接把它解释成传感器精度。

---

## 5. 三组主动修改实验：预测 → 运行 → 解释

### 实验 E1｜默认参数：一个有效簇

**输入参数：**`radius_m=0.011`，`min_neighbors=1`，`dbscan_eps_m=0.011`，`dbscan_min_points=3`，默认 EXP034 门禁。

**运行前预测：**

- Radius 不含自身：`[2,2,1,3,2]`，全部 5 点保留。
- DBSCAN 含自身：`[3,3,2,4,3]`。
- Core：P0、P1、P3、P4；Border：P2；Noise：0。
- 一个簇，`labels=[0,0,0,0,0]`，`labels.shape=(5,)`。
- 该簇点数 5、extent 合格、gap≈2.236 mm，EXP034 返回 `OK`。
- 顶层仍然 `DIAGNOSTIC_ONLY`、`position_m=None`、`publishable=False`。

**实际终端输出（节选）：**

```text
Frame: camera_optical_frame
Mask pixels: 6
Valid points: 5
Neighbor counts: [2, 2, 1, 3, 2]
Filtered points: 5
Centroid (m): [0.007000000000000001, 0.001, 1.0]
Bbox extent (m): [0.02, 0.01, 0.0]
Status: DIAGNOSTIC_ONLY
Publishable: False
DBSCAN labels: [0, 0, 0, 0, 0]
Cluster count: 1
Cluster noise count: 0
Local gate status: OK
Local gate valid count: 1
Local gate reasons: [[]]
```

**结论：**从 Mask 到有效簇的正常路径成立；几何门禁通过不改变机器人发布许可。

### 实验 E2｜增大 `min_neighbors`：过滤后四点被门禁拒绝

**仅改变：**`min_neighbors=1 → 2`。

**运行前预测：**P2 原邻居数仅为 1，被删；保留索引 `[0,1,3,4]`，`filtered_points.shape=(4,3)`，质心 `[0.010,0,1.0] m`，AABB extent `[0.010,0.010,0] m`。相对 E1 质心变化 `[+0.003,-0.001,0] m`，即 `[+3,-1,0] mm`。

**测试证据：**参数化 B1 测试确认上述点数、索引、质心及 extent。四点仍组成一个 DBSCAN 簇，但低于 EXP034 默认 `min_points=5`，所以：

```text
geometry_gate.status = ALL_INVALID
valid_count          = 0
invalid_count        = 1
reasons              = ["POINT_COUNT_OUT_OF_RANGE"]
report.status        = GEOMETRY_GATE_REJECTED
report.position_m    = None
report.publishable   = False
```

**结论：**参数变化可能不会让聚类消失，却会使局部几何门禁拒绝。过滤参数与门禁参数存在耦合。

### 实验 E3｜增大 DBSCAN `min_points`：全部 Noise

**仅改变：**`dbscan_min_points=3 → 6`，Radius 仍保留五点。

**运行前预测：**每点 DBSCAN 邻居数最多只有 4（含自身），没有 Core，因此：

```text
labels              = [-1,-1,-1,-1,-1]
cluster_count       = 0
cluster_noise_count = 5
geometry_gate.status = NO_OBJECTS  # EXP034 的局部语义：没有输入簇
geometry_gate.objects = []
report.status        = NO_CLUSTERS  # Pipeline 顶层语义
report.position_m    = None
report.publishable   = False
```

**测试证据：**`test_all_dbscan_noise_must_report_no_clusters` 最终通过；诊断质心 `[0.007,0.001,1] m` 和 AABB `[0.02,0.01,0] m` 被保留。

**结论：**“有过滤后的点”不代表“找到目标簇”。整体点云统计不能替代逐簇质量验证。

### 5.4 三组对照表

| 实验 | 过滤后点数 | DBSCAN 簇数 | Noise 数 | EXP034 | 顶层状态 |
|---|---:|---:|---:|---|---|
| E1 默认 | 5 | 1 | 0 | `OK` | `DIAGNOSTIC_ONLY` |
| E2 `min_neighbors=2` | 4 | 1 | 0 | `ALL_INVALID` | `GEOMETRY_GATE_REJECTED` |
| E3 `dbscan_min_points=6` | 5 | 0 | 5 | `NO_OBJECTS` | `NO_CLUSTERS` |

---

## 6. 两次 RED → GREEN：定位、修复、回归

### D1｜有簇但全部被门禁拒绝，顶层状态错误

**运行前预测：**自定义 `LocalizationGates(min_points=6, max_points=20, max_extent_m=(0.04,0.04,0.02), max_centroid_bbox_gap_m=0.02)` 时，五点簇点数不足，EXP034 应返回 `ALL_INVALID`，原因 `POINT_COUNT_OUT_OF_RANGE`；当前 Pipeline 顶层却仍是 `DIAGNOSTIC_ONLY`。

**RED：**测试 `test_rejected_cluster_must_be_explicit_and_fail_closed` 最后断言失败：

```text
assert 'DIAGNOSTIC_ONLY' == 'GEOMETRY_GATE_REJECTED'
1 failed in 1.00s
```

此前断言已证明聚类、门禁原因、诊断质心和 Fail-closed 安全字段均正确。**根因是状态映射缺失，不是 DBSCAN 算错。**

**GREEN：**在 `local_report["status"] == "ALL_INVALID"` 时，将 Pipeline 状态设置为 `GEOMETRY_GATE_REJECTED`；不改算法、不删除诊断、不开放发布。

**回归中发现的契约更新：**B1 `min_neighbors=2` 原测试仍期待 `DIAGNOSTIC_ONLY`，但四点簇按新门禁应被拒绝。故保留其全部几何断言，仅将状态断言改为按参数区分，同时新增门禁详情断言。之后 B1/B2 联合测试 **6 passed in 0.68s**。

### D2｜全 Noise 没有簇，顶层仍显示诊断成功

**运行前预测：**`dbscan_min_points=6` → 全部 Noise，EXP034 `NO_OBJECTS`、无对象、计数 0；但顶层错误地保留 `DIAGNOSTIC_ONLY`。

**RED：**

```text
test_rejected_cluster_must_be_explicit_and_fail_closed PASSED
test_all_dbscan_noise_must_report_no_clusters FAILED
assert 'DIAGNOSTIC_ONLY' == 'NO_CLUSTERS'
1 failed, 1 passed in 0.96s
```

**GREEN：**只在已进入聚类/门禁阶段后增加映射：

```python
if local_report["status"] == "NO_OBJECTS":
    report["status"] = "NO_CLUSTERS"
elif local_report["status"] == "ALL_INVALID":
    report["status"] = "GEOMETRY_GATE_REJECTED"

report["position_m"] = None
report["publishable"] = False
```

**实测：**B2 `2 passed in 0.54s`；B1+B2 `7 passed in 0.60s`。

**Debug 方法总结：**先读完整报错 → 确认前置断言 → 定位第一个违约的接口 → 只改状态映射 → 再执行专项和全量回归。**不要为了绿灯删除正确断言。**

---

## 7. Pipeline 状态契约与 Fail-closed

### 7.1 顶层状态表

| 条件 | `report["status"]` | 说明 |
|---|---|---|
| 目标 Mask 无 True | `NO_OBJECTS` | 没有候选目标像素 |
| Mask 有 True，但合法深度为 0 点 | `NO_VALID_DEPTH` | 有 Mask，但无可反投影点 |
| 合法点存在，但 Radius Filter 后为 0 点 | `NO_POINTS_AFTER_FILTER` | 过滤删除全部点 |
| 有过滤点，DBSCAN 仅产生 `-1` | `NO_CLUSTERS` | 有点但没有任何可用簇 |
| DBSCAN 有簇，EXP034 全部拒绝 | `GEOMETRY_GATE_REJECTED` | 有簇但均未通过局部几何门禁 |
| 至少一个簇通过门禁 | `DIAGNOSTIC_ONLY` | 仍只提供诊断信息；可包含部分无效簇 |

**局部与顶层的区别：**EXP034 的 `NO_OBJECTS` 表示“局部门禁没有簇”，不是“原始 Mask 为空”；因此 Pipeline 必须映射成 `NO_CLUSTERS`，以免混淆处理阶段。

EXP034 还可能返回 `PARTIAL_INVALID`：既有通过的簇，也有拒绝的簇。当前 Pipeline 顶层保持 `DIAGNOSTIC_ONLY`，但 `diagnostics.geometry_gate.status` 保留 `PARTIAL_INVALID`。是否增设更细的顶层状态，留给后续多目标场景的接口设计，不伪装为已实现。

### 7.2 报告结构与安全出口

```python
report = {
    "frame_id": "camera_optical_frame",
    "status": "DIAGNOSTIC_ONLY",  # 示例；会随状态分支改变
    "publishable": False,
    "position_m": None,
    "mask_pixel_count": 6,
    "valid_point_count": 5,
    "filtered_point_count": 5,
    "diagnostics": {
        "neighbor_counts": [2, 2, 1, 3, 2],
        "kept_indices": [0, 1, 2, 3, 4],
        "centroid_camera_m": [0.007, 0.001, 1.0],
        "bbox_extent_m": [0.02, 0.01, 0.0],
        "cluster_labels": [0, 0, 0, 0, 0],
        "cluster_count": 1,
        "cluster_noise_count": 0,
        "geometry_gate": {
            "profile": "EXP034_DEMO",
            "status": "OK",
            "valid_count": 1,
            "invalid_count": 0,
            "objects": [
                {
                    "cluster_label": 0,
                    "local_gate_passed": True,
                    "reasons": [],
                    "centroid_camera_m": [0.007, 0.001, 1.0],
                    "bbox_extent_m": [0.02, 0.01, 0.0],
                }
            ],
        },
    },
}
```

以上是**正常路径示意结构**，不是逐字复制终端 JSON。浮点输出可能有末位差异；空数据早退分支的诊断字段可能为 `None` 或空列表。

### 7.3 为什么局部 `publishable=True` 不能原样传给机器人

EXP034 返回的某个簇 `publishable=True` 仅说明它通过**当前配置的局部几何门禁**。它不能证明采集是真实的、标定可信、坐标变换正确、时间同步或安全策略通过。

本单元实施的安全边界是：

```python
report["position_m"] = None
report["publishable"] = False
```

**未来目标设计（本单元尚未实现）：**真实传感器数据来源核验、可信内外参、标定有效期、frame 变换链、单位和时间一致性、独立物理真值、工作空间覆盖、碰撞与机器人控制策略检查；任一必要条件未知或不满足则拒绝发布。`frame_id` 字符串本身不是变换合法性的证明。

### 7.4 保留诊断 ≠ 提供有效目标

被拒簇的质心、AABB、点数、错误码仍可用于回放、调参、故障定位和质量分析。保留的是**必要的诊断记录**，不是执行许可。当前 `centroid_camera_m` 还是**过滤后整片点云**的质心；没有簇时它绝不能冒充目标实例位置。

---

## 8. 综合 Boss：A 概念、B 推理、C 安全、D Debug 与迁移

> 本节是 **11 点、两簇、一个噪声点的假设题**；已完成独立作答和理论核对，**没有声称运行过新的双目标 RGB-D 集成测试**。

### Boss A｜Shape、索引与单位

```text
filtered_points.shape = (11,3)
labels.shape          = (11,)
labels = [0,0,0,0,0,0, 1,1,1,1, -1]
```

簇 0 为 6 点，簇 1 为 4 点，最后一点为 Noise。`-1` 不计作第三个簇，也不参加 `np.unique(labels[labels>=0])` 的逐簇循环。点云单位为 m；Labels 无物理单位。

### Boss B｜门禁预测

默认 `LocalizationGates`：每簇点数 `[5,20]`，extent 上界 `(0.04,0.04,0.02) m`，gap 上界 `0.02 m`。

- 簇 0：6 点，extent `[0.02,0.01,0.01] m`，gap `0.003 m` → `VALID`。
- 簇 1：4 点，extent `[0.01,0.01,0] m`，gap `0.001 m` → `INVALID_CLUSTER`，`reasons=["POINT_COUNT_OUT_OF_RANGE"]`。
- 帧级门禁：`status="PARTIAL_INVALID"`，`valid_count=1`，`invalid_count=1`，`noise_count=1`。

首次独立作答把错误码写成了 `POINT_COUNT_TOO_LOW`；源码的实际契约是 `POINT_COUNT_OUT_OF_RANGE`，已纠正。

### Boss C｜部分通过簇的安全出口

现有 Pipeline 对 `PARTIAL_INVALID` 没有额外的顶层状态映射，所以：

```text
report.status      = DIAGNOSTIC_ONLY
report.position_m  = None
report.publishable = False
diagnostics.geometry_gate.status = PARTIAL_INVALID
```

一个簇通过局部门禁，不代表坐标已经通过真实标定、物理真值、时间/单位/坐标系以及机器人安全策略审计。

### Boss D｜错误数据源与 Shape 不一致

**错误：**

```python
labels = dbscan_numpy(points, eps=0.011, min_points=3)
local_report = validate_cluster_localization(filtered_points, labels)
```

若原始 `points.shape=(11,3)`，过滤后 `filtered_points.shape=(9,3)`，则 `labels.shape=(11,)` 不满足 EXP034 接口 `(9,)`。当前 `cluster_localization.py` **首先抛出 `ValueError: labels must be (N,) integer values`**，不会先运行到布尔索引；首次答题时误写成 `IndexError`，已纠正。

**正确：**

```python
labels = dbscan_numpy(filtered_points, eps=0.011, min_points=3)
local_report = validate_cluster_localization(filtered_points, labels)
```

除了长度一致，也要保证行顺序完全对应。数据契约检查应放在模块边界，拒绝错误输入，不能让错误配对静默传播。

---

## 9. 已知局限、常见误区与工程迁移

### 9.1 已知局限

1. **只有一个 6×8 合成 RGB-D 场景。** 没有真实传感器采集、实际内外参验证或独立物理真值；不能宣称毫米级定位精度。
2. **Mask 是确定性生成的。** 尚未评估模型分割错误、RGB/Depth 未配准、遮挡、反光、透明物体或深度缺失的真实分布。
3. **EXP034 阈值只是教学示例。** `min_points=5`、`max_points=20` 等不能直接移植到真实高密度点云。
4. **当前 NumPy DBSCAN 生成 `(M,M)` 的全距离矩阵。** 时间与内存开销随点数至少呈二次规模，不适合未经控制的高分辨率整帧直接调用。
5. **质心不是物理真值。** 它是观测点集的均值，受采样密度、可见表面与 Mask 影响。
6. **当前只输出相机坐标系诊断。** 没有可信 `camera→robot_base/world` 变换链，也没有机器人执行授权。
7. **`PARTIAL_INVALID` 仅在内部诊断中显式保留。** 后续真实多目标场景可考虑更细粒度顶层状态，但本关未实现。
8. **20 个可追溯 RGB-D 样本评测未完成。** 这是 OS-01-C 的工作，不计作 OS-01-B 完成证据。

### 9.2 常见误区与纠正

| 误区 | 正确理解 |
|---|---|
| “Radius 计数和 DBSCAN 计数一样” | Radius 排除自身；DBSCAN 包含自身 |
| “聚类 1 个簇就定位成功” | 还须通过 EXP034，且最多得到诊断意义上的几何候选 |
| “`labels.shape` 对了就万事大吉” | 还要确保标签与同一组点的行索引一一对应 |
| “Noise `-1` 是第三个簇” | `-1` 表示未分配，不能当作普通簇 |
| “无簇等于空 Mask” | 前者顶层是 `NO_CLUSTERS`，后者为 `NO_OBJECTS` |
| “门禁拒绝就删除所有几何” | 保留必要诊断，但隔离发布路径 |
| “局部 `publishable=True` 可直接发机器人” | 只表示演示性局部门禁通过，不授予执行权限 |
| “成功执行 Pytest 就完成 Level” | 还需预测、修改、解释、笔记、Boss、Git/GitHub |

### 9.3 后续迁移方向

在真实 RGB-D 数据上建立可追溯采集与对齐契约；增加多目标、多距离和噪声场景；独立记录物理参考真值并做统计评测；考虑将密集点云聚类改为受控采样或空间索引实现；最后才讨论可信变换及机器人安全授权。**任何步骤都不能通过把一个布尔值改成 `True` 取代。**

---

## 10. 复现、测试证据、文件清点和 Git 状态

### 10.1 环境

- Windows PowerShell；项目根目录：`D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer`。
- Conda：`D:\conda_envs\ai_3d`。
- 实测 Pytest 环境：Python `3.11.17`、pytest `9.1.1`；本轮未单独记录 NumPy / SciPy 的版本。

### 10.2 复现命令

```powershell
# 在项目根目录执行
python -m src.pipeline.os01_single_sample

# 兼容的 EXP033 DBSCAN 回归
python -m pytest tests/test_exp033_dbscan_numpy.py -q

# OS-01-B1/B2 关键状态验收
python -m pytest tests/test_os01_b1_pipeline.py tests/test_os01_b2_geometry_gate.py -q

# 全量项目测试及 Git 变更检查
python -m pytest -q
git diff --check
git status --short
```

### 10.3 真实观察到的测试记录

| 命令 / 阶段 | 实际输出 | 判定 |
|---|---|---|
| EXP033 迁移回归 | `29 passed in 0.34s` | PASS |
| B1 初期专项测试 | `5 passed in 0.64s` | PASS |
| 首次 B2 门禁拒绝 RED | `1 failed in 1.00s` | **预期失败已复现** |
| 首次 GREEN 的 B1 回归 | `1 failed, 4 passed` | 旧状态断言与新门禁冲突，已修正 |
| D1 联合回归 | `6 passed in 0.68s` | PASS |
| D2 无簇 RED | `1 failed, 1 passed in 0.96s` | **预期失败已复现** |
| D2 GREEN | `2 passed in 0.54s` | PASS |
| B1+B2 最终联合回归 | `7 passed in 0.60s` | PASS |
| 全项目 Pytest | **`703 passed, 106 subtests passed in 2.51s`** | PASS |
| `git diff --check` | 无错误输出 | PASS |

注：29 个 EXP033 测试及 7 个 B1/B2 测试包含在全仓测试之中，**不能与 703 重复相加**；全仓此前在 OS-01-A 收尾时为 `696 passed`。

### 10.4 Git 状态清点

本轮用户提供的 `git status --short` 为：

```text
 M src/experiments/exp033_dbscan_numpy.py
?? src/geometry/dbscan.py
?? src/pipeline/
?? tests/test_os01_b1_pipeline.py
?? tests/test_os01_b2_geometry_gate.py
```

另外还出现 3 个无关 `ScreenCamera_*.json` 和 3 个 `ScreenCapture_*.png`，**不可使用 `git add .` 误提交**。本笔记的本地仓库文件保存状态、README 更新状态尚待确认。

建议后续在仓库根目录使用显式路径：

```powershell
git add src/experiments/exp033_dbscan_numpy.py src/geometry/dbscan.py src/pipeline/__init__.py src/pipeline/os01_single_sample.py tests/test_os01_b1_pipeline.py tests/test_os01_b2_geometry_gate.py experiments/OS01_B_rgbd_pipeline.md
# 如确实修改了 README.md，再单独 git add README.md
git diff --cached --check
git diff --cached --stat
git status --short
# 人工复核暂存内容后，再 commit / push
```

**当前事实：**OS-01-A 已在 `e35ff69` 完成提交与推送；OS-01-B 尚未见到新 commit、`origin/master` 同步或 `git push` 成功证据，**不能记为 COMPLETE**。

---

## 11. 我自己的理解（独立复盘，保留关键论证）

### Q1：为什么在 OS-01 Pipeline 中 DBSCAN 必须使用过滤后的点云？

因为 DBSCAN 输出的 `labels` 是**逐点对齐**的数据，不是与点云无关的独立列表。`labels[i]` 描述的必须是参与此次聚类的 `filtered_points[i]`。假如对原始 11 点聚类，却拿得到的 11 个标签验证过滤后 9 点，那么标签长度和点行号都会违约；即使数量碰巧相等，只要顺序发生变化也可能出现错误配对。因此当前 Pipeline 必须让聚类与门禁使用同一组、同顺序的 `filtered_points`。这不是 DBSCAN 在任何系统中都必须预过滤，而是本系统对**数据源、Shape 与索引关系**的明确约定。

### Q2：为什么几何门禁拒绝后仍保留诊断信息？

因为拒绝的是该结果进入**有效定位与执行路径的资格**，不是否认这些观测点及其统计曾被计算。点数、质心、AABB 和 `POINT_COUNT_OUT_OF_RANGE` 等理由可用于调参、故障定位、回放和审计；删除它们会模糊“目标不存在”和“目标候选被拒绝”的区别。应保留必要且可追溯的诊断事实，但绝不能把失败簇的几何统计混入可发布位置流。**诊断可存在，发布可被拒绝，两者并不矛盾。**

### Q3：为什么 EXP034 的局部 `publishable=True` 不足以授权机器人运动？

因为那只是局部几何门禁的判断，最多表示点数、尺寸与质心偏移满足当前阈值。当前输入是合成数据；真实相机来源、标定有效性、独立物理真值、时间同步、坐标变换链和机器人安全策略并未在 OS-01-B 中完成验证。机器人执行需要的是整条可信验证链，而不只是一个几何模块输出的布尔值。**本单元已实现的是顶层 `position_m=None`、`publishable=False` 的 Fail-closed 隔离；完整物理安全验证链仍是后续工程任务。**

### 三句话总结

1. **点和标签必须同源、同序、同长度**，不能把原始点聚类结果套到过滤后点云上。
2. **门禁拒绝不等于删除诊断事实**，但诊断信息无权冒充有效机器人目标。
3. **局部几何通过 ≠ 物理准确 ≠ 机器人可发布**；缺少必要可信证据时必须 Fail-closed。

---

## 12. Learning Unit 通关核对与下一步

| SOP 验收项 | 本轮状态 | 证据 |
|---|---|---|
| 概念、Shape、单位、公式 | PASS | 反投影、Radius、DBSCAN、AABB、逐点标签解释 |
| 参数理解与主动修改 | PASS | `min_neighbors`、`dbscan_min_points`、门禁 `min_points` |
| 先预测再验证 | PASS | E1～E3 与两次状态 Debug；门禁初次运行前的独立预测未留记录，未追认 |
| 最小工程实现 | PASS | 正式 Pipeline、DBSCAN 迁移、兼容入口 |
| 独立 Debug | PASS | 两次 RED→GREEN 与旧测试契约更新 |
| 综合 Boss | 核心 PASS | A～D；精确错误码与异常类型已纠正 |
| 全项目回归 | PASS | `703 passed, 106 subtests passed` |
| Markdown 学习笔记 | **本稿已生成，待保存到仓库核验** | `experiments/OS01_B_rgbd_pipeline.md` |
| README 更新 | 待核验 | 尚无改动证据 |
| Git commit | **未完成** | 未提供新 commit 证据 |
| GitHub push | **未完成** | 未提供远端同步证据 |

**当前结论：OS-01-B 已完成核心学习与工程验收，但因笔记落盘 / README / Git / GitHub 闭环尚未验证，状态保持 `IN PROGRESS`；Level 2 整体为 `CONTINUE`。**

**下一步：**将本文件放入项目 `experiments/`，核对真实仓库文件结构和 README；检查 `git diff --check` 与暂存差异，逐文件提交并推送；确认 `HEAD`、`master` 与 `origin/master` 一致后，再更新 OS-01-B 为 `COMPLETE`，进入 OS-01-C 的可追溯多样本评测。
