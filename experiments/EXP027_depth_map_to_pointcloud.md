# EXP027 — Depth Map to Point Cloud

> Project：`Project_002_RGBD_Object_Localizer`
> Level：Level 2 · 3D Vision & Geometry
> Learning Unit：EXP027 · Depth Map → Point Cloud
> 前置单元：EXP026 · Camera Intrinsics & Pixel to 3D（COMPLETE）
> 记录日期：2026-10-08
> 当前状态：**EXP027 CONTINUE**（独立通关测试及 Git/GitHub 闭环尚未完成）

## 1. 学习目标与完成标准

本单元把 EXP026 的**单个像素反投影**扩展到**整张深度图生成点云**，形成可重复使用的相机几何模块。

必须能够：解释相机内参与 Z-depth；手算多个像素的三维坐标；处理无效深度；保证像素和点云的行优先对应关系；分别实现循环版 V0 和向量化版 V1；用独立参考、参数实验和故障注入验证正确性；实测时间及内存；用 Open3D 构建、检查并显示彩色点云；完成自动化测试、笔记、独立验收、Git commit 和 GitHub push。

**判定原则：程序运行成功 ≠ 几何正确 ≠ 学习通关。**

## 2. 知识地图与工程意义

```text
EXP026：Pixel (u,v) + Z-depth + K → Camera XYZ
                           ↓
EXP027：Depth Map [H,W] → Valid Pixels → Point Cloud [N,3]
                           ↓
后续：Target Mask + Depth → Target Point Cloud
                           ↓
点云过滤 / 平面去除 / 聚类 → 3D Center / Pose Clues
```

EXP027 的本质不是引入新投影公式，而是把单点几何拓展为批量数据处理，并验证 **Shape、单位、数值、顺序、颜色及性能**。

## 3. 输入、输出与 API Contract

| 项目 | 定义 | 单位 / dtype / Shape |
|---|---|---|
| `depth_m` | 每个像素的相机 Z-depth | m；转换后为 `float64`；`[H,W]` |
| `u` | 图像列索引，向右增大 | pixel，`0…W-1` |
| `v` | 图像行索引，向下增大 | pixel，`0…H-1` |
| `fx, fy` | 水平和垂直焦距 | pixel，正且有限 |
| `cx, cy` | 主点坐标 | pixel，有限 |
| `valid` | 有效深度掩码 | bool；`[H,W]` |
| `points` | 相机坐标系 XYZ 点集 | m；`float64`；`[N,3]` |
| `rgb` | 与深度逐像素对齐的颜色图 | `uint8`；`[H,W,3]` |
| `colors` | 与 `points` 一一对应的颜色 | 浮点 `[0,1]`；`[N,3]` |

约定：相机坐标系 **X 向右、Y 向下、Z 向前**；这里的 Z 是光轴方向深度，**不是**从相机中心到点的欧氏距离。`depth_m[v,u]` 的下标顺序是 `[row,col]`，而像素通常记为 `(u,v)`。输出仅保留有效点，按行优先顺序排列。

输入必须是非空二维深度数组；`fx/fy` 必须为有限正数，`cx/cy` 必须有限。纯几何层不负责猜测毫米与米的单位、不自动做传感器标定，也不按业务场景推测距离合理性。

## 4. 最小理论与核心公式

相机内参矩阵：

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

像素 `(u,v)` 与相机三维点 `(X,Y,Z)` 的反投影：

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth_m[v, u]

P_camera = Z * K^(-1) * [u, v, 1]^T
```

`K^(-1)[u,v,1]^T` 是归一化成 **z 分量为 1** 的视线方向向量，并不一定是单位长度向量。若像素和内参固定，Z 翻倍，则 XYZ 全部翻倍；但 `X/Z`、`Y/Z` 不变。

参数变化规律：

- `cx` 减小 `Δ`：每个点的 `X` 增加 `ΔZ/fx`，不同 Z 的绝对增量可能不同；Y、Z 不变。
- `fx` 增大：在 `u≠cx` 且 Z 固定时，`|X|` 变小；`X=0` 的主点列不受影响。
- `fy` 增大：在 `v≠cy` 且 Z 固定时，`|Y|` 变小。
- `Z` 增大：同一像素的 XYZ 沿同一条相机射线成比例扩大。

## 5. 从 Depth Map 到点云的流程

```text
Depth Map [H,W]
  ├─ np.indices → v_grid [H,W], u_grid [H,W]
  ├─ valid = isfinite(depth_m) & (depth_m > 0)
  ├─ u_grid[valid] → u [N]
  ├─ v_grid[valid] → v [N]
  └─ depth_m[valid] → z [N]
        ↓
   x = (u-cx)*z/fx
   y = (v-cy)*z/fy
        ↓
   column_stack(x,y,z) → points [N,3]
```

无效深度包括 `0`、负数、`NaN`、`+Inf`、`-Inf`。仅用 `depth_m > 0` 不够，因为 `+Inf > 0` 为真。全无效时应返回 `(0,3)` 而非 `(0,)`，保持下游的三维坐标结构契约。

**Organized Point Cloud** 通常保留像素网格结构 `[H,W,3]`；当前输出为过滤无效点后的 **Unorganized Point Cloud** `[N,3]`。从当前 `[N,3]` 不能单凭点下标恢复原始像素位置，除非同时保存有效 Mask 或像素索引。

## 6. V0：循环实现及代码拆解

文件：`src/geometry/depth_to_pointcloud.py`。

核心逻辑：

```python
points = []
for v in range(height):
    for u in range(width):
        z = float(depth_m[v, u])
        if not np.isfinite(z) or z <= 0:
            continue
        point = pixel_to_camera(
            u=float(u), v=float(v), depth_m=z,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )
        points.append(point)
return np.asarray(points, dtype=np.float64).reshape(-1, 3)
```

V0 的价值：代码直接对应数学公式，能复用 EXP026 的 `pixel_to_camera()`，适合作为可阅读、可调试的参考实现。代价：每个有效像素都要执行 Python 层循环和函数调用，还会累积 Python 对象。

`for v` 在外、`for u` 在内，是为了保持行优先顺序；`.reshape(-1,3)` 保证全无效输入也保持二维三列输出。

## 7. V1：NumPy 向量化实现及代码拆解

文件：`src/geometry/depth_to_pointcloud_vectorized.py`。

```python
v_grid, u_grid = np.indices(depth_m.shape)
valid = np.isfinite(depth_m) & (depth_m > 0)

u = u_grid[valid].astype(np.float64)
v = v_grid[valid].astype(np.float64)
z = depth_m[valid]

x = (u - cx) * z / fx
y = (v - cy) * z / fy
points = np.column_stack((x, y, z))
```

V1 与 V0 使用相同的输入验证与几何公式。`np.indices()` 第一个输出是行网格 `v_grid`，第二个是列网格 `u_grid`。布尔索引按 C 顺序，即行优先，筛出的 `u/v/z` 保持彼此一一对应。当前 `u_grid/v_grid` 是整数网格，`u/v` 经过 `.astype(np.float64)` 是浮点向量。

**复杂度**：V0、V1 均为 `O(HW)`。V1 不会跳过处理深度图的基本工作量，而是减少逐像素 Python 开销；同时付出网格、Mask 和中间数组的分配成本。

## 8. 向量化前的独立手算参考

输入：

```text
depth_m = [[1.0, 0.0, 3.0],
           [2.0, NaN, 4.0]]
fx=2, fy=2, cx=1, cy=0
```

对应：

```text
v_grid = [[0,0,0], [1,1,1]]
u_grid = [[0,1,2], [0,1,2]]
valid  = [[T,F,T], [T,F,T]]

u = [0,2,0,2]
v = [0,0,1,1]
z = [1,3,2,4]

points = [[-0.5, 0.0, 1.0],
          [ 1.5, 0.0, 3.0],
          [-1.0, 1.0, 2.0],
          [ 2.0, 2.0, 4.0]]
shape = (4,3)
```

这份参考由像素位置和内参独立计算，不由 V0 或 V1 生成，能避免两份实现犯相同错误却彼此验证通过的问题。

## 9. 预测、纠错与主动修改记录

### 9.1 初始预测纠错

首次 5 题预测为 **2/5 正确**，问题是把像素坐标直接当作相机三维坐标，忽略主点偏移和焦距缩放。针对性补测 Q6–Q8 **3/3 PASS**：

```text
fx=4, fy=2, cx=2, cy=1, (u,v)=(1,2), Z=2
→ (-0.5,1.0,2.0)m
Z 改成 4 → (-1.0,2.0,4.0)m
```

纠错结论：像素坐标不等于三维坐标；先减主点，再按焦距归一化，最后乘以 Z-depth。

### 9.2 实验 A：无效深度变有效

原始 `2×3` 深度图的 `(u,v)=(2,0)` 从 `Z=0` 改成 `3.0m`。

- 预测：输出从 `(5,3)` 变成 `(6,3)`；新增点 `(1.5,0,3)`；位于索引 2。
- 实测：预测完全一致，**Case A PASS**。

### 9.3 实验 B：混合无效深度

```text
[[NaN, Inf, -1],
 [  0,   2,  3]]
```

- 预测：仅 2 个有效点，`(0,1,2)` 和 `(1.5,1.5,3)`；Shape `(2,3)`。
- 实测：预测完全一致，**Case B PASS**。

### 9.4 实验 C：修改主点 cx

把 `cx: 1 → 0`，其他量不变：

```text
ΔX = (cx_old - cx_new) * Z / fx = Z/2
ΔY = 0, ΔZ = 0
```

- 预测：点数不变；Z=1m 的点 X 增加 0.5m，Z=2m 的点 X 增加 1m。
- 实测：Delta X=`[0.5,0.5,1,1,1]`，Y/Z 均不变，**Case C PASS**。

主动修改实现与预测验证：**3/3 PASS**。脚本：`src/experiments/exp027_depth_parameter_behavior.py`。

## 10. V0 / V1 正确性对比

脚本：`src/experiments/exp027_vectorization_comparison.py`。

| 场景 | 输入 Shape | 输出 Shape | 结果 |
|---|---|---|---|
| 独立手算参考 | `(2,3)` | `(4,3)` | PASS |
| 混合无效深度 | `(2,3)` | `(2,3)` | PASS |
| 全无效深度 | `(2,3)` | `(0,3)` | PASS |
| 单行输入 | `(1,4)` | `(3,3)` | PASS |
| 单列输入 | `(4,1)` | `(3,3)` | PASS |
| 固定种子随机图 | `(7,11)` | `(73,3)` | PASS |

V0/V1：**6/6 PASS**。逐点比较采用 `np.testing.assert_allclose(..., rtol=0, atol=1e-12)`；形状、数值及行优先顺序都在这些测试输入上保持一致。注意这不等于证明了所有可能输入的正确性。

## 11. 性能 Benchmark

脚本：`src/experiments/exp027_vectorization_benchmark.py`；使用预热、多次运行、交替先后顺序、`perf_counter_ns()` 计时及中位数。

| 分辨率 | 总像素 | 有效点 | 重复 | V0 中位数 ms | V1 中位数 ms | V0/V1 |
|---|---:|---:|---:|---:|---:|---:|
| 64×64 | 4,096 | 3,700 | 7 | 8.9002 | 0.0963 | **92.42×** |
| 240×320 | 76,800 | 69,144 | 5 | 170.2374 | 2.6743 | **63.66×** |
| 480×640 | 307,200 | 276,696 | 3 | 689.8255 | 11.5339 | **59.81×** |

公式：`speedup = median(T_v0) / median(T_v1)`；当大图像素为中图 4 倍时，V0 时间约 4.05 倍，V1 时间约 4.31 倍，结果与两者均为 `O(HW)` 的理论一致。小图到大图的加速比下降，可能与分配、缓存和内存流量等因素有关；**本实验并未直接测定具体瓶颈**。

结论仅适用于当前环境、实现与输入。VGA 图像的 11.5339ms 是**点云转换函数**的时间，不是采集、配准、分割、过滤在内的 RGB-D 端到端时延。

CSV：`results/exp027/vectorization_benchmark.csv`。

## 12. Memory Profiling

脚本：`src/experiments/exp027_memory_profile.py`；相同输入下两种实现分别运行，`tracemalloc` 跟踪峰值及返回时仍被跟踪的分配，3 次取中位数。大图由小图每像素扩成 `2×2`，有效比例保持不变。

| 分辨率 | 有效点 | V0 Peak MiB | V1 Peak MiB | V0 Retained MiB | V1 Retained MiB | Peak V0/V1 |
|---|---:|---:|---:|---:|---:|---:|
| 240×320 | 69,151 | 13.199 | 5.468 | 1.708 | 1.584 | 2.41× |
| 480×640 | 276,604 | 52.854 | 21.865 | 6.456 | 6.332 | 2.42× |

内存实验使用单独生成的输入，**有效点数不同于 Benchmark 脚本**，不能误当成同一批数据。两组的有效点数恰好相差 4 倍，峰值近似增长 4 倍。

解释：V0 使用 Python 容器收集逐点结果，可能保留大量 Python 对象，并在转为 NumPy 时与新数组同时存在；V1 尽管有多个网格与中间数组，在本次实测中仍有更低的**跟踪峰值**。不能据此推断所有 V0 实现都比 V1 占内存，更不能把 `tracemalloc` 峰值直接称作进程峰值 RSS。`Retained at return` 不是所有临时对象的大小，它与峰值差异反映了运行时部分分配已被释放。

综合本次性能和内存数据：**V1 适合作为当前主实现，V0 保留为可理解的参考实现。**

## 13. u/v Fault Injection

实验文件：`src/experiments/exp027_uv_fault_injection.py`。

正确赋值：

```python
v_grid, u_grid = np.indices(depth_m.shape)
```

人为错误：

```python
u_grid, v_grid = np.indices(depth_m.shape)
```

针对 `(u,v)=(2,0)`、`Z=3`、`fx=fy=2`、`cx=1,cy=0`：

```text
正确：( 1.5, 0.0, 3.0)
错误：(-1.5, 3.0, 3.0)
```

错误实现仍然满足 `shape=(4,3)`、Z 有效、全部坐标有限，因此这些结构检查会**误放行**。独立 XYZ 参考断言成功检测错误：`Independent XYZ check: FAULT DETECTED`；故障注入实验整体 **PASS**。

还发现 `(u,v)=(0,0)` 在对调后坐标可能不变，说明仅测试左上角无法保证发现该错误。应选择不对称位置并测试已知坐标与方向关系；平面约束和三维可视化可作辅助，但不能独立保证发现 u/v 对调。

**可迁移原则：Shape Correct ≠ Numeric Correct ≠ Geometry Correct。**

## 14. RGB-D 与 Open3D 彩色点云

新 Conda 环境：`D:\conda_envs\ai_3d`；已验证 `NumPy 2.2.6`、`Open3D 0.20.0`、`pip check` 无依赖冲突；在新环境中 V0/V1 6 项比较均通过。

实验文件：`src/experiments/exp027_open3d_visualization.py`。

合成场景：

```python
depth_m = np.full((120, 160), 2.0)
depth_m[40:80, 60:100] = 1.2
depth_m[10:20, 20:30] = 0.0
```

RGB 与 Depth 已假定逐像素对齐；背景为灰色 `[150,160,175]`，目标为橙色 `[245,135,50]`，颜色进入 Open3D 前除以 255，转为 `[0,1]` 浮点值。

```python
valid = np.isfinite(depth_m) & (depth_m > 0)
points = depth_to_pointcloud_vectorized(depth_m, **params)
colors = rgb[valid].astype(np.float64) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
```

关键数值：

| 项目 | 结果 |
|---|---|
| `depth_m.shape` | `(120,160)` |
| 总像素 | 19,200 |
| 无效点 | 100 |
| `points.shape`、`colors.shape` | `(19100,3)` |
| 橙色目标 | 1,600 点；Z=1.2m |
| 灰色背景 | 17,500 点；Z=2.0m |
| Open3D PointCloud | PASS |
| PLY 保存 | PASS |
| 三维窗口 | 成功打开，已观察 |

截图观察：橙色目标表面位于灰色背景前方；灰色背景中的较大缺口对应被近处目标取代的像素，不表示数据丢失；另一个较小的缺口来自无效深度区域。两组点是不同深度的**表面点集**，不等于一个具有体积的实心物体。

注意：**相同有效 Mask、相同行优先顺序、相同 RGB/Depth 像素坐标系** 是颜色正确配对的前提。仅仅 `points.shape[0] == colors.shape[0]` 不能证明对齐。当前实验使用合成且已对齐数据，不代表真实深度相机的彩色图与深度图天然配准。

输出：`results/exp027/synthetic_rgbd_colored.ply`，本机已保存并显示；`.ply` 通常受现有 `.gitignore` 排除，需通过脚本复现。

## 15. Contract Tests 与全量回归

新增：`tests/test_depth_to_pointcloud.py`，共 **12 项测试**；EXP026 `tests/test_pixel_to_camera.py` 原有 **7 项测试**。

新增的 12 项测试覆盖：

1. `test_independent_xyz_reference`
2. `test_invalid_depth_filtering`
3. `test_all_invalid_returns_zero_by_three`
4. `test_single_row_and_column`
5. `test_row_major_depth_order`
6. `test_reject_non_2d_input`
7. `test_reject_empty_depth_map`
8. `test_reject_invalid_focal_lengths`
9. `test_reject_nonfinite_intrinsics`
10. `test_cx_shift_depends_on_depth`
11. `test_accept_2d_python_list`
12. `test_v0_v1_random_consistency`

运行结果：

```text
Ran 19 tests in 0.125s
OK
```

同时，新增测试文件与相关实验脚本 `py_compile` 通过。这个结论表示**当前已覆盖条件下未发现回归**，不是对任意输入和未来修改的绝对保证。

## 16. 易错点与排错规则

| 易错点 | 错误后果 | 如何验证 / 修复 |
|---|---|---|
| 把像素 `(u,v)` 直接当 XYZ | 三维坐标数值错误 | 独立手算主点偏移与焦距缩放 |
| `depth_m[u,v]` 与 `[v,u]` 搞反 | 点深度错位或越界 | 选非方形输入与已知像素 |
| 交换 `u_grid/v_grid` | Shape 正确但空间几何错误 | 非对称点的独立 XYZ 断言 |
| 只用 `z>0` | `Inf` 被错误保留 | `np.isfinite(z) & (z>0)` |
| 无效深度生成零深度点 | 污染点云和统计 | Mask 过滤及有效点数断言 |
| 空点云变 `(0,)` | 下游 `[N,3]` 契约破坏 | 输出 `.reshape(-1,3)` 或 `column_stack` |
| 假定 Z 是欧氏距离 | 点坐标缩放或物理解释错误 | 区分 Z-depth 与 ray length |
| RGB 和点坐标筛选顺序不同 | 颜色贴到错误的三维点 | 同一个 Mask、同一行优先规则 |
| 只检查 V0 == V1 | 可能两者同错 | 加入独立手算参考 |
| 从视觉截图认定几何正确 | 错误可能看起来合理 | 数值断言与几何约束 |
| 认为向量化一定是 O(1) | 错误理解复杂度 | 基于像素数分析 O(HW) |
| 把追踪峰值视为进程 RSS | 错报内存指标 | 明确工具和测量口径 |

## 17. 可复现命令

在项目根目录 `D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer` 执行；Open3D 实验使用 `ai_3d` 环境：

```powershell
conda activate D:\conda_envs\ai_3d

python -m src.experiments.exp027_depth_map_to_pointcloud_minimal
python -m src.experiments.exp027_depth_parameter_behavior
python -m src.experiments.exp027_vectorization_comparison
python -m src.experiments.exp027_vectorization_benchmark
python -m src.experiments.exp027_uv_fault_injection
python -m src.experiments.exp027_memory_profile
python -m src.experiments.exp027_open3d_visualization
python -m src.experiments.exp027_open3d_visualization --show

python -m unittest discover -s tests -p "test_*.py" -v
python -m py_compile tests\test_depth_to_pointcloud.py
```

环境自检：

```powershell
python -c "import numpy as np; import open3d as o3d; print(np.__version__, o3d.__version__)"
python -m pip check
```

运行 Benchmark 时不保证不同机器得到相同毫秒数。可视化 `--show` 需要可用图形环境。

## 18. 个人理解复盘参考稿（尚需独立确认）

> 本节为学习整理的**参考答案**，不是已经完成的独立通关证据；独立验收仍需本人脱离笔记复述或实作。

### Explain：为什么 Depth Map 能生成 Point Cloud？

Depth Map 的每个像素位置 `(u,v)` 在已知相机内参时确定一条相机射线，Z-depth 给出该射线上的一个三维点。对所有有效像素执行同一反投影并按顺序收集，就得到 `[N,3]` 点云。

### Shape：为什么 `depth_m[v,u]` 不能写成 `[u,v]`？

NumPy 二维图像先行后列，即 `[row,column]`。图像坐标 `(u,v)` 则先列后行；二者顺序不同。非方形图像下交换还可能导致越界，更危险的是坐标未越界却悄悄取错位置。

### Vectorization：为什么 V1 更快但仍是 O(HW)？

V1 把逐元素操作交给 NumPy 底层批量实现，减少 Python 循环、函数调用和对象构建开销；但仍需扫描深度图、形成有效 Mask 并处理有效像素，因此时间复杂度不变。

### Memory：为什么当前 V1 的跟踪峰值更低？

本次 V0 使用 Python 对象列表收集大量点，再复制为 NumPy 数组，可能在峰值时同时保留原对象和结果数组。V1 虽有多个中间数组，但数据较紧凑，因此实测追踪峰值低于当前 V0；不同 V0 写法或测量工具可能改变结果。

### Debug：为什么 u/v 对调后仍能运行？

`u_grid`、`v_grid` 都是同样的 `[H,W]` Shape，交换后数据类型、输出行数和 Z 列仍合法；X/Y 的意义却已改变。应使用独立已知 XYZ 的非对称像素检测，而不能仅检查 Shape 和有限性。

### Transfer：怎样生成只属于目标的点云？

在真实 RGB 与深度已配准且 Mask 同尺寸的条件下，建立 `select = valid_depth & target_mask`，仅对选中的 `(u,v,Z)` 做反投影，并用同一 `select` 选出颜色。这是后续 RGB-D Object Localizer 的目标点云输入。需要额外处理 Mask 质量、遮挡、深度噪声以及配准误差。

## 19. 已知局限与后续问题

- 仅使用合成 RGB-D 数据，未引入真实相机深度标定、畸变校正和 RGB/Depth 配准误差。
- 尚无目标级 Mask 点云提取模块，未实现离群点滤波、背景平面去除、聚类、三维质心或 6D Pose。
- 性能数据不是完整感知流水线延迟；内存数据为追踪分配，不是峰值 RSS。
- 大型点云的生成、存储、可视化与精度在不同设备上的表现尚需验证。
- 当前点云是相机坐标系结果，尚未进行外参变换到机器人基座或世界坐标系。

## 20. 文件清单与版本管理注意事项

**几何代码：**

```text
src/geometry/depth_to_pointcloud.py
src/geometry/depth_to_pointcloud_vectorized.py
```

**实验与测试：**

```text
src/experiments/exp027_depth_map_to_pointcloud_minimal.py
src/experiments/exp027_depth_parameter_behavior.py
src/experiments/exp027_vectorization_comparison.py
src/experiments/exp027_vectorization_benchmark.py
src/experiments/exp027_uv_fault_injection.py
src/experiments/exp027_memory_profile.py
src/experiments/exp027_open3d_visualization.py
tests/test_depth_to_pointcloud.py
```

**实验结果：**

```text
results/exp027/vectorization_benchmark.csv
results/exp027/synthetic_rgbd_colored.ply
```

此前 `git status --short` 显示 EXP027 源文件、测试文件、结果目录及根目录的 `ScreenCamera_*.json` / `ScreenCapture_*.png` 处于未跟踪状态。最终提交之前应明确挑选源码、测试、笔记、必要 CSV 和可复现说明；不要盲目 `git add .`。根目录截图与辅助 JSON 应先检查用途，再决定保留、迁移或排除。PLY 现有忽略规则通常排除，不必强行提交大文件。

## 21. 当前验收记录与下一步

| 项目 | 状态 |
|---|---|
| EXP026 前置知识 | COMPLETE |
| 最小理论及公式 | PASS |
| V0 参考实现 | PASS |
| 参数预测与主动修改 | PASS |
| V1 向量化与 6 项对比 | PASS |
| 性能 Benchmark | PASS |
| Memory Profiling | PASS |
| u/v Fault Injection | PASS |
| Open3D 颜色、几何、窗口及 PLY | PASS |
| 新增契约测试 | 12/12 PASS |
| 含 EXP026 在内的回归测试 | 19/19 PASS |
Independent Acceptance Q1-Q10: PASS
Minimal Experiment: PASS
Full Regression: 19/19 PASS
Experiment Notes: COMPLETE
Git Commit: LOCAL DONE
GitHub Push: PENDING
Remote Verification: PENDING

EXP027: CONTINUE
Level 2: CONTINUE

**当前单元：EXP027 CONTINUE；当前 Level：Level 2 CONTINUE。**

下一步：独立复述第 18 节关键理解，完成包含概念、推理、实战、Debug、迁移的正式通关测试；必要时补弱项；最后进行全量回归、精确 Git 暂存、commit、push、远端核验与最终 Closure Record。只有完成这些，才能将 EXP027 标为 `COMPLETE`。
