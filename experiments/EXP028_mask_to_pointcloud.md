# EXP028 — Target Mask → 3D Object Point Cloud（完整学习与实验笔记）

> **项目**：`Project_002_RGBD_Object_Localizer`
> **Level**：Level 2 · 3D Vision & Geometry
> **Learning Unit**：EXP028 · Target Mask → 3D Object Point Cloud
> **前置**：EXP026 Camera Intrinsics & Pixel to 3D（COMPLETE）；EXP027 Depth Map → Point Cloud（COMPLETE）
> **笔记整理日期**：2026-10-09（根据本 Project 已提供的实验日志与独立答题记录）
> **当前状态**：EXP028 最终封关。理论、实验和独立验收已通过；源码已同步 GitHub。本文档远端核验后正式 COMPLETE。

> 记录说明：本笔记区分 **已提供的终端实测**、**已完成的独立预测/答题** 与 **尚未验证的事项**。示例代码是原理性复盘或已给出的教学实现，不把未提供日志的运行写成 PASS。本文综合整理 SOP 中的原理、参数、实验、源码阅读、预测验证、Debug、独立验收与 Git 工作流。

---

## 1. 本关为什么要学

EXP027 能将整张深度图转成全部有效点云。但是，一帧 RGB-D 图像里通常同时包含背景、桌面以及多个物体。工业视觉定位需要的往往不是“所有点”，而是**某一个目标物体所对应的点**。

因此 EXP028 的任务是：已知 RGB、Depth、目标分割 Mask，把**目标区域且深度有效**的像素反投影为相机坐标系下的彩色三维点云，同时保留像素来源，以便以后进行 3D 中心估计、离群点过滤、聚类和多帧对应。

### 1.1 与已学知识的连接

```text
EXP026：单像素 (u,v,Z) + Camera Intrinsics → (X,Y,Z)
                    ↓  扩展到全部有效像素
EXP027：Depth [H,W] → Valid Depth → Scene Points [N,3]
                    ↓  增加目标像素筛选及同步 RGB
EXP028：RGB + Depth + Target Mask → Target Points + Colors + Select Mask
                    ↓  后续（尚未在 EXP028 实现）
点云去噪 / 聚类 / 平面过滤 → 鲁棒 3D Center → 物体定位与姿态线索
```

**本关不是学习一条新反投影公式，而是学习正确的数据筛选、像素对应、接口契约与故障验证。**

### 1.2 当前学习边界

- **必须掌握**：Mask 交集、Depth 有效性、`v/u` 索引、XYZ、RGB 逐点对齐、输入异常与空目标。
- **建议掌握**：可见表面和几何中心的差异、行索引变化、独立手算与故障注入。
- **知道存在即可**：深度图与 RGB 的实际配准、3D 点云的统计滤波/聚类。
- **本关不实现**：真实分割模型、相机外参标定、完整 3D Center Estimator、6D Pose、ICP、ROS2。

---

## 2. 可验收学习目标、产物与状态

| 目标/交付物 | 验收要求 | 当前记录 |
|---|---|---|
| 解释 Mask 筛选 | 区分“是目标”与“深度有效” | PASS |
| 批量反投影 | 独立手算及实现 `[N,3]` 点云 | PASS |
| 颜色对齐 | 用不同 RGB 验证点和颜色同序 | PASS |
| 主动修改 | A/B/C 每次先预测后验证 | PASS |
| 核心 API | 输入 Shape、dtype、内参与空目标行为明确 | PASS（已做 Smoke + 单元测试） |
| 自动化回归 | EXP026 7 + EXP027 12 + EXP028 11 | **30/30 PASS**（已提供日志） |
| Open3D | 1500 个目标点，颜色正确，PLY 导出与读回 | PASS（已提供日志与截图） |
| 个人理解/独立验收 | Explain / Shape / Calculate / Predict / Implement / Debug / Transfer | **Q1～Q7 PASS**（回答及批改已完成） |
| Markdown 笔记 | 20 节完整学习记录 | PASS（已保存并核验） |
| Git commit、push、远端核验 | 源码与笔记版本管理 | 源码 PASS；封关文档待核验 |

学习闭环状态：知识与工程实验已通过；最终封关文档远端同步后 EXP028 COMPLETE。

---

## 3. 输入、输出和数据契约（Shape First）

生产版函数位置：`src/geometry/mask_to_pointcloud.py`。

```python
def mask_to_pointcloud(depth_m, rgb, target_mask, fx, fy, cx, cy):
    # returns points, colors, select_mask
    ...
```

| 名称 | 含义 | Shape | dtype / 单位 | 约束 |
|---|---|---|---|---|
| `depth_m` | 相机 Z-depth | `[H,W]` | 可转换为 `float64`，米 | 非空二维 |
| `rgb` | 与深度**已配准**的颜色图 | `[H,W,3]` | `uint8`，通道 RGB | 高宽匹配；非 BGR |
| `target_mask` | 目标的像素选择图 | `[H,W]` | `bool` | 仅 True 属于目标 |
| `fx, fy` | 焦距 | 标量 | pixel | 正且有限 |
| `cx, cy` | 主点 | 标量 | pixel | 有限 |
| `valid_depth` | 有效深度图 | `[H,W]` | `bool` | `isfinite & >0` |
| `select_mask` | **目标且深度有效** | `[H,W]` | `bool` | 对应最后真正参与反投影的像素 |
| `u` / `v` | 列坐标 / 行坐标 | `[N]` / `[N]` | 像素索引 | `np.nonzero` 返回 `v,u` |
| `z` | 有效目标的 Z-depth | `[N]` | `float64`，米 | 与 `u/v` 顺序一致 |
| `points` | 目标彩色点云的几何部分 | `[N,3]` | `float64`，米 | 列固定为 X、Y、Z |
| `colors` | 逐点颜色 | `[N,3]` | `float64`，`[0,1]` | 列固定为 R、G、B |

**N 不等于目标 Mask 的 True 总数**，而是：

```text
N = count_nonzero(target_mask & isfinite(depth_m) & (depth_m > 0))
```

**两种必须区分的“空”**：

1. `depth_m.shape=(0,3)` 或 `(3,0)`：输入数组本身不合法，应 `ValueError`。
2. `depth_m.shape=(2,3)` 合法，但目标全无有效深度：应正常返回 `points.shape=(0,3)`、`colors.shape=(0,3)`，`select_mask` 全 False。

此外：**相同 Shape ≠ RGB 与 Depth 完成空间配准**。不同镜头可能具有不同内参/外参，真正工程接入时需要确认配准后的像素确实指向同一场景位置。

---

## 4. 最小理论：射线、Z-depth 与反投影

假设采用针孔相机模型，且深度是相机光轴方向的 **Z-depth（米）**，不是斜距。相机内参：

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

像素 `(u,v)` 反投影：

\[
X = \frac{(u-c_x)Z}{f_x},\qquad
Y = \frac{(v-c_y)Z}{f_y},\qquad
Z = D[v,u]
\]

- `u`：列，向右增长；`v`：行，向下增长。
- `depth_m[v,u]`：数组写法先行后列；`(u,v)`：像素坐标通常先列后行。
- 本项目相机坐标系约定：**X 向右，Y 向下，Z 向前**。
- 同一像素的 `u/v` 和相机内参固定，Z 增大时，XYZ 沿**同一条相机射线**成比例变化。
- 仅有 `(u,v)` 与内参只能确定射线方向；还需要 Z-depth 才能确定射线上的一个三维点。

### 4.1 为什么只加 Mask 就能得到“目标点云”

在 EXP027：对所有有效深度像素反投影。EXP028：先用 `target_mask` 排除非目标像素，再对剩余有效目标像素使用**完全相同的公式**。变化发生在**参与计算的像素集合**，不是投影数学本身。

```python
valid_depth = np.isfinite(depth_m) & (depth_m > 0)
select_mask = target_mask & valid_depth
```

- **只用 `target_mask`**：可能保留 Z=0、NaN、Inf、负数，污染点云。
- **只用 `valid_depth`**：会引入背景、桌面、其他物体。
- 两者取交集：只保留“属于目标且可进行反投影”的像素。

---

## 5. 核心算法总流程

```text
Depth [H,W] ────── isfinite & depth>0 ──────┐
                                            ├─ AND → select_mask [H,W]
Target Mask [H,W] ──────────────────────────┘
                                               │
                   ┌───────────────────────────┴──────────────────────────┐
                   │                                                      │
       np.nonzero(select_mask)                                   RGB[select_mask]
         v [N], u [N]                                            colors_u8 [N,3]
       depth[select_mask]                                                │
         z [N]                                             astype(float64) / 255
                   │                                                      │
     x=(u-cx)*z/fx; y=(v-cy)*z/fy                                colors [N,3]
                   │                                                      │
         column_stack((x,y,z))                                           │
            points [N,3] ─────────────────── 一一对应 ─────────────────────┘
                   │
          + select_mask [H,W]
                   ↓
     points, colors, select_mask
```

这里有一个重要的不变量（invariant）：

```text
points[i] 与 colors[i] 源于同一个 select_mask 中选中的像素。
```

只有对两者使用同一布尔 Mask 且同一排序，才能保证这个不变量。

---

## 6. NumPy 核心 API 与逐行代码拆解

### 6.1 `np.asarray(depth_m, dtype=np.float64)`

把输入转为统一的数值数组，便于用米进行浮点计算。`rgb` 不强制转换到 `uint8`，而是检查当前 `dtype` 是否恰当，避免悄悄改变原始颜色值。**注意**：若输入完全无法转成数值数组（例如某些字符串），底层可能在类型转换处抛异常，具体异常类型要单独测试，不能声称所有非法输入都自动返回同一个错误类型。

### 6.2 `np.isfinite(depth)` 与 `(depth > 0)`

```python
valid_depth = np.isfinite(depth) & (depth > 0)
```

- `isfinite`: 排除 `NaN`、`+Inf`、`-Inf`。
- `> 0`: 排除零和负深度。
- 单用 `depth > 0` 不够，因为 `+Inf > 0` 是 True。

### 6.3 `select_mask = mask & valid_depth`

逐元素逻辑与操作。注意这里的 `mask` 必须是 `bool`，不能把 `uint8` 的 0/1 不经检查就当作布尔 Mask：NumPy 的整数索引与布尔索引有不同语义。

### 6.4 `v, u = np.nonzero(select_mask)`

**输出次序是“行索引 v、列索引 u”**，不是 `(u,v)`：

```python
select_mask = np.array([
    [ True, False],
    [False,  True],
])
v, u = np.nonzero(select_mask)
# v = [0,1]
# u = [0,1]
```

对二维 Mask，`np.nonzero()` 按行优先次序给出所有 True 的坐标。返回的坐标是过滤后的 N 个像素的索引数组，每项对应一处原始像素。

### 6.5 `z = depth[select_mask]`

从深度图取出**完全同序**的有效目标深度。它与 `v/u` 的第 i 个元素一起表示同一个像素：

```text
(u[i],v[i],z[i]) 是一个目标像素的完整几何输入。
```

### 6.6 批量反投影

```python
x = (u.astype(np.float64) - cx) * z / fx
y = (v.astype(np.float64) - cy) * z / fy
```

向量化使 N 个像素一次完成数值运算。这里不是 N 次 Python `for` 循环，虽然仍需要处理 N 个点，时间复杂度仍随输入规模增长。

### 6.7 `np.column_stack((x,y,z))`

把三个 `[N]` 数组合成 `[N,3]`：

```python
points = np.column_stack((x, y, z))
# 每行顺序必须是 [X,Y,Z]
```

如果写成 `(z,y,x)`，程序通常仍能正常运行、Shape 也不变，但不再满足几何列语义。

**空目标**：`x/y/z` 为空一维数组时，`column_stack` 返回 `(0,3)`；不应把它变成 `(0,)`。

### 6.8 RGB 提取与归一化

```python
colors = image[select_mask].astype(np.float64) / 255.0
```

- 输入 `[H,W,3]`，筛选后 `[N,3]`。
- `uint8` 的 `[0,255]` 转为 Open3D 传统 PointCloud 使用的浮点 `[0,1]`。
- `image[select_mask]` 采用同一 Mask 和同一行优先顺序，与 points 一一对应。
- **RGB 与 OpenCV 常见 BGR 不同**。如果来源是 `cv2.imread()`，需先确认通道顺序并转换，而不是把 BGR 直接当 RGB。

### 6.9 `points[2]`、`points[1,2]`、`points[:,2]`

```text
points[2]   = 第 3 个点的完整 [X,Y,Z]（行索引从 0 开始）
points[1,2] = 第 2 个点的 Z 数值
points[:,2] = 全部点的 Z 向量
```

学习期间曾混淆“行”与“列”。经过针对性补测 3/3 PASS，后续 Q2 已能独立正确解释。

---

## 7. 生产版函数：接口设计与参考实现

已使用的核心模块：`src/geometry/mask_to_pointcloud.py`。下面记录与已通过 Smoke/Contract 测试的教学版本一致的核心实现结构；以本地 Git 工作区文件为最终源码事实来源。

```python
import numpy as np


def mask_to_pointcloud(depth_m, rgb, target_mask, fx, fy, cx, cy):
    depth = np.asarray(depth_m, dtype=np.float64)
    image = np.asarray(rgb)
    mask = np.asarray(target_mask)

    # 输入契约
    if depth.ndim != 2 or 0 in depth.shape:
        raise ValueError("depth_m must be non-empty 2D")
    h, w = depth.shape

    if image.shape != (h, w, 3):
        raise ValueError("rgb must have shape (H,W,3)")
    if image.dtype != np.uint8:
        raise ValueError("rgb must be uint8")
    if mask.shape != (h, w) or mask.dtype != np.bool_:
        raise ValueError("target_mask must be bool [H,W]")

    intrinsics = np.asarray([fx, fy, cx, cy], dtype=np.float64)
    if not np.isfinite(intrinsics).all():
        raise ValueError("intrinsics must be finite")
    fx, fy, cx, cy = intrinsics
    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive")

    # 先选择后投影
    valid_depth = np.isfinite(depth) & (depth > 0)
    select_mask = mask & valid_depth

    v, u = np.nonzero(select_mask)
    z = depth[select_mask]

    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    points = np.column_stack((x, y, z))

    # 与 points 共享同一 select_mask
    colors = image[select_mask].astype(np.float64) / 255.0
    return points, colors, select_mask
```

### 7.1 为什么要把检查放在计算前

- RGB 与 Depth Shape 不一致：相同 Mask 无法安全用于 RGB 索引。
- Mask 是 `uint8` 而不是 `bool`：存在整数索引误用风险。
- `fx/fy` 非正：除零、几何模型无效。
- `cx/cy`、`fx/fy` 非有限：坐标可能污染为 NaN/Inf。
- 深度数组维度为零：是非法输入，不能悄悄当作“未检测到目标”。

### 7.2 需要知道的 API 范围与局限

- 本函数**不判断物体类别**，只消费外部提供的目标 Mask。
- 本函数**不验证真实 RGB/Depth 配准是否正确**，仅验证 Shape。
- 本函数**不自动处理毫米到米**。如果传入毫米值却称为米，XYZ 也会被放大 1000 倍。
- 本函数**没有自动去掉目标 Mask 内的错误但正且有限的深度值**；它们仍会影响下游质心。
- RGB 这里约定为 `uint8` 的 RGB 通道顺序，而非任意范围/通道布局。

---

## 8. 初始最小实验：2×2 Mask → XYZ

实验：`src/experiments/exp028_mask_to_xyz_step1.py`。**已提供实际终端输出：PASS；py_compile 无报错。**

输入：

```python
depth_m = np.array([[1.0, 0.0],
                    [2.0, 3.0]])
target_mask = np.array([[True, True],
                        [False, True]])
fx = fy = 2.0
cx = cy = 0.0
```

预测与实测对照：

```text
select_mask = [[True, False],
               [False, True]]
v = [0,1]
u = [0,1]
z = [1.0,3.0]

points = [[0.0,0.0,1.0],
          [1.5,1.5,3.0]]
points.shape = (2,3)
```

日志结果：`EXP028 Mask to XYZ: PASS`。

独立解释：位于 `(u=0,v=1)` 的深度 2.0 虽然有效，却因目标 Mask 为 False 被排除；将 `column_stack((x,y,z))` 错写为 `(z,y,x)`，Shape 仍 `(2,3)`，但独立手算 XYZ 断言会失败。

另有 `src/experiments/exp028_mask_to_pointcloud_minimal.py` 的教学脚本，使用 3×4 深度图及彩色 Mask 构造 4 个目标点。**当前提供的终端记录尚未单独展示该脚本的运行结果，不能将该脚本的最终回归擅自写成已实测 PASS**。可以在工程封关前执行并保留日志。

---

## 9. 主动修改 A/B/C：先预测，再验证

SOP 规定核心实验原则上至少主动修改三次。EXP028 的 A/B/C 均有**独立预测和实际终端记录**。

### 9.1 Modification A：扩大 Target Mask

**修改前**：

```text
depth  = [[1,0],
          [2,3]]
mask   = [[T,T],
          [F,T]]
points = [[0,0,1],
          [1.5,1.5,3]]
```

**唯一修改**：`target_mask[1,0]=True`；相机内参 `fx=fy=2,cx=cy=0` 不变。

| 项目 | 先预测 | 实际 |
|---|---|---|
| `select_mask` | `[[T,F],[T,T]]` | 一致 |
| 点数 | 2→3，`(3,3)` | 一致 |
| 新增点 `(u=0,v=1,Z=2)` | `(0,1,2)` | 一致 |
| 新点行索引 | 1（第 2 行） | 一致 |
| 原有 XYZ | 不变 | 一致 |

```text
points_after = [[0,0,1],
                [0,1,2],
                [1.5,1.5,3]]
```

脚本：`src/experiments/exp028_active_modification_a.py`。日志：`EXP028 Modification A: PASS`；语法检查无报错。

**结论**：修改 Mask 只改变参与计算的像素集合，不会改变原来像素的内参/深度，因此原有点坐标不变；新点也**不保证排在末尾**。

### 9.2 Modification B：固定像素，改变目标深度

沿用 A 后的全 True Mask。修改右下角深度：`depth[1,1]:3→6m`；相机内参固定。

预测：

```text
P_before=(1.5,1.5,3.0)
P_after =(3.0,3.0,6.0)
Delta   =(1.5,1.5,3.0)
points.shape: (3,3) → (3,3)
```

实测（摘自用户终端）：

```text
Points before:
[[0.  0.  1. ]
 [0.  1.  2. ]
 [1.5 1.5 3. ]]
Points after:
[[0. 0. 1.]
 [0. 1. 2.]
 [3. 3. 6.]]
Delta:
[[0.  0.  0. ]
 [0.  0.  0. ]
 [1.5 1.5 3. ]]
EXP028 Modification B: PASS
```

**原因**：像素固定，`X/Z=0.5`、`Y/Z=0.5` 不变；深度翻倍，XYZ 同比例翻倍。这里点**沿射线移动**，并非仅沿 Z 轴平移。

### 9.3 Modification C：有效深度变成无效

修改前 `depth_before=[[1,0],[2,6]]`，Mask 全 True。唯一修改：`depth_after[1,0]=0`。

预测：

```text
mask_before=[[T,F],[T,T]]   → 3 points
mask_after =[[T,F],[F,T]]   → 2 points

points_before = [[0,0,1], [0,1,2], [3,3,6]]
points_after  = [[0,0,1], [3,3,6]]
```

实测：

```text
Points before:
[[0. 0. 1.]
 [0. 1. 2.]
 [3. 3. 6.]]
Points after:
[[0. 0. 1.]
 [3. 3. 6.]]
Expected broadcasting error detected:
operands could not be broadcast together with shapes (2,3) (3,3)
EXP028 Modification C: PASS
```

**核心工程结论**：过滤点后，`points[i]` 是压缩数组的行号，不是原始像素永久 ID。原来索引 2 的 `(3,3,6)` 点，现在索引变为 1。两次点数不同，直接相减会触发广播错误；**即使两次点数恰好相等，也不保证行号已经正确对应**。

两帧要逐点计算位移，需要先建立对应关系（例如同坐标系下共同有效像素、已校正的像素映射或三维匹配）；相机或物体运动时不能把“同一个 `(u,v)`”简单认作“同一个物理点”。

---

## 10. RGB 与 XYZ 独立对齐验证

使用不同颜色而不是同一灰色，避免颜色乱序仍蒙混过关。

```python
depth = np.array([[1.0,0.0],
                  [2.0,3.0]])
target_mask = np.array([[True,True],
                        [False,True]])
rgb = np.array([
    [[255,0,0], [0,255,0]],        # 红、绿
    [[0,0,255], [255,255,0]],      # 蓝、黄
], dtype=np.uint8)
```

选择条件：

```text
select_mask = [[True,False],
               [False,True]]
```

独立正确结果：

| 行号 | 原像素 `(u,v)` | XYZ（m） | RGB uint8 | RGB 浮点 |
|---|---|---|---|---|
| 0 | `(0,0)` | `(0,0,1)` | 红 `[255,0,0]` | `[1,0,0]` |
| 1 | `(1,1)` | `(1.5,1.5,3)` | 黄 `[255,255,0]` | `[1,1,0]` |

- 绿 `(1,0)`：Mask True，但深度 0，无效。
- 蓝 `(0,1)`：深度 2 有效，但 Mask False。

`test_distinct_rgb_colors_align_with_xyz` **已运行 PASS**；专项对齐测试和全量回归均有日志。它验证独立 XYZ、独立颜色和完整选择 Mask，不是仅比较 `points.shape == colors.shape`。

### 10.1 Debug 示例：颜色反序

错误：

```python
colors = rgb[select_mask][::-1].astype(np.float64) / 255.0
```

结果：

```text
colors_wrong   = [[1,1,0],[1,0,0]]
colors_correct = [[1,0,0],[1,1,0]]
```

两者 `Shape=(2,3)`，但颜色被颠倒。正确修复：

```python
colors = rgb[select_mask].astype(np.float64) / 255.0
expected_colors = np.array([[1,0,0],[1,1,0]], dtype=np.float64)
np.testing.assert_allclose(colors, expected_colors, rtol=0, atol=1e-12)
```

此 Bug 在**独立通关 Q6 中被分析并通过**；本项目已有使用不同颜色的自动测试能检测类似问题。不要把该教学性故障示例误记为已另行运行一个独立故障脚本。

---

## 11. API Contract：为什么合法输入不一定有点

### 11.1 正确输入但没有有效目标

```python
depth = np.zeros((2,2), dtype=np.float64)
mask = np.ones((2,2), dtype=bool)
# RGB 与内参符合契约
```

`select_mask` 全 False，返回：

```text
points.shape=(0,3); points.dtype=float64
colors.shape=(0,3); colors.dtype=float64
select_mask.shape=(2,2); select_mask.dtype=bool
```

也可以是所有 Mask 均 False，而 Depth 仍为有效数值；依然返回合法空目标。

### 11.2 非法输入应拒绝

| 情况 | 示例 | 预期行为 | 测试覆盖 |
|---|---|---|---|
| RGB 尺寸不一致 | Depth `(2,2)`、RGB `(3,2,3)` | `ValueError` | 已通过 |
| RGB dtype 不对 | `float32` 而非 `uint8` | `ValueError` | 已通过 |
| Mask dtype 不对 | `uint8` 而非 `bool` | `ValueError` | 已通过 |
| Mask Shape 不对 | `(2,3)` 而 Depth `(2,2)` | `ValueError` | 已通过 |
| 焦距非法 | `fx=0` | `ValueError` | 已通过 |
| 内参非有限 | `fx/fy/cx/cy = nan/inf` | `ValueError` | 已通过 |
| Depth 本身尺寸为零 | `(0,3)` / `(3,0)` | `ValueError` | 已通过 |
| Depth 包含 NaN/Inf/负数/零 | 混合合法及非法 | 只过滤无效点 | 已通过 |
| 目标区域全无有效深度 | 非空 Depth 图但有效点零 | 返回两个 `(0,3)` | 已通过 |

对于**非数值字符串类型的内参、尺寸极端巨大、完全未对齐 RGB 等**尚无本关明确自动验收记录，不能宣称覆盖完全。

---

## 12. 自动化测试与实际回归日志

文件：

- `tests/test_mask_to_pointcloud.py`：**5 个测试**，分别检查不同颜色对齐、RGB Shape、非布尔 Mask、零焦距、全无效目标深度。
- `tests/test_mask_to_pointcloud_boundaries.py`：**6 个测试**，分别检查空目标 Mask、混合无效深度、空 Depth、非有限内参、Mask Shape、RGB dtype。
- EXP026 测试：**7 个**。
- EXP027 测试：**12 个**。

实际提供的最后一轮完整日志：

```text
=== EXP028 Boundary Tests ===
Ran 6 tests in 0.078s
OK

=== Full Regression Tests ===
Ran 30 tests in 0.131s
OK

=== Syntax Check ===
(no error)
```

在 Open3D 可视化同一轮提供的另一份全量回归日志：

```text
Ran 30 tests in 0.152s
OK
```

**计数**：EXP028 = 5+6 = **11**；全项目 = 7+12+11 = **30**。

原则：测试通过说明这些测试覆盖的契约行为被验证，**不意味着真实相机下标定、配准、精度、实时性能全部完成**。

---

## 13. Open3D 可视化与 PLY 复现

文件：`src/experiments/exp028_open3d_target_visualization.py`。

### 13.1 场景设计（合成 RGB-D）

```python
depth = np.full((80,100), 2.0)        # 背景 2.0m
mask = np.zeros((80,100), dtype=bool)
mask[20:60,30:70] = True            # 40x40 = 1600 个目标像素
depth[mask] = 1.2                   # 目标平面 Z=1.2m
depth[30:40,40:50] = 0.0            # 10x10 = 100 个无效点
```

设定 RGB 背景灰色、目标橙色 `[245,135,50]`。使用内参 `fx=100, fy=100, cx=49.5, cy=39.5`，仅返回被选中的目标点。

**先预测**：

- Mask True：`40×40=1600`。
- 目标内无效深度：`10×10=100`。
- 有效目标点：`1600−100=1500`，`points.shape=(1500,3)`。
- 只保留目标 Z=1.2m，不应该出现 Z=2.0m 的灰色背景点。
- 从三维窗口观察，是**带缺口的橙色平面点集**，不是有体积的立方体。

### 13.2 独立几何参考

第一个有效目标像素 `(u=30,v=20,Z=1.2)`：

```text
X = (30−49.5)×1.2/100 = −0.234m
Y = (20−39.5)×1.2/100 = −0.234m
Z = 1.2m
```

应得到 `(-0.234,-0.234,1.2)`。

### 13.3 实际终端结果

```text
=== EXP028 Open3D Target Visualization ===
Depth shape: (80, 100)
Target mask pixels: 1600
Selected points: 1500
Points shape: (1500, 3)
Colors shape: (1500, 3)
Shape and Mask: PASS
First target point: [-0.234 -0.234  1.2  ]
Target Z: [1.2]
Geometry: PASS
RGB alignment: PASS
Open3D PointCloud: PASS
PLY saved: results\exp028\target_pointcloud_colored.ply
PLY reload: PASS
EXP028 Open3D checks PASSED.
Opening Open3D window...
```

同时提供了 Open3D 窗口截图：看到橙色平面、内部缺口、无灰色背景，画面左下角的彩色坐标轴是 Open3D 坐标框架，不属于目标点云。

**结论**：预测与数值断言、PLY 导出读回、可视化观察一致。单张深度图只描述被相机看见的表面，不自动生成物体的不可见背面和实体厚度。

### 13.4 Open3D 核心调用

```python
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

assert pcd.has_points() and pcd.has_colors()
assert len(pcd.points) == len(pcd.colors) == 1500

o3d.io.write_point_cloud(str(output), pcd)
loaded = o3d.io.read_point_cloud(str(output))
```

`PLY` 是可复现的实验输出，不意味着必须把二进制点云作为 Git 源码提交。先检查项目 `.gitignore` 和文件体积再决定。

---

## 14. 独立通关测试 Q1～Q7 的完整知识记录

> 本节不是新增测试题，是对**已经提交并经过逐题批改的答案**进行归档。状态：**7/7 PASS**。

### Q1：Explain — 为什么双 Mask 必须取交集？

- `target_mask` 判断像素是否属于目标；`valid_depth` 判断 Z 是否可用于反投影。
- 只用目标 Mask 会保留无效 Z，生成零坐标/NaN/Inf 等污染数据。
- 只用有效深度会把背景/其他物体也纳入目标点云。
- 正确：`select_mask = target_mask & isfinite(depth) & (depth>0)`。

### Q2：Shape — 行列索引与压缩顺序

- `np.nonzero()` 对二维数组返回 `(row,col)`，即 `(v,u)`。
- `points[2]`：第三个点的完整 XYZ；`points[:,2]`：全部 Z-depth。
- 一旦中间点被过滤，后续点的行索引可能变化。要回溯像素位置，可保留 `select_mask`，用 `np.nonzero(select_mask)` 按同一规则重新取得 u/v。

### Q3：Calculate — 新数据独立手算

给定：

```python
depth = np.array([[0,1,2],
                  [3,np.nan,4],
                  [5,0,6]], dtype=float)
mask = np.array([[True, True, False],
                 [True, True, True],
                 [False,True,True]])
fx=2.0; fy=4.0; cx=1.0; cy=1.0
```

```text
select_mask = [[F,T,F],
               [T,F,T],
               [F,F,T]]
```

有效像素（u,v,Z）按行优先：

```text
(1,0,1), (0,1,3), (2,1,4), (2,2,6)
```

独立 XYZ：

```text
( 0.0, −0.25, 1.0)
(−1.5,  0.0,  3.0)
( 2.0,  0.0,  4.0)
( 3.0,  1.5,  6.0)
```

最终 `points.shape=(4,3)`。例如 `(u=0,v=1,Z=3)` 得到 `X=(0−1)×3/2=−1.5`，`Y=(1−1)×3/4=0`。

### Q4：Predict — 扩大 Mask 的插入位置

将 `mask[2,0]=False→True`，新增 `(u=0,v=2,Z=5)`：

```text
X=(0−1)×5/2=−2.5
Y=(2−1)×5/4=1.25
P_new=(−2.5,1.25,5)
```

`select_mask=[[F,T,F],[T,F,T],[T,F,T]]`，共有 5 点；新增点出现在**行索引 3**，原来最后一个点移至索引 4。其余 XYZ 不变。

### Q5：Implement — 独立重写 API

独立实现正确完成：非空二维 Depth、RGB Shape/dtype、Mask Shape/dtype、内参有限且正焦距、无效深度过滤、`np.nonzero` 行优先、XYZ、颜色归一化、三个返回值及空目标 Shape。

代码复盘：`np.column_stack((x,y,z))` 结果已经是 `float64` 时，额外 `.astype(np.float64)` 通常是一份不必要的拷贝。独立答案对 `fx="invalid"` 一类非数值输入尚未确保统一为 `ValueError`；这是值得改进的边界，但不影响 Q5 基础算法验收。

### Q6：Debug — 颜色 `[::-1]` 乱序

同序正确颜色 `[红,黄]` 变成错误 `[黄,红]`，Shape 都是 `(2,3)`；不能用 Shape 替代语义断言。正确修复为取消 `[::-1]`，使用独立的 `expected_colors` 做 `assert_allclose`。

### Q7：Transfer — 目标点云质心与异常点

```python
target_points = np.array([
    [0,0,1], [2,0,1], [0,2,1], [2,2,1]
], dtype=float)
center = np.mean(target_points, axis=0)  # [1,1,1]
```

加入离群点 `(1,1,6)` 后共有 5 点，新的质心 `(1,1,2)`：X/Y 不变，Z 从 1 被异常点拉到 2。

质心是**观测点集的均值**，不一定是完整实体的几何中心。可能偏离的原因：可见表面不完整、采样不均、遮挡、深度噪声及离群点。可考虑中位数、截断均值、统计滤波、基于已知几何模型的 RANSAC 或密度聚类等方式，但每种方法都有适用条件。

**空点云 `points.shape=(0,3)` 不可直接把 `np.mean(...,axis=0)` 的 NaN 结果当作有效中心**。后续正式 Center API 必须约定返回 `None`、异常或显式无效标记。

> **边界声明**：本单元完成了“质心概念与异常值敏感性”的迁移推理，**没有实现并验收生产版 3D Center Estimator**；这属于后续单元。

---

## 15. Debug 速查与常见误区

| 问题/现象 | 可能根因 | 应如何验证 | 预防办法 |
|---|---|---|---|
| `points` 包含背景 | 只用 valid depth，没与 target_mask 求交集 | 检查 `select_mask` 与独立目标位置 | 双 Mask 与独立像素参考 |
| `nan` 或 `inf` 点 | 没检查 `np.isfinite` | `np.isfinite(points).all()` | `isfinite & >0` |
| X/Y 反了 | 把 `v,u` 写成 `u,v` | 非对称像素独立手算 XYZ | 明确行列与 u/v |
| XYZ 列错位 | `column_stack((z,y,x))` | 独立 XYZ 断言 | 固定 `[X,Y,Z]` 契约 |
| 颜色落在错误点上 | 使用不同 Mask 或 `[::-1]` 排序 | 区分 RGB 的像素级独立参考 | 同 Mask、同顺序 |
| Mask 是 `uint8` | 误把整数索引当布尔索引 | `mask.dtype` 测试 | 输入 dtype 检查 |
| 没有目标时报错或形状 `(0,)` | 空目标路径缺失 | 空 Mask / 全零 Depth 两种测试 | 返回 `(0,3)` |
| 两帧点云直接相减失败 | 有效点数不同 | 打印 Shape、源像素索引 | 先建立点对应 |
| 两帧能相减但结果不可信 | Shape 巧合相同、实际对应不同 | 对比原像素 u/v 或做配准 | 禁用“同数组行号=同物理点”假设 |
| 颜色值过大 | 忘记 `/255` | `colors.max()<=1` | Open3D 归一化 |
| 颜色通道倒置 | 把 BGR 当 RGB | 对单个已知红/蓝像素检查 | 明确通道约定 |
| XYZ 尺度整体错 | 深度单位 mm 被当作 m | 对已知距离点独立验证 | 明确 m 与 depth_scale |
| RGB 尺寸相同但仍错位 | 未做真正 RGB-D 配准 | 真实标定图/重投影误差 | 区分 Shape 与 Registration |

### 15.1 故障排查固定顺序

1. 先读完整异常类型、文件、行号、日志；
2. 打印输入 Shape、dtype、内参、深度单位；
3. 比对 `target_mask / valid_depth / select_mask`；
4. 从一两个已知像素独立手算 XYZ 和 RGB；
5. 一次只修改一个假设；
6. 修复后添加自动化回归测试，不只让程序恢复运行。

---

## 16. 当前实验目录与源码资产

核心模块：

```text
src/geometry/mask_to_pointcloud.py
```

实验：

```text
src/experiments/exp028_mask_to_xyz_step1.py
src/experiments/exp028_mask_to_pointcloud_minimal.py
src/experiments/exp028_active_modification_a.py
src/experiments/exp028_active_modification_b.py
src/experiments/exp028_active_modification_c.py
src/experiments/exp028_open3d_target_visualization.py
```

测试：

```text
tests/test_mask_to_pointcloud.py
tests/test_mask_to_pointcloud_boundaries.py
```

笔记及输出：

```text
experiments/EXP028_mask_to_pointcloud.md
results/exp028/target_pointcloud_colored.ply
```

Git 工程记录：10 个 EXP028 文件已提交并同步到 GitHub，源码 Commit 为 138bdc2。六个 ScreenCamera / ScreenCapture 临时文件未提交。

---

## 17. 环境与可复现命令（Windows PowerShell）

实验环境已确认：

```text
Conda: D:\conda_envs\ai_3d
Python: 3.11（该环境创建时指定）
NumPy: 2.2.6
Open3D: 0.20.0
Project: D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer
```

激活与版本检查：

```powershell
conda activate D:\conda_envs\ai_3d
python -c "import numpy as np; import open3d as o3d; print(np.__version__,o3d.__version__)"
python -m pip check
```

最小实验、主动修改：

```powershell
python -m src.experiments.exp028_mask_to_xyz_step1
python -m src.experiments.exp028_active_modification_a
python -m src.experiments.exp028_active_modification_b
python -m src.experiments.exp028_active_modification_c
```

完整最小实验（建议封关前单独留一份日志，当前无已提交运行记录）：

```powershell
python -m src.experiments.exp028_mask_to_pointcloud_minimal
```

EXP028 专项与全量回归：

```powershell
python -m unittest discover -s tests -p "test_mask_to_pointcloud*.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
```

Open3D：

```powershell
python -m src.experiments.exp028_open3d_target_visualization
python -m src.experiments.exp028_open3d_target_visualization --show
```

语法检查与 Git 状态：

```powershell
python -m py_compile src\geometry\mask_to_pointcloud.py
python -m py_compile tests\test_mask_to_pointcloud.py
python -m py_compile tests\test_mask_to_pointcloud_boundaries.py
git status --short
```

> 说明：使用 `python -m unittest discover -s tests -p "test_mask_to_pointcloud*.py"` 时，glob 匹配两个 EXP028 测试文件。`--show` 依赖 Windows 图形环境；无窗口模式的断言和 PLY 导出可独立运行。

---

## 18. 真实工业视觉迁移前的注意事项

1. **Depth 单位**：设备可能输出 `uint16` 毫米、比例因子深度或已转换的米。要在传感器适配层统一成米，不能在几何函数里猜。
2. **RGB/Depth 配准**：相机光心、内参和分辨率可能不同。Mask 与 RGB 像素索引一致，不代表能直接用于原始、未配准的 Depth。
3. **目标 Mask 来源**：本关使用人工合成布尔 Mask，真实系统可能来自 YOLO Seg、SAM、工业分割模型；应单独验证 Mask 与输入图像的坐标一致性。
4. **深度噪声和缺失**：反光、透明、遮挡等可能造成 0/NaN 或**有限但错误**的值；本关仅处理前者，不解决全部噪声。
5. **中心定义**：观测点质心、3D Bounding Box 中心、物体体积几何中心、可抓取中心并不是一回事，需要明确业务指标。
6. **多帧运动**：点云数组下标不构成跨帧 ID；相机运动与物体运动需要几何匹配或时空跟踪。
7. **实时性**：本关未单独做 Target Mask → Point Cloud 的性能 Benchmark，也未测端到端相机+分割+定位延迟。不能沿用 EXP027 的转换速度推断完整 EXP028 流水线速度。

---

## 19. 我现在能够独立回答的七类能力

| 能力 | 已独立证明的行为 | 证据 |
|---|---|---|
| Explain | 解释双 Mask、Z-depth、可见表面 | Q1、Q7 |
| Shape | 区分 `points[2]` / `points[:,2]`，解释行优先 | 索引补测、Q2 |
| Calculate | 手算多像素目标 XYZ | Q3 |
| Predict | 预测 Mask 扩大后的数值与插入位置 | 修改 A、Q4 |
| Implement | 从零写接口验证+目标反投影+颜色 | Q5 |
| Modify | 主动控制 Mask、Z、失效深度并验证 | 修改 A/B/C |
| Debug | 识别颜色乱序与广播错误 | Q6、修改 C |
| Transfer | 质心与异常点，空点云处理 | Q7 |
| Verify / Reproduce | 30 tests、Open3D、PLY | 用户终端日志与截图 |

**仍不稳定 / 尚未证明**：真实 RGB-D 相机配准、目标中心生产实现和鲁棒性、真实数据上的评价指标、跨帧点匹配、端到端延迟。它们不是 EXP028 在本次教学场景下的通关阻塞项，但必须在后续关卡独立验收。

---

## 20. SOP Closure Record 与 Git/GitHub 门槛

### 20.1 已完成项

```text
Entry Protocol                      PASS
Knowledge Map                       PASS
Minimum Theory                      PASS
Minimum 2x2 XYZ Experiment          PASS (provided terminal log)
Active Modification A               PREDICT + VERIFY PASS
Active Modification B               PREDICT + VERIFY PASS
Active Modification C               PREDICT + VERIFY PASS
Core Production API Smoke Test      PASS
RGB Alignment Test                  PASS
Contract & Boundary Tests           11/11 PASS
Full Project Regression              30/30 PASS
Open3D Colored Target Visualization PASS
PLY Export / Reload                  PASS
Independent Acceptance Q1-Q7        7/7 PASS
```

### 20.2 最终工程验收

| 验收项目 | 结果 |
|---|---|
| 最小实验 | PASS |
| 主动修改 A/B/C | PASS |
| Production API | PASS |
| EXP028 专项测试 | 11/11 PASS |
| 全量回归 | 30/30 PASS |
| Open3D 可视化 | PASS |
| PLY 导出与读回 | PASS |
| 独立通关 Q1～Q7 | 7/7 PASS |
| 完整实验笔记 | 20 节，PASS |
| Git 文件审查 | 10 个文件，PASS |
| Git Whitespace Check | PASS |
| 源码 Commit | 138bdc2 |
| 源码 GitHub 同步 | PASS |

源码 Commit：

138bdc279b5127edeb51efe55f22ae8685ca5f03

### 20.3 最终封关规则

- EXP028 的理论、实验、测试和独立能力验收已通过。
- 源码、测试、实验与笔记已经完成第一次 GitHub 同步。
- 本次封关笔记单独创建一个文档 Commit。
- 此文档 Commit 推送到 GitHub 后，直接查询远端 master 的 SHA。
- 只有远端 SHA 与本地 HEAD 一致，EXP028 才正式标记 COMPLETE。
- 最终文档 Commit SHA 以 Git 历史为准，不在自身内容中循环引用。

### 20.4 下一阶段衔接（只做知识地图，不抢跑）

```text
EXP028: Target Mask → Target Colored Point Cloud
                    ↓
候选后续单元：Target Point Cloud → 3D Center Estimation
                    ↓
鲁棒统计 / 背景平面去除 / 目标簇选择 / 坐标变换
```

**最终阶段结论**：EXP028 理论、实验、30/30 回归、7/7 独立验收和源码 GitHub 同步已通过。最终笔记的提交与远端核验完成后，EXP028 正式 COMPLETE。Level 2 继续。
