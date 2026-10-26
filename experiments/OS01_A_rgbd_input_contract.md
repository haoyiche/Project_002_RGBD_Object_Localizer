# OS-01-A｜RGB-D 输入契约、合成样本与扰动实验

> 项目：`Project_002_RGBD_Object_Localizer`
> 路线：Level 2 · 3D Vision / OS-01 RGB-D Object Localizer
> 学习单元：OS-01-A（数据与接口契约）
> 整理日期：2026-10-26
> **状态：CONTINUE（概念复盘、A1/A2/A3 与综合 Boss 已通过；最终回归、Git/GitHub 待验收）**
> 数据边界：本单元全部输入为**确定性合成 RGB-D**；未接入真实 RGB-D 传感器，也没有独立物理真值。
> 记录原则：预测与实测来自本轮聊天和用户提供的 Windows PowerShell 运行结果；未执行的项目不得写作 PASS。

## 0. Entry Protocol：目标、前置与验收

### 0.1 为什么学

EXP026～EXP037 分别训练了反投影、Mask 到点云、点云统计、滤波、聚类、SE(3) 和误差审计。OS-01 的目标不是再造新公式，而是为完整 `RGB-D Object Localizer` 建立**可重复读入、可以验证 Shape/单位/来源的数据契约**，从孤立实验迈向可集成的工程入口。

### 0.2 前置知识

- **必须掌握**：EXP026 针孔相机反投影；EXP028 Mask × 有效深度；EXP029 点云质心；NumPy Shape、布尔掩码与 `axis=0`。
- **建议掌握**：EXP034 几何有效性门禁；EXP035 坐标系约定；EXP036～037 误差与真实性边界。
- **本单元不展开**：真实相机内外参标定、6D Pose、RGB/Depth 配准算法、传感器时间同步与机器人发布认证。

### 0.3 本单元目标和产物

1. 定义 RGB、Depth、Mask、K 与 `frame_id` 的输入契约。
2. 生成一个确定性合成样本 `os01_sample_000.npz`，并能重新加载。
3. 复用现有 `src.geometry.mask_to_pointcloud.mask_to_pointcloud()`，计算目标点云与质心。
4. 分别通过 A1 内参、A2 深度、A3 Mask 的**预测 → 实测 → 解释**。
5. 保存代码、独立回读测试与学习笔记；完成后续综合 Boss、Git/GitHub 闭环。

本单元只验证**一个标准样本和输入协议**，不把它冒充 OS-01 主项目的 20 样本评测完成。

## 1. Knowledge Map：上下游连接

```text
EXP026：内参 K + (u,v,Z) → 相机系三维点
        ↓
EXP028：Mask ∩ 有效深度 → 目标点云及 RGB 颜色
        ↓
EXP029：点云统计 → 质心 / AABB
        ↓
OS-01-A：保存、重载、验证 RGB-D 输入契约；做三种受控扰动
        ↓
OS-01-B：单样本端到端目标定位流水线
        ↓
OS-01-C：不少于 20 个可追溯 RGB-D 样本的正式评测
        ↓
OS-01-D / Level 2 Boss：鲁棒性、交付、独立验收
```

与 EXP036～037 的关系：误差数值、数据来源及物理真值必须独立审计。当前得到的是合成几何结果，而不是已认证的物理坐标。

## 2. Minimum Theory：核心概念与公式

### 2.1 输入与输出契约

| 参数 | 类型 / Shape | 单位 | 来源及工程语义 |
|---|---|---|---|
| `rgb` | `uint8`, `(H,W,3)` | 通道整数 0～255 | 与 Depth/Mask **已经对齐**的 RGB 图像，本文不处理配准 |
| `depth_m` | 浮点数组，`(H,W)` | **m** | 相机光轴方向的 Z-depth；不是视线长度 |
| `target_mask` | `bool`, `(H,W)` | 无 | 指定目标候选像素；本次手工合成，非模型推理 |
| `fx, fy` | 有限正浮点数 | px | X/Y 方向焦距参数 |
| `cx, cy` | 有限浮点数 | px | 像素主点偏移 |
| `frame_id` | 非空字符串 | 无 | 本例为 `camera_optical_frame` |
| `reference_kind` | 字符串 | 无 | 本例为 `synthetic_input`，**不是独立真值** |
| `selected` | `bool`, `(H,W)` | 无 | Mask 与合法深度共同决定的实际选点集合 |
| `points` | 浮点数组，`(N,3)` | m | 相机坐标系下的 XYZ，每行一个点 |
| `colors` | 浮点数组，`(N,3)` | 0～1 | 与 `points` 逐行对应的 RGB 颜色 |
| `centroid` | 浮点数组，`(3,)` | m | 有效目标点的**采样质心**，非真实体积质心 |

约定相机坐标：X 向右、Y 向下、Z 向前。`H` 为图像行数，`W` 为列数，`u` 是列坐标，`v` 是行坐标。

### 2.2 像素反投影：符号、单位、运算、变化

对已校正且 RGB/Depth/Mask 对齐的针孔相机像素：

\[
X=\frac{(u-c_x)Z}{f_x},\qquad
Y=\frac{(v-c_y)Z}{f_y},\qquad Z=\operatorname{depth\_m}[v,u]
\]

- **符号**：`u,v,cx,cy,fx,fy` 均按像素尺度解释；`X,Y,Z` 是相机系三维位置。
- **单位**：`(u-cx)/fx`、`(v-cy)/fy` 无量纲，乘 `Z[m]` 得到 `X,Y[m]`；乘 `1000` 后才是 mm。
- **运算**：像素偏移 → 归一化射线方向 → 按光轴深度恢复三维点。
- **变化**：只增大 `fx`，在固定 `u,Z,cx` 下 `|X|` 缩小；统一按比例增大 `Z`，单点 XYZ 同比例变化；改变 Mask 则改变参加统计的点集。

注意：`fx` 增大不代表所有 X 数值都减小。若 X 原为负数，乘以小于 1 的比例后会**向零靠近**，其数值反而增大。

### 2.3 像素合法性与点云质心

```python
valid_depth = np.isfinite(depth_m) & (depth_m > 0)
selected = target_mask & valid_depth
```

`Mask=True` 仅表示候选目标像素；它还必须有有限且正的深度，才能被本函数转换为有效三维点。

\[
N=\operatorname{count\_nonzero}(selected),\qquad
\mathbf C=\frac{1}{N}\sum_{i=1}^{N}\mathbf P_i\quad(N>0)
\]

如果 `N=0`，不能盲目对空数组求质心并发布 NaN，应由后续集成层给出显式的无效目标/无可用点结果。本单元没有声称已经验证该集成层的完整处理。

## 3. 实验环境、代码与数据

- Windows PowerShell，项目根目录：`D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer`。
- Conda 环境：`D:\conda_envs\ai_3d`。
- 自动测试输出显示 Python `3.11.17`、pytest `9.1.1`；本轮未记录 NumPy 版本。
- 统一从**项目根目录**运行 `python -m src.experiments.<module>`。

### 3.1 文件清单

| 文件 | 作用 | 本轮证据 |
|---|---|---|
| `src/experiments/os01_a_synthetic_rgbd_sample.py` | 生成、计算并保存合成 RGB-D 标准样本 | 运行 PASS |
| `data/raw/synthetic/os01_sample_000.npz` | RGB/Depth/Mask/K/frame/source 元数据 | 本地生成及测试重载 PASS |
| `tests/test_os01_a_synthetic_rgbd_sample.py` | 临时目录生成、重载、输入契约和独立 XYZ 数值断言 | `1 passed` |
| `src/experiments/os01_a_intrinsics_ablation.py` | A1 仅修改 `fx` | 预测+实测 PASS |
| `src/experiments/os01_a_depth_ablation.py` | A2 仅增加有效目标深度 | 预测+实测 PASS |
| `src/experiments/os01_a_mask_ablation.py` | A3 仅移除一个有效 Mask 像素 | 预测+实测 PASS |
| `experiments/OS01_A_rgbd_input_contract.md` | 本学习笔记 | 本文草稿，待保存到项目并核验 |

此处的文件路径及实测证据来自本轮对话；Git 追踪、提交和远程同步尚未核验。轻量合成 `.npz` 是否纳入版本控制，需要在暂存前结合仓库数据策略确认；生成脚本与测试必须可复现。

### 3.2 标准样本定义

```python
rgb = np.zeros((6, 8, 3), dtype=np.uint8)
rgb[2:4, 3:6] = [200, 60, 40]

depth_m = np.zeros((6, 8), dtype=np.float64)
depth_m[2:4, 3:6] = 1.0
depth_m[2, 3] = 0.0  # 注入无效深度

target_mask = np.zeros((6, 8), dtype=bool)
target_mask[2:4, 3:6] = True

fx, fy, cx, cy = 100.0, 100.0, 3.5, 2.5
```

Mask 有 `2×3=6` 个 True 像素。`(v=2,u=3)` 虽属于 Mask，深度却为 0，故实际点数 `N=5`。

有效像素按行优先顺序得到：

| 像素 `(v,u)` | X (m) | Y (m) | Z (m) |
|---|---:|---:|---:|
| `(2,4)` | +0.005 | -0.005 | 1.0 |
| `(2,5)` | +0.015 | -0.005 | 1.0 |
| `(3,3)` | -0.005 | +0.005 | 1.0 |
| `(3,4)` | +0.005 | +0.005 | 1.0 |
| `(3,5)` | +0.015 | +0.005 | 1.0 |

质心：

\[
\mathbf C_0=[0.007,\ 0.001,\ 1.0]\ \mathrm{m}
\]

### 3.3 最小实验：先预测、再运行

**本人运行前预测**：Mask True 数为 6，合法点为 5，`points.shape=(5,3)`，`colors.shape=(5,3)`，`selected.shape=(6,8)`。

**运行命令**：

```powershell
python -m src.experiments.os01_a_synthetic_rgbd_sample
```

**本人提供的实际输出**：

```text
RGB shape: (6, 8, 3)
Depth shape: (6, 8)
Mask pixels: 6
Valid selected pixels: 5
Points shape: (5, 3)
Colors shape: (5, 3)
Centroid (m): [0.007 0.001 1.   ]
Saved: data\raw\synthetic\os01_sample_000.npz
```

**结论**：预测与实测一致。输出是合成相机系点云，并非真实空间精度认证。

### 3.4 保存 / 回读的自动化测试

测试使用 `tmp_path` 和 `monkeypatch.chdir(tmp_path)`，在隔离临时目录中调用 `main()`，避免依赖或覆盖已存在的本地 `.npz`。随后执行 `np.load(..., allow_pickle=False)`，确认数据和元信息能够安全回读，重新调用已有 `mask_to_pointcloud()`。

检查内容：

- RGB/Depth/Mask 的 Shape 与 dtype；
- `frame_id="camera_optical_frame"` 和 `reference_kind="synthetic_input"`；
- Mask True = 6，Selected True = 5；
- Points 与 Colors 都为 `(5,3)`；
- `np.testing.assert_allclose(points, expected_points, rtol=0, atol=1e-12)`，用**独立列出的五个 XYZ**核对反投影数值。

```powershell
python -m pytest tests/test_os01_a_synthetic_rgbd_sample.py -v
```

**实测**：`1 passed in 0.23s`。该测试验证本次定义的确定性数据和路径，并非项目全量回归。

## 4. Active Modification：三组受控扰动

三组均使用同一保存样本，先手算预测后运行，单次仅改变一种因素。`centroid delta` 的定义统一为：

\[
\Delta\mathbf C_{\mathrm{mm}}=1000(\mathbf C_{\mathrm{after}}-\mathbf C_{\mathrm{before}})
\]

它是**相对于合成基线的变化量**，不是独立物理真值误差。

### A1：仅将 `fx: 100 → 200 px`

**目标**：验证相机焦距参数对 X 反投影的影响，隔离 Mask/Depth 的变化。

**控制条件**：`fy,cx,cy`、RGB、Depth、Mask 均不变。

**预测推导**：

\[
X_{after}=\frac{(u-c_x)Z}{2f_x}=\tfrac12X_{before}
\]

由于初始质心 X 为正：

\[
C_{after}=[0.0035,0.001,1.0]\ \mathrm m,
\quad \Delta C=[-3.5,0,0]\ \mathrm{mm}
\]

点数不变、`points_after.shape=(5,3)`，`Selected mask unchanged=True`。

**实测命令**：

```powershell
python -m src.experiments.os01_a_intrinsics_ablation
```

**本人提供的实测输出**：

```text
fx before: 100.0
fx after: 200.0
Points before: (5, 3)
Points after: (5, 3)
Selected mask unchanged: True
Centroid before (m): [0.007 0.001 1.   ]
Centroid after (m): [0.0035 0.001  1.    ]
Centroid delta (mm): [-3.5  0.   0. ]
```

**对照结论**：PASS。固定其他量，`fx` 加倍使每个点的 `X` **向零缩半**，并不保证所有带符号 X 数值都下降；Y/Z 不变。

### A2：仅将有效目标深度 `1.0 → 1.2 m`

**目标**：验证深度影响整个三维射线而非仅 Z 分量。

**控制条件**：使用 `depth_modified=depth_m.copy()`；只对 `target_mask & isfinite(depth_m) & (depth_m>0)` 增加 `0.2m`；Mask、RGB、K 均不变；保持 `(2,3)` 处零深度。

**预测推导**：本样本所有有效深度原本都是 `1.0m`，统一加 `0.2m` 后恰好相当于每个有效 XYZ 乘 `1.2`：

\[
C_{after}=1.2C_0=[0.0084,0.0012,1.2]\ \mathrm m
\]

\[
\Delta C=[+1.4,+0.2,+200]\ \mathrm{mm}
\]

预测 `N=5`、Selected 不变、无效深度保留。

**实测命令**：

```powershell
python -m src.experiments.os01_a_depth_ablation
```

**本人提供的实测输出**：

```text
Valid points before: 5
Valid points after: 5
Selected mask unchanged: True
Centroid before (m): [0.007 0.001 1.   ]
Centroid after (m): [0.0084 0.0012 1.2   ]
Centroid delta (mm): [  1.4   0.2 200. ]
Invalid depth preserved: True
```

**对照结论**：PASS。只因本例五个有效点全部具有相同初始深度，统一加 `0.2m` 才等价于整体乘 `1.2`；不同初始深度时不应套用此简化比例。

### A3：只将一个有效 Mask 像素 `(v=3,u=5)` 设为 False

**目标**：模拟分割漏检对点数量及采样质心的影响。

**控制条件**：`mask_modified=target_mask.copy()`；只改 `mask_modified[3,5]=False`；Depth、RGB、K 均不变。

**运行前预测**：

- Mask True：6 → 5；有效点：5 → 4；`points_after.shape=(4,3)`。
- `Selected mask unchanged=False`；原深度 `depth_m[3,5] == 1.0` 为 True。
- 被移除点：`P_k=[0.015,0.005,1.0]m`。

剩余四点均值：

\[
C_{after}=[0.005,0,1.0]\ \mathrm m
\]

从质心定义可直接得到删除一点的公式：

\[
C_{after}=\frac{NC_{before}-P_k}{N-1},\quad
\Delta C=\frac{C_{before}-P_k}{N-1}\quad(N>1)
\]

代入本例：

\[
\Delta C=[-2,-1,0]\ \mathrm{mm},\qquad
\|\Delta C\|_2=\sqrt5\approx2.236\ \mathrm{mm}
\]

**实测命令**：

```powershell
python -m src.experiments.os01_a_mask_ablation
```

**本人提供的实测输出**：

```text
Mask pixels before: 6
Mask pixels after: 5
Points before: (5, 3)
Points after: (4, 3)
Selected mask unchanged: False
Centroid before (m): [0.007 0.001 1.   ]
Centroid after (m): [0.005 0.    1.   ]
Centroid delta (mm): [-2. -1.  0.]
Original depth unchanged: True
```

**对照结论**：PASS。虽然其余四点不移动，点集及平均分母改变，质心仍向被移除点的反方向移动。单像素漏检引起的 `2.236mm` **相对偏移**是否可接受，取决于任务阈值、成像尺度及真实系统的验证，而不是“只少一个像素”的主观判断。

### 4.4 三组对照汇总

| 试验 | 唯一改变量 | 点数变化 | 质心偏移 (mm) | 学习结论 |
|---|---|---:|---|---|
| A1 | `fx × 2` | 5 → 5 | `[-3.5,0,0]` | 内参改变 X 尺度 |
| A2 | 有效深度 `+0.2m` | 5 → 5 | `[+1.4,+0.2,+200]` | 沿像素射线改变三维位置 |
| A3 | 移除一个有效 Mask 像素 | 5 → 4 | `[-2,-1,0]` | 采样集合变化导致质心漂移 |

## 5. Code Reading：关键实现与 Shape 流

从 `src/geometry/mask_to_pointcloud.py` 的实现观察：

```python
valid = np.isfinite(depth) & (depth > 0)
select_mask = mask & valid
v, u = np.nonzero(select_mask)
z = depth[select_mask]
x = (u.astype(np.float64) - cx) * z / fx
y = (v.astype(np.float64) - cy) * z / fy
points = np.column_stack((x, y, z))
colors = image[select_mask].astype(np.float64) / 255.0
```

| 操作 | 输入 Shape | 输出 Shape | 关键理解 |
|---|---|---|---|
| `np.isfinite(depth) & (depth > 0)` | `(H,W)` | `(H,W)` bool | 过滤零值、负值、NaN、Inf |
| `mask & valid` | `(H,W)` × `(H,W)` | `(H,W)` bool | 同位置逐像素与运算 |
| `np.nonzero(select_mask)` | `(H,W)` | 两个 `(N,)` | 先 `v` 后 `u`，分别对应行与列 |
| `depth[select_mask]` | `(H,W)` | `(N,)` | 与 `u,v` 同一选点顺序 |
| `np.column_stack((x,y,z))` | 3 个 `(N,)` | `(N,3)` | 每行一个 XYZ |
| `image[select_mask]` | `(H,W,3)` | `(N,3)` | 与点云逐行对应的 RGB |
| `.mean(axis=0)` | `(N,3)` | `(3,)` | 分别对 XYZ 求均值 |
| `np.savez_compressed()` | 各原始数组和标量 | `.npz` | 保存确定性可回读输入 |
| `np.load(..., allow_pickle=False)` | `.npz` | 可按键访问 | 拒绝 pickle 反序列化路径；加载后检查契约 |

关键条件：所有输入图像必须空间对齐、Depth 已换算为米。单纯 Shape 一致不能证明 RGB/Depth 真正对齐；`frame_id` 字符串也不能证明外参已标定。

## 6. Debug / Common Mistakes：问题、正确理解与验证

| 易错假设 | 正确认识 | 本次验证或未来检查 |
|---|---|---|
| Mask True 数等于点云数 | 还需要有限且正的 Z 深度 | 6 个 Mask 像素只有 5 个点 |
| `(u,v)` 与数组 `[v,u]` 可互换 | `u` 是列、`v` 是行 | 独立核对五个点的 XYZ |
| Depth 数字可直接假定为 m | 必须检查传感器单位与 Z-depth 语义 | 合成样本显式 `depth_m` |
| `fx` 增大，所有 X 都数值减小 | 带符号坐标向零靠近；负数数值可能增大 | A1 公式分析 |
| 深度变化只影响 Z | 反投影 X/Y 也依赖 Z | A2 三轴变化 |
| 任意 `Z += 0.2` 都使 XYZ 整体乘 1.2 | 仅本例初始 Z 全为 1.0m | A2 适用条件 |
| 少一个 Mask 像素可以忽略 | 采样质心取决于样本几何与数量 | A3 `2.236mm` 相对偏移 |
| 质心变化量就是物理真值误差 | 本例只是相对合成基线 | 所有 A 组结果 `delta_mm` 定义 |
| 合成测试 PASS 就能声称代码完全正确 | 仅说明被测路径及已断言条件符合预期 | 只有 1 个专项 Pytest，尚未全量回归 |
| 计算出漂亮的毫米数字即可发布给机器人 | 真实标定、独立 GT、frame/时序、误差门禁未验证 | 继续维持跨 frame / 机器人发布禁止状态 |

尚未执行的故障注入、空目标测试、错单位测试等不得写成 RED/GREEN 已验证。这些可在后续集成阶段按风险补充，避免本单元无限膨胀。

## 7. My Understanding：本人概念复盘（依据已提交作答整理）

### Q1｜为什么 Mask=True 的像素不一定生成三维点？

`target_mask=True` 只说明这个像素属于候选目标，还必须满足 `np.isfinite(depth)` 且 `depth>0`。在本合成样本里，`(v=2,u=3)` 的 Mask 为 True，但 Depth 为 0，所以 6 个 Mask 像素只有 5 个可反投影点。**Mask=True 是本函数选点的必要条件，不是充分条件。**

### Q2｜A1、A2、A3 的影响为什么不同？

- **A1（内参）**：只改 `fx`，其值只进入 X 反投影，因此 X 尺度改变而有效点集、Y、Z 不变；本例质心偏移 `[-3.5,0,0]mm`。
- **A2（深度）**：深度进入 X、Y、Z 的三维重建关系，固定像素对应从相机光心出发的射线；本例所有有效点从 `1.0m` 增为 `1.2m`，质心偏移 `[1.4,0.2,200]mm`。
- **A3（Mask）**：不修改任何保留点的 XYZ，只改变参与均值的样本集合；由五点变为四点，质心偏移 `[-2,-1,0]mm`。

归纳：**内参改变重建尺度；深度改变沿射线的位置；Mask 改变统计样本集合。**

### Q3｜为什么合成质心正确，仍不能宣称真实毫米级精度？

本合成实验验证的是规定输入、假设与测试覆盖范围内的代码行为。现实中还存在深度噪声、量化、畸变与标定误差、反光/透明/黑色材质、遮挡、边缘飞点、RGB/Depth 不对齐与不同步、安装外参、温漂和振动等问题。

若要宣称真实物理定位精度，需要可信真实采集、**独立测得的真值**、标定与外参验证记录、明确工作空间及材料/姿态/光照覆盖、可追溯样本与 Mean/Median/P95/Max 等指标、预定义阈值和失败处理。即使某批合成样本恰好误差为 0，也不能替代这些证据。

> **修正后的表达**：合成测试通过，不是“全部代码正确”；只能证明**被测代码路径在这些输入和断言下符合预期**。物理精度需要独立物理证据。

## 8. Reproducibility：复现命令与证据

从仓库根目录、激活 `ai_3d` 环境后，按次序执行：

```powershell
python -m src.experiments.os01_a_synthetic_rgbd_sample
python -m pytest tests/test_os01_a_synthetic_rgbd_sample.py -v
python -m src.experiments.os01_a_intrinsics_ablation
python -m src.experiments.os01_a_depth_ablation
python -m src.experiments.os01_a_mask_ablation
```

注意 A1/A2/A3 从已保存样本加载，故首次运行需要先生成 `.npz`。自动化测试在 `tmp_path` 重新生成样本，独立于项目根目录的已有 `.npz` 文件。

**本轮已观察证据**：最小样本输出与质心 PASS；Pytest `1 passed in 0.23s`；A1/A2/A3 预测及终端实测均 PASS。**新增已观察证据**：本人完成 Boss B/C 独立作答并获导师判定 PASS；C3 是设计方案，未在仓库实现。**尚未观察**：OS-01-A 相关文件完整暂存、仓库全量回归和 GitHub 远程同步。

## 8.1 综合 Boss B/C：独立预测、工程排错与 Fail-Closed（作答 PASS）

> **证据属性**：以下 Boss B 的结果为本人独立推导并经导师复核；Boss C 的故障行为与修复方案为代码审查层面的推断与方案验收，**未声称已在仓库中实现或执行新测试**。2026-10-26 综合 Boss B/C 均获判定 PASS。

### Boss B｜删去“原本就没有有效深度”的 Mask 像素

基线有 `6` 个 `target_mask=True` 像素，但 `(v=2,u=3)` 的 `depth_m[2,3]=0.0`，所以实际只有 `5` 个有效三维点，质心为 `[0.007,0.001,1.0] m`。

仅修改：

```python
mask_new = target_mask.copy()
mask_new[2, 3] = False
```

**运行前推导（无须运行即可得到的预测）：**

| 项目 | 修改前 | 修改后 |
|---|---|---|
| Mask True 数 | 6 | **5** |
| 有效三维点数 | 5 | **5** |
| `points.shape` | `(5,3)` | **`(5,3)`** |
| `selected` 有效选点位置 | 原有 5 个 | **完全相同** |
| `centroid_m` | `[0.007,0.001,1.0]` | **`[0.007,0.001,1.0]`** |
| `delta_mm` | — | **`[0,0,0]`** |

因为实际选点为：

```python
selected = target_mask & np.isfinite(depth_m) & (depth_m > 0)
```

被删除位置之前就因 `depth_m=0.0` 而被过滤。虽然原始 Mask 发生了变化，但**参与反投影的有效点集合并未变化**；所以点云和质心保持不变。

对比 A3：A3 删除的是一个真实参与点云计算的有效像素，质心偏移 `[-2,-1,0] mm`；Boss B 删除的是本来已被排除的像素，质心偏移为零。**不能将“Mask 数量变化”直接等同于“有效点数或三维位置变化”。**

### Boss C1/C2｜原始报告代码的故障预测及根因

错误条件是：Mask 只有一个 True，但对应深度为零；其余输入和相机参数合法。

原代码通过 `np.count_nonzero(target_mask)>0` 判断是否计算质心。实际 `mask_to_pointcloud()` 返回 `points.shape==(0,3)`，但因为 Mask 非空仍进入 `VALID` 分支。对空 `(0,3)` 数组执行 `points.mean(axis=0)`，在 NumPy 默认警告处理下通常产生 `[nan,nan,nan]` 并发出 `RuntimeWarning`，于是报告可能成为：

```python
{
    "status": "VALID",
    "position_m": [float("nan"), float("nan"), float("nan")],
    "publishable": True,
}
```

这不是“静默成功”：NumPy 会发出警告；若配置将 RuntimeWarning 提升为异常，执行也可能中断。无论哪种配置，**都不应让它进入合法定位与发布分支**。

本人识别出四个设计缺陷：

1. 用 **Mask 非空**冒充 **有效三维点非空**。
2. 对空点云求均值，没有防止非有限坐标流入报告。
3. 将 `publishable=True` 与基础定位计算绑定，跳过真实标定、外参和安全门禁。
4. `status` 混淆了目标候选、三维位置有效性和机器人坐标发布许可。

### Boss C3｜独立提出的最小 Fail-Closed 接口方案

以下为**设计参考代码**，还未集成到 `src/` 或执行自动化测试。增加 `frame_id` 用来说明诊断质心所属坐标系；所有分支均拒绝机器人坐标发布。

```python
import numpy as np

points, colors, selected = mask_to_pointcloud(
    depth_m, rgb, target_mask, fx, fy, cx, cy
)

mask_count = int(np.count_nonzero(target_mask))
valid_count = int(points.shape[0])

if mask_count == 0:
    report = {
        "status": "NO_OBJECTS",  # 仅指当前目标 Mask 为空
        "reason": "EMPTY_TARGET_MASK",
        "mask_count": 0,
        "valid_point_count": 0,
        "position_m": None,
        "frame_id": "camera_optical_frame",
        "publishable": False,
    }
elif valid_count == 0:
    report = {
        "status": "MASK_ONLY_NO_VALID_DEPTH",
        "reason": "NO_VALID_3D_POINTS",
        "mask_count": mask_count,
        "valid_point_count": 0,
        "position_m": None,
        "frame_id": "camera_optical_frame",
        "publishable": False,
    }
else:
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")

    centroid = points.mean(axis=0)
    if not np.isfinite(centroid).all():
        raise ValueError("centroid must be finite")

    report = {
        "status": "DIAGNOSTIC_ONLY",
        "reason": "SYNTHETIC_OR_UNCALIBRATED",
        "mask_count": mask_count,
        "valid_point_count": valid_count,
        "position_m": centroid.tolist(),
        "frame_id": "camera_optical_frame",
        "publishable": False,
    }
```

**验收要点：**

- 空 Mask → `NO_OBJECTS`；但它只是“目标掩码无候选像素”，**不证明真实场景绝对无物体**。
- Mask 非空且没有有效点 → `MASK_ONLY_NO_VALID_DEPTH`，`position_m=None`，绝不对空点云求均值。
- 有有效点且数值有限 → 只记录相机系的 `DIAGNOSTIC_ONLY` 诊断质心。
- 当前样本为合成输入，缺乏独立真实外参验证，**所有分支 `publishable=False`**，不允许冒充机器人基坐标或其他目标 frame 位置。
- `NO_OBJECTS` / `MASK_ONLY_NO_VALID_DEPTH` / `DIAGNOSTIC_ONLY` 是**本 Boss 提出的候选接口命名**，不是仓库已统一采用的正式枚举。OS-01-B 应与 EXP034/035 的状态和发布门禁统一。
- 此方案的异常处理及后续报告契约仍需在 OS-01-B 变成真实代码和自动化测试；**Boss 方案 PASS 不代表实现 PASS**。

### 最终回归验收（2026-10-26）

- 合成 RGB-D 样本重新生成：PASS
- 仓库全量 Pytest：696 passed，106 subtests passed，2.99s
- A1 内参扰动：PASS
- A2 深度扰动：PASS
- A3 Mask 扰动：PASS
- 综合 Boss B/C：PASS（设计层面）
- Git Commit：PENDING
- GitHub Push：PENDING

结论：知识、实验与代码回归验收 PASS；
OS-01-A 尚待 Git/GitHub 工程闭环。

### 综合 Boss 结论

| 能力项 | 判断 |
|---|---|
| B：从选点逻辑预测新 Mask 场景 | PASS |
| C1：推断空点云与 `nan` 故障 | PASS |
| C2：定位接口语义与安全门禁错误 | PASS |
| C3：独立提出可实现的 Fail-Closed 最小修复 | PASS（设计层面） |

**本单元知识与综合 Boss 验收已通过；尚未完成最终回归与 Git/GitHub，OS-01-A 仍为 CONTINUE。**

## 9. Remaining Risks / 下一阶段边界

1. 输入目前为手工构造的小尺寸样本，并未验证真实图像、深度传感器单位或 RGB/Depth 对齐。
2. `target_mask` 目前手工合成，未评估分割模型的假阳性、假阴性或目标遮挡。
3. 当前只验证采样质心，不等于物体真实几何中心，也未输出主项目所需完整点云处理及 3D 包围范围报告。
4. 尚未验证该输入协议下的端到端滤波、RANSAC、聚类、SE(3) 审计链。
5. 尚未形成至少 20 个可追溯 RGB-D 样本的主项目评测；当前单样本与 EXP036 的 20 个合成位置误差样本不能混为一谈。
6. 没有独立物理外参校验，任何机器人坐标输出仍应维持保守拒绝发布策略。

## 10. Acceptance / Exit Checklist

### 概念、公式、参数与实验

- [x] 明确当前目标、前置知识、产物与单元验收边界。
- [x] 解释 RGB/Depth/Mask/K 的 Shape、dtype、单位和 frame。
- [x] 解释反投影、有效深度过滤与质心公式。
- [x] 手算并验证 6 个 Mask 像素、5 个有效三维点与基线质心。
- [x] 完成合成 `.npz` 的保存/读取专项 Pytest，`1 passed`。
- [x] A1：`fx` 控制变量，预测与实测 PASS。
- [x] A2：深度控制变量，预测与实测 PASS。
- [x] A3：Mask 控制变量，预测与实测 PASS。
- [x] 本人用自己的话回答 Q1～Q3，导师概念复盘评定 PASS。

### 工程闭环（未完成项严禁提前勾选）

- [ ] 将本笔记保存到本地仓库 `experiments/OS01_A_rgbd_input_contract.md` 并本人复核。
- [x] 完成事先约定的**简短综合 Boss**（B/C1/C2/C3 均 PASS；C3 为设计验收，未声明实现）。
- [ ] 检查数据文件策略、`git status`、全量回归和必要测试记录。
- [ ] 仅暂存 OS-01-A 有关的正确文件；`git diff --cached --check` PASS。
- [ ] Git Commit；记录 SHA。
- [ ] GitHub Push；确认本地/远程同步。

**OS-01-A 当前状态：CONTINUE。** “实验全绿”和“文档已整理”均不等于单元已完成。

## 11. Next / 下一步

综合 Boss 已完成。先将本笔记保存并复核，再执行最终测试、Git 与 GitHub 闭环。通过后进入 **OS-01-B：现有模块组成单样本端到端 RGB-D Object Localizer**，不另起冗余 demo，也不提前宣布 Level 2 毕业。
