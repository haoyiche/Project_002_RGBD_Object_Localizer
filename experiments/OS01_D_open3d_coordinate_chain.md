# OS-01-D｜Open3D 集成、背景分割、坐标链与同帧一致性

> Project：`Project_002_RGBD_Object_Localizer`（P02）  
> 当前 Level：**Level 2 · CONTINUE**  
> Learning Unit：**OS-01-D · 工程实验、主动预测、综合 Boss、全量回归 PASS；Git/GitHub 待闭环**  
> 实测环境：Windows PowerShell / Python 3.11.17 / Open3D 0.20.0 / Conda `ai_3d`  
> 最近一次本机全量回归（2026-10-29）：**732 passed, 106 subtests passed in 6.41s**  
> 参考的已完成阶段：OS-01-A/B/C；EXP028/030/031/032/035/037。

## 1. 目标、知识链、最终产物和验收标准

**目标：** 将已经完成的 NumPy 点云处理、Open3D 操作、背景分割和 SE(3) 坐标变换串联为一套可检查、可复现、始终 Fail-closed 的合成 RGB-D 诊断流程；进一步防止“D2 验证帧 A，D3 处理帧 B”的数据版本混用问题。

```text
OS-01-A 输入契约 + OS-01-B 点云与聚类 + OS-01-C 可追溯评估
       ↓
D1：NumPy / Open3D 对照 + Voxel 改参数
       ↓
D2：完整场景背景 RANSAC + Mask 逐点身份检验
       ↓
D3：camera_optical_frame → world_demo 数值变换
       ↓
D4：同一 Capture 快照 / 帧 UID / 内容哈希 / 有效期绑定
       ↓
诊断结果；始终 position_m=None、publishable=False
```

**产物：** D1/D2/D3/D4 四个源文件、四个对应测试文件、本学习笔记、README 更新、最终 Git/GitHub 记录。

**验收门槛：** 理论、公式与参数可解释；先预测后运行；主动改变体素大小；故障注入（身份交换、错误变换、上游失败、A/B 帧混用、内容变化、票据过期）；全部专项与全仓测试通过；笔记归档、Git Commit、Push、远端 SHA 核对。真实物理精度与机器人发布**不在此合成单元的完成声明中**。

## 2. 最小必要理论和数据契约

### 2.1 点云 / 像素 / 颜色

| 变量 | Shape | 类型 | 单位 / 语义 |
| --- | --- | --- | --- |
| `depth_m` | `(H,W)` | 数值 | 米，Z 深度；0、负值、NaN、Inf 无效 |
| `rgb` | `(H,W,3)` | `uint8` | RGB 0～255 |
| `target_mask` | `(H,W)` | `bool` | 目标像素的教学标签 |
| `points_camera` | `(N,3)` | `float64` | 相机系 XYZ，米 |
| `colors` | `(N,3)` | `float64` | RGB 归一化至 [0,1] |
| `dbscan_labels` | `(N,)` | 整型 | -1 是 Noise；非负是簇 ID |
| `R` | `(3,3)` | 数值 | 无量纲，右手系刚体旋转 |
| `t_m`、`centroid` | `(3,)` | 数值 | 米 |

反投影（`fx,fy` 单位 px，`u,v,cx,cy` 单位 px）：

\[
X=\frac{(u-c_x)Z}{f_x},\qquad Y=\frac{(v-c_y)Z}{f_y},\qquad Z=\mathrm{depth_m}[v,u].
\]

### 2.2 Radius Filter、DBSCAN、Voxel

- Radius Filter 的 `min_neighbors` 统计**其他点**；DBSCAN 的 `min_points` 邻居数**包含自身**。
- DBSCAN 的 Core / Border / Noise 判定依赖输入点的数量、密度、`eps`（米）及 `min_points`；更改 Voxel 与 DBSCAN 的顺序不能假定结果不变。
- Open3D `voxel_down_sample(voxel_size)` 为非空体素输出代表点（本例为体素内部质心）。若直接对**各体素代表点等权**求质心，不一定等于原始点的等权质心。保留原始每体素点数后可进行加权恢复。
- RANSAC `distance_threshold` 的单位是米。它找的是符合几何模型的内点，**无法自动辨认“桌面”与“目标”**；背景平面移除必须基于场景先验及目标身份校验。

### 2.3 坐标变换方向与安全含义

`RigidTransform(source_frame_id="camera_optical_frame", target_frame_id="world_demo", R, t_m)` 定义：

\[
P_{world\_demo}=R_{world\_demo\leftarrow camera}P_{camera}+t_m.
\]

此处 `world_demo` 为**人为设定的合成坐标系**，不是经过标定的机器人 `base_link`；Round-trip 接近零仅证明数值正逆变换自洽。

所有 D1～D4 合格诊断仍不满足实际机器人发布资格：

```python
report["publishable"] is False
report["position_m"] is None
```

## 3. D1｜五点 Open3D 对照与体素主动修改

**代码：** `src/pipeline/os01_d_open3d_compare.py`  
**测试：** `tests/test_os01_d1_open3d_compare.py`。

### 3.1 预测和实测基线

基于 OS-01-A 的五个有效点（单位 m）：

```text
P0=( 0.005,-0.005,1.0)
P1=( 0.015,-0.005,1.0)
P2=(-0.005, 0.005,1.0)
P3=( 0.005, 0.005,1.0)
P4=( 0.015, 0.005,1.0)
```

在 `radius=0.011 m`、`min_neighbors=1`，DBSCAN `eps=0.011 m`、`min_points=3` 下：

- NumPy / Open3D 点和颜色 Shape 均为 `(5,3)`。
- DBSCAN 含自身邻居数 `[3,3,2,4,3]`；Core 为 P0、P1、P3、P4；P2 为 Border。
- NumPy 与 Open3D 均实测标签 `[0,0,0,0,0]`，为 **1 簇、0 Noise**（簇 ID 数值本身没有固定物理意义）。
- `voxel_size=0.005 m`：实测保留五个点。

### 3.2 主动改变 `voxel_size` 为 0.020 m

**先预测：** 五点合并为三个非空体素；各代表点：

```text
[ 0.010,-0.005,1.0]：P0/P1
[-0.005, 0.005,1.0]：P2
[ 0.010, 0.005,1.0]：P3/P4
```

实测：下采样点数 **5 → 3**，与预测一致。

| 项目 | XYZ 质心（m） |
| --- | --- |
| 原始五点 | `[0.007,0.001,1.0]` |
| 三个体素代表点直接等权 | `[0.005,0.0016666667,1.0]` |
| 偏移 | `[-0.002,+0.0006666667,0]` m |

位移模长：

\[
\|\Delta P\|=\sqrt{2^2+(2/3)^2}\ \mathrm{mm}\approx2.108185\ \mathrm{mm}.
\]

如果按体素内原始点数 `[2,1,2]` 加权：

\[
\frac{2\bar P_1+\bar P_2+2\bar P_3}{5}=[0.007,0.001,1.0]\ \mathrm m.
\]

**工程结论：** 体素降采样既改变空间采样密度，也可能改变**直接等权计算**的定位质心。若下采样发生在 DBSCAN 之后，不会逆向改变已经计算的 DBSCAN 标签。

**自动化：** `test_d1_full_pipeline`、`test_d1_voxel_centroid_shift`，专项 **2 passed**。

## 4. D2｜完整 RGB-D 场景的 RANSAC 与身份检查

**代码：** `src/pipeline/os01_d2_plane_background.py`（最终版本允许 `analyze_scene(scene=(depth_m,rgb,target_mask))`，仍保留无参数演示接口）。  
**测试：** `tests/test_os01_d2_plane_background.py`。

场景直接复用 EXP028：`H=80,W=100`；背景深度 `Z=2.0 m`，目标深度 `Z=1.2 m`，目标像素 40×40，左上部有 10×10 无效深度块。

| 项目 | 先预测 | 实测 |
| --- | ---: | ---: |
| 全场景像素 | 8000 | 8000 |
| 无效深度 | 100 | 100 |
| 有效场景点 | 7900 | 7900 |
| 背景主平面内点 | 6400 | 6400 |
| 保留目标点 | 1500 | 1500 |
| `target_membership_matches` | True | True |
| `target_xyz_matches` / `target_rgb_matches` | True | True / True |
| 参考平面误差 | 近 0 | `0.0 m` |

正常报告 `status="DIAGNOSTIC_ONLY"`。本合成场景的 RANSAC `distance_threshold=0.01 m`、`ransac_n=3`、`num_iterations=300`；验收同时使用硬编码的场景规模和真值结构，因此**不是通用的生产平面分割器**。

### 4.1 故障注入：数量正确、点身份错误

交换两个像素深度：`(v=20,u=30)` 从 1.2 改为 2.0 m；`(v=0,u=0)` 从 2.0 改为 1.2 m。故障后：

```text
scene_point_count=7900
plane_inlier_count=6400
remaining_point_count=1500
mask_target_point_count=1500

target_membership_matches=False
target_xyz_matches=False
target_rgb_matches=False
status=SEGMENTATION_MISMATCH
publishable=False
position_m=None
```

**结论：** 计数一致不能证明点身份一致；同时检查原始像素归属、XYZ、RGB 才能发现这次故障。正确身份仍只是可信物理定位的必要条件之一，不是充分条件。

**自动化：** 基线与等量交换专项 **2 passed**。

## 5. D3｜SE(3) 合成坐标链与失败阻断

**代码：** `src/pipeline/os01_d3_coordinate_chain.py`；**测试：** `tests/test_os01_d3_coordinate_chain.py`。

合成外参：

\[
R=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix},\quad
 t=[0.10,-0.20,0.30]^T\ \mathrm m.
\]

无效深度块偏向左上方。目标完整 1600 像素中心为 `(49.5,39.5)`，去掉均值 `(44.5,34.5)` 的 100 个无效像素后，1500 个有效像素中心为 `(49.833333,39.833333)`。

\[
P_c=[0.004,0.004,1.2]^T\ \mathrm m,
\qquad P_{world\_demo}=[0.096,-0.196,1.5]^T\ \mathrm m.
\]

本机实测 `roundtrip_residual_mm = 5.057544569639795e-15`，仅反映浮点条件下的数值自洽。

| 情况 | D3 状态 | 后续行为 |
| --- | --- | --- |
| 合成流程全部通过 | `DIAGNOSTIC_ONLY` | 仅在 diagnostics 返回合成位置 |
| 逆向外参错误使用 | `FRAME_MISMATCH` | 原因 `TRANSFORM_DIRECTION_OR_FRAME_MISMATCH`，不产生位置 |
| 上游 D2 校验失败 | `UPSTREAM_REJECTED` | 原因 `D2_SEGMENTATION_NOT_VERIFIED`，不执行目标反投影 |
| `calibration_verified=True` 但无物理证据 | 仍仅 `DIAGNOSTIC_ONLY` | 绝不开放发布 |

**工程边界：** D3 的演示实现分别重新生成 D2 与反投影输入的同一确定性场景。虽然本例数值相同，但真实异步采集不能据此推断同帧一致性。这是 D4 要修正的接口问题。

**自动化：** 正向与 round-trip、逆向变换、上游阻断、伪造标定声明，专项 **4 passed**。

## 6. D4｜同一采集帧验证快照（Validated Capture）

**代码：** `src/pipeline/os01_d4_same_capture.py`；**测试：** `tests/test_os01_d4_same_capture.py`。

### 6.1 结构、身份、哈希与时间

```text
Capture(
    frame_uid, frame_id, capture_timestamp_ns,
    depth_m, rgb, target_mask
)
       ↓ _snapshot() 复制三个数组
       ↓ D2.analyze_scene(scene=(snapshot.depth_m, snapshot.rgb, snapshot.target_mask))
       ↓ _digest() SHA-256 绑定 frame_uid / frame_id / 时间戳 / 内参 / dtype / shape / bytes
ValidatedCapture(snapshot, input_sha256, d2_report, validated_at_ns, expires_at_ns)
       ↓ analyze_validated_capture(ticket, candidate=?, now_ns=?, transform=?)
       ↓ 上游状态 → 有效期 → 快照哈希 → 候选 UID / 元数据 / 哈希 → 外参方向
       ↓ 只使用 ticket.snapshot 进行反投影和合成数值变换
       ↓ DIAGNOSTIC_ONLY / explicit reject；始终 fail-closed
```

D4 相比 D3 的关键改进：D2 验证的点云和 D4 消费的目标像素来自**同一份快照**，而不是“先检查 A 帧、后来重新从相机读 B 帧”。传入 `candidate` 仅用于校验；实际投影仍使用 `ticket.snapshot`。

**注意：** `@dataclass(frozen=True)` 只冻结字段赋值，不会自动冻结嵌套 NumPy 数组与 `d2_report` 字典。模块采用防御性拷贝与再哈希检测变化，但这并非防恶意伪造的不可篡改票据。

### 6.2 测试与实际结果

| 自动化场景 | 预期合同 | 2026-10-29 本机结果 |
| --- | --- | --- |
| 同一 Capture 消费 | `DIAGNOSTIC_ONLY`，合成质心正确 | PASS |
| A 帧票据 + B 帧 UID | `FRAME_UID_MISMATCH`；反投影前阻断 | PASS |
| 同 UID 但 Depth 改动 | `INPUT_DATA_HASH_MISMATCH` | PASS |
| 票据有效期已过 | `STALE_VALIDATION_TICKET` | PASS |
| D2 点归属失败 | `UPSTREAM_REJECTED`；反投影前阻断 | PASS |
| 逆向外参 | `FRAME_MISMATCH` | PASS |
| 人为声明标定通过 | 仍 `publishable=False` | PASS |

专项输出：**7 passed in 0.40s**。

直接执行 `python -m src.pipeline.os01_d4_same_capture` 输出：`DIAGNOSTIC_ONLY`；`frame_uid=os01-d4-demo-frame-0001`；`target_point_count=1500`；`centroid_camera_m≈[0.004,0.004,1.2]`；`centroid_world_demo_m≈[0.096,-0.196,1.5]`；`publishable=False`、`position_m=None`；终行 `D4 same-capture diagnostic: PASS`。

### 6.3 不能夸大的安全范围

1. 票据未由外部认证方签发或签名；攻击者若同时改写内容、哈希和报告，不受纯 SHA-256 保护。
2. 有效期的 `now_ns` 由调用方提供，并非可信时钟或真实采集硬件的同步证明。
3. 内参为固定合成教学配置；没有可信相机标定链。
4. D2 的状态检查针对**此固定合成场景**，不能外推为生产级分割质量保证。
5. 模型内没有真实物理测量、工作空间覆盖、姿态标定、安全规划或机器人执行接口。

因此 D4 的结论是：**同进程、固定合成样本上的数据身份与内容一致性机制通过验证**，不是生产级安全认证。

## 7. OS-01-D 综合 Boss A～D

### Boss A：流程顺序

把 DBSCAN 之前加入 `voxel_size=0.020 m` 的下采样，输入三个体素代表点时，`eps=0.011 m`、`min_points=3`：邻居（含自身）数 `[2,1,2]`，无 Core，三个点均 Noise。代表点 `(0.010,-0.005,1)` 与 `(0.010,0.005,1)` 距离 10 mm；约 18.03 mm 指左侧单点与第一个双点代表点的距离。**Boss A PASS（细节已修正）。**

### Boss B：失败传播

D2 `SEGMENTATION_MISMATCH` → D3 `UPSTREAM_REJECTED`，拒绝反投影。不能凭 1500 这个数量放行。**Boss B PASS（采用实际状态名）。**

### Boss C：坐标链

`P_c=[0.004,0.004,1.2] m` → `P_world_demo=[0.096,-0.196,1.5] m`；反向外参返回 `FRAME_MISMATCH`。Round-trip 为零和 `calibration_verified=True` 不构成物理标定证据。**Boss C PASS。**

### Boss D：跨帧混用迁移

`frame_id` 只表明坐标系，不表明采集实例。需要绑定 UID、时间戳、输入内容及验证状态；A/B 帧混用、同 UID 改动和过期均应拒绝。已通过 D4 自动测试验证**本阶段设计的最小实现**。**Boss D PASS。**

## 8. 可复现命令与回归证据

在仓库根目录、Open3D 已安装的 `ai_3d` 环境中执行：

```powershell
python -m src.experiments.os01_a_synthetic_rgbd_sample
python -m src.pipeline.os01_d_open3d_compare
python -m src.pipeline.os01_d2_plane_background
python -m src.pipeline.os01_d3_coordinate_chain
python -m src.pipeline.os01_d4_same_capture

python -m pytest tests/test_os01_d1_open3d_compare.py -v
python -m pytest tests/test_os01_d2_plane_background.py -v
python -m pytest tests/test_os01_d3_coordinate_chain.py -v
python -m pytest tests/test_os01_d4_same_capture.py -v
python -m pytest -q

git diff --check
git status --short
```

2026-10-29 本机实测结果：

```text
D1: 2 passed
D2: 2 passed
D3: 4 passed
D4: 7 passed
Full repository: 732 passed, 106 subtests passed in 6.41s
Git diff --check: no reported error
```

本地尚未跟踪 D1～D4 的 4 个源文件及 4 个测试文件；六个 `ScreenCamera_*.json` / `ScreenCapture_*.png` 与本单元无关，**禁止纳入本关 Commit**。`git diff --check` 无错误不代表新文件已经通过暂存检查，必须在 `git add` 后重新运行 `git diff --cached --check`。

## 9. 易错点与排错记录

| 误区 / 故障 | 根因 | 验证与处理 |
| --- | --- | --- |
| `voxel_size` 增大不影响质心 | 把体素等权代表点误认为原始点等权 | 5 点 vs 3 代表点，质心差约 2.108 mm |
| DBSCAN 与 Voxel 可任意交换 | 下采样改变点数与密度 | `[3,3,2,4,3]` → `[2,1,2]`，从 1 簇变全 Noise |
| 目标平面可直接删除 | RANSAC 不认识“目标”标签 | 先用完整场景与明确上下文识别背景，再对照 Mask |
| 6400/1500 数量一致便通过 | 可能出现一进一出的身份交换 | 比对同源像素布尔归属、XYZ、RGB |
| 正逆变换 Round-trip 0 就是真实外参正确 | 错误外参与其逆也可能自洽 | 必须独立物理真值和标定证据 |
| `calibration_verified=True` 就能发布 | 布尔声称不等于可信证据 | 强制 `publishable=False` |
| 两帧有相同 `frame_id` 就是同帧 | `frame_id` 仅是坐标系身份 | 绑定 UID、时间戳、内容指纹和 D2 报告 |
| SHA-256 就是认证签名 | 哈希不证明签发方可信 | D4 明确只是教学数据完整性演示 |
| D1 Pytest 找不到测试文件 | 文件名误粘贴为重复拼接 | 重命名文件后专项 2 passed；全仓由 717→719 |

## 10. SOP 结束验收与下一步

| 验收项 | 当前状态 |
| --- | --- |
| 目标、知识地图、最小理论 | PASS |
| Shape、单位、核心参数解释 | PASS |
| 最小实验、主动修改与先预测后验证 | PASS |
| D1/D2/D3/D4 实测与故障注入 | PASS |
| 综合 Boss A/B/C/D | PASS |
| 全仓回归 732 + 106 subtests | PASS |
| 学习笔记 | 已整理，待本人保存到仓库 |
| README | 待更新入库 |
| Git Commit | **PENDING** |
| GitHub Push / 远端 SHA | **PENDING** |
| OS-01-D | **IN PROGRESS（仅工程收尾未完成）** |
| Level 2 | **CONTINUE** |

**真正掌握：** 输入与输出 Shape；像素反投影；体素权重、密度与 DBSCAN；背景分割与目标身份；SE(3) 方向；同帧数据绑定与 Fail-closed。

**仍不稳定 / 未实证：** 生产级可信票据和时间源、不可变状态容器、真实内外参校准、真实采集、多目标身份、物理误差与工作空间覆盖；Open3D RANSAC 在噪声场景下的鲁棒性。

**下一步：** 保存本笔记和 README，精确暂存 D1～D4 的八个文件及文档，审查 `git diff --cached --check`，重新核对测试，Commit、Push，并确认 `HEAD == origin/master`。只有完成 GitHub 闭环，才把 **OS-01-D 标为 COMPLETE**；之后再回到 Level 2 总体能力验收，不继续无限延长本合成演示。
