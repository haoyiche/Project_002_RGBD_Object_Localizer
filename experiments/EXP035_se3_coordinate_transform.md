# EXP035｜SE(3) 坐标变换、坐标链契约与安全定位集成

> Project：`Project_002_RGBD_Object_Localizer`
> 学习线：Level 2 / T1 / OS-01 / EXP035
> 验收日期：2026-10-19
> 环境：Windows PowerShell，`D:\conda_envs\ai_3d`，Open3D 0.20.0，NumPy，pytest
> **当前 Learning Unit 状态：CONTINUE**（A～E 与 Boss PASS，待笔记落库、Git commit、GitHub push、远程 SHA 核验）
> **重要安全边界：**全部相机→基座→世界外参为教学合成数据，未经真实标定验证；诊断坐标不能作为机器人的控制/发布坐标。

## 0. 目标、知识地图、前置知识、最终验收

**目标：**承接 EXP034 的 `camera_optical_frame` 三维物体位置，能够解释、手算、实现、修改、调试、批量处理和组合 SE(3) 变换，并为错误输入、错误坐标链、无效簇、污染缓存、伪造标定声明提供 fail-closed 行为。

**知识地图：**EXP029 三维几何统计 → EXP033 DBSCAN 标签 → EXP034 逐簇几何定位与 `audit_and_extract()` → **EXP035 相机坐标→基座坐标→世界坐标、SE(3)、安全审计** → 相机外参标定/手眼标定、坐标链维护、机器人空间定位。

**前置条件：**已理解 NumPy `(N,3)`、矩阵乘法、EXP034 的 `VALID`/`INVALID_CLUSTER` 和 `position_m`/`diagnostics` 契约；能区别相机系、基座系与世界系。

**学习产物：**A～E 实验脚本、严格 SE(3) 接口、诊断专用位置流水线、故障注入、Boss 独立代码与测试、5 份 JSON 结果、Markdown 笔记和 Git/GitHub 记录。

**验收门槛（事先约定）：**完成 A～E 预测与实测；Boss ≥85/100，工程实现 C ≥20/30、Debug D ≥12/20，安全关键测试零跳过；完整回归通过；笔记归档；Git commit、push 与本地/远端 SHA 对齐。

## 1. 原理与公式（对应 SOP `01_原理.md`）

### 1.1 SO(3)、SE(3)、方向约定

- `SO(3)`：三维**纯旋转**，要求 `R.T @ R ≈ I` 且 `det(R) ≈ +1`。`det(R)=-1` 的镜像反射不是合法旋转。
- `SE(3)`：三维**刚体变换**，由旋转 `R` 与平移 `t` 组成。坐标采用**列向量**定义，长度统一为米：

\[
\mathbf p_{target}=R_{target\leftarrow source}\,\mathbf p_{source}+\mathbf t_{target\leftarrow source},\qquad
T=\begin{bmatrix}R&t\\0&1\end{bmatrix}.
\]

- `t_target<-source` 是**源坐标系原点在目标坐标系中的坐标**。不能把不同坐标系下的 `p` 与 `t` 随意相加。
- `T_base_camera` 表示 `camera_optical_frame → robot_base`；右侧先作用，`T_world_camera=T_world_base @ T_base_camera`。

### 1.2 逆变换和复合变换

\[
R^{-1}=R^T,\qquad t^{-1}=-R^Tt,\qquad
\mathbf p_{source}=R^T(\mathbf p_{target}-t).
\]

\[
R_{wc}=R_{wb}R_{bc},\qquad t_{wc}=R_{wb}t_{bc}+t_{wb}.
\]

必须**先旋转第一级平移**。纯旋转保持欧氏范数；完整平移+旋转一般不保持点到原点的距离（但保持任意两点之间的距离）。

### 1.3 NumPy 行向量、Shape、单位

| 项目 | Shape | 单位 | 工程意义 |
|---|---|---|---|
| `point_m` | `(3,)` | m | 单点，代码用 `R @ p + t` |
| `points_m` | `(N,3)` | m | 每行是一个点，代码用 `points @ R.T + t` |
| `R` | `(3,3)` | 无量纲 | 源系坐标到目标系的旋转 |
| `t_m` | `(3,)` | m | 源系原点在目标系的坐标 |
| `T` | `(4,4)` | 混合 | 齐次变换，最后一行为 `[0,0,0,1]` |
| `frame_id` | 字符串 | 不适用 | 坐标语义，改字符串不会发生物理变换 |
| `calibration_verified` | bool | 不适用 | **声明**，不等于可信标定证据 |

NumPy `points @ R.T + t` 中，`t` 从 `(3,)` 广播到 `(N,3)`；不能漏掉行向量约定下的 `.T`。毫米数据须在输入适配层换算为米；仅凭 Shape、正交性、Round-trip、数值合理性不能证明单位或标定真实正确。

### 1.4 三种可靠性不能混淆

1. **数值正确：**运算、矩阵、Shape、有限性和数值真值正确。
2. **坐标系语义正确：**变换方向、源/目标 `frame_id`、组合顺序、单位契约一致。
3. **物理可信且允许发布：**真实标定证据、安装状态、时效、误差验证、时间同步、工作空间和业务发布门禁全部满足。

EXP035 实际实现只授权**诊断计算**，不授权真实机器人发布：`publishable=False`，`published_positions=[]`，变换目标的 `position_m=null`；诊断位置保存在 `diagnostics` 中。即使 `calibration_verified=True`，仍需外部可信验证链。

## 2. 输入检查与安全契约（对应 SOP `02_核心参数.md`）

| 参数/检查 | 规则 | 错误影响/检测 |
|---|---|---|
| `R` | `(3,3)`，有限、实数，`R.T@R≈I`，`det≈+1` | 反射或非正交输入抛 `ValueError`；教程容差 `1e-9` |
| `t_m` | `(3,)`，有限实数，米 | 单位错但数值合法时，仍可能悄然输出错误坐标 |
| `point_m` | `(3,)`；`points_m` 为 `(N,3)` | 拒绝布尔、NaN/Inf、错误 Shape |
| `frame_id` | 必须等于 `source_frame_id` | 不匹配时拒绝，防止语义错帧 |
| `compose(outer, inner)` | `outer.source_frame_id == inner.target_frame_id` | 断链抛 `ValueError` |
| `published_positions` | 缺失时可重建；存在则与 `objects` 审计结果逐项比较 | 多余 INVALID、重复标签、位置/坐标系矛盾须拒绝；比较 `rtol=0` |
| `calibration_verified` | 必须为 bool；视为未认证的状态声明 | True 也不能替代标定记录，更不能直接设发布权限 |
| 诊断数据 | `demo_only=True`、记录链路和失败原因 | 不允许将 `p_target_demo_m` 接到下游控制输出 |

**关于容差：**`np.allclose(a,b,atol=1e-12)` 仍带默认相对容差；对于安全缓存一致性必须显式写 `rtol=0, atol=1e-12`（本关演示契约），但容差不是签名/完整性认证。

## 3. 实验记录：预测 → 实测 → 解释（对应 SOP `03_实验记录.md`）

### A：最小 SE(3) 正逆变换

**输入：**EXP034 物体 B，`p_camera=[0.51,0.01,1.152]` m；绕 Z +90°，`t=[0.30,-0.10,0.20]` m。**运行前手算：**旋转 `[-0.01,0.51,1.152]`，基座 `[0.29,0.41,1.352]`，错误 `R@(p+t)` 得到 `[0.09,0.81,1.352]`，与正确差 `[-0.20,+0.40,0]` m；逆变换恢复原点。

**Windows 实测：**NumPy 独立真值、齐次逆、Open3D 正逆一致，**15 passed**；当时全量 **263 passed，106 subtests passed**。数值长尾是浮点舍入误差。

### B：三组主动修改与多级坐标链

| 控制变量 | 运行前预测 | Windows 验证 | 分析 |
|---|---|---|---|
| B1 `R=I` 只平移 | `[0.81,-0.09,1.352]` | PASS | 相对 A 差 `[+0.52,-0.50,0]` m |
| B2 `t=0` 只旋转 | `[-0.01,0.51,1.152]` | PASS | 旋转保持 `||p||²=1.587304` |
| B3/B4 再加 base→world，Z +90° | `p_world=[-0.61,0.69,1.352]`，`t_wc=[-0.10,0.70,0.20]` | PASS | 复合与逐级一致；错误交换完整矩阵导致 world Y 差 `-1.00 m` |
| B5 第二级未验证 | 不发布世界坐标 | PASS | 诊断数值与可信发布分离 |

**Windows 实测：**Open3D B1/B2/B3 正逆一致，**28 passed**；全量 **291 passed，106 subtests passed**。

### C：严格 SE(3) 工程模块

新增 `src/geometry/rigid_transform.py`，实现不可变的 `RigidTransform`、`transform_point()`、`transform_points()`、`inverse()`、`compose()`、`as_matrix()` 和 `inspect_numeric_position()`。

**先预测：**`diag(1,1,-1)` 正交但反射，`det=-1` 必须拒绝；断开坐标链必须拒绝；100 个点经 `points @ R.T + t` 后仍是 `(100,3)`；逆平移是 `[+0.10,+0.30,-0.20]` m；`calibration_verified=True` 不自动发布。

**Windows 实测：**所有预期行为与 Open3D 对照 PASS，**64 passed**；全量 **355 passed，106 subtests passed**。

### D：故障注入 RED → PATCH → GREEN

| 故障 | 运行前预测：错误结果及偏差 | 实测 |
|---|---|---|
| D1 行向量漏 `.T` | 错误 base `[0.31,-0.61,1.352]`，偏差 `[+0.02,-1.02,0]` m | RED 检出，GREEN 通过 |
| D2 逆平移使用 `-t` | 错误 camera `[0.11,-0.19,1.152]`，偏差 `[-0.40,-0.20,0]` m | RED 检出，GREEN 通过 |
| D3 复合平移不旋转 | 错误 world `[-0.41,0.29,1.352]`，偏差 `[+0.20,-0.40,0]` m | RED 检出，GREEN 通过 |
| D4 删除 frame_id 检查 | 数值正常返回，但坐标语义错误 | RED 检出，GREEN 拒绝错帧 |
| D5 发布由 bool 直接决定 | 自声明 True 可导致非法放行 | RED 检出，GREEN fail-closed |

**Windows 实测：**五组 RED 均被预期 oracle 检出、五组 GREEN 通过、Open3D PASS，**31 passed**；全量 **386 passed，106 subtests passed**。

### E：EXP034 → SE(3) 端到端审计

来源：`results/exp034/exp034_C_validated_localization.json`。EXP034 物体 A `cluster_label=0` 为 `INVALID_CLUSTER`（`position_m=null`），B `cluster_label=1` 为 `VALID`（`[0.51,0.01,1.152]` m）。`audit_and_extract()` 只返回 B 的局部可用坐标；A 的诊断质心不得绕过门禁。

- **E1：**只提取 label 1，有效簇与无效簇隔离 PASS。
- **E2：**未验证相机→基座外参，base **诊断** `[0.29,0.41,1.352]`，状态 `BLOCKED_UNCALIBRATED`、`position_m=null`、`publishable=False`，PASS。
- **E3：**仅把标定声明置 True，状态 `READY_FOR_GLOBAL_VALIDATION`，理由 `GLOBAL_VALIDATION_REQUIRED`，仍不可发布，PASS。
- **E4：**错误 `frame_id` 与被注入无效簇的 `published_positions` 缓存均拒绝，PASS。
- **E5：**两级链中第二级未验证，world **诊断** `[-0.61,0.69,1.352]`，最终发布列表保持空，PASS。

**Windows 实测：**E1～E5、Open3D PASS，**54 passed**；全量 **440 passed，106 subtests passed**。`validated_position_pipeline.py` 严格核对缓存，不把 `calibration_verified` 声明当认证。

## 4. Boss Test 与 Debug 记录（对应 SOP `04_常见问题与误区.md`）

### 4.1 全新独立手算（Boss B）

输入：`p_camera=[0.20,0.10,1.00]`，`t_bc=[0.40,-0.20,0.10]`，`R_bc=Rz(+90°)`；`R_wb=diag(1,-1,-1)`，`t_wb=[-0.30,0.50,0.20]`。

手算得 `p_base=[0.30,0,1.10]`、`p_world=[0,0.50,-0.90]`；复合旋转 `R_wc=[[0,-1,0],[-1,0,0],[0,0,-1]]`，复合平移 `t_wc=[0.10,0.70,0.10]` m。直接复合、逐级、逆变换关系一致。

### 4.2 Boss C/D 首次补测缺陷、根因、修复

1. **整帧/全对象 frame 检查漏项**：只审计 VALID 对象，漏检无效簇的错误 `frame_id`；修复为检查 `report.frame_id` 与**全部** `objects[*].frame_id`。
2. **缓存比较容差漏洞**：`np.allclose(...,atol=1e-12)` 未禁用默认 `rtol`，可能接受 `1e-7` m 级污染；修复为显式 `rtol=0`。
3. **变换对象过宽的兼容策略**：只检查有无 R/t 属性，可能绕过 `det(R)=+1`；修复为严格 `isinstance(tr,RigidTransform)`，复用经过校验的 `transform_point()`。
4. **冻结对象测试误用**：`RigidTransform` 是 `@dataclass(frozen=True)`；不能通过 `obj.calibration_verified=True` 篡改测试对象，原始测试因捕获异常而 skip。修复为**构造一个新的** `calibration_verified=True` 实例，禁止跳过关键安全测试。
5. **RED/GREEN 对照不够强**：只测新实现拒绝，还要真正调用故意不安全的 `unsafe_pipeline()`，确认它错误接受污染缓存/错误 frame，再验证新实现拒绝。对布尔声明误发布，应断言旧实现可能返回 `publishable=True`，新实现恒 False。

**Boss 初评：84/100**，C/D 补测后复评 **90/100**，补丁通过后达到题目既定门槛。Windows 用户于 **2026-10-19** 提供最终实测：Boss **13 passed，0 skipped**；全量回归 **453 passed，106 subtests passed**。

**注意：**Boss 的实际通过以 Windows 的 13 项和全量回归结果为依据，90/100 是此前评审给出的评分，而非 pytest 自行计算出的分数。

### 4.3 个人可复述要点（根据本轮独立预测及 Boss 作答整理）

- 列向量写 `R@p+t`，行批量写 `points@R.T+t`；`.T` 绝不能省略。
- 逆变换平移为 `-R.T@t`，复合平移为 `R_outer@t_inner+t_outer`；Shape 通过并不代表物理语义正确。
- `SO(3)` 检查正交和 `det=+1`，只能保证数学旋转合法，不能保证标定真实。
- 单位、时间、标定记录、坐标链、簇级几何门禁和最终发布权限互相独立；必须在系统边界逐层核对。
- `diagnostics` 可以提供用于工程排查的 demo 坐标；不能回填 `position_m` 或发送给机器人运动控制器。

### 4.4 后续仍需加强

- 真实相机外参/手眼标定及标定复验（重复安装、碰撞、温漂、时间漂移）。
- 可信标定证据管理、版本/签名/有效期、坐标系链登记与时钟同步。
- 目标位置误差的实测标定（物理 GT、多姿态多距离、多样本和统计误差）。

## 5. 代码资产与复现实验

### 5.1 文件清单

```text
src/geometry/rigid_transform.py
src/geometry/validated_position_pipeline.py
src/experiments/exp035_se3_minimal.py
src/experiments/exp035_transform_ablation_b.py
src/experiments/exp035_se3_contract_c.py
src/experiments/exp035_fault_injection_d.py
src/experiments/exp035_safe_pipeline_e.py
src/experiments/exp035_boss_student.py
tests/test_exp035_se3_minimal.py
tests/test_exp035_transform_ablation_b.py
tests/test_exp035_se3_contract_c.py
tests/test_exp035_fault_injection_d.py
tests/test_exp035_safe_pipeline_e.py
tests/test_exp035_boss_student.py
results/exp035/exp035_A_se3_minimal.json
results/exp035/exp035_B_transform_ablation.json
results/exp035/exp035_C_rigid_transform_contract.json
results/exp035/exp035_D_fault_injection.json
results/exp035/exp035_E_safe_pipeline.json
experiments/EXP035_se3_coordinate_transform.md
```

`results/exp035/*.json` 为教学合成实验报告；应检查尺寸并确认不包含原始大文件或隐私数据后再提交。

### 5.2 PowerShell 最小复现

在 `D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer`，使用 `D:\conda_envs\ai_3d` 环境：

```powershell
python -m pytest -q tests/test_exp035_boss_student.py
if ($LASTEXITCODE -ne 0) { throw 'EXP035 Boss failed' }

python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'P02 regression failed' }

git status --short --untracked-files=all
```

**2026-10-19 最后一次 Windows 结果：**专项 `13 passed in 0.19s`；全量 `453 passed, 106 subtests passed in 1.66s`。实验过程中的 Open3D 均在 Windows `0.20.0` 验证通过。

## 6. Git / GitHub / Exit Protocol

- **知识/理论/公式：**PASS。
- **最小实验/主动修改/先预测后验证：**A～E PASS。
- **Bug 预测、可复现 RED/PATCH/GREEN：**PASS。
- **最终 Boss：**90/100；Windows `13 passed, 0 skipped`，PASS。
- **本笔记：**已整理，待保存进项目 `experiments/` 并人工确认。
- **Git Commit / Push / SHA：**本笔记撰写时尚未执行；不得把本文件的存在误认为远程已同步。

**明确排除的历史未追踪文件：**`ScreenCamera_*.json`、`ScreenCapture_*.png`、`*.bak_*`、`src/experiments/exp034_cluster_geometry_minimal.py`，以及未经审核的其他未追踪文件。使用精确 `git add -- <paths>`，不使用 `git add .`。

**待 Git 完成后，在对话中核验：**提交信息、`git log -1 --oneline`、`git rev-parse HEAD`、`git ls-remote origin refs/heads/master`，确认本地与远程 master SHA 完全一致；如当前分支不是 master，则以实际分支为准，不做盲推。

**下一阶段：**在本关完整归档后，转向外参真实标定、坐标变换误差评估与机器人/世界系可信定位，不把教学示例视为真实部署方案。
