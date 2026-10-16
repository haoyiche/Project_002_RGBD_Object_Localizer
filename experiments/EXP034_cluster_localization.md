# EXP034：DBSCAN 簇到三维物体定位与发布门禁

> Project：`Project_002_RGBD_Object_Localizer`
>
> 学习线：Level 2 / T1 / OS-01 / EXP034
>
> 日期：2026-10-16
>
> 坐标系：`camera_optical_frame`；除特别标注，长度统一为 **m**
>
> **Learning Unit 状态：CONTINUE**（Boss Test 通过，待笔记确认、Git commit、GitHub push 和 SHA 核验）

## 0. 本关目标、前置知识与验收

**目标：**将 EXP033 的 DBSCAN 标签转换为可解释、可验证的三维物体定位结果，并设计局部门禁及输出契约，避免把明显异常的几何结果直接传给下游。

**知识地图：**EXP029 三维几何统计（均值、极值、AABB）→ EXP033 DBSCAN（`labels` 和噪声 `-1`）→ **EXP034 逐簇定位、门禁、报告审计** → 后续坐标变换、标定与机器人空间任务。

**输入：**`points: (N,3)`、`labels: (N,)`、`frame_id: str`、`LocalizationGates`。`points` 单位为 m，标签 `-1` 表示噪声，非负整数代表有效簇编号；标签允许不连续。

**最终产物：**最小定位实验、采样偏差/标签污染/阈值消融实验、工程接口、报告审计、专项测试、可复现 JSON 结果、本学习笔记、Git/GitHub 历史。

**验收条件：**能手算和预测、解释代码和参数、主动修改并比较预测与实测、定位并修复契约缺陷、通过 Boss Test、完成笔记与 GitHub 同步。仅 pytest 全绿不足以通关。

---

## 1. 原理（对应 SOP `01_原理.md`）

### 1.1 每个簇的几何统计

对第 `k` 个簇，令 `cluster = points[labels == k]`，Shape `(M,3)`，`M>=1`。对 `cluster` 执行 `axis=0` 计算 XYZ 各轴：

\[
\mathbf c=\frac{1}{M}\sum_{i=1}^M\mathbf p_i,\qquad
\mathbf l=\min_i\mathbf p_i,\qquad
\mathbf u=\max_i\mathbf p_i
\]
\[
\operatorname{extent}=\mathbf u-\mathbf l,\qquad
\operatorname{bbox\_center}=\frac{\mathbf l+\mathbf u}{2},\qquad
\operatorname{gap}=\|\mathbf c-\operatorname{bbox\_center}\|_2
\]

- `centroid_m`：**点样本均值**。对采样密度敏感，不等于真实体积质心。
- `bbox_min_m` / `bbox_max_m`：各轴极值，最容易受外侧错误点影响。
- `bbox_center_m`：包围盒中心，不依赖内部点的重复计数。
- `extent_m`：轴对齐包围盒的长宽高，方向与当前坐标轴绑定；不等于物体在任意方向上的精确尺寸。
- `centroid_bbox_distance_m`：样本分布与包围盒中心的偏移线索，不能单独证明簇有效/无效。

### 1.2 输入合法性与几何可靠性是不同层次

- `ValueError`：输入违反函数契约，如 NaN/Inf、非法 Shape、点与标签数目不一致、负标签小于 `-1`、无效门禁参数。
- `INVALID_CLUSTER`：输入合法，某个簇却没有通过所配置的局部几何门禁；保留 `diagnostics`，设 `publishable=false`、`position_m=null`。
- `VALID`：簇通过当前局部门禁，可供**下一阶段处理**；不代表真实世界绝对定位正确，更不证明抓取安全。
- `NO_OBJECTS`：空场景或全噪声，是合法结果，`objects=[]`，不是 API 输入错误。

### 1.3 状态机与逐簇隔离

`validate_cluster_localization()` 逐个遍历 `np.unique(labels[labels>=0])`；噪声跳过、簇 ID 不重编号。每簇累积 `reasons`，`accepted = not reasons`。

| 条件 | 整帧 `report['status']` |
|---|---|
| 不存在非负簇 | `NO_OBJECTS` |
| 所有簇都 `VALID` | `OK` |
| 所有簇都 `INVALID_CLUSTER` | `ALL_INVALID` |
| 部分有效、部分无效 | `PARTIAL_INVALID` |

**注意：**当前代码使用 `OK`，不是 `ALL_VALID`。整帧状态和单个簇的 `status` 是两个不同字段。A 失败不应自动导致几何上无关的 B 失败；但全局标定、时间同步或坐标系错误可要求所有结果禁止发布。

### 1.4 输出契约

每个簇的结构包括：

```json
{
  "cluster_label": 0,
  "frame_id": "camera_optical_frame",
  "status": "INVALID_CLUSTER",
  "publishable": false,
  "position_m": null,
  "reasons": ["EXTENT_EXCEEDED", "CENTROID_BBOX_GAP"],
  "diagnostics": {"point_count": 10, "centroid_m": [0.099, 0.059, 1.16]}
}
```

示例省略了 `diagnostics` 的其余字段；实际模块包含 `bbox_min_m`、`bbox_max_m`、`bbox_center_m`、`extent_m`、`centroid_bbox_distance_m`。

**`frame_id` 只是声明坐标系。** 更改这个字符串不会执行单位转换或 SE(3) 变换。当前 `VALID` 局部门禁没有验证真实标定质量，生产部署需要独立的全局安全链路。

---

## 2. 关键参数（对应 SOP `02_核心参数.md`）

| 参数 | 类型、单位 | 默认演示值 | 增大/放宽的影响 | 减小/收紧的影响 |
|---|---|---|---|---|
| `min_points` | int、个 | 5 | 更容易拒绝稀疏簇 | 更多小簇通过 |
| `max_points` | int、个 | 20 | 更多密集簇通过 | 更容易拒绝高点数簇 |
| `max_extent_m` | 三元长度、m | `(0.04, 0.04, 0.02)` | 容许更大的包围盒，可能漏检污染 | 容易拒绝正常大零件 |
| `max_centroid_bbox_gap_m` | float、m | 0.02 | 容忍更强分布不对称，可能误接受 | 对正常遮挡/非均匀采样更容易误拒绝 |
| `frame_id` | 非空字符串 | `camera_optical_frame` | 非数值参数，无旋转平移效果 | 必须与实际坐标一致 |

本关默认值是**专为合成样例设计的教学阈值**，不是通过真实产线标定的生产参数。门禁可以互补，但门禁通过不是定位准确的充分条件。

---

## 3. 实验记录（对应 SOP `03_实验记录.md`）

### 3.1 环境与控制变量

用户 Windows 环境：`D:\conda_envs\ai_3d`；Open3D `0.20.0`；PowerShell。脚本运行位置：`D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer`。

各实验先给出人工预测，再用 NumPy 手算真值/断言、Open3D 对照和 pytest 验证。对照实现一致只能说明数值计算一致，不能替代真实物体真值。

### 3.2 EXP034-A：最小簇定位

**输入：**A=5、B=5、噪声=2，共 12 点，label 为 `0,1,-1`。

**预测与实测：**两个有效簇，A 质心 `(0.01,0.01,1.152)` m，B 质心 `(0.51,0.01,1.152)` m；A/B 的 AABB Z 中心均约 `1.155` m；噪声不进入几何统计。独立人工计算与 Open3D 对照 PASS。

**结论：**通过聚类标签把混合点云拆解为独立几何实体，`np.unique` 和布尔索引不得把噪声当物体。

### 3.3 EXP034-B1：重复采样引起的质心偏差

**唯一修改：**在 A 簇重复增加四个 `(0.01,0.01,1.160)` m 点，标签均为 0。A 从 5 → 9 点，B=5，噪声=2。

**运行前预测：**A 新质心 `(0.01,0.01,1.155556)` m，Z 上移约 `3.555556` mm；AABB 的 min/max/center/extent 均不变。

**实测：**偏移 `+3.555556` mm，质心 Z 超过原 AABB 中心约 `0.555556` mm；Open3D PASS；B1 专项 `8 passed`。

**结论：**点云样本均值受采样密度影响；包围盒对在其范围内添加重复点不敏感，但**不能因此声称对外侧离群点鲁棒**。

### 3.4 EXP034-B2：单个标签误归属

**唯一修改：**B1 的坐标完全不变，只将 `P11=(0.90,0.50,1.200)` 的标签 `-1→0`。A=10，B=5，噪声=1。

**运行前预测与实测：**

| 指标 | B1 | B2 | 变化 |
|---|---|---|---|
| A 质心（m） | `(0.01,0.01,1.155556)` | `(0.099,0.059,1.160)` | `(+89,+49,+4.444444)` mm |
| A AABB extent（m） | `(0.02,0.02,0.01)` | `(0.90,0.50,0.05)` | `(45,25,5)` 倍 |
| AABB 中心（m） | `(0.01,0.01,1.155)` | `(0.45,0.25,1.175)` | 极值被污染点劫持 |

示范门禁：5～20 点的计数门禁错误放行 A；尺寸门禁和质心-AABB 中心距离门禁拦截 A。后者约 `0.3998837331` m。B 未变；Open3D PASS；B2 专项 `11 passed`。

**结论：**点数看起来正常不代表分布正常；应保留失败原因，不向下游发布无效位置。标签误归属是**人为控制变量**，并非断言原 DBSCAN 必然产生该错误。

### 3.5 EXP034-C：工程输出契约

`LocalizationGates` + `validate_cluster_localization()` + `extract_publishable_positions()`：

- A：`INVALID_CLUSTER`，`position_m=null`，`reasons=[EXTENT_EXCEEDED,CENTROID_BBOX_GAP]`。
- B：`VALID`，`position_m=[0.51,0.01,1.152]`。
- 整帧：`PARTIAL_INVALID`，`valid_count=1`，`invalid_count=1`，`noise_count=1`。
- 全噪声：`NO_OBJECTS`；非法 Shape/NaN 等：`ValueError`。

Open3D PASS；C 专项 `40 passed`。**诊断数据仍可保留错误坐标，但不能当作发布位置。**

### 3.6 EXP034-D：门禁阈值消融

**控制变量：**保持 B2 的 `points`、`labels`、`frame_id` 不变，只修改门禁配置。

| 配置 | 参数变动 | A | B | 整帧 | 发布标签 |
|---|---|---|---|---|---|
| D0 默认 | Gap 0.02 m，Extent `(0.04,0.04,0.02)` | INVALID（两原因） | VALID | `PARTIAL_INVALID` | `[1]` |
| D1 严格 Gap | Gap 0.002 m | INVALID（两原因） | INVALID（Gap） | `ALL_INVALID` | `[]` |
| D2 放宽 Gap | Gap 0.50 m | INVALID（Extent） | VALID | `PARTIAL_INVALID` | `[1]` |
| D3 两门禁都放宽 | Extent `(1.0,1.0,0.10)`；Gap 0.50 m | VALID | VALID | `OK` | `[0,1]` |

**预测与实测：**全部一致；D 专项 `16 passed`，Open3D PASS。D1 展示正常 B 的误拒绝；D2 展示多门禁互补；D3 展示被污染 A 的误接受。**阈值改变不等于点云质量改变。**

### 3.7 EXP034-E：源码理解与独立实现

- B2 的 `points.shape=(16,3)`，`labels.shape=(16,)`；单个簇 `(M,3)`，`axis=0` 输出 3 维质心。
- `np.unique(labs[labs>=0])` 去掉噪声、保留不连续的实际标签。
- `np.any(extent > max_extent)` 表示任何一轴越界都拒绝，误写 `np.all` 会漏过单轴异常。
- `reasons=[]` → `accepted=True`；`position_m` 只在接受时返回数组。
- 自行实现发布结果提取函数，独立核查 `VALID`、`publishable` 和非空位置。

### 3.8 Boss Test 及 Boss C 的 RED → PATCH → GREEN

**Boss A～E：**概念、几何计算、独立代码、Debug、坐标单位/坐标变换迁移；最终成绩 `98/100`，Boss PASS。

**旧版审计器缺陷：**`.get()` 默认值导致缺字段时静默放行；`bool` 被 Python 视为 `int`；簇标签和空白 `frame_id` 校验不严。旧版缺失 `objects` 的初始反例存在状态不一致干扰，随后已改为确实能证明“缺 `objects` 被静默视为空”的最小反例。

**PATCH：**要求 `objects/status/valid_count/invalid_count` 显式存在；计数为非负整数；3D 坐标有限且拒绝布尔值；簇标签为非负整数且不重复；有效 `frame_id` 不能只含空白；状态机与计数必须一致；可发布位置复制输出不与原数组共享可变引用。

**验证：**修订 RED 复现 8 个旧版契约缺陷；修复后 13 项专项全部 GREEN。Windows 项目回归于 2026-10-16 显示 **`248 passed, 106 subtests passed`**，Boss C 专项 **`13 passed`**。

**尚待后续增强：**审计模块对 `report['status']` 非字符串、不可哈希等类型可进一步统一抛 `ValueError`；整帧 `frame_id`、外参版本、标定有效性等全局门禁不在本关局部门禁的职责范围内。

---

## 4. 错误与误区（对应 SOP `04_常见问题与误区.md`）

| 误区或故障 | 真实原因 | 修复 / 预防 |
|---|---|---|
| NumPy 与 Open3D 一致，就证明真实物体中心正确 | 相同统计公式对同样的有偏样本给出相同结果 | 区分实现一致、统计估计与几何真值；引入真值评测 |
| AABB 不受 B1 重复点影响，所以 AABB 鲁棒 | B1 仅重复已有范围内的点 | 用 B2 外侧错误标签反例测试极值敏感性 |
| 门禁越严越安全 | D1 会误拒绝干净簇 | 用真值数据平衡误拒绝/误接受，分对象标定阈值 |
| 门禁通过证明定位可抓取 | D3 放宽参数后，污染位置同样成为 `VALID` | 局部门禁与标定、抓取碰撞、全局禁止发布分层 |
| `frame_id` 改名等于坐标变换 | 字符串不改变坐标数值 | 统一 mm→m，使用经验证的 `R @ p + t` |
| `np.all` 检查每轴越界 | 要求全部轴越界才拒绝，会漏掉单轴异常 | 使用 `np.any` |
| `.get()` 让审计代码更安全 | 必填键缺失会被默认值吞掉 | 在安全契约中显式检查必填键和类型 |
| `bool` 不会被当作坐标 | `bool` 是 `int` 子类 | 显式拒绝布尔值 |
| pytest 报 import error 就是算法算错 | Boss C 两文件曾被放在项目根目录，模块路径不符 | 放至 `src/geometry/` 与 `tests/`，再分开跑专项和全部回归 |

**Debug 过程复盘：**读错误及收集阶段 Traceback → 核对项目目录与模块路径 → 检查 `git status` 找到根目录的两个误放文件 → 移至正确位置 → 重新运行专项和完整回归。本次错误属于工程路径，不属于定位算法数学错误。

---

## 5. 代码位置、复现方式与结果文件

### 5.1 核心工程资产

```text
src/geometry/cluster_localization.py
src/geometry/localization_report_audit.py
src/experiments/exp034_cluster_localization_minimal.py
src/experiments/exp034_density_bias_b1.py
src/experiments/exp034_label_contamination_b2.py
src/experiments/exp034_localization_contract_c.py
src/experiments/exp034_gate_ablation_d.py
tests/test_exp034_density_bias_b1.py
tests/test_exp034_label_contamination_b2.py
tests/test_exp034_localization_contract_c.py
tests/test_exp034_gate_ablation_d.py
tests/test_exp034_report_audit.py
```

仓库中还存在 `src/experiments/exp034_cluster_geometry_minimal.py`。**其来源与职责尚未在本轮验收确认，提交前需单独检查，不应未经审查自动纳入。**

### 5.2 实验结果

```text
results/exp034/exp034_A_localization.json
results/exp034/exp034_B1_density_bias.json
results/exp034/exp034_B2_label_contamination.json
results/exp034/exp034_C_validated_localization.json
results/exp034/exp034_D_gate_ablation.json
```

以上为各阶段实验已报告保存的结果路径；在 Git 提交前仍需在 Windows 中逐个确认实际存在，并检查是否有其他不应提交的输出。

### 5.3 复现命令（PowerShell，仓库根目录）

```powershell
python -m src.experiments.exp034_cluster_localization_minimal
python -m src.experiments.exp034_density_bias_b1
python -m src.experiments.exp034_label_contamination_b2
python -m src.experiments.exp034_localization_contract_c
python -m src.experiments.exp034_gate_ablation_d
python -m pytest -q tests/test_exp034_report_audit.py
python -m pytest -q
```

*运行脚本会更新对应 `results/exp034/*.json`；如需保留某次原始记录，先备份或比对内容。*

### 5.4 本轮验收记录

| 节点 | 结果来源 | 状态 |
|---|---|---|
| A 独立手算、Open3D | Windows 用户日志 | PASS |
| B1 | Windows：8 专项 / Open3D | PASS |
| B2 | Windows：11 专项 / Open3D | PASS |
| C | Windows：40 专项 / Open3D | PASS |
| D | Windows：16 专项 / Open3D | PASS |
| Boss C | Windows：13 专项 | PASS |
| P02 最终回归 | Windows：248 passed + 106 subtests | PASS |
| Git commit | 未执行 | PENDING |
| GitHub push、远程 SHA | 未验证 | PENDING |

---

## 6. 工程边界、风险与下一步

1. **当前只是局部几何筛选：**无法证明是否识别了正确物体，无法排除标定偏差、时间同步故障、遮挡与坐标系错配。
2. **质心是样本统计量：**工业定位要评估更稳定的几何中心估计、密度纠偏、多帧融合或已知模型配准。
3. **AABB 对单点极值敏感：**可研究更稳健的统计特征，但必须评估是否会意外裁去真实目标结构。
4. **阈值需要真实数据：**按零件类别/尺寸/传感器误差建立验证集，测量误拒绝和误接受，而不是直接沿用教学数值。
5. **发布应有分层门禁：**局部几何合格 → 位置报告审计 → 相机内外参、时间同步、坐标链健康检查 → 下游动作安全确认。

### 我的理解与迁移（依据本关本人作答整理，建议提交前核对）

> 几何公式算得对，不等于被测物体确实定位得对。B1 的重复采样说明点云质心受采样分布影响；B2 的错误标签说明 AABB 可能被极值严重扭曲；D1/D3 说明门禁过严、过松都会造成不同风险。正确的工程实现既要返回结果，也要能够明确拒绝不可信的结果，并且把诊断信息与可发布位置分开。`VALID` 的定义必须受门禁配置与上层安全链路约束。

### 最终收尾清单

- [x] 理论、Shape、关键公式与参数解释
- [x] A/B1/B2/C/D 预测、主动修改、实测、Open3D 对照
- [x] E 源码拆解、独立实现
- [x] Boss Test PASS（98/100）
- [x] Boss C 契约修复，专项 13 passed
- [x] P02 完整回归 248 passed + 106 subtests
- [ ] 本笔记与个人理解校对，复制到 `experiments/EXP034_cluster_localization.md`
- [ ] 审核结果 JSON、未知 EXP034 文件、Git staged diff
- [ ] Git commit
- [ ] GitHub push，核对本地/远程 SHA

**当前 Learning Unit：CONTINUE；满足剩余清单后才能写 `LEVEL COMPLETE`。**
