# OS-01-C｜RGB-D 多样本可追溯评估与合成参考差异

> 项目：`Project_002_RGBD_Object_Localizer`
> Level：2 · 3D Vision · OS-01 Open3D / RGB-D 定位实战
> Learning Unit：OS-01-C（C1 单样本评估 + C2 二十样本追溯、批量统计）
> 记录日期：2026-10-27
> **数据边界**：本单元使用确定性**合成** RGB-D 数据和合成名义参考位置；无独立物理真值，无机器人运动授权。
> **当前状态**：理论、预测、C1/C2 专项测试、20 样本批量运行、误差推导与综合 Boss 已有验收证据。最终全仓回归、笔记入库、Git commit / GitHub push 尚待确认，因此 **OS-01-C 仍为 IN PROGRESS**。

## 1. Entry Protocol：目标、知识地图、前置条件、产物与验收

### 1.1 目标

从 OS-01-B 已完成的单样本诊断 Pipeline 出发，建立可复现、可追溯、不会隐藏失败样本的评估链：

```text
OS-01-A：合成 RGB-D 输入契约
  → OS-01-B：Mask 反投影 → Radius Filter → DBSCAN → EXP034 几何门禁
  → OS-01-C1：单样本评估资格 → 合成参考差异
  → OS-01-C2：20 样本生成 → Manifest 校验 → 批量执行
             → 逐样本 CSV → 汇总 JSON → 失败分析与解释
  → 后续：真实采集、独立物理真值、工作空间与标定验证
```

### 1.2 前置知识

- 必须掌握：NumPy 数组 Shape、相机反投影、frame_id、m/mm 单位、质心与 AABB、DBSCAN Core/Border/Noise、几何门禁状态与 Pytest。
- 建议掌握：JSON/CSV、SHA-256、百分位数、合成数据生成、实验复现。
- 目前不要求：真实相机标定、机器人下发、真实物理误差评测、FoundationPose 工业闭环。

### 1.3 核心产物

| 类型 | 路径 / 对象 | 用途 |
|---|---|---|
| C1 评估函数 | `src/evaluation/os01_c1_metrics.py` | 单样本资格筛选与欧氏误差（mm） |
| C2 数据生成器 | `src/evaluation/os01_c2_generate.py` | 确定性生成 20 个输入、清单与校验 |
| C2 批量评估器 | `src/evaluation/os01_c2_batch_evaluate.py` | 真实调用 Pipeline 与 C1，导出 CSV/JSON |
| 测试 | `tests/test_os01_c1_metrics.py` | 10 个 C1 用例，含真实 Pipeline 集成 |
| 测试 | `tests/test_os01_c2_generation.py` | 两项生成与哈希测试 |
| 测试 | `tests/test_os01_c2_batch_evaluate.py` | 两项 20 样本端到端和状态区分测试 |
| Manifest | `data/manifests/os01_c_samples.json` | 样本身份、输入路径、参数、参考、哈希 |
| 单样本数据 | `data/raw/synthetic/os01_c/*.npz` | 20 个不同的合成 RGB-D 样本，可重新生成 |
| 逐样本输出 | `outputs/os01_c/evaluation_results.csv` | 每份样本的状态、门禁、排除原因与差异 |
| 汇总输出 | `outputs/os01_c/evaluation_summary.json` | 比率、Mean/Median/P95/Max、哈希、口径 |

数据文件与输出报告是否纳入 Git 应以仓库追踪策略为准；**应保证脚本能再生文件**。不要把未经版本控制的数据路径当成不可变归档。

### 1.4 验收标准

1. 20 个不同的合成 RGB-D 内容、唯一 ID、五组场景及哈希校验。
2. 单样本评估器覆盖正常、门禁拒绝、参考缺失、Frame 不匹配、多目标歧义与非法数值。
3. 能批量运行真实 OS-01-B Pipeline，输出完整逐样本结果与排除原因，不把缺失误差写成 0。
4. 计算 Mean、Median、P95、Max，明确只使用 `eligible=True` 样本。
5. 至少一项先预测后验证的公式推导，并用逐样本输出核对。
6. 完成综合 Boss、笔记、全量回归与 Git/GitHub 闭环。**代码运行成功不等于通关。**

## 2. 最小理论：什么才是可评估误差

### 2.1 输入 / 输出契约

C1 核心函数：

```python
evaluate_case(
    *, sample_id, report, reference_position_m, reference_frame_id
)
```

- `sample_id`：样本标识，字符串。
- `report`：OS-01-B 的真实诊断报告，不是机器人可发布命令。
- `report["status"]`：必须为 `DIAGNOSTIC_ONLY` 才可能评估。
- `report["diagnostics"]["geometry_gate"]["objects"]`：必须恰有一个 `local_gate_passed=True` 的簇（当前单目标限制）。
- `centroid_camera_m`：有效簇的诊断质心，Shape `(3,)`，单位米。
- `reference_position_m`：同一目标、同一坐标系下有明确来源的参考位置，Shape `(3,)`，单位米；可为 `None`。
- `reference_frame_id`：参考坐标系名，需与 `report["frame_id"]` 一致；当前评估器**不会自动坐标变换**。
- 输出：`{"sample_id", "eligible", "reason", "error_mm"}`，不满足资格时 `error_mm=None`。

**合法但不可评估**：返回 `eligible=False` 加明确的 `reason`，例如 `PIPELINE_REJECTED`、`NO_UNIQUE_VALID_CLUSTER`、`MISSING_REFERENCE`、`FRAME_MISMATCH`。
**在实际检查到 Shape 不合法或 NaN/Inf 的路径上**：抛 `ValueError`。注意函数有早退逻辑，不能声称对所有被早退的记录都会完成后续字段验证。

### 2.2 欧氏误差与单位

同一坐标系、同一目标匹配时：

\[
e_{\mathrm{mm}}=1000\sqrt{(\hat X-X_r)^2+(\hat Y-Y_r)^2+(\hat Z-Z_r)^2}
\]

三个差值的坐标单位为 m；因 `1 m = 1000 mm`，输出以 mm 计。

**正确命名**：`synthetic_nominal_reference_discrepancy_mm`，即**合成名义参考差异**，不是独立物理测量定位误差。参考值在生成器中使用 `reference_kind="synthetic_nominal_patch_center"`，不是 `physical_ground_truth`。

### 2.3 三类分母必须区分

\[
\text{geometry\_pass\_rate}=\frac{N_{\mathrm{geometry\ pass}}}{N_{\mathrm{total}}}
\]

\[
\text{evaluable\_rate}=\frac{N_{\mathrm{eligible}}}{N_{\mathrm{total}}}
\]

\[
\text{Mean}_{\mathrm{mm}}=\frac{\sum_{i\in\mathrm{eligible}}e_i}{N_{\mathrm{eligible}}}
\]

几何门禁通过不代表有参考值，`evaluable_count` 不必等于 `geometry_pass_count`。拒绝或缺少参考的样本应计入排除报告，但**绝不能当作 0 mm 填入均值**。

## 3. 关键参数 / 数据身份与来源

| 项目 | 来源 / 取值 | 作用 / 单位 / 警告 |
|---|---|---|
| `fx,fy,cx,cy` | 当前生成器 `100,100,3.5,2.5` | 相机内参；`fx,fy` 是像素尺度，不是米 |
| `depth_m` | `(6,8)` `float64` | 深度单位 m，无效值包括 0、NaN、Inf |
| `target_mask` | `(6,8)` `bool` | 目标像素筛选条件 |
| `radius_m` | 通常 `0.011`；正常组按 `Z` 缩放 | 半径过滤（不计自身），单位 m |
| `dbscan_eps_m` | 通常 `0.011`；正常组按 `Z` 缩放 | DBSCAN 邻域（计自身），单位 m |
| `dbscan_min_points` | 通常 3；`all_noise` 为 6 | 核心点门槛，整数 |
| `min_neighbors` | 通常 1；`filter_all` 为 4、`filter_four` 为 2 | Radius 过滤门槛（不含自身） |
| `LocalizationGates` | 教学默认 `min_points=5,max_points=20` 等 | 局部几何门禁，不是物理精度验证 |
| `sample_id` | `OS01_C_001` 等 | 唯一身份，和 CSV/Manifest 对齐 |
| `generation_params` | 深度、像素平移、掩码修改或异常模式 | 区分具体场景，不把复制输入当 20 个独立输入 |
| `source_kind` | `synthetic` | 来源标签，不等于真实相机数据 |
| `input_sha256` | `.npz` 原始文件字节哈希 | 检测文件变化，依赖可信参考哈希 |
| `content_sha256` | RGB/Depth/Mask 数组内容指纹 | 检查生成内容不同、数据是否匹配 |
| `reference_kind` | `synthetic_nominal_patch_center` / `none` | 防止冒称独立物理真值 |
| `reference_frame_id` | `camera_optical_frame` 或缺失 | 不一致不能直接相减 |
| `pipeline_params`/`gate_params` | 逐样本 Manifest | 保证参数可追溯 |

实际 Manifest 结构（不是要求手工重复新建）：`schema_version=os01_c2_v1`，包含 20 条 `samples`。

## 4. C1：从四样本预测到真实 Pipeline 集成

### 4.1 运行前预测

| Case | Pipeline 与参考条件 | eligible | reason | error_mm |
|---|---|---|---|---:|
| S001 | 通过门禁，差值 0.002 m | True | None | 2.0 |
| S002 | 通过门禁，差值 0.005 m | True | None | 5.0 |
| S003 | `GEOMETRY_GATE_REJECTED` | False | `PIPELINE_REJECTED` | None |
| S004 | 几何通过、无参考 | False | `MISSING_REFERENCE` | None |

只有 S001、S002 可以计算误差：Mean=3.5 mm、Median=3.5 mm、Max=5.0 mm。若算四个样本的几何通过率和覆盖率，分别是 3/4=75%、2/4=50%。

### 4.2 主动扩展与异常预测

- S005：Frame 不一致 → `eligible=False`、`FRAME_MISMATCH`。
- S006：两个有效簇但无目标身份匹配 → `eligible=False`、`NO_UNIQUE_VALID_CLUSTER`。
- S007：参考值 NaN、Inf 或 Shape `(2,)` → 在有效报告路径上抛 `ValueError`。
- 从 OS-01-A 的 `.npz` 进入正式 `localize_single_sample()`，再把真实报告交给 `evaluate_case()`：预测 `DIAGNOSTIC_ONLY`、1 个有效簇、合成差异 2.0 mm；同时顶层 `position_m=None`、`publishable=False`。

### 4.3 真实测试证据

```text
C1 初版 4 项：4 passed in 0.16s
C1 异常版本 9 项：9 passed in 0.19s
C1 加真实 Pipeline 集成 10 项：10 passed in 0.58s
对应阶段全仓回归：713 passed, 106 subtests passed in 2.77s
```

上述全仓数字发生在 **C2 新增测试之前**，不能当成 C2 完成后的最终全仓结果。

## 5. C2：20 个不同合成输入的生成、追溯与批量评估

### 5.1 生成分组与场景

| 场景组 | 数量 | 主要变化 |
|---|---:|---|
| `normal_geometry` | 6 | 深度 Z、目标像素平移；固定相对缺失深度像素 |
| `mask_perturbation` | 4 | 掩码缺点或完全空 Mask |
| `depth_anomaly` | 4 | 额外零深度、NaN、Inf、全零深度 |
| `cluster_gate_pressure` | 4 | `filter_all`、`all_noise`、`gate_reject`、`filter_four` |
| `missing_reference` | 2 | 仍执行几何 Pipeline，但参考位置为空 |
| **合计** | **20** | 20 种不同的 RGB-D 输入内容 |

生成器：`src/evaluation/os01_c2_generate.py`；校验器：`validate_manifest()`。

生成实测记录：

```text
Samples: 20
Scenario groups: {'normal_geometry': 6, 'mask_perturbation': 4,
 'depth_anomaly': 4, 'cluster_gate_pressure': 4, 'missing_reference': 2}
Missing references: 2
Unique RGB-D contents: 20
All sample files and SHA-256 digests verified
```

C2 生成器专项测试：`2 passed in 0.38s`（输入区分与 Manifest 篡改校验）。

### 5.2 真实批量 Pipeline 与误差评估

`run_batch()` 先验证 Manifest 与文件，再逐条读取 `.npz`，调用真实 `localize_single_sample()`，并调用已验证的 `evaluate_case()`。CSV 保留每样本结果和原因；JSON 输出汇总、代码与 Manifest 哈希。

C2 批量专项测试：`2 passed in 0.92s`。

生成报告：

```text
outputs/os01_c/evaluation_results.csv
outputs/os01_c/evaluation_summary.json
```

**真实汇总**：

| 指标 | 实测 |
|---|---:|
| 总样本数 | 20 |
| 几何门禁通过 | 8 |
| 几何门禁通过率 | 40% |
| 几何未通过 | 12 |
| 有资格计算差异 | 6 |
| 差异评估覆盖率 | 30% |
| 被排除 | 14 |
| 其中：Pipeline 拒绝 | 12 |
| 其中：参考缺失 | 2 |
| Mean（仅 6 个） | 2.2360679775 mm |
| Median（仅 6 个） | 2.2360679775 mm |
| P95（仅 6 个） | 2.4820354550 mm |
| Max（仅 6 个） | 2.5043961348 mm |

原始 JSON 标注：

```json
{
  "schema_version": "os01_c2_evaluation_v1",
  "metric_name": "synthetic_nominal_reference_discrepancy_mm",
  "physical_accuracy_claimed": false,
  "percentile_method": "numpy.percentile(method='linear')",
  "total_samples": 20,
  "geometry_pass_count": 8,
  "evaluable_count": 6,
  "excluded_count": 14
}
```

**请勿将结果写为「20 个样本平均误差 2.2361 mm、成功率 100%」。** 2.2361 mm 仅适用于六个合格、具有合成名义参考值的样本。

### 5.3 分组失败审计

| 组别 | 样本数 | 诊断通过 | 门禁拒绝 | 其他未通过 |
|---|---:|---:|---:|---:|
| `normal_geometry` | 6 | 6 | 0 | 0 |
| `missing_reference` | 2 | 2 | 0 | 0 |
| `mask_perturbation` | 4 | 0 | 3 | 1 × `NO_OBJECTS` |
| `depth_anomaly` | 4 | 0 | 3 | 1 × `NO_VALID_DEPTH` |
| `cluster_gate_pressure` | 4 | 0 | 2 | 1 × `NO_CLUSTERS` + 1 × `NO_POINTS_AFTER_FILTER` |
| **总计** | **20** | **8** | **8** | **4** |

全局 Pipeline 状态数：

```text
DIAGNOSTIC_ONLY        8
GEOMETRY_GATE_REJECTED 8
NO_CLUSTERS            1
NO_OBJECTS             1
NO_POINTS_AFTER_FILTER 1
NO_VALID_DEPTH         1
```

**解释边界**：压力测试组故意注入失败。几何通过率 40% 是该合成测试集的状态分布，不能外推为真实场景系统成功率或鲁棒性指标。

## 6. 重点预测实验：深度导致的结构性质心偏差

### 6.1 观察和先验预测

目标模板是 2×3 个像素，名义几何中心 `(u,v)=(4.0,2.5)`。固定目标相对左上角像素深度无效时，剩余 5 像素质心为 `(4.2,2.6)`。

因此：

```text
Δu = +0.2 px
Δv = +0.1 px
fx = fy = 100
```

在目标点等深、内参固定的理想条件下：

\[
\Delta X=0.002Z,\quad\Delta Y=0.001Z
\]

\[
e_m(Z)=Z\sqrt{0.002^2+0.001^2}=0.0022360679775\,Z
\]

\[
\boxed{e_{mm}(Z)=\sqrt 5\,Z},\quad Z\text{ 的数值以 m 为单位}
\]

### 6.2 六组先预测后验证的结果

| sample_id | Z (m) | 解析预测 (mm) | 实测 (mm) |
|---|---:|---:|---:|
| OS01_C_001 | 0.88 | 1.9677398202 | 1.9677398202 |
| OS01_C_002 | 0.92 | 2.0571825393 | 2.0571825393 |
| OS01_C_003 | 0.96 | 2.1466252584 | 2.1466252584 |
| OS01_C_004 | 1.04 | 2.3255106966 | 2.3255106966 |
| OS01_C_005 | 1.08 | 2.4149534157 | 2.4149534157 |
| OS01_C_006 | 1.12 | 2.5043961348 | 2.5043961348 |

预测与实测在显示精度下逐条一致。由于六个深度的均值正好是 1 m，Mean 恰为 `sqrt(5) ≈ 2.2361 mm`。

这是一种**固定缺失像素导致的结构性偏差**：位置偏差方向稳定，随着 Z 增大而线性增大。不能笼统地声称所有深度噪声都满足此规律；真实点云表面起伏、变动的 Mask 和深度噪声会破坏这些前提。若缺失模式每帧固定，多帧简单平均通常不能消除该系统偏差。

## 7. P95 与统计边界

六个升序差异值（mm）：

```text
[1.967740, 2.057183, 2.146625,
 2.325511, 2.414953, 2.504396]
```

NumPy 使用线性插值：`np.percentile(errors, 95, method='linear')`。对于 `n=6`：

\[
h=(n-1)\times0.95=4.75
\]

0 基索引 4 和 5 之间做 75% 插值：

\[
P95=e_4+0.75(e_5-e_4)\approx2.4820354550\ \mathrm{mm}
\]

这只是六个确定性样本的**描述性统计**，不是工业可靠性上界或 P95 性能保证。样本独立性、场景覆盖及不确定性需要另行评估。

## 8. 可追溯性与安全隔离

1. Manifest 含 `sample_id`、`input_path`、`input_sha256`、`content_sha256`、`generation_params`、`pipeline_params`、`gate_params`、参考值与 `reference_kind`。
2. 运行前的 `validate_manifest()` 会核对场景计数、唯一 ID、内容唯一性、文件字节哈希、数组内容指纹和 Shape/DType。
3. 汇总 JSON 含 Manifest SHA-256 和生成器、批量评估器、C1 评估器、Pipeline 的代码 SHA-256。
4. 文件名不变但 `.npz` 字节改变，若与可信的 Manifest 中预期值不一致，应当被校验拒绝；若文件和清单一起被改写且无外部可信锚点，哈希本身不能证明来源真实。
5. SHA-256 既不能证明参考位置来自独立测量，也不能证明时间同步、标定有效、目标匹配正确或机器人安全。
6. 本单元所有批量结果均要求顶层 `position_m is None`、`publishable is False`；即使合成差异很小也不能改变。

## 9. 综合 Boss A～D：独立作答要点与校正

### Boss A：统计口径

**通过**。误导性陈述「20 样本平均误差 2.2361 mm、成功率 100%」的问题在于隐藏了 12 个几何未通过样本、2 个缺参考样本，并将合成差异冒称真实定位精度。正确表述必须同时列 20 总数、8 几何通过（40%）、6 可评估（30%）、14 排除，并将 2.2361 mm 限定为这 6 个可评估样本的合成名义参考差异。

### Boss B：P95

**通过**。`h=4.75`，在索引 4 与 5 之间按 0.75 插值得约 2.4820 mm。六个确定性样本无法支撑真实尾部风险保证，不能机械地说样本数达到某个固定数量就足够。

### Boss C：哈希

**通过，附可信锚点修正**。重新计算输入字节哈希与 Manifest 的 `input_sha256` 比较能够检测不一致。哈希匹配依赖参考摘要可信；SHA-256 不能证明参考值是独立物理真值，若要追踪真实性还需要独立测量过程、设备与标定记录、时间戳和可信的数据签名/存档。

### Boss D：工业迁移

**通过**。拒绝仅凭 Mean≈2.2361 mm 改成 `publishable=True`。尚无真实相机采集、可信内外参、独立物理真值、足够场景覆盖、同步与机器人运动安全验证。这些是未来目标，不是当前已经实现的功能。

### 我的三句话总结

1. **几何通过率**衡量当前测试集产生几何合格候选的比例；**评估覆盖率**衡量具有合法配对参考、可计算差异的比例，二者不得混写。
2. 统计量必须带分母、来源与适用范围；失败和缺参考样本保留记录，但误差字段写 `None` 而非 `0`。
3. **合成解析一致 ≠ 独立物理精度 ≠ 机器人可发布**，顶层安全出口始终保持 Fail-closed。

## 10. 复现命令与证据

在 `D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer` 根目录、`ai_3d` 环境中运行：

```powershell
# C1 专项与单样本集成
python -m pytest tests/test_os01_c1_metrics.py -v

# C2 输入生成与完整性检查
python -m pytest tests/test_os01_c2_generation.py -v
python -m src.evaluation.os01_c2_generate

# C2 批量真实 Pipeline 与统计报告
python -m pytest tests/test_os01_c2_batch_evaluate.py -v
python -m src.evaluation.os01_c2_batch_evaluate
Get-Content .\outputs\os01_c\evaluation_summary.json -Encoding UTF8

# 正式收尾：全项目回归和 Git 检查
python -m pytest -q
git diff --check
git status --short
```

现有实际证据：C1 `10 passed`、C2 生成 `2 passed`、C2 批量 `2 passed`，20 样本输出如第 5 节所列。**最后一次已报告的全仓回归为 C2 新增前的 `713 passed, 106 subtests passed`；完整 C2 后的全仓回归仍需再运行确认。**

## 11. 已知局限与下一步迁移

- 6×8 的小型确定性合成 RGB-D 与名义参考值不能代表真实相机的复杂噪声、遮挡、反光、畸变、标定误差或深度偏置。
- 故障注入组全部未通过门禁，能够测试拒绝路径，但不能量化真实场景的自然故障率。
- 当前 C1 要求恰好一个有效簇。若多目标场景需增加稳定的目标身份关联，不能挑距离参考值最近的簇造成乐观偏差。
- DBSCAN NumPy 实现是成对距离矩阵，内存随点数以 O(N²) 增长；真实密集点云需要有界输入或更高效的邻域搜索。
- 未来真实物理精度验证需要独立真值设备、可信标定、时间同步、工作空间覆盖、重复测量和安全分离；任何一项没有证据时不能发布机器人位置。

## 12. SOP Exit Protocol：本关通关清单

| 项目 | 状态 | 已有证据 / 待办 |
|---|---|---|
| 目标、知识地图、最小理论 | PASS | 已明确统计口径与信任边界 |
| 公式、参数、Shape、单位解释 | PASS | C1 欧氏差异、C2 深度偏差、P95 推导 |
| 最小实验和代码复用 | PASS | C1→OS-01-B→C2 接口与脚本 |
| 主动修改与异常验证 | PASS | Frame 错配、多簇、NaN/Inf/Shape；过滤、聚类、门禁压力 |
| 先预测后运行 | PASS | C1 资格、C2 分母、深度解析偏差逐条对照 |
| 20 样本与审计报告 | PASS | 唯一内容、校验、CSV/JSON、分组统计 |
| 综合 Boss | PASS | A～D；哈希可信锚点及描述性 P95 已修正 |
| 笔记生成 | PASS | 本文件已整理；待放入本地仓库确认 |
| **C2 后最终全仓回归** | **待验证** | `python -m pytest -q` |
| **Git diff / 状态复核** | **待验证** | 避免附带无关截图与非预期生成文件 |
| **Git commit** | **待完成** | 本关代码与文档待提交 |
| **GitHub push / SHA 对齐** | **待完成** | `HEAD == origin/master` 后才正式 COMPLETE |

**阶段结论：OS-01-C 综合 Boss 与功能/分析验收通过；Learning Unit 尚未完成 SOP 的 Git/GitHub 闭环。Level 2 保持 CONTINUE。**
