# EXP036｜RGB-D 三维定位误差分析与安全评估

> AI Lab / Level 2 / P02 `Project_002_RGBD_Object_Localizer`
> 日期：2026-10-21
> 学习状态：**Boss PASS / Notes Review + Git/GitHub Pending**
> 前置：EXP034 几何审计、EXP035 SE(3) 与 Fail-Closed 定位报告

## 1. Goal / 目标

由「能计算一个三维坐标」升级为「能解释、预测、计算、诊断、审计三维定位误差」。本单元重点验证：

1. 像素反投影、深度、相机内参及外参扰动如何影响定位误差。
2. 有符号偏差、三维欧氏误差、准确度和重复性之间的区别。
3. 多样本 Mean、Median、P95（`linear`）、Max 与异常样本敏感性。
4. 无效簇、缺失真值、跨坐标系比较、样本来源不独立时如何 fail-closed。
5. 复用 EXP034/035 的实际安全接口，而不是另起一条绕过审计的发布链。

本单元**没有**完成真实设备标定或真实物理精度认证。所有数值基准均使用 `synthetic_gt`，不得宣称机器人定位达到相应毫米精度。

## 2. Knowledge Map / 知识地图

```text
RGB-D 像素 (u,v,Z) + 相机内参 K
    ↓ 反投影
相机系三维点 (EXP031～EXP034)
    ↓ 几何审计 / 有效性门禁 (EXP034)
相机系合法定位报告
    ↓ RigidTransform + audit_chain_and_compute (EXP035)
数学上合法的基座系诊断点（始终不可直接发布）
    ↓ 参考真值、frame / 单位 / 来源审计 (EXP036)
合成误差评估报告 + 故障诊断 + 禁止发布标记
    ↓ 后续：真实标定、可信独立真值、时序与系统安全审计
真实物理定位精度及机器人使用资格（本关不覆盖）
```

## 3. Minimum Theory / 最小必要理论

### 3.1 反投影与误差

对已校正的针孔相机像素（`Z` 为沿光轴的深度，不是射线距离）：

\[
X=\frac{(u-c_x)Z}{f_x},\qquad Y=\frac{(v-c_y)Z}{f_y},\qquad Z=Z
\]

同一 `frame_id`、同一物理单位下，估计点与独立参考点的三维位置误差：

\[
\Delta\mathbf p_i = \hat{\mathbf p}_i-\mathbf p_i^{GT}, \qquad
 e_i =1000\,\lVert\Delta\mathbf p_i\rVert_2\ \mathrm{mm}
\]

这里 \(\Delta\mathbf p_i\) 先使用米，乘 `1000` 后转换为毫米。`e_i >= 0`；有符号 XYZ 偏差保留方向。

### 3.2 SE(3) 与误差评估并不等价

\[
\mathbf p_{base}=R_{base\_camera}\mathbf p_{camera}+t_{base\_camera}
\]

合法 `R`（正交、行列式 +1）、合法 Shape 和有限数值只能证明数学契约成立。它不证明外参对应真实相机安装位置。`calibration_verified=True` 在 EXP035 中只是声明，不是可验证的第三方标定证明。EXP035 的全局接口保持 `publishable=False`、`published_positions=[]`。

### 3.3 指标及统计口径

- **Mean**：所有有效样本三维距离误差的算术平均，单位 mm。
- **Median**：有效误差的中位数，单位 mm。
- **P95**：`np.percentile(errors_mm, 95, method="linear")`；报告必须记录方法和样本数。
- **Max**：有效样本最大三维误差，单位 mm。
- **Mean signed delta XYZ**：按方向统计的 `(3,)` 平均误差向量，单位 mm，可能因为正负抵消而接近零。
- **Repeatability**：同一目标/条件多次测量的离散程度，不能由不同空间位置的单次测量直接推断。

20 个合成样本的样本 P95 不是物理系统 P95 置信保证；`n=1` 时计算出的 P95 仅等于该唯一误差。

## 4. Parameters, Units & Shapes / 核心参数

| 参数 / 数据 | 单位 | 类型或 Shape | 说明 |
|---|---|---|---|
| `u_px, v_px` | px | 标量 | 校正后像素 |
| `fx_px, fy_px, cx_px, cy_px` | px | 标量 | 相机内参 |
| `z_m` | m | 标量，`>0` | 光轴深度 |
| `R` | 无量纲 | `(3,3)` | 合法旋转 |
| `t_m` | m | `(3,)` | 坐标变换平移 |
| `gt_points, estimate_points` | m | `(N,3)` | 同一 frame 的三维位置 |
| `signed_delta_xyz_mm` | mm | `(N,3)` | 保留 XYZ 方向的有符号误差 |
| `signed_delta_x_mm` | mm | `(N,)` | 仅 X 方向偏差 |
| `euclidean_errors_mm` | mm | `(N,)` | 每样本三维距离误差 |
| `frame_id` | 无 | 非空字符串 | 不允许跨 frame 直接相减 |
| `sample_id` | 无 | 唯一非空字符串 | 对齐索引；还需核对独立来源记录 |
| `reference_kind/gt_source` | 无 | 明确字符串 | 合成、独立测量、自证来源必须区分 |
| `percentile_method` | 无 | `linear` | 固定 P95 算法以实现复算 |

## 5. Predictions → Experiments → Verification / A～E

### A：单点最小实验

| 受控条件 | 预测 | Windows 实测结论 |
|---|---|---|
| 反投影：`u=380,v=270,Z=1m,fx=fy=600,cx=320,cy=240` | `(0.10,0.05,1.00)` m | PASS |
| `Z=1.02m` | `Δ=(2,1,20)` mm；`e≈20.124612mm` | PASS |
| `fx=630` | `ΔX≈-4.761905mm` | PASS |
| 平移 `t_x +10mm` | 三维误差 `10mm` | PASS |
| 误差 `[0,2,4,6,18]` | Mean `6`，Median `4`，P95 `15.6`，Max `18` mm | PASS |

专项 **41 passed**。关键认识：深度变化沿像素射线同时改变 XYZ；浮点尾数如 `2.0000000000000018` 属于数值表示误差。

### B：主动修改 / 控制变量

- B1：深度 `+10/-10mm`：平均有符号误差向量为零，平均欧氏误差约 `10.062306mm`，两者不能互换。
- B2：仅 `fx +5%`，`u=320/380/500` 时误差分别为 `0/4.761905/14.285714mm`。中心误差为零仅适用于这一种 `fx` 扰动，不能泛化所有内参错误。
- B3：三个不同空间点的外参 `t_x+10mm`，各点欧氏误差均 `10mm`；这不能证明固定目标的重复性。

专项 **41 passed**。每次改动先给出方向、大小与根因预测，再执行代码验证。

### C：20 样本基准及异常值敏感性

合成有符号 X 偏移（mm）：

```text
[+1,-2,+3,-4,+5, -2,+4,-6,+8,-10, +10,-10,+10,-10,+10, -3,+4,-5,+6,-50]
```

`synthetic_gt`：`GT_i = [0.10 + 0.01*(i%5), 0.05 + 0.01*(i//5), 1.00] m`，估计点 X 叠加 `delta_i/1000` m；无随机抽样：`rng_used=false`, `random_seed=null`。

| 指标 | 含 S19（20 样本，正式整体统计） | 去掉 S19（19 样本，仅敏感性分析） |
|---|---:|---:|
| Mean | **8.15 mm** | **113/19 ≈ 5.947368 mm** |
| Median | **5.5 mm** | **5.0 mm** |
| P95 (`linear`) | **12.0 mm** | **10.0 mm** |
| Max | **50.0 mm** | **10.0 mm** |
| 平均 X 有符号偏差（全量） | **-2.05 mm** | 不作为整体基线报告 |

专项 **81 passed**。S19 是有效的故意注入大误差，不可以悄悄剔除后把新的统计称为“性能提升”。错帧、缺失 GT、非有限值必须留下原因而非零误差。

### D：评估器故障注入 / RED → PATCH → GREEN

| 故障 | RED 错误结果 | GREEN/防护 |
|---|---|---|
| D1 用 `abs(mean(signed X))` 当平均距离 | Mean `2.05`（应 `8.15`）mm | 使用逐点三维欧氏距离均值 |
| D2 P95 改用 `higher` | `50`（应 `12`）mm | 固定并记录 `method="linear"` |
| D3 静默删 S19 | `n=19, Mean≈5.947` | 全量保留；另列敏感性分析 |
| D4 S00/S01 预测值错配 | `e=8/9` mm；Mean `8.85` | 独立来源追踪、样本配对检查 |
| D5 缺失真值填 estimate | `n=21, Mean≈7.761905` | `MISSING_GROUND_TRUTH`，有效样本仍 `n=20` |

专项 **38 passed**。仅 `sample_id` 唯一性不足以防止内容配错，必须保留可信数据来源和采集身份。

### E：复用 EXP034/035 的安全定位链

复用既有 `audit_and_extract()`、`RigidTransform` 和 `audit_chain_and_compute(report, transforms)`。后者内部执行几何审计、frame、坐标链与缓存检查；输出**诊断**而非机器人可发布目标。

| 场景 | 实测 | 诊断/安全 |
|---|---|---|
| E1 有效簇，独立设计的合成 GT | `e≈20.124612mm` | `publishable=false` |
| E2 无效簇 | `evaluated=false`, `error=null` | `INVALID_CLUSTER`，排除统计 |
| E3 未验证外参 | 仅 demo 坐标 `[0.40,-0.05,1.20]`m | `UNVERIFIED_TRANSFORM_FOR_CROSS_FRAME_GT` |
| E4 变换的标定声明为 True | 合成对齐比较 `e=0mm` | `calibration_authenticated=false`，仍不可发布 |
| E5 自证真值 | 诊断数值可能 `0mm` | `SELF_DERIVED_REFERENCE`，不进入统计 |

专项 **33 passed**，跨阶段专项回归 **119 passed**。任何合成评估通过都不能将 `publishable` 改为 True。

## 6. Independent Boss / 独立实现与补测

- 2026-10-21，A～E Boss 综合评分：**92/100**，达到 **85/100** 门槛。
- 独立实现 `evaluate_localizations(records)`，包括输入筛查、有效样本统计、缺失/错误原因、`publishable=False`。
- 新增独立硬化 RED：**2 passed / 3 failed**（故意暴露边界缺陷）。
- 本人完成 PATCH：
  1. 非字符串 `gt_source` 在集合成员检查前拒绝，避免不可哈希 `TypeError`。
  2. 不仅检查输入，还检查相减、毫米转换与范数计算的非有限结果，拒绝 `inf/NaN`。
  3. 正常与零有效样本统计均记录 `percentile_method="linear"`。
- Windows 硬化 GREEN：**5 passed**，已归档为 `src/experiments/exp036_boss_student.py` 与 `tests/test_exp036_boss_hardening.py`。
- 项目整体最新实测：**692 passed, 106 subtests passed in 2.54s**。

> GREEN 证明上述最小测试通过；真实生产级 API 尚需结合业务规则、可信数据来源和认证标定进一步验收。

## 7. Common Mistakes & Debug / 易错点

1. `abs(mean(signed_delta))` 不等于 `mean(norm(delta_xyz))`：正负抵消掩盖位置误差。
2. `P95` 依赖方法和样本数。`n=20` 时一个极端值即可显著影响尾部指标。
3. 统计分母仅包含通过有效性/真值审计的样本；缺失 GT ≠ 0 mm。
4. 错误的 GT/estimate `frame_id` 即使 Shape 都是 `(3,)` 也不能直接相减。
5. 几何 `VALID` 不代表 GT 准确、相机标定正确或允许机器人使用。
6. 自证 GT（由估计管线生成）不能用来证明物理准确度。
7. `sample_id` 是必要的关联键，但不能代替来源真实性、时间戳和采集记录校验。
8. 输入各自有限，计算过程仍可能溢出，需要再次检查结果。
9. S19 等有效尾部大误差必须保留原始统计；过滤实验须透明标识。

## 8. Reproducibility & Outputs / 环境、结果和文件

### 环境与执行

- 平台：Windows / PowerShell。
- Python 环境：`D:\conda_envs\ai_3d\python.exe`。
- NumPy：`2.2.6`（A/B/C Windows 日志确认）。
- 项目根目录：`D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer`。

从 P02 根目录执行：

```powershell
$py = 'D:\conda_envs\ai_3d\python.exe'
& $py -m src.experiments.exp036_error_minimal
& $py -m src.experiments.exp036_error_ablation_b
& $py -m src.experiments.exp036_error_benchmark_c
& $py -m src.experiments.exp036_fault_injection_d
& $py -m src.experiments.exp036_pipeline_integration_e
& $py -m pytest -q tests/test_exp036_boss_hardening.py
& $py -m pytest -q
```

每条命令应检查退出码，任何步骤失败不得继续宣告验收。所有结果目录使用 `results/exp036/`，JSON 均为合成数值实验报告。

### 核心新增源码与测试

```text
src/geometry/localization_error_metrics.py
src/geometry/localization_evaluation_adapter.py
src/experiments/exp036_error_minimal.py
src/experiments/exp036_error_ablation_b.py
src/experiments/exp036_error_benchmark_c.py
src/experiments/exp036_fault_injection_d.py
src/experiments/exp036_pipeline_integration_e.py
src/experiments/exp036_boss_student.py
tests/test_exp036_error_minimal.py
tests/test_exp036_error_ablation_b.py
tests/test_exp036_error_benchmark_c.py
tests/test_exp036_fault_injection_d.py
tests/test_exp036_pipeline_integration_e.py
tests/test_exp036_boss_hardening.py
```

### 结果 JSON

```text
results/exp036/exp036_A_error_minimal.json
results/exp036/exp036_B_error_ablation.json
results/exp036/exp036_C_error_benchmark.json
results/exp036/exp036_D_fault_injection.json
results/exp036/exp036_E_pipeline_integration.json
```

## 9. Mastery / 本人复盘（提交 Git 前请用自己的话复核）

# EXP036｜第 9 节 Mastery 复盘

---

## Q1｜为什么系统每帧都很稳定时，仍可能偏离真实目标 10 mm？

稳定，指的是**重复性**；偏离真实目标，指的是**绝对准确度**。两者不是一回事。

举个 EXP036-B3 的例子：

```text
t_correct = [0.30, -0.10, 0.20]
t_wrong   = [0.31, -0.10, 0.20]

三个不同空间点，用 t_wrong 计算后：
每个点的误差向量都是 [10, 0, 0] mm
```

每次测量都偏移同一个 10 mm，帧间几乎没有波动，看起来非常“稳定”。
但这个稳定是**稳定地错**：系统带着一个固定系统偏差（bias），每次都以同样的方式偏离真值。

Boss A2 也是同一个逻辑：

```text
100 次重复测量，全部沿 X 偏移 15 mm，测量间几乎无波动。
```

结论：

```text
重复性好 = 测量彼此接近（precision 高）
绝对准确度 = 测量接近真值（accuracy 高）
高重复性完全不能推出高准确度。
```

多帧重复只能降低随机噪声，不能消除固定系统偏差。要发现这种偏差，必须有**独立真值**或**可信标定验证**。

---

## Q2｜一份 P95 很低的报告，至少检查哪三种证据才能判断它可信？

至少检查三类证据：

### 1. 样本筛选是否透明

```text
- 输入样本数 vs 评估样本数；
- 排除了多少样本、排除原因；
- 是否悄悄删除了大误差样本；
- 缺失 GT、INVALID_CLUSTER、跨帧样本是否被正确排除，
  而不是被填成 0 mm。
```

D3 已经证明：悄悄删掉 S19 后，P95 从 12 mm 变成 10 mm，看起来更好，但这是选择性报告。

### 2. 真值来源是否独立

```text
- GT 是真实独立测量，还是由同一条预测管线生成；
- 是否使用同一条未验证外参；
- 是否属于自证循环（estimate 生成 GT，再用 GT 验证 estimate）。
```

E5 已经证明：两条路径共用同一外参时，误差可以是 0 mm，但这是自证，没有信息量。

### 3. 百分位算法与统计口径是否固定

```text
- P95 用的是 linear / nearest / higher；
- 样本数 n 是多少；
- 是否记录了 percentile_method；
- Mean 是有符号平均还是欧氏平均；
- 是否同时报告 Median、Max、P95。
```

D2 已经证明：同一批 20 个样本，`linear` 给 12 mm，`higher` 给 50 mm。不记录算法，P95 不可复算、不可比较。

补充还需要检查：

```text
- frame_id 是否一致；
- 单位是否一致；
- 外参是否可信、是否过期；
- 是否只是合成数据。
```

---

## Q3｜在什么情况下即使测得 0 mm，仍必须 `publishable=False`？

至少两种情况：

### 情况 1：真值不独立（EXP036-E5）

```text
estimate_base_m     = [0.40, -0.05, 1.20]
synthetic_gt_base_m = [0.40, -0.05, 1.20]
computed_error_mm   = 0.0
```

但两者都来自**同一个相机点 + 同一条未验证外参**。
0 mm 只证明“两条路径用了同一个公式”，不证明接近物理真值。

```text
gt_source = "derived_from_estimate_pipeline"
gt_independent = false
→ SELF_DERIVED_REFERENCE
→ 不进入正式统计
→ publishable = False
```

### 情况 2：外参未经验证（EXP036-E4 / E3）

```text
R = I, t = [0.30, -0.10, 0.20]
calibration_verified = False
```

即使跨帧对齐后数值误差为 0 mm：

```text
status      = BLOCKED_UNCALIBRATED
publishable = False
position_m  = null
reasons     = ["UNVERIFIED_EXTRINSIC"]
```

因为：

```text
数值对齐正确 ≠ 外参对应真实安装关系；
calibration_verified=True 只是自声明，没有验证记录；
没有可信记录 → 坐标系不可信 → 不能发布。
```

其他也应 `publishable=False` 的情况：

```text
- 几何审计 INVALID_CLUSTER；
- 目标缺失、深度全无效；
- frame_id 链不连续；
- 外参过期、签名无效；
- 时间同步失败；
- 多帧一致性差；
- 处于标定 / 调试模式。
```

核心原则：

```text
0 mm 是“数值一致”，
不是“物理准确”，
更不是“可发布”。

publishable 必须由完整验证链决定，
不能被一个漂亮的误差数字覆盖。
```

## 10. Acceptance / Exit Checklist

- [x] 最小理论、公式、单位/Shape 与运行前预测（A）。
- [x] 三组主动修改、预测再验证（B）。
- [x] 20 样本统计、异常样本敏感性（C）。
- [x] RED / PATCH / GREEN 故障注入（D）。
- [x] EXP034/035 安全门禁复用与集成（E）。
- [x] Boss 总分 92/100，附加硬化 5 GREEN。
- [x] Windows 全量回归 `692 passed, 106 subtests passed`。
- [x] 用自己的话完成第 9 节复盘并检查 Markdown 准确性。
- [ ] 审核暂存文件，不包含截图、`.bak` 和历史遗留 EXP034 文件。
- [ ] Git Commit。
- [ ] GitHub Push，确认本地与远程 SHA 一致。

**当前结果：`CONTINUE`。未完成笔记复核和 GitHub 双 SHA 核验，不得标记 EXP036 LEVEL COMPLETE。**

## 11. Next Level / 后续能力边界

后续转向真实相机内/外参校验、独立物理真值、误差覆盖范围与时序一致性。在可信标定和正式发布策略到位前，本项目只产生安全诊断，不输出机器人可执行目标。
