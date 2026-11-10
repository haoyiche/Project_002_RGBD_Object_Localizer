# T1-07-A｜T1 80 分职业级证据盘点与标定缺口审计

> AI Lab / Project_002_RGBD_Object_Localizer
> 依据：`AI_Lab_SOP.md`、升级攻略 v2.0 第 33 章；GitHub `ffb2f902...` 是上一关已验证的历史基线，并非本关提交。
> 当前状态：**CONTINUE｜E1、E2、E3 实验与 Final Boss 已通过；Git/GitHub 和 T1 80 分硬门槛尚未验收（2026-11-09）**。本笔记记录现有证据和缺口，不宣称 T1 职业级 80 分通过。

## 1. Entry Protocol

- **上一关**：T1-06-A FoundationPose 50 帧推理、GT 隔离、ICP 配对对照、消融及 Boss 已通关，`ffb2f90` 已与远端 master 对齐；全仓 `902 passed, 106 subtests passed`（上一关历史回归）。
- **本关目标**：逐项审查 G0–G4、E1–E4、5 项 100 分评审维度及硬性门槛；发现并排序真实缺口；为后续标定误差传播与鲁棒性实验制定可验收任务。
- **上游知识**：针孔投影、SE(3)、点云、ICP、BOP RGB-D、模型到相机 6D Pose、GT 预测隔离、统计和 Git。
- **不做的事**：不再运行完整 50 帧 FoundationPose；不把合成外参称为真实标定；不拿同一组已研究数据充作新留出集。
- **最终产物**：证据矩阵（文件/Commit/测试/原始输出/可信程度/缺口）、硬门槛核查表、候选实验与先验预测、笔记和最终 Git/GitHub 记录。
- **验收规则**：每项必须有可核查的证据；文件存在只证明文件存在，不能替代实验有效性或理论掌握；当前 T1 得分暂不赋值。

## 2. 知识地图

`G0 像素→3D 和 SE(3) → G1 RGB-D 点云与定位 → G2 标定/外参/尺度可信性 → G3 6D Pose → G4 扰动鲁棒性 → T1 80 分综合 Boss`

## 3. 入关时的证据地图（历史基线；后续状态以第 8.5～11 节为准）

| 阶梯 | 已能定位的仓库证据 | 目前可支持的结论 | 缺口/重点审计 |
|---|---|---|---|
| G0 | `src/geometry/pixel_to_camera.py`、`src/geometry/rigid_transform.py`、`tests/test_pixel_to_camera.py`、`experiments/EXP035_se3_coordinate_transform.md` | 投影、SE(3) 代码及故障注入已经覆盖 | 复核数学与方向独立测试、单位规则；不能由 round-trip 宣称物理标定 |
| G1 | `src/geometry/mask_to_pointcloud.py`、`src/geometry/cluster_localization.py`、`experiments/OS01_D_open3d_coordinate_chain.md` | Open3D / mask+Depth / 几何处理已有实践 | 核对 OBB/聚类/滤波及 ≥50 样本的证据是否满足评分表 |
| G2 | `experiments/EXP037_extrinsic_validation.md`、`tests/test_exp037_extrinsic_validation.py`、`experiments/EXP036_localization_error_analysis.md` | 合成参考点下外参方向错误、验证盲区已有严格实验证据 | **未证实真实物理标定**；畸变、真实深度尺度、外参可信锚点和独立几何靶标仍需核查 |
| G3 | `src/evaluation/t1_06_a_fp_single_frame.py`、`src/evaluation/t1_06_a_fp_batch_posthoc.py`、`experiments/T1_06_A_FoundationPose_iteration_ablation.md` | 50 帧 GT 隔离 FoundationPose vs PCA+ICP 已完成 | Oracle Mask、已知 CAD、单场景、非 held-out、非对称性感知；需正确限定结论 |
| G4 | `src/evaluation/t1_03_c_three_ablations.py`、`experiments/T1_03_real_RGBD_geometry_and_sensitivity_learning_note.md`、`experiments/EXP037_extrinsic_validation.md` | 深度加性偏置、`fx × 1.10`、缺失深度及外参错误已有受控实验 | 需要审查 ≥4 种扰动是否各具独立结论、整批 P95/失败归因、独立样本与先验预测证据 |

**标记约定**：`证据已定位` ≠ `该项满分`；`合成验证` ≠ `物理精度认证`。

## 4. 入关时 E1–E4 对照（历史基线；完成情况以第 8.5～11 节为准）

| 实验 | 路线图要求 | 已有基础 | 仍需检查或补强 |
|---|---|---|---|
| E1 | 深度噪声 σ 与距离 Z，先预测后用同一批样本配对比较 | T1-03 深度整体 +10 mm；EXP036 数学扰动 | **随机噪声 σ × 不同 Z** 的正式配对设计、统计和不确定性 |
| E2 | 内参 fx/fy/cx/cy 或畸变注入，预测点云形变并验证 | T1-03 `fx × 1.10` 一次受控修改；EXP036 fx 示例 | 多参数、方向/量级、全批配对误差与标定来源；避免重复单帧 fx 演示 |
| E3 | Mask 边界、遮挡、反光/缺失深度；鲁棒中心、聚类与 P95 | T1-03 每十个有效像素置零，OS-01 点云处理已有 | Mask 形态扰动、遮挡和错误来源分组、鲁棒性统计 |
| E4 | 原始 6D Pose 基线与改进，平移 mm 与旋转角 °，必要时 ADD(-S) | T1-04 PCA+ICP、T1-06-A FoundationPose，50 帧配对 | 需明确对称性、已研究数据和未留出测试的限制；不重新跑已冻结基线 |

## 5. 100 分评分表（暂不打分）

| 维度 | 满分 | 得分 | 证据审核 |
|---|---:|---:|---|
| 投影 / 变换理论 | 20 | 待评 | 公式、独立实现、方向测试 |
| 3D 工程基线 | 20 | 待评 | RGB-D→点云/定位/误差，≥50 样本，GT 来源 |
| 6D Pose 复现 | 20 | 待评 | 固定版本，CAD/mask/GT 与对称性口径 |
| 鲁棒性与实验 | 25 | 待评 | ≥4 种扰动，Mean/Median/P95，失败归因与独立测试集 |
| 工程复现 / 展示 | 15 | 待评 | README、环境、测试、性能日志、Git 历史和演示 |
| **合计** | **100** | **未评分** | 分数 ≥80 且全部硬门槛通过才可 HARD PASS |

### 硬性门槛（待逐项核证）

- [ ] ≥50 个可追溯 RGB-D 样本；数据身份与来源可核查。
- [ ] 正确 6D Pose 基线和评价指标；保留对称性与 BOP GT 局限。
- [ ] 至少一次坐标方向错误注入，需有预测/测试/失败定位。
- [ ] 几何测试可独立运行。
- [ ] 至少三次**确实提前写下**预测的受控实验；不得事后补造。
- [ ] GitHub 可复现，外部数据与模型权重须给出来源及环境说明。

## 6. 第一次独立核查任务（不跑 GPU）

在仓库根目录、`ai_3d` 环境下核对下述文件是否**被 Git 跟踪**，然后运行 G0/G2 现有专项测试，保留完整输出（见聊天中的 PowerShell 命令）。完成后把硬门槛标记为「已验收 / 部分证据 / 缺失」，并说明原因。

## 7. 最小理论与预测题

针孔模型：`X=(u-cx)Z/fx`。固定像素 `u`、深度 `Z` 与 `fx`，只把 `cx` 增加 `5 px`：先手算 `ΔX` 的符号和大小，再写一份自己的预测，最后才运行代码。参考输入可选 `u=380 px, cx=320 px, fx=600 px, Z=1.0 m`。

不要将已有的 `fx×1.10` 结果当作本题的答案；两个受控变量不同。

## 8. 本关新增实验与验收记录（2026-11-09，用户 Windows 本机日志）

### 8.1 G0/G2/E2 基础代码与回归

- 核对七个指定文件，全部 `TRACKED`：`src/geometry/pixel_to_camera.py`、`src/geometry/rigid_transform.py`、`src/geometry/mask_to_pointcloud.py`，及 EXP035/EXP037/T1-03/T1-06-A 的四份笔记。
- 专项命令：`python -m pytest -v tests/test_pixel_to_camera.py tests/test_exp035_se3_contract_c.py tests/test_exp037_extrinsic_validation.py tests/test_t1_03_c_three_ablations.py`。
- 本机日志 `80 passed in 0.94s`、`T107_A_FOUNDATION_TESTS_PASS`。这仅验证所列契约和样例，不能证明真实相机标定可信。

### 8.2 E2-1：单点主点偏差（先预测、后运行）

- 在运行前，学员独立预测 `u=380, cx=320, fx=600, Z=1m` 条件下 `cx+5 px`：`X_old=100mm`、`X_new=91.6667mm`、`ΔX=-8.3333mm`，Y/Z 不变；明确这是受控内参敏感性，不是真实物理标定误差。
- 随后通过既有 `src.geometry.pixel_to_camera.pixel_to_camera()` 验证，日志 `T107_A_E2_1_MINIMAL_PASS`。
- 一般解析式：`ΔX_i = -(Δcx/fx) Z_i`（要求 `fx`、有效像素集合、深度固定）。

### 8.3 E2-2：深度翻倍与非刚体性质（先预测、后运行）

- 学员提前预测：`Z=2m` 时 `ΔX=-16.6667mm`，与 `1m` 对比为两倍；不同深度点的 X 位移不同，不是统一刚体平移。
- 本机验证日志 `T107_A_E2_2_SHEAR_PASS`：近点 `−8.333333mm`、远点 `−16.666667mm`、比值 `2.0`；两点距离从 `1.004987562112089m` 变为 `1.0041925999417531m`。这一反例的点间距离不守恒，说明此组点的扰动非刚体运动。不能把一个反例说成对所有特殊点集距离都必然变化。
- 固定 `fx`、只偏移 `cx` 的点云坐标关系为 `X'=X-(Δcx/fx)Z, Y'=Y, Z'=Z`，即 X-Z 剪切。

### 8.4 E2-3：真实 BOP LM 50 帧主点敏感性（先预测、后运行）

- 学员运行前推导：`Δmean(X)=−(5/fx) mean(Z)<0`；X extent 不保证不变。首次给出的旧极值点深度公式遗漏了**极值点身份可能交换**这一前提；导师指出后，正式采用扰动后重新 `argmax`/`argmin` 的正确口径。不能事后将原预测描述为无缺陷。
- 数据及来源：`data/manifests/t1_02_lm_50.json`，Manifest SHA-256 `b5ba15667c803b2f235d0de3f40a7edfe8b584d5ffaef8037cd45392c85b1877`，在本机 `verify_manifest(...)=True`，50 个帧；Oracle 可见 Mask。实验沿用 `read_instance` / `mask_to_pointcloud`；仅改变 `cx + 5px`，有效像素与顺序固定。
- `T107_A_E2_3_REAL50_PASS`，输出保存在本机 `outputs/t1_07_a/e2_3_cx5_50frames.json`；这里记录的是用户日志中的**汇总结果**，不是该 JSON 的完整副本。

| 检查项 | 用户实测结果 | 解读 |
|---|---:|---|
| 样本数 | 50 | 单场景 BOP LM，不能当独立跨场景测试 |
| X 质心负移 | 50 / 50 | 与解析符号预测一致 |
| 最大 `abs(预测−实测)` 质心变化残差 | `4.336808689942018e-14 mm` | 数值浮点量级的一致性 |
| X AABB 宽度增大 | 44 / 50 | 由真实深度与 X 极值重排决定 |
| X AABB 宽度减小 | 6 / 50 | 没有统一方向结论 |
| 以容差 `1e-8mm` 判断不变 | 0 / 50 | 是此次样本和设定下的观察 |
| `argmax(X)` 索引变化 | 1 / 50 | 仅指索引变化，遇到并列最大值还须检查身份歧义 |
| `argmin(X)` 索引变化 | 1 / 50 | 同上；不能推断与 max 变化是同一帧 |
| AABB 变化绝对值上界 | 50 / 50 满足 | `|Δwidth_X| ≤ (5/fx)*(max Z−min Z)`，前提为同一固定点集合 |

- **解释限制：**这证明公式在相同实现管线与固定输入下数值成立，不能证明 `cx` 真实误差为 5px、实际物理定位精度，也没有证明 6D Pose 精度变化；BOP `scene_gt.json` 由读取契约使用，因此不是纯 GT-blind 位姿实验。
- 该实验已经完成一次真实 50 帧受控内参扰动，无需继续重复同类小实验；后续转向 T1 全局证据审计和缺口补强。

## 8.5 E1：随机深度噪声、共享扰动与异方差（先预测，再运行）

### E1-A 独立噪声

- 预测：固定有效点数 `N`、噪声 `ε_i~N(0,σ²)` 独立零均值，`Std(mean(ε))=σ/√N`；`σ` 翻倍则 Std 翻倍，`N` ×4 则 Std 减半。学员先给出推导以及系统偏置/空间相关/缺失采样限制。
- 使用 `SEED=20261109`、`TRIALS=20000` 的 Monte Carlo，在本机报告 `outputs/t1_07_a/e1_independent_noise.json`。执行日志：`T107_A_E1_INDEPENDENT_NOISE_PASS`。

| σ(mm) | N | 理论质心 Std(mm) | 实测质心 Std(mm) |
|---:|---:|---:|---:|
| 2 | 100 | 0.200000 | 0.199370 |
| 4 | 100 | 0.400000 | 0.398369 |
| 2 | 400 | 0.100000 | 0.099341 |
| 4 | 400 | 0.200000 | 0.199860 |

- 实测倍数：`σ_ratio = 1.9981444786185154`，`N_ratio = 0.4982764819961422`。**局限**：此实验目前以交互式 PowerShell 代码执行，尚未定位一个正式可执行的 E1-A Python 文件；报告生成并不能替代脚本归档。

### E1-B 固定偏置与共享随机误差

- 预测：`ε_i=b+c+η_i`；`b=10mm`，每帧 `c~N(0,1²)mm` 共享，每点 `η_i~N(0,2²)mm` 独立。因此 `E(mean ε)=10mm`，`Std(mean ε)=sqrt(1+4/N)mm`；`N` ×4 不会导致总 Std 减半。
- 源码 `src/experiments/t1_07_a_e1_b_shared_noise.py`；报告 `outputs/t1_07_a/e1_b_bias_shared_noise.json`，`T107_A_E1_B_SHARED_NOISE_PASS`、`T107_A_E1_B_REPORT_AUDIT_PASS`。

| N | 理论 Mean(mm) | 实测 Mean(mm) | 理论 Std(mm) | 实测 Std(mm) |
|---:|---:|---:|---:|---:|
| 100 | 10.000000 | 10.005209 | 1.019804 | 1.019079 |
| 400 | 10.000000 | 10.003902 | 1.004988 | 1.002298 |

- 实测 `Std_ratio N400/N100=0.9835329874469777`；报告 SHA-256 `65c03edcaafe3601b4905c42de427e33ad4ff67cdb67ef79c130356cbd0fcd6b`。可见共享扰动的方差下界，但**这是受控模拟而非真实传感器实测**。

### E1-C 深度相关异方差

- 预测：明确合成模型 `σ(Z)=2(Z/1m)² mm`；不同深度时 `Std(mean ε)=sqrt(Σσ(Z_i)²)/N`，不能用 `σ(mean Z)/sqrt(N)` 代替。
- 源码 `src/experiments/t1_07_a_e1_c_depth_noise.py`，测试 `tests/test_t1_07_a_e1_c_depth_noise.py`（`9 passed`）；报告 `outputs/t1_07_a/e1_c_depth_noise.json`；日志 `T107_A_E1_C_HETEROSCEDASTIC_PASS` 和 `T107_A_E1_C_REPORT_AUDIT_PASS`。

| 组别 | 理论 Std(mm) | 实测 Std(mm) |
|---|---:|---:|
| 1m × 100 点 | 0.200000 | 0.199370 |
| 2m × 100 点 | 0.800000 | 0.796739 |
| 2m × 400 点 | 0.400000 | 0.397365 |
| 混合 1m/2m ×100 点 | 0.583095 | 0.584215 |

- `σ(mean Z)/sqrt(N)` 在混合组仅给 `0.450000mm`，低估受控模型的真实标准差。报告 SHA-256 `6d325b1ecacd2f5df3898c1eec0d12d1b02b58bc59b63f722ed021de69c2ece8`。
- **残余缺口**：目前 E1 是合成噪声 Monte Carlo，并未在固定真实 50 帧上对 `σ × Z` 做完整的预注册配对统计，也未测得真实相机噪声模型。因此 E1 原理与合成实验通过，T1 路线图“真实/固定样本 σ×Z 鲁棒性”要求不可直接视为满额完成。

## 8.6 E2-3 报告独立审计补记

- `T107_A_E2_3_REPORT_AUDIT_PASS`；报告 SHA-256 `a85e36801c28c7d345f48ba9950473a0423cd56f1e615c5f3ae31bd41dae4699`。
- 50 帧质心 X 均左移，X extent 增 44、减 6；最大、最小点索引变化均见于 `im_id=66`。极值点身份是否实质变化需注意并列值情况。完整报告在本机 `outputs/t1_07_a/e2_3_cx5_50frames.json`；主实验的可重跑脚本在正式提交前需确认是否落盘，不能只依赖聊天中的临时代码。

## 8.7 E3：真实 50 帧配对 Mask / Depth 扰动及工程加固

- **先预测**：腐蚀/膨胀对应的有效点数满足 `N_erode ≤ N_base ≤ N_dilate`；质心位移方向不能单凭 Mask 面积判断；相同竖带上 Mask=False 与 Depth=0 可导致相同最终 `S=Mask∧ValidDepth`，但不同上游原因。
- **控制变量**：同一 BOP LM 单场景、50 帧、Oracle 可见 Mask、同一内参和真实采集深度，单独进行 `3×3` 腐蚀、膨胀、中心竖带 Mask 删除和同竖带 Depth 置零。竖带为人为模拟，不代表重新采集到的物理遮挡。
- **V1**：`src/experiments/t1_07_a_e3_mask_depth.py` 原版在 50 帧上成功，报告 `outputs/t1_07_a/e3_mask_depth_real50.json` 的独立审计 `T107_A_E3_REPORT_AUDIT_PASS`；V1 报告 SHA-256 `147e13b55224c5926eb2acbc6ff29c72bc944c220116d8e79a354f434e008288`；V1 源码 SHA-256 `93ddda4b342d862af9a2dd911a3be4cdeba604d3fd9de28b5b19cecfd0f1b30c`。
- **V2**：将关键断言替换为显式异常和输入契约；同名源码覆盖、旧报告不覆盖，保存新报告 `outputs/t1_07_a/e3_mask_depth_real50_v2.json`，含独立 `source_sha256` 与 Manifest SHA。V2 专项 `8 passed in 1.23s`，含 `python -O` 子进程注入 `STRIP_OUTPUT_MISMATCH`；真实 50 帧 `T107_A_E3_V2_REAL50_PASS`；新源码 SHA-256 `7f44aac3bb2a88adaddc4349c70c55135250af697431edf59070397a005a104d`。V2 新报告 SHA 和内容仍需独立核对。

| 受控扰动 | Mean shift(mm) | Median(mm) | P95(mm) | Max(mm) | 50 帧合计点数 |
|---|---:|---:|---:|---:|---:|
| erode | 2.315836 | 2.047278 | 5.281616 | 6.103755 | 87446 |
| dilate | 2.927592 | 2.880128 | 5.166566 | 6.232704 | 111280 |
| mask_strip | 1.090984 | 1.002466 | 2.111438 | 2.612461 | 85954 |
| depth_strip | 1.090984 | 1.002466 | 2.111438 | 2.612461 | 85954 |

- C/D 全 50 帧 XYZ、RGB、选中像素完全相同；对应统计一致。**Mean/Median/P95/Max 是相对原始点云的质心位移幅值，不是 GT 定位误差/6D Pose/物理标定精度。** 不得以单场景 50 帧声称工业泛化或独立 held-out。
- 新旧源码版本哈希不同：V1 报告仅是 V1 的历史证据；V2 应与 V2 源码和 Manifest 核验。不在 Windows / Git 换行归一化后未经检查地假设字节级 SHA 永远不变。

## 8.8 Final Boss、排错与全仓回归

- **Boss Q1～Q5**：已独立作答且通过理解验收。重点包括：故障源与最终可见点云不可辨识、Mean/P95 指标不可混淆、拒绝路径与 mutation/negative control、`python -O` 不可依赖 assert，以及工业自动分割 + 深度 + FoundationPose 的控制变量/2×2 交互设计。
- **Q5 修正**：Oracle Mask + 原始深度是受控参考基线而非理论精度上限；同条件误差差值是配对条件效应而非独立误差源唯一分解；交互应使用 `I=e_D-e_C-e_B+e_A`、可能正负；GT-blind、frozen、held-out 是三个不同属性。
- 曾出现**粘贴的测试代码缺 `depth[:]=0`** 与本机 `6 passed` 矛盾；实际本机测试文件包含该赋值，使用 `Select-String` + 单项 `pytest -vv` 证实拒绝路径 `1 passed`，并非实际测试错误。
- 全仓最终回归（用户 2026-11-09 本机日志）：`919 passed, 106 subtests passed in 10.91s`，`T107_A_FULL_REGRESSION_PASS`。旧历史回归为 902（T1-06-A）与 917（E3 V1）；测试数增加不等于已证明物理精度。

## 9. G0–G4 与路线图维度：截至本次会话的证据矩阵

| Gate | 现有证据 | 当前结论 | 未关闭的真实性/复现缺口 |
|---|---|---|---|
| G0 投影与坐标变换 | T1-02/T1-03 几何、EXP035 SE(3)、E2-1/2/3、80 项专项回归 | **教学与代码契约已有强证据** | 实测相机内参、物理方向校验不是 G0 单独完成的 |
| G1 RGB-D 点云基线 | 50 帧 Manifest、`mask_to_pointcloud`、ICP、聚类处理，E3 50 帧受控扰动 | **点云链路有证据** | OBB 直观展示/指标和独立复查需对照仓库证据最终确认 |
| G2 标定与尺度 | EXP037 合成外参方向故障、EXP036 敏感性、E2 `cx+5px` | **仅敏感性与合成契约成立** | 真实靶标、畸变、深度尺度、物理外参及 `calibration_report.md` 不足 |
| G3 6D Pose | T1-06-A FP GPU 50 帧 + ICP 冻结预测 GT 后评 | **受控已知 CAD + Oracle Mask 基线成立** | 对称性感知指标、单场景/非 held-out、外部部署与权重许可证复现限制 |
| G4 鲁棒性 | T1-03 depth bias/fx/dropout、E1 合成噪声、E2 cx、E3 Mask/Depth 真实50帧 | **多种故障已开展** | E1 固定真实样本 σ×Z、独立 held-out、不同材质/场景/遮挡及完整姿态失败分层缺口 |

| 路线图实验 | 结论 | 核心限制 |
|---|---|---|
| E1 随机深度噪声 σ×Z | **合成机制和公式已验证** | 非真实传感器标定，非固定真实 50 帧的全部配对协议 |
| E2 相机内参误差 | **cx 解析与真实50帧验证**，此前 fx 敏感性亦有记录 | 尚无畸变/真实标定误差测量 |
| E3 Mask/遮挡/缺失 | **真实50帧几何敏感性 + Mean/Median/P95 + C/D 等价** | 没有自动分割真实系统、鲁棒中心/聚类对比和姿态端到端的独立测试 |
| E4 6D Pose 对照 | **FP vs ICP、FP Refiner 1 vs 5 的先前记录** | 非 held-out，原始旋转角未按对称性修正；本关不重跑 GPU |

## 10. 100 分评分与硬门槛：暂不宣称 T1 HARD PASS

| 评分维度 | 满分 | 当前评审结论 |
|---|---:|---|
| 投影 / 坐标变换 | 20 | 证据较充分，待正式按条目给分 |
| 3D 工程基线 | 20 | 有 50 帧和工程实验，OBB / 展示部分待终审 |
| 6D Pose 基线 | 20 | 有 FP 与 ICP 对照，但对称性/独立性口径需保留 |
| 鲁棒性 / 独立实验 | 25 | 多种扰动，但真实 σ×Z 与 held-out 未关口 |
| README / 环境 / 可复现展示 / Git | 15 | 尚未验证文档、环境复现及本关远端提交 |
| **合计** | **100** | **暂不评分：不可凭 919 passed 推断达到 80** |

**硬门槛状态：**

- [x] ≥50 帧可追溯 RGB-D 输入：BOP LM Manifest + 50 帧配对扰动（样本局限已披露）。
- [x] 6D Pose 基线与可核查原始指标：T1-06-A FP/ICP，但不宣称对称性感知物理准确率。
- [x] 坐标方向错误注入：EXP035/EXP037。
- [x] 独立几何测试：本机 919 passed，106 subtests；并含 E3 `python -O` 故障注入。
- [x] ≥3 次先预测再验证：E2-1/2/3，E1-A/B/C，E3（聊天顺序可溯源）。
- [ ] GitHub 复现：本关未提交/推送，README 与环境重建流程仍需核对。
- [ ] T1 80 分职业评价所要求的独立测试集与真实标定完整性：尚无充分证据。

## 11. 此次 Learning Unit Exit Protocol

- **已经完成**：投影/SE(3) 与误差敏感性审计；E1-A/B/C 先预测 + 合成 Monte Carlo；E2-1/2/3 解析与真实50帧；E3 Mask/Depth 真50帧 V1→V2、显式异常、`python -O` 注入与配对统计；Final Boss Q1～Q5；全仓 919 测试通过。
- **能够解释、预测、修改、排错、迁移的内容**：随机/共享/异方差传播，固定像素集合下内参剪切，Mask 与 Depth 有效集合等价而故障不可辨识，配对实验和 GT 隔离，fail-closed 和版本哈希。
- **仍不稳定/未闭口**：E1 真 50 帧 σ×Z 配对、G2 物理标定、独立 held-out、鲁棒中心/OBB、README/环境重建、第三方模型获取与复现。
- **文件**：实验源文件 `src/experiments/t1_07_a_e1_b_shared_noise.py`、`t1_07_a_e1_c_depth_noise.py`、`t1_07_a_e3_mask_depth.py`，两项新测试，5 份 V1/旧实验 JSON + 新 V2 JSON；E1-A/E2-3 可复现脚本待独立确认或归档。
- **最后一次验证**：E3 V2 `8 passed`、真实50帧 `T107_A_E3_V2_REAL50_PASS`、全仓 `919 passed, 106 subtests`；**V2 报告独立审计仍待做**。
- **Git**：以用户上次 `git status --short --untracked-files=all` 为准，本关源码和结果尚未跟踪。备份 `.bak_*`、ScreenCapture、历史 outputs、raw data 和权重不提交；精确 `git add --`；最终核对本地 HEAD 与 `origin/master` 一致。
- **当前 Level 状态**：**T1-07-A · CONTINUE**；理论/实验/Boss 通过，笔记+Git+GitHub 未闭环。
- **下一步**：独立审计新 V2 JSON（source/manifest hash，逐帧聚合、V1/V2 指标一致）；复制本笔记到仓库；核对 README/环境；精确提交 + Push + 远端 SHA 验收。之后 T1 职业级 80 分是否达标应另行据完整证据评分，不自动通关。


<!-- T107-E3-V2-FINAL-AUDIT -->

## T1-07-A 最终验收补充记录

本节为最新审计状态，取代前文关于
E3 V2 报告“待审计”的历史描述。

### E3 V2 最终审计

- 真实 BOP LM 50 帧验证：PASS
- Mask/Depth 等价性：50/50 PASS
- V1/V2 逐帧结果一致：PASS
- Manifest SHA-256 验证：PASS
- Source SHA-256 验证：PASS
- 独立报告统计复核：PASS
- 专项测试：8 passed
- 全仓回归：919 passed，106 subtests passed
- Final Boss Q1～Q5：PASS

E3 V2 源码 SHA-256：

7f44aac3bb2a88adaddc4349c70c55135250af697431edf59070397a005a104d

E3 V2 报告 SHA-256：

ee82ff84c545f83466b92146094ed482be9fee780d1876ed3952b1b7e27aff95

### 验收边界

以上证明受控几何实验及当前工程测试通过。

不代表真实传感器物理标定通过，
不代表独立 held-out 测试完成，
也不代表整个 T1 职业级 80 分达标。

### 下一步

审核环境配置与 Git 暂存内容，
完成 Commit、GitHub Push 和远端 SHA 核验。

当前状态：T1-07-A CONTINUE。
