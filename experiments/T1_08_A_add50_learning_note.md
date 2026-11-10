# T1-08-A｜FoundationPose 50 帧 ADD / ADD-S 几何指标审计学习笔记

> Project：`Project_002_RGBD_Object_Localizer`
> Level：Level 2 / T1 职业级 80 分强化
> Learning Unit：T1-08-A 6D Pose 评价指标与对称性
> 记录日期：2026-11-10
> 状态：**理论、运行前预测、最小实验、50 帧评价、独立统计审计、全仓回归、Boss 讨论已完成；正式 Git/GitHub 交付待验收**。

## 1. 目标与先修知识

目标：区分 ADD 与 ADD-S 的模型点对应协议，验证人为构造的对称模型，使用**已冻结**的 FoundationPose 50 帧结果及 BOP LINEMOD `scene=000001, obj_id=1` CAD 顶点进行补充后验几何评价，不重新运行模型，也不把评价指标误写成物理精度或官方 BOP 成绩。

知识链：像素反投影与 SE(3) → T1-04 ICP 基线及后验 GT 评价 → T1-06 FoundationPose 冻结预测 → 本关 ADD/ADD-S → T1 总证据审计。

## 2. 公式、单位和语义

设 CAD 模型顶点 `p_i` 的单位为 m，两个位姿均为 `T_camera_from_model`，平移 `t_e,t_g` 单位 m。

- **ADD**（同一顶点对应）：`ADD_mm = 1000/N * Σ_i ||R_e p_i + t_e − (R_g p_i + t_g)||`。
- **ADD-S 最近邻诊断**：`ADD-S_mm = 1000/N * Σ_i min_j ||R_e p_i + t_e − (R_g p_j + t_g)||`。
- 对相同点集和有限合法位姿，逐顶点最近邻距离不大于与同编号顶点距离，因而 `ADD-S ≤ ADD`。本次是离散 CAD **顶点集合**上的 ADD 与最近邻 ADD-S，不是官方 BOP-VSD/MSSD/MSPD 分数，也不是均匀表面采样指标。
- BOP `cam_t_m2c` 单位 mm，评价器使用 `PoseM2C.from_bop()` 仅转换一次；PLY 原始顶点单位 mm，使用 `×0.001` 变为 m。错误的量纲换算可把结果放大 1000 倍。

## 3. 运行前预测与最小验证

**在执行程序之前提交的预测**：对以原点为中心、边长 `2a` 的正方形四顶点集合，GT 为单位旋转，预测为绕 Z 轴 180° 且平移为零。预测 `ADD-S=0`（旋转使离散点集映射到自身），`ADD>0`（对应顶点移位），四顶点 `ADD=2a√2`。

**实测**：`a=0.05 m`，`ADD=141.421356 mm`，`ADD-S=0.000000 mm`，原始旋转差 `180.000000°`；`T108_A_MINIMAL_METRIC_PASS`。该结果只证明人工对称点集上的指标行为，不能推广为实际物体具有对称性。专项测试 `5 passed`，包括纯平移、非法或非有限顶点拒绝以及 ADD-S 上界契约。

## 4. 真实 BOP LM 50 帧后验评价

输入：原 T1-06-A 固定 50 帧 GT-blind 冻结候选及其 SHA，CAD `obj_000001.ply`，BOP 数据集 `scene_gt.json`（**只在预测冻结核验后读取**）。固定 `scene=000001`，`obj_id=1`，`gt_index=0`，Oracle Mask，已研究过的 50 帧，非 held-out。CAD 共 **5841 个顶点**。`models_info.json` 对本物体没有声明离散或连续对称性；不能据此证明物体在数学上完全不对称。

原始输出：`outputs/t1_08_a/fp_add50_v1.json`。为 GitHub 归档时建议保留一个字节相同的轻量副本：`experiments/artifacts/t1_08_a/fp_add50_v1.json`。

| 指标 | ADD（mm） | ADD-S 最近邻诊断（mm） |
|---|---:|---:|
| Mean | 5.053290347936849 | 2.5751862491334565 |
| Median | 5.083874206504339 | 2.487315460840483 |
| P95（linear） | 8.634341604731683 | 4.212666807094443 |
| Max | 10.046496240408882 | 4.968557005051291 |

按 ADD 从大到小排列的前五帧：`55: 10.0465mm`、`106: 8.8729mm`、`87: 8.8489mm`、`350: 8.3721mm`、`66: 8.0718mm`。这些是固定选取样本的失败/差异观察，不能当作模型泛化最差情形。

程序标识：`T108_A_FROZEN_50_PREFLIGHT_PASS: GT_NOT_READ_YET`、`T108_A_ADD50_PASS`。

输出原文件 SHA-256：`1489BE4D43414731A19683329BA033F16FA055A71D0D2F63802A5BB2CBEDEA8F`。该哈希必须在后续归档时按本机重新验证；不允许以文件名相同代替内容一致。

## 5. 独立审计与工程回归

通过独立 PowerShell/Python 只读审计：

1. `data["count"] == 50`、记录 50 行且 `im_id` 不重复、顺序同既有评价摘要。
2. 模型顶点数 5841；冻结预测、GT 注释、CAD PLY 的文件 SHA-256 与报告中记录一致，**3/3 PASS**。
3. 全 50 帧 `ADD-S ≤ ADD` 且非负、有限，**50/50 PASS**。
4. 从逐帧指标独立重算 Mean、Median、线性插值 P95 和 Max，与报告吻合，`T108_A_INDEPENDENT_AUDIT_PASS`。
5. Windows `ai_3d` 全仓 `python -m pytest -q`：**924 passed、106 subtests passed in 12.35s**，`T108_A_REGRESSION_AND_AUDIT_PASS`。

## 6. Boss：指标结果能说明什么、不能说明什么

已解释：ADD-S 最近邻对应总比同编号 ADD 更宽容，不能根据 ADD-S 均值较低就断言 FoundationPose 擅长识别对称性。

**必须保留的两项修正**：

- `ADD-S 小` 仅证明离散点集的最近邻距离小，**不必然**代表预测靠近某个真正对称等价姿态。`ADD-S 小、ADD 大` 是检查对称性/近似重复几何的线索，不是对称性的充分证据。
- 对称等价姿态可能有不同的旋转矩阵代表。若当前任务采用对称感知评测，**并不要求每帧选择同一代表**；是否需要时序连续的代表姿态，应根据下游跟踪或操作要求另行规定。严格的对象对称关系通常考虑模型定义的变换集合 `S` 及完整 `T_gt @ S`，不总能只比较旋转矩阵。

仍未证实：真实的对称变换集合、FoundationPose 是否主动利用对称先验、跨场景/不同物体泛化、独立 held-out、自动分割端到端性能、真实相机标定精度。

## 7. 代码、参数及复现

- 评价器：`src/evaluation/t1_08_a_add50_audit.py`；核心函数 `point_metrics_mm(vertices_m, estimate, gt)`。`vertices_m` 为 `float64 [N,3]` 米，位姿 `PoseM2C`，输出标量 `(add_mm, adds_mm)`。`cKDTree` 对 GT 变换模型顶点构建最近邻索引，逐个预测顶点查最近距离。
- 专项测试：`tests/test_t1_08_a_add50_audit.py`，共 5 项。
- 原冻结输入：`outputs/t1_06_a/fp_fixed50_v1`，原后验摘要：`outputs/t1_06_a/fp_fixed50_eval_v1/batch_summary.json`。
- 在已经生成输出 JSON 的机器上**不要重复调用默认运行入口**，因为 `fp_add50_v1.json` 已存在，代码有防覆盖门禁；复核结果应优先执行只读检查。
- 本实验依赖公开 BOP LM 数据、已有 T1-06-A 冻结预测、`numpy`、`scipy`、`open3d`。`environment.yml` 已提交，但全新环境一键重建仍未实测验证。

## 8. Learning Unit Exit Protocol

- [x] 公式、单位、点集对应与对称性理论讲解。
- [x] 学员执行前独立预测；模型实验与预测一致。
- [x] 50 帧 CAD 后验评价（无 GPU 重推理）。
- [x] 数据哈希、帧身份及统计指标独立审计。
- [x] 5 项专项测试、全仓 924 项测试及 106 subtests。
- [x] Boss 讨论及两项准确性修正。
- [ ] 学习笔记及轻量报告正式归档，逐文件安全暂存。
- [ ] Git commit、GitHub push、本地与远端 HEAD SHA 一致。

**状态：T1-08-A 仍为 IN PROGRESS，尚未完成 Git/GitHub 闭环。T1 职业级评分先维持上轮 76/100 暂评，待全项目证据再审，不自动 +分或宣布 80 分通关。**
