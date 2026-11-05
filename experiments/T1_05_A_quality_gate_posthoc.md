# T1-05-A｜GT-free 6D Pose 候选质量门禁与事后风险审计

> **Learning Unit：T1-05-A｜状态：IN PROGRESS（实验、自动化测试与最终 Boss 理解验收通过；Git/GitHub 闭环待完成）**
> 本文根据本机已提交运行日志整理。最后提交仍为 T1-04-B `7c680efdaf2dbfb748621030f182477baef168d9`。
> 本研究的 Oracle Mask、同场景同物体 50 帧、已见 GT、原始旋转角差与启发式门禁的限制必须随实验结果一并呈现。

## 1. 本关目标、前置知识与数据流

- 上游：T1-04-B 已冻结的 50 份 BOP LINEMOD `scene=000001, obj_id=1` PCA 24 初值 + Trimmed ICP 位姿候选，以及**独立** B4 GT 评测文件。
- 当前目标：用**不含 GT** 的支持比例和 ICP 裁剪 RMSE 给既有候选标记 `PASS_HEURISTIC_NOT_POSE_ACCURACY` 或 `REVIEW_HEURISTIC_NOT_REJECTED_BY_GT`；审计程序在报告冻结后读取 B4 的**GT 派生**评测结果进行事后关联。
- 下游：面向更加可靠的 GT-free 多候选一致性、遮挡/可见性与重投影检查，以及真正未参与调参的验证数据。
- 不重新配准、不使用 GT 选候选、不产生机器人可执行坐标；`publishable=False`, `position_m=None`。

数据流：

```text
B4 已冻结 50 份 candidate JSON（GT-free Pose）
    ├─ T1-05-A 质量门禁：仅使用 support_fraction / trimmed_rmse_m
    │    └─ baseline / A / B 三份门禁报告 + C 阈值不匹配拒绝
    └─ B4 既有独立 GT 评测 JSON
         └─ T1-05-A 后验审计：验证候选及评测 SHA-256 / 身份 / 阈值 / 决定
              └─ 50 帧门禁覆盖率与原始大角度误差放行统计
```

## 2. 最小公式与核心参数

以预测完成后**固定姿态**下第 `i` 个观测点到变换后模型的最近邻距离 `d_i`（单位 m）定义：

\[ s = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[d_i\le \tau] \]

- `support_fraction = s`：浮点，0～1，无量纲；基于最终姿态的最近邻距离统计。
- `support_threshold_m = τ`：8 mm = `0.008 m`，**支持率的统计口径**，不是本门禁重配准参数。
- `trimmed_rmse_m`：浮点，m，由 B4 预测代码在其裁剪后的最近邻集合上计算。
- `min_support_fraction`：门禁最低支持率；Baseline 0.80，A 0.90。
- `max_trimmed_rmse_m`：门禁最高裁剪 RMSE；Baseline 0.004 m，B 0.003 m。

\[ \mathrm{PASS}_{heuristic} = (s \ge s_{min}) \land (RMSE_{trim}\le r_{max}) \]

`PASS_heuristic` 只表示两个几何条件都满足，不是 GT 准确率、安全审批或置信概率。只有保存的 `support_threshold_m` 与门禁所需的阈值一致才有资格进行比较；不能用 8 mm 下的支持率冒充 4 mm 下的统计。

## 3. A/B/C 单变量修改：预测与实测

最初 A/B/C 运行前预测已由学员在执行前明确提交，并得到下述结果：

| 变体 | 唯一改动 | 预测 | 本机 50 帧实测 |
|---|---|---|---|
| Baseline | `support ≥0.80; RMSE ≤4mm; τ=8mm` | 固定对照 | PASS 37；REVIEW 13；覆盖率 74% |
| A | `min_support 0.80→0.90` | PASS 不增 | PASS 2；REVIEW 48；覆盖率 4% |
| B | `max_rmse 4→3mm` | PASS 不增 | PASS 4；REVIEW 46；覆盖率 8% |
| C | `support_threshold 8→4mm`，但不重算支持率 | 必须 fail-closed | 抛出 `ValueError: SUPPORT_METRIC_THRESHOLD_MISMATCH`；未生成无效报告 |

Baseline → A 减少 35 个 PASS，Baseline → B 减少 33 个 PASS。这些是覆盖率变化，不是姿态准确率改善。

真实 CLI 日志目录：`outputs/t1_05/gate_20261030_190304/`；门禁输入是 `outputs/t1_04/b4_50_20261030_091523/`。专项自动化测试 **19 passed**。

## 4. 事后 GT 审计：关键数据

事后审计**只读取 B4 已有 GT 派生评测文件**，不直接打开 `scene_gt.json`，不重新生成或选择姿态。它校验 B4 冻结清单、候选 SHA-256、评测 JSON SHA-256、门禁报告哈希/配置/身份并重算规则决定。审计文件：`outputs/t1_05/gate_20261030_190304/posthoc_gt_audit.json`。

原始旋转角差 `>90°` 的有 `27/50`，`>150°` 的有 `16/50`。

| 配置 | PASS/50 | REVIEW/50 | PASS 中 raw rot >90° | PASS 中 raw rot >150° | raw rot >90° / PASS |
|---|---:|---:|---:|---:|---:|
| Baseline | 37 | 13 | 24 | 15 | 24/37=64.8649% |
| A_support_090 | 2 | 48 | 0 | 0 | 0/2=0% |
| B_rmse_003 | 4 | 46 | 1 | 0 | 1/4=25% |

**由上表派生的后验诊断，不是额外实验：**

- Baseline 的 `>90°` 候选：PASS 24，REVIEW 3；原始旋转角差 `≤90°`：PASS 13，REVIEW 10。
- A 的 `>90°` 候选：PASS 0，REVIEW 27；`≤90°`：PASS 2，REVIEW 21。
- B 的 `>90°` 候选：PASS 1，REVIEW 26；`≤90°`：PASS 3，REVIEW 20。
- Baseline 放行了全部 27 个 `>90°` 案例中的 `24/27=88.8889%`；这证明当前简单门禁不能可靠拦截此数据上的大角度差异。
- A 的 0/2 不证明“可靠度 100%”：仅保留了总共 2 帧，覆盖率只有 4%；B 仅保留 4 帧，覆盖率 8%。
- 审计程序不提供对称性感知 BOP 官方指标；`>90°` 是选定的事后诊断事件阈值，不是工业部署失败概率。

**帧 55 的反例**：`support_fraction≈0.8384401`，`trimmed_rmse≈3.635847 mm`；Baseline PASS，但 B4 原始旋转差 `≈176.214°`；若 A `support≥0.90`，不满足支持率；若 B `RMSE≤3mm`，不满足 RMSE，上述两项门禁应将该帧标为 REVIEW。该说明是**事后推导**，不能冒称为未运行前的 D/E 预测记录。

## 5. 测试与可复现证据

- `src/evaluation/t1_05_a_gt_free_quality_gate.py` + `tests/test_t1_05_a_gt_free_quality_gate.py`：**19 passed**。
- `src/evaluation/t1_05_a_posthoc_gate_audit.py` + `tests/test_t1_05_a_posthoc_gate_audit.py`：**8 passed**。
- Windows / `ai_3d` 本机全仓 `python -m pytest -q`：**870 passed，106 subtests passed，10.96 s**。
- `git diff --check`：无错误输出（仅检查已跟踪的未暂存差异，不等于对所有未跟踪文件完成安全检查）。
- 审计验证覆盖候选文件篡改、评测文件篡改、门禁决定篡改、阈值错配、样本数错配、0 PASS 除零保护，以及隔离原始 `scene_gt.json`。

## 6. 重要边界与反例分析

1. **数据已见 GT：**当前 50 帧此前用于 B4/B5 的失败诊断和阈值讨论；本轮属于 GT-exposed post-hoc audit，不能对外宣称无偏测试或参数泛化提升。
2. **单场景、单物体、Oracle Mask：**没有跨物体泛化、端到端分割或物理精度验证证据。
3. **阈值不是准确率：**PASS 只是诊断规则条件成立，不是正确姿态的证书。0/2 的高角度错误不能被称为 100% 准确率。
4. **原始旋转误差不是 BOP 官方对称性感知分数：**LM obj_id=1 模型元数据未声明对称变换，尚不能把大角度差异归因于具体旋转对称性机理。
5. **保留覆盖率分母：**REVIEW 候选仍属于 50 帧，不得从整体性能统计中删除或当作零误差。
6. **D/E 先预测记录缺失：**上一轮要求的 D/E 预测未在这次运行日志之前单独提交，不能把事后正确解释当作 SOP 先预测后验证通过；A/B/C 的先预测记录已存在。

## 7. 最终 Boss 验收（2026-11-05）

**Boss 1 / 统计：PASS，术语已纠偏。** 学员独立构造了对 50 帧的 2×2 表：

| | raw rotation >90° | raw rotation ≤90° | 合计 |
|---|---:|---:|---:|
| PASS | 24 | 13 | 37 |
| REVIEW | 3 | 10 | 13 |
| 合计 | 27 | 23 | 50 |

- 对预定义 `raw rotation >90°` 事件的 REVIEW 比例 = `3/27 = 11.1111%`。另 `24/27 = 88.8889%` 此类案例被 PASS 放行。
- `P(raw rotation >90° | PASS) = 24/37 = 64.8649%`；`P(raw rotation ≤90° | PASS) = 13/37 = 35.1351%`。
- **重要修正：** `≤90°` 不等于“可接受位姿”或“真实准确率”。90° 只是本关的事后事件阈值，尚无旋转/平移联合成功判据，也不是 BOP 官方对称感知分数或工业容差。

**Boss 2 / 条件概率：PASS。** 学员正确给出 `PASS(new threshold) ⊆ PASS(old threshold)`，并构造 5 帧反例：旧门禁 5 PASS，其中 2 帧 >90°（40%）；收紧支持率后只余 2 PASS，二者都 >90°（100%）。该例证明**集合单调缩小，不代表条件错误比例单调减小**。反例所称“对称翻转”只能作为假设，不是对 LM obj_id=1 已证明的失败机理。

**Boss 3 / GT-free 可见性与重投影方案：设计理解 PASS，尚未编码/验证。** 学员提出以已冻结候选 R/t、PLY、相机 K、Depth、Oracle Mask 为输入，计算相机系模型点、像素投影、正深度/视野内比例、Mask 重合与深度一致性；将多初值一致性降为辅助证据，且预测端不读取 `scene_gt.json`。

进入下一 Learning Unit 实现前必须遵守以下设计修正：

1. **先处理可见性：**不能把全 PLY 所有顶点直接当成可见模型表面，投影到 Mask 外就拒绝。应基于三角网格渲染、Z-buffer / 深度遮挡判断及视野裁剪产生预测可见表面，并注意均匀表面采样，避免模型顶点密度偏置。
2. **遮挡和缺测：**对无效 Depth、投影出界、目标内外及前景遮挡分别计数，合理定义“有效可比较像素”及分母；不能把所有不可见模型点当成错误。深度比较应说明采样/像素取整方法及单位。
3. **不是独立传感器证据：**ICP 本身已使用该 RGB-D 深度和 Oracle Mask，重投影检查重用相同观测，属于 GT-free 的互补几何一致性，而非统计独立的真值验证。错误/对称位姿可能仍同时满足 ICP 与重投影条件。
4. **对称来源：**只在模型声明或经核查确认的对称变换可用时，才能枚举对称集；LM obj_id=1 官方元数据没有声明离散或连续对称，不能凭大角度误差自动标成 `SYMMETRY_AMBIGUOUS`。即使显式几何对称成立，多种姿态可能观测不可区分，不能保证重投影揭穿。
5. **阈值仍为待验证假设：**例如 `in_image_ratio≥0.5`、深度 median≤5mm、P95≤15mm、方向一致性 15° 等均为初始实验阈值，未经过独立开发/验证集论证，不可称部署认证。验证还需覆盖可见性退化、边界精度、对称不确定性、严重伪匹配和误放行/覆盖率。
6. **单元测试职责：**合成测试可用设计好的已知姿态构造输入和断言；门禁函数自身绝不能读取或利用 GT 选择候选。测试不得无条件假设“所有 180° 旋转都必须拒绝”或“正确姿态永远通过”。

**本关 Boss 最终判定：3/3 PASS（其中 Boss 3 为方案设计能力通过，不代表可见性/重投影模块已实现）。**

**预测记录边界：**A/B/C 运行前预测已提交；D/E 未在相关审计运行前单独提交，因此不计作正式“先预测再验证”，即使事后推理正确也不能补录成过去的预测。

## 8. Git/GitHub 闭环前清单

- [x] T1-05-A 最小理论、参数拆解与 A/B/C 先预测后验证。
- [x] 50 帧门禁结果和后验审计结果。
- [x] 专项 19 + 8 项测试；全仓 870 passed，106 subtests passed。
- [x] 最终 Boss 1/2/3 方案与理解验收；本笔记已据实际作答复核（待复制至用户仓库）。
- [ ] 审查运行 JSON 的敏感/GT 边界；建议将三份门禁 JSON、后验审计 JSON 按字节复制到 `experiments/artifacts/t1_05_a/`，保留当前本机 `outputs/` 原始副本；冻结候选 JSON 未入 Git，所以摘要归档不等同完整独立复现资产。
- [ ] **精确** `git add --` 两份源码、两份测试、学习笔记、四份已核查摘要；不使用 `git add .`，不加入无关六个截图文件及 `outputs/t1_04`, `outputs/t1_05` 整个目录。
- [ ] `git diff --cached --check`、全仓回归、`git commit`、`git push origin master`，验证 `HEAD == origin/master`。

> **本关当前状态：IN PROGRESS（仅剩笔记入库、实验报告精确归档与 Git/GitHub 远端核验）；完成后才能改为 COMPLETE。**
