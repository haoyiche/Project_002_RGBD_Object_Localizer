# T1-04-B｜真实 BOP LM 6D Pose 基线与单变量消融（实验与 Boss 验收通过，待 Git 闭环）

> 状态：**IN PROGRESS（仅剩 Git/GitHub 闭环）**；B4/B5 baseline 交叉审计与最终 Boss 已通过；不宣称算法姿态精度达标。
>
> 数据：BOP LM，scene `000001`，obj_id `1`，Oracle visible mask；**GT-pose-blind** 候选预测，BOP GT 仅在独立评测阶段使用。

## 1. 目标与知识连接

- T1-01：BOP `cam_R_m2c` 与 `cam_t_m2c`；GT 平移从 mm 转换为 m。
- T1-02：真实 RGB-D、实例标识和数据哈希；预测阶段不得读取包含 GT 的 T1-02 Manifest。
- T1-03：深度反投影与可见点云；可见局部 AABB 不等于完整模型 AABB。
- T1-04-A：model→camera 位姿契约、平移 mm 误差、SO(3) 原始旋转误差，以及非对称感知指标限制。
- T1-04-B：使用已知模型（PLY 顶点 mm→m）+ 深度 + Oracle Mask，GT-free PCA 多初值 + trimmed point-to-point ICP，冻结候选，再独立评测。

## 2. 模型与数据检查

- 官方 `lm_models.zip` SHA-256：`CB5B5366CE620D41800C7941C2E770036C7C13C178514FA07E6A89FDA5FF0E7F`，核验 PASS。
- `obj_000001.ply`：5,841 顶点、11,678 三角面。
- 全模型尺寸：[75.8686, 77.5992, 91.769] mm，即 [0.0758686, 0.0775992, 0.091769] m。
- `models_info.json` 中 `diameter=102.099 mm`。
- `symmetries_discrete=[]`、`symmetries_continuous=[]`：元数据未声明此物体的对称变换；不能据此断言几何完全没有近似对称。
- 模型通过元数据尺寸的单位契约检查，不允许 1000 倍尺度错误进入 ICP。

## 3. 数据流、评测隔离与关键公式

**B1 预测输入**：场景 `scene_camera.json`、RGB/Depth PNG、指定 `im_id`+`gt_index` 的 Oracle Mask、`obj_id` 对应模型 PLY 和 `models_info.json`。不读 `scene_gt.json`。样本中的物体 ID、Mask 来自标注；不是端到端检测。

**B2 评测输入**：已冻结候选 + 输入文件/模型 SHA-256 + 独立的 `scene_gt.json`。只读现有候选，不回调 ICP 或根据 GT 选择候选。

对估计位姿 `\hat{T}_{camera←model}` 与 GT 的原始误差：

- `translation_error_mm = 1000 * ||t_est_m - t_GT_m||_2`。
- `rotation_error_deg_raw = angle(R_est * R_GT.T)`；此处不是对称性感知 BOP 官方评分。
- `trimmed_rmse_m = sqrt(mean(d_nearest^2))`，只在保留的最近对应点上计算，反映**局部几何拟合**，不是位姿 GT 误差。

B1 `fit_pca_multistart_icp` 源码中的关键参数和流向：

```python
ntrim = max(30, int(np.ceil(trim_fraction * len(observed_points))))
chosen = np.argpartition(distance, ntrim - 1)[:ntrim]
# chosen 用于 SVD/Kabsch 更新；ntrim 也用于最终 trimmed_rmse 与候选选择。
# max_iterations 限制每个初始位姿的 ICP 更新次数。
support_fraction = float(np.mean(distance <= support_threshold_m))
# support_threshold_m 只影响诊断 support_fraction，
# 不参与更新、trimmed_rmse 或候选选择。
```

## 4. B4 固定参数、50 帧完整基线（已完成）

- 50 个 GT-pose-blind 候选、50 个独立 GT 评测、失败 0，冻结文件哈希/P95 审计通过。
- 专项加全仓回归至 B4 时：836 passed、106 subtests passed。

| 指标 | 平移误差 (mm) | 原始旋转误差 (°) |
|---|---:|---:|
| Mean | 12.247970 | 97.454421 |
| Median | 12.940895 | 107.819548 |
| P95（NumPy linear） | 23.407751 | 172.522562 |
| Max | 24.996436 | 176.214258 |

- 27/50 帧原始旋转误差 >90°；16/50 帧 >150°。
- Top-5 按旋转误差与按 ICP RMSE 排名重合 2 帧：im_id 228、245。
- Spearman ρ(RMSE, 原始旋转误差)=0.320528；ρ(RMSE, 平移误差)=0.070156。
- B3 与 B4 共有的 5 帧候选 R、t **差异恰为 0**，候选 JSON SHA-256 也完全一致。
- 这些数字是 **50 帧单场景、单物体、Oracle Mask、无对称性修正的 GT 标注相对诊断**；不是物理精度、BOP 官方排名或跨场景泛化性能。

## 5. B5 三次单变量消融（已运行）

诊断帧 47/55/240 **由先前 B4 的 GT 结果选择**，属于 GT-informed case selection；本次预测阶段仍不读取 GT，且 12 个候选先冻结再独立评测。不得将选帧后的消融解释成无偏基准提升。

固定基线：`max_iterations=18`、`trim_fraction=0.75`、`support_threshold_m=0.008`。

- A `trim_060`：只把 `trim_fraction` 改为 0.60。
- B `iter_036`：只把 `max_iterations` 改为 36。
- C `support_004`：只把 `support_threshold_m` 改为 0.004 m。

每一行数据为 `translation_error_mm / rotation_error_deg_raw / trimmed_rmse_mm / support_fraction`：

| im_id | 变体 | 平移 mm | 原始旋转 ° | ICP RMSE mm | Support |
|---|---|---:|---:|---:|---:|
| 47 | baseline | 3.305 | 32.364 | 3.899 | 0.756 |
| 47 | trim_060 | 18.765 | 165.875 | 3.030 | 0.746 |
| 47 | iter_036 | 23.322 | 172.829 | 3.463 | 0.838 |
| 47 | support_004 | 3.305 | 32.364 | 3.899 | 0.501 |
| 55 | baseline | 21.724 | 176.214 | 3.636 | 0.838 |
| 55 | trim_060 | 3.921 | 43.339 | 3.053 | 0.741 |
| 55 | iter_036 | 17.285 | 155.919 | 3.471 | 0.819 |
| 55 | support_004 | 21.724 | 176.214 | 3.636 | 0.513 |
| 240 | baseline | 16.744 | 169.351 | 3.351 | 0.849 |
| 240 | trim_060 | 17.324 | 177.763 | 2.843 | 0.764 |
| 240 | iter_036 | 17.174 | 164.395 | 3.190 | 0.882 |
| 240 | support_004 | 16.744 | 169.351 | 3.351 | 0.555 |

**A 实测**：保留对应点占比下降。在三个帧上裁剪后的 RMSE 都下降，但旋转误差分别变化 +133.511° / -132.875° / +8.412°。不同裁剪比例的 trimmed RMSE **使用不同对应点集合**，不能直接将降低当作拟合变得更准确。

**B 实测**：增加迭代次数后，三个帧的 RMSE 都下降，但旋转误差分别变化 +140.465° / -20.295° / -4.956°。更多迭代没有保证更低的 GT 旋转误差。

**C 实测**：三个帧的 R、t、trimmed RMSE 与 baseline 均相同；`support_fraction` 分别从 0.756→0.501、0.838→0.513、0.849→0.555。这一参数**不改变候选、不触发拒绝状态**，只改诊断统计。

B5 专项：7 passed；完整回归：**843 passed、106 subtests passed**。`git diff --check` 无报错输出。

## 6. 预测 / 实测与未完成事项

- A、B 的方向性预测与实测一致：降低 trim 不保证精度提升，增加迭代不保证 GT 旋转误差降低。
- C 需要纠偏：预先列出的四个字段实际是 `R`、`t`、`trimmed_rmse_m`、`support_fraction`。`support_count`、`support_mean_distance_m`、`support_status` 并非本代码当前生成的输出字段；不能将推测中的拒绝状态写成实测。
- 本次预测文字在聊天记录中附在运行日志之后，无法从当前记录独立证明“先预测后执行”的顺序；下一轮需要明确先提交预测，再运行。
- B5 的改善或恶化只能用于**三个 GT-informed 诊断帧**；不以 3 帧结论代替固定 B4 50 帧表现，也不以这些帧选择最优参数后再声称独立精度提升。

## 7. 最终 Boss 验收（2026-10-30）

**Boss 1：计算与参数，PASS**。帧 47 `len(b)=1596`，`ntrim=max(30,ceil(trim_fraction*len(b)))`：

- 0.75：`ceil(1197.0)=1197`；
- 0.60：`ceil(957.6)=958`；
- 少保留 `239` 个对应点。`trimmed_rmse_m` 比较的点集改变，且优化后的候选位姿也可改变，RMSE 降低不能证明 GT 位姿误差降低。

**Boss 2：失败分析与迁移，PASS**。帧 47 的原始旋转误差约 `+133.511°`，帧 55 约 `−132.875°`。三帧是由 B4 GT 结果事后挑选的诊断案例，不能据此把 0.60 设置成全局默认值。更严谨的改进评估必须使用预先固定的调参/验证划分，并使用**未参与模型或参数选择的独立测试集**报告泛化误差。对原来的 50 帧反复查看 GT 并择优调参后，再把相同 50 帧称为“独立测试集”会产生选择偏倚。所有失败帧都要进入覆盖率分母。

**Boss 3：参数流向与 GT-free 门禁，PASS**。`support_threshold_m` 只在最优候选已按 `trimmed_rmse_m` 选择后，计算最终距离数组中 `d <= threshold` 的均值 `support_fraction`；不会更改 `R`、`t_m` 或 `trimmed_rmse_m`。当前实现**没有** `support_count`、`support_status` 或基于该阈值的自动拒绝状态。若新建质量门禁，只能在预测阶段使用观测、模型、可见性、重投影和不依赖 GT 的候选一致性，并记录拒绝状态及分母，严禁把 GT 位姿误差用于在线挑选或通过门禁。

**评测边界**：官方 `models_info.json` 对 obj_id=1 未声明对称变换；不能把近似 180° 的原始旋转误差自动修正为低误差。GT-informed failure-case analysis 不等同于独立性能评估。

## 8. B4/B5 baseline 交叉审计（2026-10-30，PASS）

已在 Windows 本机对 **im_id=47、55、240** 分别比较 B4 原始候选和 B5 `baseline` 候选：

| im_id | `max(abs(R_B4-R_B5))` | `norm(t_B4-t_B5)` | 观测/模型元数据 | RMSE/支持比例 |
|---|---:|---:|---|---|
| 47 | `0.000e+00` | `0.000e+00 m` | 相同 | `atol=1e-12` PASS |
| 55 | `0.000e+00` | `0.000e+00 m` | 相同 | `atol=1e-12` PASS |
| 240 | `0.000e+00` | `0.000e+00 m` | 相同 | `atol=1e-12` PASS |

本机最终打印 `T1-04-B4/B5 baseline consistency: PASS`。B5 的 JSON 还包含消融实验元数据，**不要求**与 B4 的完整 JSON SHA 相同，但必须确保输入、模型、R/t 和相关诊断等价。此前 B3/B4 五帧候选的 R/t 差异和完整 JSON SHA 均为零差异 / 相同哈希。

## 9. 最终交付与待办

**实验验收**：理论、代码/参数解释、三项单变量消融、预测与实测分析、50 帧完整评测、指标复算、B3/B4 和 B4/B5 交叉复现、Boss 3/3 均 PASS；全仓 `843 passed, 106 subtests passed`。本单元成果是 **GT-pose-blind、Oracle-mask、已知模型的受控 6D Pose 基线和失败诊断**，**不是**高精度、端到端、对称性感知或物理真值认证的定位系统。

**环境**：Windows / Python 3.11.17 / NumPy 2.2.6 / SciPy 1.17.1 / Open3D 0.20 / pytest 9.1.1；实验机器和库版本属于复现条件。

**实验来源**：

- B4 固定 50 帧：`outputs/t1_04/b4_50_20261030_091523`；候选 `frozen_predictions.json`、结果 `batch_summary.json`。
- B5 三帧消融：`outputs/t1_04/b5_ablation_20261030_101526`；候选 `frozen_ablations.json`、结果 `diagnostic_summary.json`。
- 完整原始数据与模型位于被排除出 Git 的 `data/raw/bop/lm`；重要原始候选/评测 JSON 留在本机 `outputs/`，不把整个运行目录提交到 Git。
- 建议将 B4/B5 两份冻结清单与两份汇总报告**按原始字节复制**至 `experiments/artifacts/t1_04_b/` 并精确提交；这些摘要/清单只支持追溯，未包含全部候选 JSON，因此不等于完整实验归档。

**Git/GitHub 待办**：

- [x] 已在本机复制学习笔记审核草稿（本版本补充审计事实，需再次覆盖）。
- [x] B4/B5 baseline 三帧数值一致性与诊断审计 PASS。
- [x] 完整回归：843 passed、106 subtests passed。
- [ ] 复制本文最终版本到项目 `experiments/T1_04_B_pose_baseline_and_ablations.md`。
- [ ] 复制并检查关键 JSON 实验清单/摘要的 SHA-256；仅提交确定的源码、测试、笔记及实验摘要。
- [ ] `git diff --cached --check`、`git commit`、`git push`、校验 `HEAD == origin/master`。
- [ ] 验收远端 commit 后方可标记 `T1-04-B COMPLETE`。

**当前状态：IN PROGRESS；仅差 Git/GitHub 闭环。**
