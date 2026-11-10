# Level 2 总证据导航｜Project_002_RGBD_Object_Localizer

> 归档日期：2026-11-10
> 依据：`AI_Lab_SOP.md` 与《个人AI工程师打怪升级攻略 v2.0》第 33 章。
> 最终学习验收：Level 2 / T1 **80/100（职业级、限定范围）**。
> 证据基线：GitHub `master` 上提交 `2a62a2cfcb3476b01c973390a476c92967a1fac7` 及后续 Final Archive 提交。
> 本文是**索引**，不是对第三方新机器全流程复现、工业精度或官方 BOP 分数的保证。

## 1. 首选阅读顺序

1. [Level 2 完整学习与工程成果报告](LEVEL2_FINAL_REPORT.md)：从入门到 T1 职业级 80 分的全程解释、结果和局限。
2. [根目录 README](../README.md)：项目工程概况、使用范围和历史进展。请以最上方“最新状态”优先于旧章节中的 `CONTINUE`。
3. [T1 可复现性审计](T1_80_reproducibility_evidence_v1.md)：复现层级 A/B/C 已通过、D/E 未验证；外部 BOP/FP 依赖边界。
4. [T1-08-A ADD 学习笔记](T1_08_A_add50_learning_note.md)：对称性预测、计算、50 帧评价及纠错。
5. [T1-07-A 职业级证据审计](T1_07_A_T1_80point_evidence_audit_v0_3.md)：G0-G4、E1-E4 的证据和缺口。

## 2. 基础几何与 RGB-D 定位（OS-01）

| 主题 | 源码入口 | 笔记与验收依据 | 能证明什么 |
|---|---|---|---|
| 像素反投影与单位 | [`pixel_to_camera.py`](../src/geometry/pixel_to_camera.py) | [`test_pixel_to_camera.py`](../tests/test_pixel_to_camera.py) | K、Depth 到相机 XYZ 的数学和契约 |
| Mask→点云 | [`mask_to_pointcloud.py`](../src/geometry/mask_to_pointcloud.py) | [`OS01_B_rgbd_pipeline.md`](OS01_B_rgbd_pipeline.md) | Mask 与有效深度交集的点云生成 |
| Radius 去噪、统计、聚类 | [`pointcloud_filter.py`](../src/geometry/pointcloud_filter.py)、[`dbscan.py`](../src/geometry/dbscan.py)、[`cluster_localization.py`](../src/geometry/cluster_localization.py) | [`OS01_B_rgbd_pipeline.md`](OS01_B_rgbd_pipeline.md) | 邻域定义、几何门禁、Fail-Closed |
| 单帧工程流水线 | [`os01_single_sample.py`](../src/pipeline/os01_single_sample.py) | [`OS01_B_rgbd_pipeline.md`](OS01_B_rgbd_pipeline.md) | 诊断级 3D 位置，不是机器人发布 |
| 20 个合成样本 | [`os01_c2_batch_evaluate.py`](../src/evaluation/os01_c2_batch_evaluate.py) | [`OS01_C_multisample_evaluation.md`](OS01_C_multisample_evaluation.md) | 20 个输入，**仅 6 个**合格误差统计样本 |
| Open3D、RANSAC、坐标变换 | [`os01_d3_coordinate_chain.py`](../src/pipeline/os01_d3_coordinate_chain.py) | [`OS01_D_open3d_coordinate_chain.md`](OS01_D_open3d_coordinate_chain.md) | 合成坐标链、失败阻断和同帧检查 |
| 最终 Demo 与故障审计 | [`os01_final1_demo_report.py`](../src/pipeline/os01_final1_demo_report.py)、[`os01_final2_failure_audit.py`](../src/pipeline/os01_final2_failure_audit.py) | [`Level2_Final_Boss_and_delivery.md`](Level2_Final_Boss_and_delivery.md) | 同帧诊断与 7 场景故障审计 |

**记住分母：** OS-01-C 的 `2.2361 mm` Mean 与 `2.4820 mm` P95 是 **6 个可评估合成样本**相对于合成名义参考的位置差异；不能称为 20 个样本的真实物理准确率。

## 3. 真实 RGB-D 数据、6D Pose 与实验协议（T1）

| 主题 | 核心代码、文件或记录 | 结论与边界 |
|---|---|---|
| BOP LINEMOD 50 帧身份 | [`t1_02_lm_50.json`](../data/manifests/t1_02_lm_50.json) | 单场景 `000001`、单物体 `obj_id=1`，输入身份和哈希可追溯 |
| 位姿、外参、单位 | [`t1_04_a_pose_metrics.py`](../src/evaluation/t1_04_a_pose_metrics.py)、[`EXP037_extrinsic_validation.md`](EXP037_extrinsic_validation.md) | 明确 `T_camera←model`、SO(3)、BOP GT 平移 mm→m；方向故障注入 |
| PCA+ICP 与 FoundationPose 基线 | [`T1_06_A_FoundationPose_iteration_ablation.md`](T1_06_A_FoundationPose_iteration_ablation.md) | 相同 50 帧严格配对，FoundationPose 平移均值 **4.8952 mm**、原始旋转均值 **2.1491°** |
| FP GT 盲预测及冻结 | [`t1_06_a_fp_fixed_batch.py`](../src/evaluation/t1_06_a_fp_fixed_batch.py)、[`t1_06_a_fp_posthoc_eval.py`](../src/evaluation/t1_06_a_fp_posthoc_eval.py) | 预测阶段与 GT 隔离；评价前验证 SHA 与帧身份 |
| FP 50 帧评价对照 | [`t1_06_a_fp_batch_posthoc.py`](../src/evaluation/t1_06_a_fp_batch_posthoc.py) | 对照 PCA+ICP：平移均值 **12.2480 mm**、原始旋转均值 **97.4544°** |
| E1 深度噪声、E2 内参、E3 Mask/Depth | [`T1_07_A_T1_80point_evidence_audit_v0_3.md`](T1_07_A_T1_80point_evidence_audit_v0_3.md) | 合成/受控配对扰动、Mean/Median/P95、预测与异常归因 |
| T1-08-A ADD / ADD-S | [`t1_08_a_add50_audit.py`](../src/evaluation/t1_08_a_add50_audit.py)、[`test_t1_08_a_add50_audit.py`](../tests/test_t1_08_a_add50_audit.py) | 5,841 CAD 顶点、50 帧，ADD Mean **5.0533 mm**、P95 **8.6343 mm** |
| 已提交的 ADD 结果 | [`fp_add50_v1.json`](artifacts/t1_08_a/fp_add50_v1.json) | 轻量、可独立阅读的 50 帧逐帧结果和输入哈希 |
| 复现分级与外部资产 | [`T1_80_reproducibility_evidence_v1.md`](T1_80_reproducibility_evidence_v1.md) | 源码独立目录 15 passed；既有环境全仓 924 passed/106 subtests |

**评价术语：** ADD-S 是最近邻几何诊断；BOP LM `obj_id=1` 的 `models_info` 未声明对称性，不能声称“模型识别对称性成功”。Oracle Mask 属于 GT 提供的可见实例区域，不等于自动分割。所有该批次指标都是 BOP 标注相对诊断，**不是 held-out、独立物理精度或官方排名**。

## 4. 最终验收与 GitHub 里程碑

| 关卡 | Commit | 说明 |
|---|---|---|
| OS-01-A | `e35ff69` | RGB-D 数据契约 |
| OS-01-B | `bc7e3ff` | 单帧定位流水线 |
| OS-01-C | `41a69c5` | 20 个合成样本评价 |
| OS-01-D | `78b57fd` | Open3D 与坐标链 |
| T1-06-A | `ffb2f90` | FoundationPose 50 帧实验工程闭环 |
| T1-07-A | `454dae1` | E1/E2/E3 与职业级证据审计 |
| T1-08-A | `77b0d67` | ADD / ADD-S 指标、50 帧报告与学习笔记 |
| T1 复现审计 | `2a62a2c` | 干净源码测试与复现边界文档 |
| Level 2 Final Archive | **本次归档提交完成后由 Git 历史确定 SHA** | 最新 README、完整总结与证据索引 |

- 已验收：T1 综合 Boss 3/3；原环境 `924 passed, 106 subtests passed`；`git archive` 后在原 `ai_3d` 中核心 `15 passed`；导师职业级 **80/100（限定范围）**。
- 未认证：新机器从零安装依赖/重建 GPU 50 帧、独立场景泛化、真实物理标定及安全机器人部署。

## 5. 本地资产：**绝不因为不在 Git 中就删除**

| 资产位置（原仓库相对路径） | 策略 | 原因 |
|---|---|---|
| `outputs/t1_06_a/fp_fixed50_v1/` | **本地关键证据，保留并建议异盘备份** | 50 帧冻结预测/清单，后验审计需要原始候选 |
| `outputs/t1_06_a/fp_fixed50_eval_v1/` | **本地关键证据，保留** | 原始后验统计与逐帧评价 |
| `outputs/t1_04/`、`outputs/t1_05/` | **保留，后续按版本盘点** | ICP 等对照输出与历史实验 |
| `outputs/t1_08_a/fp_add50_v1.json` | **保留** | 已有字节一致的 Git 跟踪副本，但原始本地路径仍供程序读取 |
| `data/raw/bop/lm/` | **外部数据，不直接纳入 Git** | 数据大、许可和官方获取路径另有说明 |
| `D:\AI_Lab\_external\FoundationPose`、Docker 镜像、权重 | **本机独立资产，保留** | 重新 GPU 推理所需，未随仓库自动发布 |
| `ScreenCamera_*.json`、`ScreenCapture_*.png` | **待人工挑选，禁止自动删除** | 可能含过程证据或私人画面 |
| `T1_08_A_README.txt` 等解压说明 | **可人工归类，不自动删除** | 非核心源码，但清理须独立判断 |

归档脚本只生成本机资产清单，不移动或删除任何原始文件；该清单默认放在 `outputs/level2_archive/` 下，不参与 Git 提交。

## 6. 如何复核，不重跑 GPU

在已有 `ai_3d` 环境和仓库根目录下，执行：

```powershell
python -m pytest -q tests/test_pixel_to_camera.py tests/test_exp037_extrinsic_validation.py tests/test_t1_08_a_add50_audit.py
python -m pytest -q
Get-Content experiments/artifacts/t1_08_a/fp_add50_v1.json -Raw
```

全仓回归和 50 帧原始运行已有可追溯日志，不要求归档时重复执行；这里只是保留入口。

## 7. 最终归档状态说明

- **学习**：COMPLETE（T1 导师职业级 80/100，适用于明确声明的基础验收范围）。
- **工程**：已提交代码、测试、轻量报告；最终报告/索引在本次归档提交后进入 GitHub。
- **原始数据**：关键冻结预测仍为本地资产。未验证新机端到端重建。
- **剩余限制**：真实标定、独立 held-out、E1-A/E2-3 个别生成脚本归档不足、OBB 完整示例缺口；属后续提升事项，不应悄悄写成已完成。
