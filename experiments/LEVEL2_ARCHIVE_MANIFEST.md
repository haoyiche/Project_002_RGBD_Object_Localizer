# Level 2 Final Archive｜归档清单与验收声明

归档日期：2026-11-10
项目：`Project_002_RGBD_Object_Localizer`
归档前经用户与 GitHub 双重核验的基线：`2a62a2cfcb3476b01c973390a476c92967a1fac7`。
**归档最终 Commit SHA：以完成本次 Git Push 后 `git ls-remote` 的实际结果为准，不预填未来的 SHA。**

## 1. 本次新增或修改内容

| 路径 | 操作 | 用途 |
|---|---|---|
| `README.md` | 在开头加入当前状态入口，并把历史 Current Status 标明为旧快照 | 避免 2026-10 的历史 `CONTINUE` 与当前已毕业状态混淆 |
| `experiments/LEVEL2_FINAL_REPORT.md` | 新增 | Level 2 全阶段 20 章节正式报告，来自用户已收到的 2026-11-10 完整总结 |
| `experiments/LEVEL2_EVIDENCE_INDEX.md` | 新增 | OS-01/T1 模块、源码、笔记、50 帧证据、外部资产的单点导航 |
| `experiments/LEVEL2_ARCHIVE_MANIFEST.md` | 新增 | 归档范围、排除项、验收状态和后续长期维护规则 |

**不修改：**旧实验代码、冻结输入、数据集标注、历史 JSON/CSV、旧笔记的历史状态。已完成的 T1-08-A 与 T1 可复现性审计证据仍以原 Commit 为准。

## 2. 验收依据与分母

- 导师职业级：T1 **80/100（限定范围）**；并不代表工业准确度认证。
- RGB-D：BOP LM 固定 50 帧、单场景单物体；Oracle Mask。
- 基础 OS-01-C：20 个合成样本，其中只有 **6 个**进入位置差异统计。
- FoundationPose：50 帧平移 Mean `4.8952 mm`、原始旋转 Mean `2.1491°`，相对于当前 BOP GT。
- PCA+ICP 对照：同 50 帧平移 Mean `12.2480 mm`、原始旋转 Mean `97.4544°`。
- ADD：50 帧、5,841 CAD 顶点，Mean `5.0533 mm`、P95 `8.6343 mm`。
- 旧环境全仓回归：`924 passed, 106 subtests passed`。
- Git HEAD 独立源码目录、既有 Conda 环境：`15 passed`。

## 3. 一票否决与边界

第 33 章职业级六门槛中，GT 标签评价、先预测、坐标错误注入、几何单测、GitHub 资产与 50 帧证据均已覆盖。GitHub 复现 PASS 的**限定口径**是已提交源码的核心测试可复验、当前机器历史实验可审计以及外部资产要求透明披露。

未验证的部分不得写成通过：全新机器 GPU 重建、全新 Conda 环境从零安装、独立 held-out、真实相机外参/深度物理标定、机器人安全执行、端到端自动分割及官方 BOP 成绩。

## 4. 本地资产保留策略

**严禁自动清理** `outputs/t1_04/`、`outputs/t1_05/`、`outputs/t1_06_a/`、`outputs/t1_08_a/`、`data/raw/bop/`、外部 FoundationPose 源码/权重/Docker、用户的 ScreenCapture/ScreenCamera 等文件。

归档准备脚本会产生只读资产盘点，保存于：

`outputs/level2_archive/local_assets_inventory.json`

该文件**仅记录本机资产存在性、目录层级和未跟踪文件计数**，不采集/上传截图内容、密钥和大型数据，不自动加入 Git。若要备份冻结文件，请单独复制到受控备份位置并验证 SHA；本次归档不进行此操作。

## 5. Git 与 SOP 交付协议

1. 归档前确认 `master` 基线 `2a62a2c` 与远端同步，已跟踪文件无本地改动，暂存区为空。
2. 将报告、导航和清单复制到 `experiments/`，仅在 README 顶部加当前入口。
3. 验证报告哈希、索引关键路径与四个待提交文件，运行 `git diff --cached --check`。
4. **只提交四个指定路径**；不使用 `git add .`；不触碰用户历史 `outputs/`。
5. Commit/Push 后核实 `HEAD == ls-remote origin/master`，把 Commit 记录在 AI Lab 进度中。
6. 此时 **Level 2：学习、实验、验收、工程文档与 GitHub 归档 COMPLETE**；尚未验证的物理/泛化/新机复现仍作为下阶段目标。

## 6. 防止误解历史 README

仓库早期 README 中的 `Level 2: CONTINUE`、`Git pending` 等文字是当时历史快照，不可当作当前最新状态。完整日期、证据索引和这次归档是最新判定的优先入口；不要通过重写旧实验日期或报告篡改历史。
