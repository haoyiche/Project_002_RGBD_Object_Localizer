# T1 职业级 80 分：可复现性证据与边界审计（v1）

> Project: `haoyiche/Project_002_RGBD_Object_Localizer`
> 当前基线 Commit: `77b0d674b77d6294e56c8ae2224d1968ce23b128`
> 日期: 2026-11-10
> 范围: Level 2 / T1 3D Vision 职业级验收的**复现证据审计**。不宣称完整新机冷启动、工业物理精度、官方 BOP 排行分数或 ROS 机器人执行资格。

## 1. 复现级别与实测结果

| 级别 | 要验证什么 | 截至本报告的证据 | 判定 |
| --- | --- | --- | --- |
| A：已提交源码的独立目录测试 | 仅从 Git HEAD 取出干净源码，在已有 `ai_3d` 中执行核心几何、外参与 ADD 单测 | `git archive HEAD` 解压至 Windows 临时目录；`15 passed in 0.74s`；用户运行标记 `T1_CLEAN_SOURCE_SMOKE_PASS` | **PASS** |
| B：原环境整体回归 | 现有项目环境、依赖和数据下全部回归 | `924 passed, 106 subtests passed in 12.35s` | **PASS** |
| C：50 帧冻结位姿后验评价 | 既有原始 RGB-D/CAD/预测文件，先校验冻结哈希再读取 BOP GT，复算 ADD / ADD-S | `T108_A_FROZEN_50_PREFLIGHT_PASS`、`T108_A_ADD50_PASS`；独立审计 3/3 SHA、50/50 帧、50/50 ADD-S ≤ ADD | **PASS（原环境和既有输入）** |
| D：全新依赖环境构建 | 新机器从 `environment.yml` 安装全部依赖及运行核心测试 | 尚无在新环境实测的完整日志 | **未验证** |
| E：全新设备上从零运行 50 帧 FoundationPose | 从公开数据和外部权重重新推理、冻结、评价 | 已有 GPU 运行记录，但没有第三方/新机全链路重建记录 | **未验证** |

`15 passed` 是**已有 Conda 环境**下的源码独立目录测试，不是全新 Python 环境。`924 passed` 不证明依赖能重新安装。不要把 A/B/C 的通过写成 D/E 已通过。

## 2. 仓库内可检查的关键材料

- 核心几何：`src/geometry/pixel_to_camera.py`、`src/geometry/mask_to_pointcloud.py`、`tests/test_pixel_to_camera.py`、`tests/test_exp037_extrinsic_validation.py`。
- 数据清单：`data/manifests/t1_02_lm_50.json`（BOP LM，固定 50 帧，含身份、相机参数、RGB/Depth/Mask 输入哈希及 GT 来源声明）。
- FoundationPose 基线及协议：`experiments/T1_06_A_FoundationPose_iteration_ablation.md`；冻结的 50 帧 GT-blind 预测属于本机 `outputs/t1_06_a/fp_fixed50_v1/`，**未作为 GitHub 完整数据资产提交**。
- ADD 实现与测试：`src/evaluation/t1_08_a_add50_audit.py`、`tests/test_t1_08_a_add50_audit.py`。
- 可审阅汇总报告：`experiments/artifacts/t1_08_a/fp_add50_v1.json`，对应学习笔记 `experiments/T1_08_A_add50_learning_note.md`。
- 鲁棒性证据：`experiments/T1_07_A_T1_80point_evidence_audit_v0_3.md`，以及已提交的 `outputs/t1_07_a/` 中 E1/E2/E3 轻量报告。
- 环境快照：根目录 `environment.yml`。它是历史配置与版本线索，而非已通过新环境重建测试的保证。
- README：根目录 `README.md`。本文件补充详细的复现分级、外部资产边界与近期验收结果。

## 3. 目前已实测的命令

在原始 Windows 机器、已激活的 `ai_3d`、仓库根目录：

```powershell
# 核心 CPU 测试（不需要 FoundationPose GPU 推理或原始 50 帧冻结输出）
python -m pytest -q tests/test_pixel_to_camera.py tests/test_exp037_extrinsic_validation.py tests/test_t1_08_a_add50_audit.py

# 原环境全项目回归
python -m pytest -q

# 已存在 T1-08-A JSON 时应先只读检查，禁止用默认入口覆盖冻结报告
Get-Content experiments/artifacts/t1_08_a/fp_add50_v1.json -Raw
```

独立源码目录实测使用：从 `git archive --format=zip HEAD` 创建压缩包、在临时空目录解压、在该目录运行上述三文件核心测试。实测为 `15 passed in 0.74s`。由于用户本机临时目录是已有环境复用，记录为 `CLEAN_SOURCE_SMOKE`，不称为全新环境安装。

## 4. 如果需要自行准备完整 50 帧运行依赖

1. 检出本项目对应 Commit，使用 Python/Conda 环境配置文件作为参考；请先验证依赖安装，不要假设所有平台都能直接 `conda env create -f environment.yml` 成功。
2. 从 BOP 官方公开数据入口获取 LINEMOD 测试 RGB-D、`mask_visib`、`scene_camera.json`、`scene_gt.json`、`obj_000001.ply` 和 `models_info.json`；项目预期目录是 `data/raw/bop/lm/test/000001/` 与 `data/raw/bop/lm/models/`。原始数据不自动随 Git 源码下载。必须核对 `data/manifests/t1_02_lm_50.json` 中身份与哈希。
3. 第三方 GPU 推理依赖独立的 `NVlabs/FoundationPose` 源码，历史固定提交 `a1b694b83e633c2cb6115b9063d940a687759392`，Docker 历史镜像 ID `sha256:22837f81c0cfc1906b96daa344f8e80919c7619b816fcab487a024f1f566c215`，Refiner/Scorer 外部权重及 CUDA 编译扩展。精确权重路径、配置和边界见 `experiments/T1_06_A_FoundationPose_iteration_ablation.md`；必须遵守各原作者发布与授权条件，不能假设现有本机镜像可在另一设备直接取得。
4. 已冻结的 50 帧预测文件位于本机 `outputs/t1_06_a/fp_fixed50_v1/`，此前评价摘要位于本机 `outputs/t1_06_a/fp_fixed50_eval_v1/`。两者**不是 GitHub 源码自带的全量输入**。若想在新机器从零重算 ADD，必须先依法重建或另行取得这些 GT-blind 冻结候选，并保留源文件哈希与评估协议。仅有 GitHub 上的 ADD 汇总 JSON，不足以执行整批姿态后验重算。
5. 原始评价脚本 `python -m src.evaluation.t1_08_a_add50_audit` 将写入 `outputs/t1_08_a/fp_add50_v1.json`，已有文件时会故意拒绝覆盖。**本次验收不再执行该命令**，需要新实验时应先审计独立输出版本和入口配置。

外部来源：Open3D `https://github.com/isl-org/Open3D`；FoundationPose `https://github.com/NVlabs/FoundationPose`；BOP 数据 `https://bop.felk.cvut.cz/datasets/`。

## 5. 50 帧指标和明确的非结论

- `obj_id=1`、已知 CAD、Oracle 可见 Mask、同场景既有 50 帧、5,841 个模型顶点。
- ADD：Mean `5.053290 mm`，Median `5.083874 mm`，P95 `8.634342 mm`，Max `10.046496 mm`。
- 最近邻 ADD-S 仅作诊断：Mean `2.575186 mm`，P95 `4.212667 mm`；物体元数据未声明对称性；**不冒充官方对称感知成绩**。
- 50 帧不是独立 held-out；评价相对于 BOP 姿态标注，**不是独立物理真值**；未完成真实相机标定，不能报告工业部署精度。
- `publishable=False`：没有验证真实机器人坐标链、时间同步、碰撞和安全门禁。

## 6. T1 职业级评分与硬门槛的审计边界

依据 v2.0 攻略第 33 章，T1 100 分五维：投影/变换 20、3D 工程 20、6D Pose 20、鲁棒性/实验 25、工程复现/展示 15。此前自评 `78/100` 为待复核分数，**这份文档的存在并不自动增加分数**。

已核实的硬门槛：50 帧可追溯 RGB-D、6D Pose 数据集标注评价、坐标方向错误注入、几何测试、至少 3 次先预测后验证。GitHub 代码/笔记/汇总已成功提交；本次补充已提交源码的独立目录复现证据。

**GitHub 可复现这一硬门槛应审慎拆成：**

- 源码/核心测试从 Git 快照可复验：PASS。
- 公开数据与外部模型的获取条件、固定版本、目录、执行边界：本报告已披露，仍需审核可执行性。
- 从零重建 FoundationPose GPU 50 帧与新 Conda 环境：未验证；不得宣称通过。

当评分人接受“源码级核心复验 + 已锁定原环境的真实 50 帧证据 + 可审查外部依赖说明”为该项目 GitHub 可复现的合格范围时，可以针对工程复现维度重新审定是否达到 80 分；若要求**新机完整端到端 6D Pose 重建**，仍有明确缺口，不得把通关状态写为 `LEVEL COMPLETE`。

## 7. 本次 Exit Protocol

- **本次完成：** Git HEAD 干净源码快照 15 项核心测试通过；形成可复现分级与限制说明。
- **真正掌握：** 区分源码目录复现、原环境测试、冻结后评价、全新环境重建与 GPU 端到端复现。
- **仍不稳定：** 新 Conda 环境安装、外部 GPU/权重重建、独立 held-out、物理标定。
- **本次生成文件：** 本文档。
- **完成的实验：** `T1_CLEAN_SOURCE_SMOKE_PASS`（15 passed），此前原环境 924 passed 与 50 帧 ADD 评价继续有效。
- **当前 Level 状态：** `CONTINUE`，等待本文档入库及最终 80 分硬门槛复核。
- **Git 状态：** 本文件生成前，最近已核实的 Commit 为 `77b0d674b77d6294e56c8ae2224d1968ce23b128`；本文档尚未自动提交。
- **下一步：** 把本文档归档到 `experiments/`，给 README 增加入口并提交；随后最终评分、明确通关范围。
