# T1-02｜真实 BOP LM RGB-D 数据读取与 50 帧可追溯审计

> 项目：`Project_002_RGBD_Object_Localizer`；阶段：T1 职业级强化；Learning Unit：T1-02。
> **状态：T1-02 COMPLETE（核心实验、理解验收、GitHub 主提交均已完成）**。本文件记录本机实测证据，不将数据接入成功误写为 6D Pose 估计或物理精度认证。

## 1. 目标、上下游与验收

- **上游：**已完成 T1-01 BOP 元数据、深度单位、实例索引与模型到相机姿态契约；Level 2 基础 Boss 已完成。
- **本关目标：**解码 BOP LM 的真实 RGB / uint16 Depth / 可见 Mask PNG；核查逐帧内参、深度尺度、实例 GT 对应；生成和独立复核至少 50 个不同 RGB-D 帧的 SHA-256 Manifest；错误必须拒绝。
- **下游：**T1-03 真实数据几何/鲁棒性评价与 T1-04 6D Pose 基线；本关**不**实现姿态推理或物理精度测量。
- **产物：**`src/evaluation/t1_02_bop_rgbd.py`、`tests/test_t1_02_bop_rgbd.py`、`data/manifests/t1_02_lm_50.json`、本学习笔记。单帧 smoke Manifest 位于 `outputs/t1_02/lm_smoke_manifest.json`，可由命令重新生成。

## 2. 数据来源和明确边界

- 数据集：BOP LINEMOD（`lm`）测试数据，场景 `000001`，BOP'19 测试子集；官方数据说明：<https://bop.felk.cvut.cz/datasets/>。代码记录的许可：`CC BY 4.0`。
- 本机解压目录：`data/raw/bop/lm/test/000001`。数据根目录由 `.gitignore` 排除，**不上传真实 PNG 或完整 ZIP 到项目 Git 仓库**。
- 压缩包内容 SHA-256（本机实测，2026-10-29）：
  - `lm_base.zip`：`A1D793837D4DE0DBD33F04E8B04CE4403C909248C527B2D7D61EF5EAC3EF2C39`
  - `lm_test_bop19.zip`：`F1BA431196065FC57A412CCA22122906AFEC94281515776C70DE282E51C0927A`
- 50 帧 Manifest SHA-256（本机实测）：`B5BA15667C803B2F235D0DE3F40A7EDFE8B584D5FFAEF8037CD45392C85B1877`。
- **证明边界：**这些哈希可用于内容复核，不构成压缩包发布者认证；BOP 姿态标注不等于独立物理测量；`physical_accuracy_claimed=false`、`pose_accuracy_claimed=false`。
- **数据划分：**当前使用 `test` 子集进行只读数据接入测试；后续不能在同一测试子集上反复调姿态模型参数后仍称为独立测试。

## 3. 最小理论与关键输入契约

| 输入 | 必须满足 | 作用 / 常见误用 |
| --- | --- | --- |
| RGB PNG | `(H,W,3)`、`uint8`，OpenCV BGR→RGB | 不得直接把 BGR 当 RGB |
| Depth PNG | `(H,W)`、`uint16`，使用 `cv2.IMREAD_UNCHANGED` | 默认 `imread` 可能转换为 8 bit，导致深度数据损失 |
| `mask_visib` PNG | `(H,W)`、`uint8`、像素值仅 0/255 | 使用 `im_id_gt_index.png`，不是 `obj_id` |
| `scene_camera.json` | 按 `im_id` 读取内参 `K` 与 `depth_scale` | `depth_scale` 以 mm / 原始值计 |
| `scene_gt.json` | 同 `im_id`、正确 `gt_index` | 模型→相机 GT；模型原点不是 Mask 点云质心 |

单位与反投影公式：

\[
Z_m=\frac{D_{raw}\cdot depth\_scale}{1000},\qquad
P_c=Z_m K^{-1}[u,v,1]^T.
\]

可用目标像素集合为 `mask_visib == 255` **与** `valid_depth` 的交集；不能把可见 Mask 像素总数直接当成有效三维点数。

姿态标签中的 `cam_t_m2c` 以毫米表达模型原点在相机坐标系的位置，`cam_R_m2c` 定义模型系→相机系旋转。不能把这个原点的平移与某个像素/可见点云质心直接当作同一个物理点相减。

## 4. 最小真实实验与实测

环境：Windows / Python 3.11.17；`opencv-python-headless==4.11.0.86` (`cv2=4.11.0`)；`NumPy=2.2.6`；`Open3D=0.20.0`；`pip check` 回报 `No broken requirements found`。这是一组**本机已观察版本**，不是跨平台锁定文件。

第一次在安装 OpenCV 前运行出现 `ModuleNotFoundError: No module named 'cv2'`，pytest 于收集阶段中止；安装 headless OpenCV 后恢复，属于**依赖故障**，不是 PNG 格式或几何算法失败。

单帧命令：

```powershell
python -m src.evaluation.t1_02_bop_rgbd `
  --scene-dir ".\data\raw\bop\lm\test\000001" `
  --scene-id 1 --count 1 `
  --output ".\outputs\t1_02\lm_smoke_manifest.json"
```

实测首帧：

```text
sample_id=lm/test/000001/000015/000000
obj_id=1
valid_visible_depth_pixels=1252
depth_scale_mm_per_unit=1.0
K_row_major=[572.4114,0,325.2611,0,573.57043,242.04899,0,0,1]
physical_accuracy_claimed=false
pose_accuracy_claimed=false
```

50 帧命令：

```powershell
python -m src.evaluation.t1_02_bop_rgbd `
  --scene-dir ".\data\raw\bop\lm\test\000001" `
  --scene-id 1 --count 50 `
  --output ".\data\manifests\t1_02_lm_50.json"
```

实测 `distinct_rgbd_frames=50`，Manifest 生成 PASS。独立 Python 审计确认 50 个不同 `(scene_id, im_id)`、50 个不同 `(rgb_sha256,depth_sha256)` 组合、每个样本有效目标深度点数大于零，`verify_manifest(manifest, scene_dir)` 返回 `True`。

**范围说明：**50 是一个场景内的 **50 个不同采集帧**，不是 50 个物体类别、50 个独立场景、50 个成功姿态估计或物理准确率结论。

## 5. 主动故障注入：先预测后验证

**变更：**只在内存拷贝中，将 Manifest 第一帧 `files.depth.sha256` 改为 64 个 `0`；原始 Manifest 和磁盘 PNG 保持不变。

**先预测：**重算文件哈希与 Manifest 声明不符，应 Fail-closed，不能继续评测；错误语义属于哈希/内容不一致。

**实际：**正常 Manifest `verify_manifest` 返回 `True`；篡改拷贝抛出 `ValueError("MANIFEST_CONTENT_MISMATCH")`。当前实现**重建整个 Manifest 作比较**，错误码不细分为 `DEPTH_SHA256_MISMATCH`，也不能仅凭这个错误断言哪一方恶意修改。

其他 13 项自动化测试覆盖：RGB BGR→RGB、16-bit Depth 解码、Manifest 唯一性/复核、PNG 内容更改、8-bit 深度拒绝、RGB/Depth 尺寸不匹配、Mask 与深度无交集、损坏 PNG、帧数不足、camera/GT 图像索引不一致、非二值 Mask、不同帧 ID 但重复 RGB-D 内容、下载子集的可用文件选择、PNG 配对缺失拒绝。

测试实测：`13 passed`；全仓：**`764 passed, 106 subtests passed`**；最近一轮完整回归耗时 **6.59 秒**。`git diff --check` 未报告差错。测试中的 PNG 是真实编码但图像内容为合成夹具，不计入 50 个真实帧。

## 6. 复现与 Git 交付

1. 按 BOP LM 数据源取得并解压 `lm_base.zip`、`lm_test_bop19.zip`，核对压缩包 SHA-256 和 `data/raw/bop/lm/test/000001/` 内相机、GT、RGB/Depth/Mask 是否齐备。
2. 使用 Python 3.11 环境及 OpenCV 4.11、NumPy 2.2.6、Open3D 0.20 已验证的组合；安装依赖前后检查环境冲突（`python -m pip check`）。
3. 使用上面两条命令生成 Smoke 与 50 帧 Manifest。
4. 使用 `verify_manifest(m, Path('data/raw/bop/lm/test/000001'))` 独立复核保存后的 Manifest；故障注入只修改**内存副本**。
5. 运行 `python -m pytest tests/test_t1_02_bop_rgbd.py -v` 及 `python -m pytest -q`。
6. 主提交已包含解析脚本、测试、学习笔记、50 帧 Manifest 和单帧 Smoke Manifest 共 5 个文件；不提交原始 PNG 与 ZIP。主提交：`122405a5629dfc782d87da199a20be320b3affd3`，推送成功，用户本机 `HEAD == origin/master` 且已独立核对远端。

## 7. 最终 Boss：数据覆盖与缺失文件策略

**Boss A（PASS）｜50 帧与泛化边界：** `lm/test/000001` 的 50 份 `(scene_id, im_id)` 与 RGB/Depth 内容组合互不相同，满足本关的**帧级可追溯性**验收；但这些样本来自同一场景、同一目标 `obj_id=1`，不具备跨物体、跨场景及系统性光照/遮挡条件的验证证据。不能凭“50 帧”宣称跨条件泛化或任何 6D Pose 准确率。单一场景不意味着每帧条件完全相同，仅意味着本关未做这些条件的分组泛化评估。

**Boss B（PASS）｜Depth 缺失必须 Fail-closed：** 若某一 RGB 帧对应的 Depth PNG 被删除，`_frame_ids()` 发现 RGB 与 Depth 图像 ID 集合不一致，抛出 `ValueError("RGB_DEPTH_IMAGE_IDS_MISMATCH")`。不能静默跳过或自动选择其他帧补足 50 个；否则样本身份与覆盖、Manifest 哈希及下游统计分母都可能发生未声明变化。修复数据后应重新生成并验证清单。

**主提交与追溯证据：** Git commit `122405a5629dfc782d87da199a20be320b3affd3`，远端 `master` 已对齐；主提交前存在本笔记第 3 行尾随空格的 `git diff --cached --check` 报告。本次文档收尾删除该空格，不改动真实数据和历史 Manifest 内容。

## 8. 通关能力检查与后续限制

- [x] 可解释 BOP RGB/Depth/Mask 解码及位深、单位、Shape 契约。
- [x] 能区分 `gt_index`、`obj_id` 和深度像素/模型原点。
- [x] 真实 LM 一帧解码及 50 不同 RGB-D 帧索引通过。
- [x] 能先预测再验证哈希不一致故障，明确 Fail-closed。
- [x] 自动化专项测试与全仓回归通过。
- [x] 数据来源、压缩包/Manifest SHA-256 和精度声明边界已记录。
- [x] Boss A/B：能够解释帧数量与泛化的区别，以及 RGB/Depth 配对缺失时的明确拒绝。
- [x] Git/GitHub 主提交 `122405a`、Push、`HEAD == origin/master` 已验证；此文档作为最终状态与空白修正单独提交。

**仍未完成：**数据发布者真实性认证、跨场景/跨物体覆盖、完全冻结的跨平台环境文件、独立物理 GT、任何 6D Pose 模型推理及准确性指标。

**下一关：T1-03 真实 RGB-D 几何诊断和鲁棒性基线**。复用现有相机反投影代码，先明确参考量与目标实例语义，不要把 GT 模型原点当作可见点云质心的真值。
