# T1-03｜真实 RGB-D 可见点云几何与受控扰动诊断

> AI Lab / Project_002_RGBD_Object_Localizer / T1 职业级强化
>
> **Learning Unit：T1-03；状态：实验与最终 Boss PASS，Git/GitHub 待验收（正式 COMPLETE 以远端核查为准）。**
>
> 本记录来自学员在 Windows `ai_3d` 环境提供的终端结果，不将数据统计、可见点云质心与 BOP 的模型原点混作 6D Pose 定位精度。

## 1. 目标、依赖与产物

- **前置：**T1-01 BOP 元数据/单位契约；T1-02 LM 真实 RGB-D 读取、50 帧 SHA-256 Manifest，最终提交 `3f58e490ee8cff22dcc72c47776d9b5cd9ddad33`。
- **目标：**对 50 个可追溯真实 RGB-D 帧计算可见点云统计；在一帧上完成三次受控扰动，预测并验证敏感性；任何 Manifest/文件篡改 fail-closed。
- **产物：**`src/evaluation/t1_03_b_real_geometry_batch.py`、`src/evaluation/t1_03_c_three_ablations.py`、两个对应的测试模块，以及 `outputs/t1_03/` 下的 50 帧几何报告和首帧扰动报告。
- **验收边界：**诊断统计仅属于 `VISIBLE_CLOUD_GEOMETRY_DIAGNOSTIC` / `VISIBLE_CLOUD_PERTURBATION_SENSITIVITY_NOT_POSE_ERROR`；`physical_accuracy_claimed=False`，`pose_accuracy_claimed=False`，`publishable=False`，`position_m=None`。

## 2. 核心理论、输入契约与代码路径

真实帧来源：LM BOP'19 测试子集，`data/raw/bop/lm/test/000001`，第一帧 `im_id=15`、`gt_index=0`、`obj_id=1`；50 帧 Manifest 位于 `data/manifests/t1_02_lm_50.json`。

| 量 | 单位 / Shape | 定义 |
| --- | --- | --- |
| `depth_m` | m / `(H,W)` float | BOP 原始 uint16 深度 × 逐帧 `depth_scale` / 1000 |
| `rgb` | `(H,W,3)` uint8 | OpenCV BGR 转 RGB |
| `visible` | `(H,W)` bool | 指定 `gt_index` 的可见 Mask |
| `usable` | `(H,W)` bool | `visible & valid_depth` |
| `K` | `(3,3)` | 逐帧 `fx,fy,cx,cy`，单位为像素 |
| `points` | m / `(N,3)` | 有效 Mask 像素反投影到相机光学坐标系 |
| `centroid` | m / `(3,)` | 可见三维点的算术均值，不是模型原点 |
| `AABB extent` | m / `(3,)` | 各坐标的 `max − min` |

反投影：`X=(u−cx)Z/fx`，`Y=(v−cy)Z/fy`，`Z=depth_m[v,u]`。

已复用模块而非重写：

1. `verify_manifest()`：重新计算文件/标注 SHA-256 与完整清单，失败则拒绝。
2. `read_instance()`：真实 PNG 解码、Mask/Depth 完整性与尺度检查。
3. `mask_to_pointcloud()`：有效像素反投影，输出点、颜色、选中像素。
4. `compute_pointcloud_statistics()`：质心、坐标极值、AABB、坐标中位数。
5. `analyze_manifest_geometry()`：50 帧批量统计；全部通过后才输出完整 JSON。
6. `three_ablations()`：在内存副本上修改首帧深度或内参；不改原始 PNG、Manifest、GT。

**代码拆解提醒：**A/B/C 中 `depth_bias_m=0.010` 单位为米，`fx_multiplier=1.10` 无量纲，`dropout_stride=10` 以原始可见有效像素的行优先顺序计数；`np.flatnonzero(usable.ravel())[9::10]` 丢弃第 10、20、… 个有效像素。

## 3. T1-03-A：真实单帧最小实验

| 指标 | 实测 |
| --- | --- |
| 可见 Mask 像素 | 1,258 |
| 有效可见深度/XYZ 点数 | 1,252 |
| 无效可见深度 | 6 |
| 可见点云质心（m） | `[-0.1464050587, 0.1297794108, 0.9911421725]` |
| AABB extent（m） | `[0.0773941798, 0.0798223972, 0.098]` |
| 可见表面深度范围（m） | `[0.959, 1.057]` |
| 模型原点 GT（m） | `[-0.14492688862, 0.1241812763, 0.98935606466]` |

先验推理与运行结果对应：1,252 个有效点、6 个剔除像素、Z Extent 98 mm。模型原点与可见点云质心不是同一物理点，不能直接相减称为定位误差。

## 4. T1-03-B：50 帧真实批量诊断

- 命令 `python -m src.evaluation.t1_03_b_real_geometry_batch --scene-dir data/raw/bop/lm/test/000001 --manifest data/manifests/t1_02_lm_50.json --output outputs/t1_03/lm_50_geometry.json`。
- 真实帧数：**50**；有效可见深度比例的**逐帧算术平均值**：`0.9947785434300768`，约 **99.478%**。这不是按像素数加权的全局占比，不代表 6D Pose 准确率。
- 首帧质心与 AABB 和 T1-03-A 完全对应。
- `publishable=False`、`position_m=None`；报告是诊断结果，不提供机器人控制坐标。
- 6 项专项测试通过；前次全仓回归 **770 passed，106 subtests passed**。

## 5. T1-03-C：三次主动修改，预测与实测

固定首帧、同一可见 Mask 与 GT；仅修改当前内存中的输入。基线点数 **1252**，质心（m）`[-0.14640505867185347, 0.12977941077501276, 0.9911421725239609]`，AABB extent（m）`[0.07739417978048656, 0.07982239715530663, 0.09799999999999986]`。

### A｜有效深度整体增加 10 mm

**预测与推导：**`Z'_i=Z_i+0.010`，因此 `mean(Z')=mean(Z)+0.010`，且 `max(Z')−min(Z')` 不变。`X'_i=X_i+(u_i−cx)*0.010/fx`，`Y'` 同理，故 X/Y 质心与横向 AABB 一般改变，不能看作三维刚体平移。

**实测：**1252 点；质心 `[-0.1478825380092225, 0.13108739480518636, 1.00114217252396]` m；Extent `[0.07811044783524579, 0.08053721782693712, 0.09799999999999986]` m。相对基线质心变化：**X −1.477479 mm，Y +1.307984 mm，Z +10 mm**；Z Extent 不变。预测匹配。

### B｜只将 `fx` 乘 1.10

**预测与推导：**`X'_i=X_i/1.1`，X 质心与 X Extent 同除以 1.1；Y/Z 与各自的 Extent 不变。

**实测：**1252 点；质心 `[-0.1330955078835028, 0.12977941077501276, 0.9911421725239609]` m；Extent `[0.07035834525498777, 0.07982239715530663, 0.09799999999999986]` m。预测匹配。由于基线 X 为负值，X 数值向零靠近，不应简单描述为“坐标减少 10%”。

### C｜每第 10 个有效像素深度置零

**预测与推导：**`floor(1252/10)=125` 个剔除，余 `1127` 点。子集的每轴最小值不会减小、最大值不会增大，因此 **AABB 各轴 Extent 只会减小或不变**；均值一般变化。

**实测：**删去 125 点，余 **1127** 点；质心 `[-0.1462824156283396, 0.12975533287579058, 0.9911756876663708]` m；Extent **三个轴均不变**。相对基线质心变化：X +0.122643 mm，Y −0.024078 mm，Z +0.033515 mm。预测匹配。

**限制：**这里是确定性行优先采样，不是统计上独立的随机失效、不代表真实传感器噪声；只在单帧上开展扰动，也不能称为跨场景鲁棒性测量。

## 6. 自动化测试、失败边界和运行记录

- T1-03-C 专项 `6 passed`，覆盖 A/B/C 数值不变量、非法扰动参数、被篡改 Manifest 在扰动前被拒绝、CLI 诊断输出安全边界。
- 含 T1-03-A/B/C 的全仓回归：**776 passed，106 subtests passed（6.83 s）**。
- `git diff --check` 没有输出错误；B/C 脚本及测试、两个 JSON 报告目前未提交。
- 保留 `.gitignore` 排除的原始真实 PNG / ZIP；六个既有无关 `ScreenCamera/ScreenCapture` 文件不能误提交。

复现命令：

```powershell
python -m src.evaluation.t1_03_b_real_geometry_batch `
  --scene-dir ".\data\raw\bop\lm\test\000001" `
  --manifest ".\data\manifests\t1_02_lm_50.json" `
  --output ".\outputs\t1_03\lm_50_geometry.json"

python -m src.evaluation.t1_03_c_three_ablations `
  --scene-dir ".\data\raw\bop\lm\test\000001" `
  --manifest ".\data\manifests\t1_02_lm_50.json" `
  --output ".\outputs\t1_03\lm_first_frame_ablations.json"

python -m pytest tests/test_t1_03_b_real_geometry_batch.py tests/test_t1_03_c_three_ablations.py -v
python -m pytest -q
```

## 7. 最终 Boss A/B/C：知识迁移验收

三道题已由学员作答并经过概念复核，**Boss A/B/C：PASS**。它们是理论推导与工程方案检查，不应冒充已经新增的自动化测试或实际部署的功能。

### Boss A｜逐帧平均比例与像素加权比例

令第 i 帧 `v_i` 为可见 Mask 像素数，`e_i` 为有效可见深度像素数，`r_i=e_i/v_i`（且 `v_i>0`）。

- **宏平均 / 每帧等权**：`mean(r_i) = sum(r_i) / 50`；已实测 `0.9947785434300768`。
- **微平均 / 像素数加权**：`sum(e_i) / sum(v_i) = sum(v_i * r_i) / sum(v_i)`；本关**未实测该数值，不应捏造**。

两者使用不同的权重。**校正一个逻辑限定：**所有 `v_i` 相等是两种结果相等的**充分条件**，并非必要条件。即使 `v_i` 不等，只要所有 `r_i` 相同，或者 `v_i` 与 `r_i` 的协方差为零，两者也相等。因此不能说“只有所有 Mask 大小相等时才相等”。

示例：`(v_1,e_1)=(10000,9990)` 和 `(v_2,e_2)=(100,90)`，宏平均 `94.95%`，微平均 `10080/10100≈99.802%`。指标口径必须写进报告，不能用 99.478% 冒充逐像素总体比例。

### Boss B｜每个有效深度乘以 1.01

对固定像素、Mask、K，`Z'_i=1.01 Z_i`，反投影有 `P'_i=1.01 P_i`，因此：

- 三轴质心 `centroid'=1.01*centroid`。
- 三轴 AABB extent `extent'=1.01*extent`（正比例保持极值顺序）。
- 有效点数不变，前提为乘法后数据仍有限且为正；对本关的真实正常深度满足此条件。

这是**相机光心为中心的 1.01 倍缩放**，不同于 A 的加性深度偏置，也不同于 B 的仅 `fx` 缩放。这道题目前是新条件下的解析预测，不宣称额外进行了物理传感器实验。

### Boss C｜失败运行后仍存在旧 JSON

失败后固定路径报告仍然存在，**不能**据此认定本次运行成功。文件可能来自上一次成功执行；其存在与本次输入哈希/代码版本/验收状态无自动绑定。

工程加固方案（**建议，尚未宣称实现**）：

1. 每次运行用唯一 `run_id` 输出目录；不复用“本次成功”固定路径。
2. 报告记录 Manifest SHA-256、实际输入哈希或可复核引用、代码 commit、状态、时间与环境信息；字段只在真实采集后填写，不伪造时间戳。
3. 下游读取前复核来源和状态，失败则拒绝。
4. 在计算和复核成功后原子写入最终报告，失败时只写明确的失败记录或不发布成功报告。

这是**失效模式分析与设计迁移**，不是声称现有 T1-03 报告已具备完整 run_id/原子写入机制。

## 8. 结论、限制与 Git 清单

**已经获得：**真实 50 帧诊断，单帧三次有解释的受控扰动，数值预测与实测一致，输入完整性/安全输出验证和全仓回归。

**没有获得：**6D Pose 实际估计及对应物理精度、传感器真实噪声定标、跨物体/跨场景泛化、干扰情况下真实 GT 精度评估。

- [x] 最小理论与关键公式、单位、Shape。
- [x] T1-03-A 单帧真实几何。
- [x] T1-03-B 50 帧几何诊断。
- [x] T1-03-C 三次可控修改及解释。
- [x] 专项测试与 776 项全仓回归。
- [x] 安全边界：诊断不可发布，不生成机器人控制位置。
- [x] 最终 Boss A/B/C 已答并复核（宏/微比例相等条件已精确修正）。
- [x] 最终学习笔记已整理与内容复核，待复制到本地 Git 仓库。
- [ ] `git diff --cached --check`、精确暂存 7 个文件、Commit、GitHub Push、HEAD 与远端 SHA 对齐。

**下一步：**完成 Git/GitHub 闭环后进入 T1-04 6D Pose baseline 的模型到相机位姿评测契约；严格区分实际估计位姿与 GT 标注，不使用点云可见质心冒充模型原点。
