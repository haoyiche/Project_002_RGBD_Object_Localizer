# T1-04-A｜6D Pose 指标、单位与坐标方向契约（最终验收笔记）

> 项目：`Project_002_RGBD_Object_Localizer`；当前 Learning Unit：T1-04-A。
> 当前状态：**理论与预测 PASS；独立 A/B/C PASS；专项 19 项 PASS；全仓 795 passed / 106 subtests PASS；最终 Boss PASS；Git/GitHub 待完成**。
> 本关只验证**位姿指标契约**，没有真实的位姿估计算法、对称感知评测或独立物理准确度结果。

## 1. 目标、前置知识和交付

- 上游：T1-01 BOP 的 `cam_R_m2c`/`cam_t_m2c`（毫米）、T1-02 真实 LM 数据与 Manifest、T1-03 50 帧真实几何诊断。
- 目标：对**同一方向的 model → camera** 估计位姿与 GT 位姿，输出平移误差（mm）和原始旋转角误差（deg）；拒绝不合规输入；显式注明不考虑对称性。
- 模块：`src/evaluation/t1_04_a_pose_metrics.py`；测试：`tests/test_t1_04_a_pose_metrics.py`。
- 最终验收：能解释、预测、计算、主动修改、排查方向和单位错误，完成笔记、回归测试和 Git/GitHub。

## 2. 最小理论、单位与 Shape

| 数据 | 形状 / 单位 | 语义 |
|---|---|---|
| `PoseM2C.R` | `(3,3)`；无单位 | 从模型系到相机系的旋转，属于 SO(3) |
| `PoseM2C.t_m` | `(3,)`；米 | 模型原点在相机系的平移 |
| BOP `cam_t_m2c` | `(3,)`；毫米 | 从原始 BOP 载入时 **只除一次 1000** |
| `translation_error_mm` | 标量；毫米 | `1000 * norm(t_est_m - t_gt_m)` |
| `rotation_error_deg_raw` | 标量；角度 | `R_est @ R_gt.T` 的原始测地旋转角，**非对称感知** |

刚体变换：`P_camera = R_cam_from_model @ P_model + t_cam_from_model_m`。

旋转指标可由 `cos(theta)=(trace(R_rel)-1)/2` 取得；实现用 `atan2(sin(theta),cos(theta))` 计算，并检查矩阵形状、有限性、正交性及 `det≈+1`。

**注意：**原始旋转角度不是 BOP 官方完整的对称物体姿态分数（MSSD/MSPD 等），也不是物理设备精度。

## 3. 三次运行前预测

**A：**固定旋转，`Δt=[0.005,-0.002,0] m`，则 `e_t=sqrt(29)=5.385164807... mm`，`e_R=0°`。

**B：**固定平移，`R_est=Rz(10°) @ R_gt`，则 `e_t=0 mm`，`e_R=10°`。

**C：**显式把 `source_frame='camera', target_frame='model'` 传入要求 model → camera 的 `PoseM2C`，应抛出 `ValueError('INVALID_POSE_FRAME_CONVENTION')`。**局限：**若逆变换数值被错误标成 model → camera，类型检查本身无法证明真实方向。

**对称性边界：**一个物体旋转 180° 后可与原外观几何等价，但原始旋转角度仍可能是 180°；必须单独使用对称感知评价。

## 4. 实测与解释

本机（Windows、`ai_3d`）执行 `python -m src.evaluation.t1_04_a_pose_metrics`：

```text
translation_error_mm: 5.385164807134504
rotation_error_deg_raw: 9.999999999999998
sample_kind: SYNTHETIC_IDENTITY_GT_FIXTURE
estimation_performed: False
gt_used_to_construct_candidate: True
publishable: False
```

真实 BOP LM 场景 `000001`、`im_id=15`、`gt_index=0` 的 GT **派生组合扰动**：

```text
translation_error_mm: 5.385164807134509
rotation_error_deg_raw: 9.999998608602157
sample_kind: REAL_BOP_GT_WITH_SYNTHETIC_GT_DERIVED_PERTURBATION
estimation_performed: False
gt_used_to_construct_candidate: True
pose_accuracy_claimed: False
physical_accuracy_claimed: False
```

旋转角差约 `1.4e-6°`；实际 BOP 旋转条目有有限小数位，存在微小数值不正交性。不能把此差值称为定位精度。

专项 `python -m pytest tests/test_t1_04_a_pose_metrics.py -v`：**19 passed**。覆盖 0°/接近 0°/90°/180°、单位、三维平移欧氏距离、方向声明、非法旋转、只读值复制和 GT 派生演示的安全字段。

**独立 A/B/C 实验已实测（本机 PowerShell，先预测后运行）：**

```text
A translation_mm: 5.385164807134504
A rotation_deg: 0.0
B translation_mm: 0.0
B rotation_deg: 9.999999999999998
C rejected: INVALID_POSE_FRAME_CONVENTION
T1-04-A independent A/B/C: PASS
```

**全仓回归：**`python -m pytest -q` 得到 `795 passed, 106 subtests passed in 7.27s`。本机 `git diff --check` 无错误；现有 T1-04-A 三个文件仍未提交。

**实测解释：**A 只修改平移，旋转误差为 0；B 只修改旋转，平移误差为 0；C 显式传入 `camera → model` 声明，按 `PoseM2C` 契约抛出 `INVALID_POSE_FRAME_CONVENTION`。

## 5. 代码/参数拆解与可迁移边界

- `PoseM2C`: `frozen=True` 加显式 `np.array(copy=True)` 与 `setflags(write=False)`，隔离外部数组修改。
- `from_bop`: `translation_unit='mm'` 并除以 1000，避免把毫米误作米。
- `axis_angle_rotation`: 使用轴角构造 SO(3) 测试旋转，不依赖位姿估计器。
- `compare_pose_m2c`: 仅允许 `symmetry_policy='raw'`，不支持的对称策略必须拒绝。
- `controlled_demo`: **候选位姿是人为用 GT 构造**；`estimation_performed=False`、`publishable=False`、`position_m=None` 防止误用为真实推理或机器人位置命令。

## 6. 常见错误与 Fail-closed

1. 把 `cam_t_m2c` 毫米直接当米：误差放大 1000 倍；统一内部 m、输出误差 mm。
2. 把 camera → model 的逆变换与 model → camera 的 GT 直接比较：数值可以有限，语义却错误。
3. 旋转矩阵 `det=-1`（反射）、不正交、形状不对或 NaN：拒绝。
4. 把原始旋转角误差用于对称物体的官方 BOP 排名：不允许。
5. 把 GT+扰动演示当姿态推理：不允许。
6. 仅依赖帧标签识别逆变换：当元数据本身错误时，该保护不充分；需验证外参链和来源。

## 7. 最终 Boss 与通关清单

**Boss：逆变换数值被错误标记为正向变换，是否能由 `PoseM2C` 自动识别？**

**结论：PASS。** 学员正确指出：`T_camera_from_model` 与其逆 `T_model_from_camera` 都是合法的 SE(3)；数值 Shape、正交性、行列式、有限性及错误填写的框架标签都可能通过检验，但并不能证明位姿语义正确。逆旋转的**绝对旋转角**与正向旋转相同，并不意味着对同一 GT 的相对旋转误差一定小。

**提出的独立几何检查（设计方案，尚未实现）：**取已知模型三维点与独立测得的图像二维对应关系，使用待检验变换将模型点送入相机系，并利用相机内参投影；检查正深度、真实 2D-3D 对应点的重投影残差、可见表面与真实 Depth/Mask 的一致性。必须考虑遮挡、深度噪声和对称性：**不能要求任意模型点都落在可见 Mask 内，或都能与单个真实深度像素直接对应**。该方案要使用独立于可疑方向标签的模型/图像观测，避免循环自证。

**指标边界：**GT 派生扰动不是独立姿态估计；当前实现只提供毫米平移误差和非对称感知旋转角，不包含 BOP MSSD/MSPD、真实模型推理或物理精度验证。

- [x] 理论、输入单位与方向、最小组合实验
- [x] 三组运行前预测、对称性概念题
- [x] 专项 19 项测试
- [x] 独立 A/B/C 实测证据
- [x] 全仓回归结果
- [x] 最终 Boss
- [x] 最终学习笔记已复核；待替换项目内的草稿
- [ ] `git diff --cached --check`、精确提交、Push、`HEAD == origin/master`

**Git 当前已知状态：**上一个已同步的 Learning Unit Commit 为 `74c1bc51d6fc7070054dc208e059c1dde95149d6`（T1-03）；T1-04-A 的三个文件待精确提交。六个无关 ScreenCamera/ScreenCapture 文件禁止纳入提交。

**下一阶段：**T1-04 后续的真实姿态估计与对称感知指标契约；本次没有独立 Pose 推理结果。
