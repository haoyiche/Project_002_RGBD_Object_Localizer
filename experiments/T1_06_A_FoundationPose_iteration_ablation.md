# T1-06-A｜FoundationPose 模型式 6D Pose：GT 隔离、固定 50 帧对照与精修轮数消融

> **用途**：覆盖 `experiments/T1_06_A_FoundationPose_iteration_ablation.md` 的完整正式学习笔记。虽沿用原文件名，本文件覆盖 T1-06-A 全部实验闭环，不只记录最后一次消融。
> **依据**：`AI_Lab_SOP.md`；本项目 T1-04 / T1-05 基线协议；已运行的 Windows / Docker 日志；50 帧统计、观察性诊断及三帧 `paired_iteration1_vs5.json`；本人完成的 Boss 问答及已指出的修正。
> **状态**：`BOSS_PASS / EXPERIMENTS_COMPLETE / ENGINEERING_ACCEPTANCE_PENDING`；**Learning Unit = CONTINUE，尚未 LEVEL COMPLETE**。
> **可证范围**：BOP LINEMOD 已研究过的同一场景、同一物体、固定 50 帧，已知 CAD + Oracle Mask，GT 隔离推理与冻结后评价。**不是未见测试集、端到端检测分割系统、官方对称感知 BOP 成绩或工业物理精度认证。**

---

## 0. Entry Protocol：本关是什么、产物和验收

- **当前 Level**：T1-06-A。**主题**：官方 FoundationPose model-based `register()` 与此前 GT-free PCA+ICP 的受控对照。
- **上一关连接**：T1-04 建立了模型到相机的 SE(3)、GT 盲预测/事后评价分离与固定 50 帧框架；T1-05 分析了 ICP 质量门控及原始旋转歧义。
- **本关目标**：让第三方 FoundationPose 在 6 GiB RTX 2060 Max-Q 上真实运行；确保推理不能读取 GT；在完全相同的 50 帧上比较误差；掌握数据单位、坐标、候选、Refiner/Scorer、显存与延迟；通过主动修改和 Boss 解释结果。
- **已形成的工程产物**：单帧适配器、冻结/评价器、固定批量执行器、批量后验汇总器、相应测试、50 帧冻结预测及汇总、三帧 `iteration=1` 消融、观察性诊断 CSV、本笔记。
- **通关条件**：理论/公式/参数/代码/主动修改/预测与分析/Boss/笔记/全项目回归/Git commit/GitHub push 均需检查。现阶段 **Boss PASS，工程回归和 Git/GitHub 未核验**；不能把“推理 PASS”替代完整通关。

### 知识地图

```text
T1-01/02：相机内参 K、深度和 BOP 数据契约
       ↓
T1-03/04：3D 模型、坐标系 SE(3)、评估指标与 GT 隔离
       ↓
T1-05：GT-free PCA+ICP、门控、局部最优与旋转歧义
       ↓
T1-06-A：已知 CAD + RGB-D + Oracle Mask → FoundationPose register
       ↓
冻结预测 → 后验 BOP 标注评价 → 与 ICP 配对比较 → 参数消融
       ↓
后续：对称感知评估、分割误差传播、模块级性能剖析、工业迁移
```

---

## 1. 最小必要理论（它是什么、为什么要用、改变会怎样）

### 1.1 Model-based 6D Pose

本实验已知目标物体 ID 和 CAD 三角网格，以单张 RGB-D 观测及**可见实例 Oracle Mask**估计模型坐标系到相机坐标系的刚体变换：

\[
T_{c\leftarrow m}=\begin{bmatrix}R&t\\0&1\end{bmatrix},
\quad p_c=Rp_m+t.
\]

- `R`：`(3,3)`、无单位、`SO(3)`，满足 `R.T @ R ≈ I`、`det(R) ≈ +1`。
- `t`：`(3,)`、**米**；不能与 BOP `cam_t_m2c` 的 **毫米**混用。
- `T_camera_from_model`：`(4,4)`，最后一行为 `[0,0,0,1]`。
- 本次使用**每帧独立 `FoundationPose.register()`**，不调用 `track_one()`；即使帧来自序列，也不利用上一帧跟踪状态。

### 1.2 相机、深度、CAD 单位

BOP `scene_camera.json` 提供各帧内参 `cam_K` 和 `depth_scale`。对于原始 `uint16` 深度：

\[
Z_{\mathrm{m}}=D_{\mathrm{raw}}\frac{\mathrm{depth\_scale}}{1000}.
\]

`depth_scale` 表示每个原始数值的毫米数；若直接当作米，会产生 1000 倍尺度错误。BOP CAD PLY 的顶点单位是毫米，本适配器在核对 `models_info.json` 中 `size_x/y/z` 后执行 `mesh.apply_scale(0.001)`。原始模型边界范围在第 55 帧预检中约为 `[75.869, 77.599, 91.769] mm`，这不是米。

反投影用于理解初始平移：

\[
X=(u-c_x)Z/f_x,\qquad Y=(v-c_y)Z/f_y.
\]

公式中 `u,v,c_x,c_y,f_x,f_y` 均以像素为相关尺度，`Z,X,Y` 都以米计。距离 Z 变大时，同样像素偏差对应更大的横向位置偏差，这是“远距离可能不利”的几何直觉，但并不是本次统计的因果结论。

### 1.3 官方 FoundationPose 推理流程及坐标补偿

```text
RGB [H,W,3] uint8 RGB + depth [H,W] float32 m
+ visible mask [H,W] + K [3,3] + CAD triangular mesh (m)
    ↓
Mask 与 Depth 估计初始目标中心平移
    ↓
离散视角 / 平面内旋转候选（本次 42 × 6 = 252 个）
    ↓
Refiner：将当前位姿的渲染与观测构造成网络输入，迭代预测位姿修正
    ↓
Scorer：对精修后的候选评分、排序
    ↓
最佳 T_camera_from_model [4,4]（原始 CAD 坐标系）
```

- 在本次运行中 `make_rotation_grid()` 产生并保留 **252** 个候选。网络评分是相对排序信号，**不是正确概率**。
- Refiner 的推理调用在 `torch.inference_mode()` 下，以前向网络预测旋转/平移修正；**不是每轮直接对 BOP GT 误差做梯度下降**。它也不保证相对于 GT 的平移或原始旋转误差单调减小。
- 官方 `FoundationPose.register()` 返回时，已通过 `poses[0] @ get_tf_to_centered_mesh()` 将估计转换回**原始 CAD 模型原点**。**不能二次补偿模型中心。**
- `compute_add_err_to_gt_pose()` 在核对的官方版本中返回常数 `-1` 占位值；日志 `add_errs min:-1.0` **不是 ADD 误差，更不是模型准确率**。
- 本实验没有目标检测器，也没有自动分割器。Oracle Mask 明确是外部监督输入；不能用“检测框偏移”或“模型自身分割错误”解释本次结果。

### 1.4 后验误差定义与局限

平移误差：

\[
e_t=1000\|t_{est}-t_{GT}\|_2\quad (\mathrm{mm}).
\]

原始旋转误差：

\[
e_R=\operatorname{angle}(R_{est}R_{GT}^{\mathsf T})\quad (\mathrm{degree}).
\]

本项目的实现使用 `atan2(sinθ, cosθ)` 形式以处理接近 0° 或 180° 的情况。**原始旋转角不考虑物体对称性**；对称感知误差、ADD/ADD-S、重投影/可见几何一致性可作为后续独立指标，不能在这里冒充已经计算完成。

---

## 2. 环境与复现身份（不是把模型权重提交到 Git）

| 项目 | 本次记录 |
|---|---|
| Windows 项目根目录 | `D:\AI_Lab\02_Projects\Project_002_RGBD_Object_Localizer` |
| 本地 Python 环境 | `D:\conda_envs\ai_3d`，Python 3.11.17；用于 CPU 契约、审计与评估，不修改为 GPU 训练环境 |
| 官方第三方仓库 | `D:\AI_Lab\_external\FoundationPose`；NVlabs/FoundationPose，固定 commit `a1b694b83e633c2cb6115b9063d940a687759392` |
| Docker | `foundationpose:t106`；镜像 ID / digest 记录 `sha256:22837f81c0cfc1906b96daa344f8e80919c7619b816fcab487a024f1f566c215` |
| 容器主要环境 | Linux / Python 3.8 / PyTorch 2.0.0+cu118 / Warp 1.0.0；CUDA 可用 |
| GPU | NVIDIA GeForce RTX 2060 Max-Q，6 GiB，`sm_75` |
| 依赖关键项 | `mycpp` 原生扩展已编译；NVDiffRast CUDA 扩展可光栅化 |
| 官方权重 | Refiner：`weights/2023-10-28-18-33-37/`；Scorer：`weights/2024-01-11-20-02-45/`；实际权重哈希由冻结批量计划记录 |
| 构建约束 | `MAX_JOBS=1`、`TORCH_CUDA_ARCH_LIST=7.5`、`TORCH_EXTENSIONS_DIR=/fp_build/torch_extensions`，缓存宿主路径 `_external/fp_torch_extensions` |

**身份边界**：适配器单帧输出里的源码 commit 和镜像 digest 属于“标签”；固定 50 帧批处理的运行前检查还核对了 Git HEAD、tracked 工作树与 Docker image ID，冻结清单另记录输入、适配器、配置和权重哈希。不要把单帧输出里的字符串本身当成独立密码学认证。

---

## 3. 代码地图、Shape、参数与错误边界

### 3.1 本项目代码

| 模块 | 核心职责 |
|---|---|
| `src/evaluation/t1_06_a_pose_adapter_contract.py` | CPU 坐标与单位、GT 隔离、单帧 register/track 契约 |
| `src/evaluation/t1_06_a_fp_single_frame.py` | 六文件输入验证、CAD mm→m、官方 `register()`、SE(3) 验证、单帧预测 JSON |
| `src/evaluation/t1_06_a_fp_posthoc_eval.py` | 预测 SHA 冻结；独立评价器只在验证后读取 `scene_gt.json` |
| `src/evaluation/t1_06_a_fp_fixed_batch.py` | 与 B4 相同选帧、每帧独立 Docker、GT 隔离输入目录、失败即停、冻结、审计 |
| `src/evaluation/t1_06_a_fp_batch_posthoc.py` | 50 帧完全预检后评价、与 T1-04-B4 的逐帧配对汇总 |
| `tests/test_t1_06_a_pose_adapter_contract.py` | CPU 契约回归；先前记录 17/17 PASS |
| `tests/test_t1_06_a_fp_fixed_batch.py` | 固定批量防护；用户环境实测 5/5 PASS |
| `tests/test_t1_06_a_fp_batch_posthoc.py` | 评价器防护；此前独立合成测试记录 10 项通过，**全项目回归仍待用户执行** |

### 3.2 关键输入、输出与参数

| 名称 | 类型 / Shape / 单位 | 来源与作用 | 错误风险 |
|---|---|---|---|
| `rgb` | `numpy uint8 [H,W,3]`，RGB 顺序 | `cv2.imread` 后 BGR→RGB | BGR/RGB 搞反 |
| `depth` | `numpy float32 [H,W]`，m | `uint16 × depth_scale/1000` | mm/m 混用 |
| `ob_mask` | `[H,W]`，0/1 可见掩码 | `mask_visib` 的 `gt_index` | 不可当自动分割能力 |
| `K` | `float64 [3,3]` | 每帧 `scene_camera` | 用错相机内参 |
| `mesh.vertices` | `[N,3]`，m | CAD PLY 校验范围后乘 0.001 | 把毫米当米 |
| `mesh.faces` | `[M,3]`，三角面索引 | CAD PLY | 不能只有点云而无三角面 |
| `pose` | `float64 [4,4]`，`T_c←m` | 官方 `register()` 返回 | 二次补偿中心 / 反置变换 |
| `iteration` | `int`，单位：轮 | `register(iteration=5)`；消融改为 1 | 不保证误差/耗时单调 |
| `min_n_views / inplane_step` | 40 视角下限 / 60° 步长 | 官方旋转网格 | 改动会影响候选及成本；本次未改 |
| `torch_peak_reserved` | MiB | PyTorch 缓存分配器历史峰值 | 不等于全 GPU 显存或活跃张量 |
| `register_cold` | s | 初始化完成之后的 `register()` 调用（含可能的首调用 JIT） | 不等于 Docker 启动 + 权重加载总延迟 |

### 3.3 数据隔离与防止 GT 泄漏

每个推理容器只挂载六项输入：

```text
rgb/000XXX.png
 depth/000XXX.png
 mask_visib/000XXX_000000.png
 scene_camera.json
 models/obj_000001.ply
 models/models_info.json
```

容器**看不到 `scene_gt.json`** 或含 GT 的 manifest。推理完成后先按 SHA-256 冻结 pose JSON 与输入指纹；评价器先验证候选、冻结文件、原始六项文件，再读取 GT。每一帧 `register()` 独立运行，不偷偷使用 `track_one()` 或按 GT 重新选候选。

---

## 4. 最小实验和工程故障排查记录

### 4.1 从构建失败到真实 GPU 推理

1. 官方 PyTorch、Refiner 与 Scorer 各自可加载；联合加载记录 `AFTER_SCORER free=4755 MiB, reserved=352 MiB`。
2. 最早创建 `dr.RasterizeCudaContext()` 时，JIT 扩展并行编译导致 `c++ ... Killed signal terminated program cc1plus`，更像容器可用主存/编译任务压力，不是已被证实的 GPU OOM。
3. 使用 `MAX_JOBS=1`、`TORCH_CUDA_ARCH_LIST=7.5` 和持久 Torch 扩展缓存后，真实 NVDiffRast `32×32` 三角形光栅化通过：`COVERED_PIXELS=338`，并通过网络 + 光栅器同进程测试。由于同时涉及多个构建设置，**不能单独断言某一个设置是唯一根因**。
4. BOP 第 55 帧六文件预检通过：图像 `480×640`、可见有效深度 `1436` 像素、CAD 范围校验通过。
5. 首次真实 `register(iteration=5)`：252 候选 + 5 轮 Refiner + Scorer，`12.046 s`；`torch_peak_reserved=4920 MiB`；预测平移 `[-0.1140508, 0.0170194, 1.0750089] m`。
6. 第 55 帧冻结 SHA `6f6904374888bf44ba58d8de9736b177e3a335abcde0c348358d483c0e4597c7`，独立后验：**平移 9.8017 mm，原始旋转 3.0197°**。这是单帧观察，不代替固定批量结论。

### 4.2 已做过的主动修改 / 反事实契约检查

- 先前 CPU 契约实验包括中心坐标补偿的主动验证，以及故意破坏模型 `mm→m` 和将独立 `register` 换为 `track` 的检查，分别拦截 `MODEL_SIZE_MM_MISMATCH`、`TEMPORAL_MODE_NOT_SINGLE_FRAME`。其价值是验证**错误能否被发现**，不是准确率实验。
- `mycpp.cluster_poses` 先前以两组平移相差 `0.05 m` 的 identity 旋转做阈值试验：大阈值合为 1 组、较小 `0.01 m` 阈值为 2 组。关键是理解聚类阈值可改变离散假设数，并非本次正式 50 帧改参。
- 本笔记后述 `iteration=5→1` 是真实 GPU、改变算法参数的主动消融；**此前没有保存学习者本人运行前的正式预测**，不能事后补写成先验记录。

### 4.3 小故障与防错

- 曾将 PowerShell 占位字符串 `"这里替换为查找到的实际 batch_summary.json 路径"` 直接赋给 `$icpSummary`，触发 `FILE_NOT_FOUND`。根因是示例占位符未替换，并非模型或评价失败；随后自动筛选出真实 T1-04-B4 50 帧汇总并完成配对比较。
- `FRAME_FROZEN 50/50` 和 `AUDIT_PASS` 不等于已经读取 GT；直到后验阶段才生成准确率统计。
- 禁止把分割、检测、渲染评分、训练监督和官方 ADD 分数混成一回事。

---

## 5. 固定 50 帧：严格可比的正式结果

### 5.1 实验协议与完成证据

- 数据：`data/raw/bop/lm/test/000001`；`obj_id=1`；`gt_index=0`；固定 50 帧来自 **T1-04-B4 同一函数 `select_rgbd_frames()`**（按照 RGB/Depth 文件名排序选首 50 帧），不是根据 GT 或结果重新挑选。
- 帧 ID：`[15,19,40,41,47,55,59,66,74,84,87,98,103,104,106,112,128,129,135,141,144,145,157,176,197,202,205,209,214,228,236,239,240,245,251,268,272,274,278,283,285,308,310,316,323,326,343,345,350,358]`。
- `tests/test_t1_06_a_fp_fixed_batch.py`：本机 **5 passed**；Pilot `3/3` 预测和审计 PASS；正式批次 **50/50 预测冻结 PASS**；**50/50 AUDIT PASS、GT_EVALUATED=False**。
- 后验配对检查为 `T106_FP_VS_ICP_PAIRED_COMPARISON_PASS count=50`；比较器验证帧 ID、GT 标注 SHA、ICP 冻结汇总等契约。没有重新估计姿态。
- ICP 对照汇总：`outputs/t1_04/b4_50_20261030_091523/batch_summary.json`。只比较同一批原始、未做对称修正的注释相对指标。

### 5.2 数值（越低越好）

| 指标 | FoundationPose `iteration=5` | PCA+ICP |
|---|---:|---:|
| 平移误差均值 (mm) | **4.8952** | 12.2480 |
| 平移误差中位数 (mm) | **4.9185** | 12.9409 |
| 平移误差线性插值 P95 (mm) | **8.5942** | 23.4078 |
| 平移误差最大值 (mm) | **9.8018** | 24.9964 |
| 原始旋转误差均值 (°) | **2.1491** | 97.4544 |
| 原始旋转误差中位数 (°) | **2.0707** | 107.8195 |
| 原始旋转误差 P95 (°) | **3.4302** | 172.5226 |
| 原始旋转误差最大值 (°) | **4.5389** | 176.2143 |
| 原始旋转误差 >90° | **0/50** | **27/50** |
| 原始旋转误差 >150° | **0/50** | **16/50** |
| 逐帧平移更小 | **35/50** | 15/50；无平局 |
| 逐帧原始旋转更小 | **50/50** | 0/50；无平局 |

相对 ICP 基线的均值，FP 平移误差约下降 **60.0%**，原始旋转误差约下降 **97.8%**，只是上述协议内的描述性统计。T1-05 先前质量门控通过子集里的 `24/37` 大角度帧和这里**完整 50 帧**的 `27/50` 不是同一分母与筛选范围，不能直接替换。

### 5.3 性能与资源代价

- 50 帧 `register()`：均值 `9.1170 s`，中位数 `9.5081 s`，P95 `10.7015 s`，最大 `15.7073 s`。
- 50 帧 PyTorch 峰值保留显存：均值/中位数/P95/最大均记录 `4920 MiB`。这是每个独立容器的 PyTorch allocator reserved 峰值，并**非** GPU 总占用、真实活跃张量大小或每次实际总线峰值。
- 单帧 `register_cold` 时间不包括 Refiner/Scorer 加载、`FoundationPose` 初始化、Docker 启动与数据准备；但可能包含首次 CUDA 内核 JIT，因此不能直接等同工业端到端单帧时延。

### 5.4 实验结论的边界

FoundationPose 在本次 50 帧、已知 CAD、Oracle Mask 的受控条件下，原始旋转误差**逐帧优于** ICP，平移误差整体也更小；但 **15 帧 ICP 平移更准**。已研究过的样本不能称为无偏 held-out 测试。未测自动检测/分割、工业鲁棒性、物体类别泛化与物理标定精度。

---

## 6. 误差分析：哪些帧例外，哪些假设得到支持？

### 6.1 平移相对落后的五帧

定义 `Δt = e_FP − e_ICP`，正值表示 FP 平移误差大于 ICP：

| 帧 | FP 平移 mm | ICP 平移 mm | Δt mm |
|---:|---:|---:|---:|
| 106 | 8.8151 | 2.0158 | +6.7993 |
| 66 | 7.7830 | 1.2582 | +6.5249 |
| 74 | 6.7372 | 1.6634 | +5.0738 |
| 47 | 7.3094 | 3.3054 | +4.0040 |
| 268 | 5.4741 | 1.5171 | +3.9570 |

“相对落后”不是“定位崩溃”：FP 在这些帧的平移误差仍均小于 10 mm；应当强调 ICP 在部分帧具有更高平移精度。

### 6.2 FP 自身原始旋转误差最大的五帧

| 帧 | 平移误差 mm | 原始旋转误差 ° |
|---:|---:|---:|
| 228 | 2.7891 | 4.5389 |
| 350 | 8.3829 | 4.0860 |
| 66 | 7.7830 | 3.4303 |
| 308 | 2.6869 | 3.4300 |
| 128 | 3.9243 | 3.3009 |

“最大 4.54°”描述的是本数据集、该模型、该参数的最大样本误差，不是普遍的最坏情况保证。

### 6.3 观察性诊断（不重跑推理）

分析 `50` 个已经冻结的记录，以相对平移误差 `Δt` 为响应量：

| 特征 | FP 相对落后 15 帧的中位数 | FP 相对领先 35 帧的中位数 | 与 `Δt` 的 Spearman ρ |
|---|---:|---:|---:|
| 可见 Mask 像素数 | 1604 | 1923 | **−0.1934** |
| Mask 内有效深度比例 | 1.000 | 1.000 | **+0.1271** |
| Mask 内中位深度 | 0.992 m | 0.894 m | **+0.2507** |

- “小 Mask 可能更差”与“远距离可能更差”的**相关方向**和先前预测一致，但相关很弱；没有因果证据，也未记录置信区间/显著性检验。
- 有效深度比例的相关方向**不支持**“有效比例越低 FP 越差”的原预测；两组中位数 `1.0` 呈饱和状态。有效像素存在不等于深度噪声小。
- Mask 大小还受距离、姿态、遮挡共同影响；同场景帧并非独立总体样本。不能由这三项数据断言“训练分布偏差”“纹理问题”“对称性”“分割模型误差”已经被证实。
- 输出：`outputs/t1_06_a/fp_failure_analysis_v1.csv`；日志 `T106_FP_FAILURE_DIAGNOSTICS_PASS`、`INFERENCE_RERUN: False`。

---

## 7. 主动改参：`register(iteration=5)` 对比 `iteration=1`

### 7.1 单变量实验设计与先验记录真实性

- **改动的算法参数**：`register(iteration)` 从 **5** 降为 **1**，其余 RGB-D、Oracle Mask、CAD、Refiner/Scorer 权重、252 视角假设、同一个单帧入口不变；**三帧单独进 Docker**。
- 帧：`66`（平移落后）、`106`（相对平移劣势最大）、`228`（FP 旋转误差最高）；这些是**已看到结果后挑出的样本**，不是随机样本或新测试集。
- 单变量条件由源码/镜像标签、输入 SHA、冻结预测 SHA、GT 标注 SHA 一致性检查支持。GPU 运行时状态和内核缓存不完全可控，计时差异只能谨慎解释。
- **预测记录**：此前导师提出过“精修减少可能降低计算时间；精度不一定单调；显存不一定下降”的方向预期，但**本次运行前没有单独提交学习者本人书面预测**。不能把实测之后的解释改写为本人先验预测。SOP 记录保留这一欠项；下一次新实验必须先记录。
- 先预测再验证的完整闭环在其他早期概念/契约实验有实施；本次消融本身此项不补造证据。

### 7.2 冻结与独立评价

三个 `iteration=1` 帧分别成功 `FOUNDATIONPOSE_SINGLE_FRAME_REGISTER_PASS`，立即冻结 SHA，`GT_READ_DURING_FREEZE: False`；之后逐帧 `T106_FP_POSTHOC_EVAL_PASS`。配对检查源码、镜像、输入、候选与 GT SHA 后通过 `T106_FP_ITERATION_ABLATION_PAIRED_PASS`。

`Δ平移 = 1轮误差 − 5轮误差`，`Δ旋转 = 1轮误差 − 5轮误差`，`节省 = 5轮耗时 − 1轮耗时`。误差差值为正表示 1 轮更差；节省为负表示 1 轮反而更慢。

| 帧 | 平移 5轮/1轮 (mm) | Δ平移 mm | 原始旋转 5轮/1轮 (°) | Δ旋转 ° | register 5轮/1轮 (s) | 节省 s |
|---:|---:|---:|---:|---:|---:|---:|
| 66 | 7.7830 / 7.8622 | +0.0792 | 3.4303 / 2.0949 | **−1.3354** | 9.5009 / 7.7548 | +1.7461 |
| 106 | 8.8151 / 8.9185 | +0.1034 | 0.8782 / 0.7101 | **−0.1681** | 9.5572 / 9.8701 | **−0.3130** |
| 228 | 2.7891 / 3.6103 | +0.8211 | 4.5389 / 9.3370 | **+4.7981** | 8.9525 / 8.3988 | +0.5536 |
| **均值** | **6.4624 / 6.7970** | **+0.3346** | **2.9491 / 4.0473** | **+1.0982** | **9.3368 / 8.6746** | **+0.6623** |

三帧均值节省约 `0.6623/9.3368 = 7.09%`；**不是减少 80% 的总耗时**。5 轮在这三帧平移全更准；旋转方面 1 轮在帧 66、106 反而更准，在帧 228 显著更差。帧 106 少四轮还慢 `0.313 s`，反映计时中的运行波动或非理想因素。每帧目前只各测一次，不能推广为稳定速度结论。

本次 1 轮三帧 PyTorch `torch_peak_reserved=4972 MiB`，正式 5 轮批量记录为 `4920 MiB`；差 52 MiB **不可归因为减少迭代导致显存增加**。`reserved` 是缓存分配器峰值，不代表活跃张量或设备总显存。

### 7.3 为什么更少轮数不保证全面更好？

Refiner 每轮对当前渲染与观测进行网络前向并预测 SE(3) 修正，而非对 GT 误差进行保证下降的优化。平移和旋转是不同自由度，受可见几何、纹理与遮挡约束的强弱不同。更多轮数可能降低某指标但增大另一个指标；**弱约束方向漂移只是可检验假说，尚未对这三帧做出因果证明**。

---

## 8. Boss 验收：本人理解与导师修正（不是事后编造运行前预测）

### Q1. 为什么精修更多轮可能平移变好、旋转变差？

我的理解：平移与旋转分别衡量，Refiner 是用训练好的网络基于渲染/观测预测修正量；推理时没有 GT 作为在线监督，因此更多轮数不保证原始旋转误差下降。遮挡或对称等情况可能造成弱约束，但本次没有确认具体帧的真实机制。

### Q2. 为什么精修少 80% 不代表 `register()` 快 80%？

`register()` 还包括深度预处理、假设生成、渲染/裁剪、评分、调用开销以及可能的首次 JIT。**模型权重加载位于本次 `register_cold` 计时之外；实验没有运行自动检测器**。简化表达：

\[
T(n)=T_{\rm fixed}+n t,\quad
 t=\frac{T(5)-T(1)}{4},\quad T_{\rm fixed}=T(1)-t.
\]

三帧均值 `T(5)=9.3368 s`，`T(1)=8.6746 s`，在**假定线性模型严格成立**的条件下得到：

- `t ≈ 0.16555 s/轮`（约 165.55 ms/轮）
- `T_fixed ≈ 8.50905 s`
- 验算 `8.50905 + 5 × 0.16555 = 9.3368 s`

**这两个值是从两个总时间均值反推的模型参数，不是模块计时实测。** 两个点总能拟合直线；帧 106 的少轮反而更慢也说明没有证明确定性线性成本结构。要确认瓶颈须做同一帧多次重复、预热、CPU/GPU 同步与模块级 profiler。

### Q3. 帧 66 的“1轮更准”成立吗？

不作无条件结论：在帧 66，1 轮**平移差 +0.079 mm**、**原始旋转更好 1.335°**。若验收以平移优先，则 5 轮更好；若以原始旋转误差优先，则 1 轮更好。整体最优须先定义阈值、指标、权重、对称性政策及速度目标，并对固定帧集做配对；不能事后按对自己有利的维度挑结论。

### Q4. 显存如何排错？

- `torch.cuda.max_memory_allocated()` 是 PyTorch 张量实际分配峰值，`max_memory_reserved()` 是缓存分配器保留峰值；`torch.cuda.mem_get_info()` 和 `nvidia-smi` 可补充查看设备/进程显存；Warp/NVDiffRast、CUDA context、workspace 等不一定全部在 PyTorch allocator 里。
- `torch.cuda.memory_snapshot()` 可辅助分析 PyTorch allocator；`empty_cache()` 可释放无活跃引用的缓存，但**不会释放活跃张量，也不会自动清除历史峰值**。新测量窗口要用 `torch.cuda.reset_peak_memory_stats()` 并做同步；不能把一次 `reserved` 差异当作根因。
- 固定输入/容器镜像/缓存/后台占用，重复运行比较 `allocated`、`reserved` 与设备实际占用，必要时模块级剖析。

### Q5. 从 Oracle Mask 迁移到工业自动分割，怎样拆分误差？

在相同 RGB-D、CAD、位姿模型和帧集上使用 Oracle Mask 与自动 Mask 两个受控条件，再做腐蚀/膨胀/空洞等可控扰动；同时记录 IoU、边界误差、漏检/误检、平移/旋转/对称感知指标与失败率。**位姿不能把两个 4×4 矩阵直接作普通数值减法来代表位姿差**；可比较 `T_A^{-1}T_B`、平移差范数和相对旋转角。`E_auto−E_oracle` 仅是两条件指标差，**不是严格可相加的独立分割误差**；Oracle 条件是受控参考，不保证每帧数值都优于自动 Mask。

### Boss 结果与理解层级

- 概念/推理/公式/参数/排错/迁移五题及补充线性模型计算均已被验收为 **Boss PASS**。
- 本人能够区分配对差异与因果解释、独立预测与 GT 后验评估、模型估计与真实模块计时。
- **下一步仍需完成**：正式笔记保存、工程回归、Git/GitHub 验收；不因 Boss PASS 直接宣告关卡完成。

---

## 9. 常见问题与误区（错在哪里，怎样验证）

| 误区/现象 | 正确认识与验证 |
|---|---|
| “权重加载成功就是 FoundationPose 实验通关” | 必须真实 `register()`、冻结、后验评价、解释、回归、Git |
| “GT-free 就不允许 Oracle Mask” | 这里 GT-free 特指**位姿 GT 不可见**；Oracle Mask 属于明确声明的分割监督，不能写成端到端无监督 |
| “模型 CAD 顶点、BOP 深度、GT 平移都是同一单位” | CAD mm→m；深度 `raw×depth_scale/1000`；GT `cam_t_m2c` mm→m；各转换仅做一次 |
| “再补偿一次 CAD 中心更稳妥” | 官方 `register` 已返回原始 CAD 坐标系；双重补偿会错 |
| “Score 81.9 = 正确率 81.9%” | Scorer 输出用于排序，不是校准后的概率 |
| “`add_errs=-1` 说明 ADD 很优秀” | 官方占位返回值，不是 ADD |
| “原始旋转 150° 必定几何对齐失败” | 对称与可见性可能造成差异；要另测对称感知误差和渲染一致性 |
| “`mask_pixels` 相关系数 −0.19 已经证明 Mask 小导致问题” | 仅弱相关，存在距离/遮挡/姿态等混杂因素 |
| “少做四轮必定每帧更快” | 实测帧 106 相反；必须重复测量控制方差 |
| “`empty_cache` 后 reserved 峰值变小就证明全部是缓存问题” | 活跃张量无法清；历史峰值要 reset；还需看非 PyTorch 显存 |
| “只比较 50 帧均值就能宣称工业级效果” | 已研究数据、同一物体、Oracle Mask，非未见/对称感知/现场认证 |
| “路径占位符能自动找到 B4 汇总” | PowerShell 只传字面字符串；用 `Get-ChildItem` 查真实路径，验证唯一匹配 |
| “失败帧可跳过后算成功率” | 固定批量失败即停，冻结/审计全 50 后才能发布 50 帧汇总 |

---

## 10. 证据文件索引、复现命令与 Git 边界

### 10.1 本地证据目录

```text
outputs/t1_06_a/fp_single_frame_000055_v1/
    pose_000055.json, frozen_prediction.json, annotation_eval.json
outputs/t1_06_a/fp_pilot3_v1/
    frames/..., frozen_predictions.json
outputs/t1_06_a/fp_fixed50_v1/
    input_bundles/, frames/, frozen_predictions.json, ...
outputs/t1_06_a/fp_fixed50_eval_v1/
    batch_summary.json, paired_comparison.json, per-frame evaluation files
outputs/t1_06_a/fp_failure_analysis_v1.csv
outputs/t1_06_a/fp_iter1_3frames_v1/
    000066/, 000106/, 000228/, paired_iteration1_vs5.json
outputs/t1_04/b4_50_20261030_091523/batch_summary.json
```

已生成日志和结果保存在用户 Windows 项目下；**本笔记没有自行核验该机器上这些文件此刻仍全部存在**，其存在性以用户提供的运行日志为证。原则上 `outputs/` 不进 Git，保留本地审计证据与必要的轻量指标摘要。

### 10.2 复现已有证据的**只读**检查

在 Windows PowerShell、项目根目录、`conda activate ai_3d` 下：

```powershell
python -m src.evaluation.t1_06_a_fp_fixed_batch plan --count 50
python -m src.evaluation.t1_06_a_fp_fixed_batch audit `
  --count 50 --run-dir outputs/t1_06_a/fp_fixed50_v1

# 已评价文件只读取，不重新生成结果
Get-Content outputs/t1_06_a/fp_fixed50_eval_v1/batch_summary.json -Raw
Get-Content outputs/t1_06_a/fp_fixed50_eval_v1/paired_comparison.json -Raw
Get-Content outputs/t1_06_a/fp_iter1_3frames_v1/paired_iteration1_vs5.json -Raw
```

`plan` 会进行文件哈希与运行参数预检；`audit` 验证冻结记录，不读取位姿 GT。已有真实预测及评价文件时**不必重跑 GPU**，避免覆盖或改变记录。

### 10.3 工程回归与 Git/GitHub（执行前必须人工核查）

```powershell
python --version
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'T106_FULL_REGRESSION_FAILED' }

git branch --show-current
git status --short
git diff --check
git diff --stat
git remote -v
git log -1 --oneline
```

提交必须检查 `.gitignore`，逐文件 `git add`，用 `git diff --cached --name-status` 和 `git diff --cached --check` 审核，避免误提交数据集、权重、容器缓存、日志和完整 `outputs/`。未核对 `git status` 和远端状态时，**不声称已 Commit/Push**。若本地分支仍是 `master`，按实际分支 push，不机械沿用 SOP 举例的 `main`。

---

## 11. Exit Protocol：当前真实完成度与下一步

### 本次已完成

- FoundationPose 官方环境、Refiner/Scorer、原生扩展及 CUDA rasterizer 运行。
- 首次 BOP 第 55 帧真正调用 `register()`，输出独立冻结、GT 后验评价。
- 固定 50 帧同一选帧协议、**50/50 预测冻结与审计**、后验与 PCA+ICP 严格配对。
- 50 帧误差分布、相对落后帧定位、Mask/距离/有效深度观察性诊断。
- 3 帧精修轮数 `5→1` 主动消融、冻结、后验和严格配对。
- Boss 理论、参数、公式、排错、迁移题及补充推导已通过。

### 已理解并能够复述

SE(3) 坐标与单位、Oracle Mask 与位姿 GT 的区别、候选 Refiner/Scorer 的角色、预测冻结与后验评价的隔离、原始旋转与对称感知指标的区别、迭代数和误差/延迟非单调性、相关与因果的边界、预测模型参数与 profiler 实测的区别。

### 仍不稳定或需要另关深挖

- `register()` 各模块耗时的真实分布与稳定性能基准；目前只有 `register_cold` 单次数据。
- 完整对称感知指标和工业现场自动分割鲁棒性。
- `iteration=1` 消融缺少学习者本人运行**前**的正式书面预测；不允许伪补。下一次参数实验先落盘预测，再执行。
- 全项目回归、Git 当前状态、commit 与远端 push **尚未验证**。

### 通关清单

- [x] 概念、输入输出、公式和单位可解释
- [x] 最小真实 GPU 推理
- [x] 修改参数并进行真实实验
- [x] 严格配对/冻结与误差分析
- [x] 代码错误防护与 Debug 解释
- [x] Boss 测试及补充性能推导 PASS
- [x] 正式完整笔记已编写（**待用户保存进项目工作区**）
- [ ] `python -m pytest -q` 在用户本机全量通过
- [ ] `git status`、变更文件与 `.gitignore` 核验
- [ ] 仅提交必要源码、测试和笔记，`git commit` 成功
- [ ] `git push` 成功，并核对远程 commit

**本关最终状态：`CONTINUE`。下一步：覆盖本笔记 → 完整回归 → Git 状态检查 → 逐文件暂存 → Commit/Push 核验。**
