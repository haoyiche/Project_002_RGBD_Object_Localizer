# EXP026 — Camera Intrinsics & Pixel to 3D

## 1. Experiment Goal

本实验的目标是理解并实现：

```text
Pixel Coordinate
+ Camera Intrinsics
+ Camera Z-depth
        ↓
Camera Coordinate (X, Y, Z)
```

本实验不只要求公式能够运行，还需要掌握：

- 像素坐标 `(u, v)` 与相机坐标 `(X, Y, Z)` 的关系
- `fx / fy / cx / cy` 的几何意义
- Camera Z-depth 的作用
- 为什么一个 Pixel 本身只确定一条 3D Ray
- 为什么加入 Depth 后才能得到唯一 3D Point
- 内参变化对反投影结果的影响
- Depth 单位错误对 3D 几何结果的影响
- Geometry Layer 与 Sensor / Dataset Layer 的职责边界
- API Contract 的设计方法
- API 重构之后为什么必须做 Regression Test
- Syntax Error、Runtime Error、Regression Error 的区别

---

# 2. Coordinate Definitions

当前实验采用常见 Camera Coordinate Convention：

```text
X：向右
Y：向下
Z：向前
```

Pixel Coordinate：

```text
u：图像水平方向坐标
v：图像垂直方向坐标
```

Camera Coordinate：

```text
X：相机坐标系中的水平位置
Y：相机坐标系中的垂直位置
Z：相机坐标系中的前向深度
```

因此：

```text
u > cx  → X > 0
u < cx  → X < 0

v > cy  → Y > 0
v < cy  → Y < 0
```

这里的正负关系依赖当前采用的 Camera Coordinate Convention。

---

# 3. Camera Intrinsics

相机内参矩阵：

```text
K =
[ fx   0   cx ]
[  0  fy   cy ]
[  0   0    1 ]
```

其中：

## 3.1 fx

`fx` 是水平方向的焦距，单位通常是 Pixel。

它决定：

```text
同样的 Pixel 横向偏移
会对应多大的 X/Z
```

因为：

```text
X / Z = (u - cx) / fx
```

所以在其他参数不变时：

```text
fx 增大
→ X/Z 减小
→ 相同 Pixel 偏移对应更小的横向角度
```

---

## 3.2 fy

`fy` 是垂直方向的焦距，单位通常是 Pixel。

因为：

```text
Y / Z = (v - cy) / fy
```

所以：

```text
fy 减小
→ Y/Z 增大
```

在相同 Pixel 和 Depth 下：

```text
fy 越小
→ |Y| 越大
```

---

## 3.3 cx

`cx` 是 Principal Point 的水平坐标。

它决定图像中的哪个位置对应：

```text
X = 0
```

因为：

```text
X = (u - cx) * Z / fx
```

当：

```text
u = cx
```

时：

```text
X = 0
```

无论 `Z` 是多少，只要参数合法，这一点都成立。

---

## 3.4 cy

`cy` 是 Principal Point 的垂直坐标。

因为：

```text
Y = (v - cy) * Z / fy
```

所以当：

```text
v = cy
```

时：

```text
Y = 0
```

---

# 4. Projection Model

3D Camera Coordinate：

```text
(X, Y, Z)
```

投影到 Pixel Coordinate：

```text
(u, v)
```

对应：

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

从这里可以得到：

```text
u - cx = fx * X / Z
v - cy = fy * Y / Z
```

进一步：

```text
X / Z = (u - cx) / fx
Y / Z = (v - cy) / fy
```

这两个比例非常重要，因为它们实际上描述了从 Camera Center 指向该 Pixel 的射线方向。

---

# 5. Back-projection

已知：

```text
u
v
Z
fx
fy
cx
cy
```

可以反推出：

```text
X = (u - cx) * Z / fx

Y = (v - cy) * Z / fy

Z = Z
```

因此：

```text
(X, Y, Z)
=
Z *
(
    (u - cx) / fx,
    (v - cy) / fy,
    1
)
```

---

# 6. Why One Pixel Defines a Ray

对于固定 Pixel：

```text
(u, v)
```

以及固定 Camera Intrinsics：

```text
fx
fy
cx
cy
```

有：

```text
X / Z = (u - cx) / fx

Y / Z = (v - cy) / fy
```

右边全部是固定值。

所以：

```text
X/Z
Y/Z
```

也是固定的。

也就是说：

```text
(X, Y, Z)
```

只能按照同一个比例变化。

因此：

```text
Pixel
→ 确定一个方向
→ 但不能确定距离
```

这个方向就是 Camera Ray。

---

# 7. Why Depth Is Needed

只有 `(u, v)` 时：

```text
(X, Y, Z)
=
Z * ray_direction
```

其中 `Z` 仍然未知。

所以同一个 Pixel 可以对应：

```text
Z = 0.5 m
Z = 1.0 m
Z = 2.0 m
Z = 10.0 m
...
```

也就是说，同一个 Pixel 上存在无数个可能的 3D Point。

加入 Camera Z-depth 后：

```text
Z 已知
```

于是：

```text
X
Y
Z
```

全部唯一确定。

因此：

```text
Pixel
→ 决定 Ray Direction

Depth
→ 决定 Point 在 Ray 上的位置
```

这是本实验最核心的几何结论之一。

---

# 8. Z-depth vs Euclidean Range

当前 `pixel_to_camera()` 使用的是：

```text
Camera Z-depth
```

也就是：

```text
点在 Camera Z Axis 方向上的坐标
```

它并不是：

```text
Camera Center 到该点的 Euclidean Distance
```

对于：

```text
P = (X, Y, Z)
```

Euclidean Range 为：

```text
R = sqrt(X^2 + Y^2 + Z^2)
```

一般情况下：

```text
R != Z
```

只有当：

```text
X = 0
Y = 0
```

时：

```text
R = Z
```

因此：

```text
Camera Z-depth
```

和：

```text
Euclidean Range
```

是两个不同的几何量。

如果 Sensor 提供的是 Range，而函数却按照 Z-depth 使用，那么即使单位完全正确，也仍然会产生几何错误。

这属于：

```text
Semantic Error
```

而不是 Unit Error。

---

# 9. Minimal Experiment

实验参数：

```text
fx = 600
fy = 600
cx = 320
cy = 240
```

---

## 9.1 Principal Point

输入：

```text
u = 320
v = 240
Z = 1.2 m
```

因为：

```text
u = cx
v = cy
```

所以预测：

```text
X = 0
Y = 0
Z = 1.2
```

实际：

```text
camera = (0.000000, 0.000000, 1.200000)
```

结果：

```text
PASS
```

---

## 9.2 Move Pixel Right

输入：

```text
u = 380
v = 240
Z = 1.2 m
```

计算：

```text
X
= (380 - 320) * 1.2 / 600
= 60 * 1.2 / 600
= 0.12 m
```

因此预测：

```text
X = 0.12
Y = 0
Z = 1.2
```

实际：

```text
camera = (0.120000, 0.000000, 1.200000)
```

结果：

```text
PASS
```

---

## 9.3 Double Depth

固定 Pixel：

```text
u = 380
v = 240
```

将：

```text
Z = 1.2
```

改为：

```text
Z = 2.4
```

因为：

```text
X ∝ Z
```

所以预测：

```text
X:
0.12 → 0.24
```

实际：

```text
camera = (0.240000, 0.000000, 2.400000)
```

结果：

```text
PASS
```

---

## 9.4 Double fx

将：

```text
fx = 600
```

改为：

```text
fx = 1200
```

因为：

```text
X = (u - cx) * Z / fx
```

所以：

```text
fx × 2
→ X ÷ 2
```

预测：

```text
0.12 → 0.06
```

实际：

```text
camera = (0.060000, 0.000000, 1.200000)
```

结果：

```text
PASS
```

---

## 9.5 Pixel Left of Principal Point

输入：

```text
u = 260
cx = 320
```

因此：

```text
u - cx < 0
```

预测：

```text
X < 0
```

实际：

```text
X = -0.12
```

结果：

```text
PASS
```

---

# 10. Intrinsics Parameter Behavior Experiment

## 10.1 Change v

输入：

```text
u = 320
v = 300
Z = 1.2
```

因为：

```text
u = cx
```

所以：

```text
X = 0
```

而：

```text
Y
= (300 - 240) * 1.2 / 600
= 0.12
```

实际：

```text
X = 0.000000
Y = 0.120000
Z = 1.200000
```

因此：

```text
改变 v
→ 改变 Y
→ 不直接改变 X
```

---

## 10.2 Change cx

原始：

```text
u = 380
cx = 320
```

所以：

```text
u - cx = 60
X = 0.12
```

修改：

```text
cx = 350
```

则：

```text
u - cx = 30
```

因此：

```text
X = 0.06
```

实际：

```text
cx = 320 → X = 0.12
cx = 350 → X = 0.06
```

说明：

```text
cx
```

本质上改变了 Pixel 相对于 Principal Point 的横向偏移。

---

## 10.3 Change fy

固定：

```text
u = 320
v = 300
Z = 1.2
```

当：

```text
fy = 600
```

时：

```text
Y = 0.12
```

当：

```text
fy = 300
```

时：

```text
Y = 0.24
```

因此：

```text
fy ↓ 50%
→ Y × 2
```

符合：

```text
Y ∝ 1 / fy
```

---

## 10.4 Change Depth

固定：

```text
u = 380
v = 300
```

第一次：

```text
Z = 1.2

X = 0.12
Y = 0.12
```

第二次：

```text
Z = 0.6

X = 0.06
Y = 0.06
```

但是：

```text
X/Z = 0.1
Y/Z = 0.1
```

始终不变。

因此：

```text
Depth 改变
→ X/Y/Z 一起按比例缩放

但
X/Z
Y/Z
不变
```

说明：

```text
Ray Direction 没有变化
```

只是 Point 在同一条 Ray 上的位置发生变化。

---

# 11. Parameter Behavior Summary

可以总结为：

```text
fx ↑
→ |X| ↓

fy ↑
→ |Y| ↓

cx ↑
→ 固定 u 时 (u-cx) ↓
→ X 相应变化

cy ↑
→ 固定 v 时 (v-cy) ↓
→ Y 相应变化

Z ↑
→ X / Y / Z 同比例 ↑

固定 Pixel 时：
X/Z 不变
Y/Z 不变
```

因此：

```text
Intrinsics
→ 决定 Ray Direction

Depth
→ 决定沿 Ray 的尺度 / 位置
```

---

# 12. Depth Unit Debug Experiment

假设真实深度：

```text
1200 mm
```

转换为 meter：

```text
1.2 m
```

正确调用：

```text
depth_m = 1.2
```

得到：

```text
(X, Y, Z)
=
(0.12, 0.12, 1.2) m
```

如果错误地将：

```text
1200 mm
```

直接当成：

```text
1200 m
```

则：

```text
(X, Y, Z)
=
(120, 120, 1200) m
```

比例：

```text
X scale = 1000x
Y scale = 1000x
Z scale = 1000x

Euclidean distance scale = 1000x
```

实验结果：

```text
PASS
```

---

# 13. Why Unit Error Is Dangerous

这种错误危险的原因是：

```text
Pixel 没变
Intrinsics 没变
```

所以：

```text
X/Z
Y/Z
```

仍然保持正确。

这意味着：

```text
Ray Direction
```

仍然正确。

因此整个 Point Cloud 可能仍然具有：

```text
正确的方向关系
正确的大致形状
正确的相对几何结构
```

但：

```text
整个 3D Scene 的尺度错误
```

例如：

```text
真实 1.2 m
```

被解释为：

```text
1200 m
```

这种错误不一定会导致：

```text
Crash
Exception
NaN
```

程序甚至可能完全正常运行。

因此：

```text
Code Runs
!=
Geometry Correct
```

---

# 14. Error Classification

当前实验将错误分成三类。

## A. Geometry Function Can Reliably Detect

例如：

```text
fx = 0
fy <= 0
depth_m <= 0
```

这些会直接违反当前 Geometry Function 的数学契约。

因此函数可以可靠拒绝。

---

## B. Sanity Check

例如：

```text
depth > 某个业务阈值
```

这可能在特定项目中很有用。

例如：

```text
室内 RGB-D Camera
合理量程只有几米
```

那么：

```text
depth_m = 1200
```

显然非常可疑。

但是这个判断依赖：

```text
Sensor
Scene
Application
```

因此不应该作为通用几何规则写死。

---

## C. Geometry Function Cannot Know

例如：

```text
depth = 1200
```

函数不知道：

```text
1200 mm
还是
1200 m
```

再例如：

```text
u = 800
```

如果函数不知道 Image Width，就无法知道这个 Pixel 是否越界。

再例如：

```text
depth 参数实际是 Euclidean Range
```

纯数值本身无法告诉函数它的真实语义。

因此这些问题必须依赖：

```text
API Contract
Sensor Metadata
Dataset Metadata
Adapter Layer
Business Logic
```

解决。

---

# 15. Engineering API Design

最初 API：

```python
pixel_to_camera(
    u,
    v,
    depth,
    fx,
    fy,
    cx,
    cy,
)
```

其中：

```text
depth
```

没有明确单位，也没有明确语义。

因此重构为：

```python
pixel_to_camera(
    u,
    v,
    depth_m,
    fx,
    fy,
    cx,
    cy,
)
```

---

# 16. Current API Contract

当前函数规定：

```text
depth_m
=
Camera Z-depth in meters
```

输入：

```text
u / v / cx / cy
→ pixel units

fx / fy
→ focal length in pixel units

depth_m
→ Camera Z-depth in meters
```

输出：

```text
X / Y / Z
→ meters
```

---

# 17. Geometry Layer Responsibilities

Geometry Layer 负责：

```text
数学反投影

fx > 0

fy > 0

depth_m > 0
```

Geometry Layer 不负责：

```text
Sensor Raw Depth Unit Conversion

mm / cm / m Conversion

Image Width / Height Validation

Sensor Working Range

Scene-specific Max Depth

Depth Semantic Inference

Sensor Metadata Parsing
```

因此推荐架构：

```text
Sensor / Dataset
       ↓
Adapter Layer
       ↓
Raw Depth
       ↓
Unit Conversion
       ↓
depth_m
       ↓
Geometry Layer
       ↓
pixel_to_camera()
```

---

# 18. Why Use `depth_m`

相比：

```python
depth=1.2
```

使用：

```python
depth_m=1.2
```

能够把单位直接写进 API。

这不能完全阻止错误，例如用户仍然可以写：

```python
depth_m=1200
```

但它至少明确告诉调用者：

```text
这里要求 meter
```

属于一种低成本但有效的 API Contract。

---

# 19. Why Not Hard-code `depth > 100`

例如：

```python
if depth_m > 100:
    raise ValueError("depth too large")
```

不适合作为通用 Geometry Rule。

因为不同系统的合法范围完全不同：

```text
显微视觉
工业近距离视觉
室内 RGB-D
自动驾驶
遥感
天文
```

都可能具有完全不同的 Depth Range。

所以：

```text
depth_m > 100
```

不代表数学非法。

它最多只能表示：

```text
在某个具体 Sensor / Application 中可疑
```

因此更合理的设计是：

```text
Geometry Layer
→ Mathematical Validity

Sensor / Business Layer
→ Reasonable Range
```

核心原则：

```text
Geometry Layer 管“能不能算”

Sensor / Business Layer 管“这个值合不合理”
```

---

# 20. Implementation

当前核心实现：

```python
def pixel_to_camera(
    u: float,
    v: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:

    if fx <= 0:
        raise ValueError("fx must be positive.")

    if fy <= 0:
        raise ValueError("fy must be positive.")

    if depth_m <= 0:
        raise ValueError("depth_m must be positive.")

    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m

    return x, y, z
```

---

# 21. API Refactor Debug Record

本实验过程中进行了：

```text
depth
↓
depth_m
```

的 API 重构。

目的是明确：

```text
输入必须是 meter
```

但是重构后出现了两个问题。

---

# 22. Debug Bug 1 — TypeError

错误：

```text
TypeError:
pixel_to_camera() got an unexpected keyword argument 'depth_m'
```

以及：

```text
TypeError:
show_case() got an unexpected keyword argument 'depth_m'
```

## Root Cause

调用方已经从：

```python
depth=
```

改成：

```python
depth_m=
```

但：

```text
pixel_to_camera()
show_case()
```

的 Function Signature 仍然保留旧参数：

```text
depth
```

因此发生：

```text
Caller API
!=
Function Definition
```

---

# 23. Debug Bug 2 — NameError

修复 Signature 后，又出现：

```text
NameError:
name 'depth' is not defined
```

问题代码类似：

```python
f"depth_m={depth:.3f}"
```

Function Parameter 已经变成：

```python
depth_m
```

但 f-string 内部仍然引用：

```python
depth
```

因此在 Runtime 时找不到该变量。

---

# 24. Why `py_compile` Passed

在 Bug 2 中：

```text
python -m py_compile
```

仍然 PASS。

原因是：

```python
f"depth_m={depth:.3f}"
```

从 Python Syntax 的角度完全合法。

Python Compiler 只能确认：

```text
语法结构合法
```

但不能保证：

```text
运行到这里时 depth 一定存在
```

因此：

```text
Syntax Correctness
!=
Runtime Correctness
```

---

# 25. Text Replacement Is Not Semantic Refactoring

本次使用了：

```text
Replace("depth=", "depth_m=")
```

这种批量文本替换。

它可以修改：

```text
depth=
```

但不会自动理解：

```text
哪些 depth 是变量
哪些 depth 是字符串
哪些 depth 是注释
哪些 depth 是 Function Signature
```

因此：

```text
Text Replacement
!=
Semantic Refactoring
```

API 修改之后必须检查整个调用链：

```text
Function Definition
        ↓
Wrapper
        ↓
Direct Caller
        ↓
Experiments
        ↓
Tests
        ↓
Display / Logging
        ↓
Documentation
```

---

# 26. Regression Testing

API 修复完成以后重新执行：

```text
exp026_pixel_to_3d_minimal

exp026_intrinsics_parameter_behavior

exp026_depth_unit_debug
```

结果：

```text
All EXP026 minimal checks PASSED.

All EXP026 parameter behavior checks PASSED.

EXP026 depth unit debug check PASSED.
```

说明：

```text
depth → depth_m
```

重构没有继续破坏已有实验行为。

因此：

```text
Regression Test PASS
```

---

# 27. Why One Passing Test Is Not Enough

例如修改：

```text
pixel_to_camera()
```

以后，只运行最新脚本成功，并不能说明旧功能没有被破坏。

所以需要：

```text
New Test PASS
+
Old Tests Still PASS
```

才能说明修改没有引入明显 Regression。

因此：

```text
Single Experiment PASS
!=
Regression PASS
```

---

# 28. Contract Tests

创建：

```text
tests/test_pixel_to_camera.py
```

当前共：

```text
7 tests
```

测试结果：

```text
Ran 7 tests

OK
```

---

# 29. Contract Test Coverage

当前测试覆盖：

```text
1. Valid Principal Point

2. Reject fx = 0

3. Reject fy < 0

4. Reject depth_m = 0

5. Reject depth_m < 0

6. Large depth is not automatically a geometry error

7. Out-of-image pixel is not automatically a geometry error
```

---

# 30. Why `depth_m = 1200` Is Accepted

调用：

```python
pixel_to_camera(
    ...,
    depth_m=1200,
)
```

从纯几何角度：

```text
1200 m
```

并不是数学非法值。

函数无法知道它其实是不是：

```text
1200 mm
```

被调用者错误地当成：

```text
1200 m
```

因此 Geometry Function 不应该擅自拒绝。

是否合理，需要：

```text
Sensor Range
Scene
Application
```

决定。

---

# 31. Why `u = 800` Is Accepted

如果：

```text
Image Width = 640
```

那么 `u = 800` 对当前图像来说可能越界。

但是 `pixel_to_camera()` 当前 API 并没有接收：

```text
image_width
image_height
```

因此函数没有足够的信息做 Image Bounds Validation。

而且反投影公式本身对：

```text
u = 800
```

仍然是数学可计算的。

因此：

```text
Image Bounds Validation
```

属于上层职责，而不是当前纯 Geometry Function 的职责。

---

# 32. Three Levels of Correctness

通过本实验，我进一步区分了三种不同的“正确”。

## 32.1 Syntax Correctness

例如：

```text
py_compile PASS
```

只能说明：

```text
Python Syntax 合法
```

---

## 32.2 Runtime Correctness

程序真正执行：

```text
没有 NameError
没有 TypeError
没有 Runtime Exception
```

说明当前执行路径能够运行。

---

## 32.3 Regression Correctness

修改代码之后：

```text
旧实验
旧测试
新实验
新测试
```

仍然能够保持正确。

因此：

```text
py_compile PASS
!=
Runtime PASS

Runtime PASS
!=
Regression PASS
```

---

# 33. My Current Understanding

## Explain

我可以解释：

```text
Pixel Coordinate
Camera Coordinate
fx / fy
cx / cy
Camera Z-depth
Euclidean Range
Back-projection
Camera Ray
```

以及它们之间的关系。

---

## Predict

我可以在不运行代码之前预测：

```text
fx 改变
fy 改变
cx 改变
cy 改变
depth 改变
Pixel 改变
```

会如何影响：

```text
X
Y
Z
X/Z
Y/Z
```

---

## Implement

我可以实现：

```text
Pixel + Camera Intrinsics + Z-depth
→ Camera XYZ
```

即：

```python
pixel_to_camera()
```

---

## Modify

我可以主动修改：

```text
depth
→ depth_m
```

通过 API Naming 明确单位契约。

---

## Debug

我已经实际处理过：

```text
Depth Unit Error

Partial API Migration

Unexpected Keyword Argument

Runtime NameError

Stale Variable Reference

Text Replacement Side Effects
```

---

## Transfer

我可以将当前知识迁移到后续：

```text
RGB-D Depth Map
        ↓
每个有效 Pixel
        ↓
Back-projection
        ↓
3D Point
        ↓
Point Cloud
```

也能够理解为什么真实 RGB-D Pipeline 需要：

```text
Sensor Adapter
Depth Scale
Camera Intrinsics
Geometry Layer
Point Cloud Layer
```

分层处理。

---

# 34. Key Lessons

本实验目前最重要的结论：

```text
1.
Pixel 本身只确定 Ray，
不能确定唯一 3D Point。

2.
Depth 决定 Point 在 Ray 上的位置。

3.
X/Z = (u-cx)/fx
Y/Z = (v-cy)/fy

4.
Intrinsics 决定 Ray Direction。

5.
Depth 改变不会改变固定 Pixel 的 Ray Direction。

6.
Z-depth != Euclidean Range。

7.
Depth Unit Error 可能让整个 3D 世界按比例缩放，
但程序仍然完全正常运行。

8.
Geometry Layer 不能替调用者猜单位和语义。

9.
明确的 API Contract 比模糊参数名更安全。

10.
Text Replacement != Semantic Refactoring。

11.
py_compile PASS != Runtime PASS。

12.
Runtime PASS != Regression PASS。

13.
Code Runs != Geometry Correct。
```

---

# 35. Experiment Files

当前 EXP026 相关代码：

```text
src/
├── geometry/
│   └── pixel_to_camera.py
│
└── experiments/
    ├── exp026_pixel_to_3d_minimal.py
    ├── exp026_intrinsics_parameter_behavior.py
    └── exp026_depth_unit_debug.py
```

测试：

```text
tests/
└── test_pixel_to_camera.py
```

实验笔记：

```text
experiments/
└── EXP026_pixel_to_3d.md
```

---

# 36. Verification Results

## Minimal Experiment

```text
PASS
```

## Parameter Behavior Experiment

```text
PASS
```

## Prediction vs Verification

```text
PASS
```

## Depth Unit Debug Experiment

```text
PASS
```

## API Refactor Regression

```text
PASS
```

## Contract Tests

```text
Ran 7 tests
OK
```

## py_compile

```text
PASS
```

---

# 37. Current EXP026 Status

```text
Minimum Theory               PASS
Prediction Round 1           PASS
Minimal Implementation       PASS
Minimal Experiment           PASS
Parameter Prediction         PASS
Active Modification          PASS
Prediction vs Verification   PASS
Depth Unit Debug             PASS
Engineering Guardrails       PASS
API Hardening                PASS
API Refactor Debug           PASS
Regression Test              PASS
Invalid Input Test           PASS
Contract Test                PASS
Notes                        DONE

Acceptance Test              PENDING
Git Commit                   PENDING
GitHub Push                  PENDING
```

Therefore:

```text
EXP026                       CONTINUE
Level 2                      CONTINUE
```

EXP026 还没有正式通关。

剩余闭环：

```text
Acceptance Test
↓
Fix Weak Points If Any
↓
Final Verification
↓
Git Status
↓
Git Commit
↓
GitHub Push
↓
Verify Remote
↓
EXP026 COMPLETE
```

---

# 38. Acceptance Test

EXP026 Acceptance Test：**PASS**

本轮验收覆盖：

```text
Explain
Predict
Calculate
Derive
Implement
Modify
Debug
Design
Transfer
```

---

## 38.1 Q1 — Pixel / Ray / Depth

**PASS**

一个 Pixel `(u, v)` 与 Camera Intrinsics 可以确定：

```text
X / Z
Y / Z
```

因此它确定的是一条从 Camera Center 出发的 Ray Direction，而不是唯一的 3D Point。

关系为：

```text
Pixel + Intrinsics
→ Ray Direction

Depth
→ 确定该 Ray 上的具体 3D Point
```

只有：

```text
Pixel + Intrinsics + Depth
```

才能得到唯一：

```text
(X, Y, Z)
```

---

## 38.2 Q2 — Camera Intrinsics

**PASS**

当前理解：

```text
fx
→ x 方向焦距，单位为 pixel

fy
→ y 方向焦距，单位为 pixel

cx
→ Principal Point 的 x 坐标

cy
→ Principal Point 的 y 坐标
```

反投影：

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

因此：

```text
fx 增大
→ 在其他参数不变时 |X| 减小

fy 增大
→ 在其他参数不变时 |Y| 减小
```

---

## 38.3 Q3 — Z-depth vs Euclidean Range

**PASS**

给定：

```text
P = (0.3, 0.4, 1.2) m
```

Camera Z-depth：

```text
Z-depth = 1.2 m
```

Euclidean Range：

```text
R
= sqrt(X^2 + Y^2 + Z^2)
= sqrt(0.3^2 + 0.4^2 + 1.2^2)
= sqrt(1.69)
= 1.3 m
```

因此：

```text
Z-depth = 1.2 m
Euclidean range = 1.3 m
```

一般情况下：

```text
Z-depth != Euclidean range
```

---

## 38.4 Q4 — Independent Back-projection

**PASS**

给定：

```text
fx = 800
fy = 600
cx = 320
cy = 240

u = 400
v = 300
Z = 1.5 m
```

计算：

```text
u - cx = 80
v - cy = 60
```

所以：

```text
X
= 80 * 1.5 / 800
= 0.15 m

Y
= 60 * 1.5 / 600
= 0.15 m

Z
= 1.5 m
```

归一化比例：

```text
X / Z = 0.1
Y / Z = 0.1
```

---

## 38.5 Q5 — Depth Behavior

**PASS**

当：

```text
Z:
1.5 → 3.0
```

固定 Pixel 和 Intrinsics 时：

```text
X:
0.15 → 0.30

Y:
0.15 → 0.30

Z:
1.50 → 3.00
```

但：

```text
X / Z = 0.1
Y / Z = 0.1
```

保持不变。

因此：

```text
Depth 改变
→ Point 沿同一条 Camera Ray 移动

Ray Direction 不变
```

---

## 38.6 Q6 — fx Behavior

**PASS**

恢复：

```text
Z = 1.5 m
```

将：

```text
fx:
800 → 400
```

则：

```text
X:
0.15 → 0.30
```

因为：

```text
X ∝ 1 / fx
```

而：

```text
Y
```

不会发生变化，因为：

```text
Y = (v - cy) * Z / fy
```

不依赖 `fx`。

---

## 38.7 Q7 — Matrix Form

**PASS**

Pixel 齐次坐标：

```text
p =
[u, v, 1]^T
```

Camera Intrinsics：

```text
K =
[ fx   0   cx ]
[  0  fy   cy ]
[  0   0    1 ]
```

则：

```text
K^-1 p
=
[
    (u - cx) / fx,
    (v - cy) / fy,
    1
]^T
```

其几何意义是：

```text
去除 Camera Intrinsics 后，
得到 Pixel 对应的归一化 Camera Ray Direction。
```

因此：

```text
P_camera = Z * K^-1 * p
```

得到：

```text
[
    X,
    Y,
    Z
]^T
=
[
    (u - cx) * Z / fx,
    (v - cy) * Z / fy,
    Z
]^T
```

这与标量形式完全一致：

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

### Precision Note

这里的：

```text
Z
```

是：

```text
Camera Z-depth
```

不是沿单位 Ray 的 Euclidean Distance。

因为：

```text
K^-1 p
```

一般并不是 Unit Vector。

更准确地说：

```text
Z 将该 Ray Representation 缩放，
使结果 Point 的第三个坐标等于目标 Camera Z-depth。
```

---

## 38.8 Q8 — Raw Depth Scale Debug

**PASS**

给定：

```text
raw_depth = 1500
depth_scale = 0.001 m / unit
```

正确：

```text
depth_m
= 1500 * 0.001
= 1.5 m
```

如果错误调用：

```text
depth_m = 1500
```

则相对于正确结果：

```text
X × 1000
Y × 1000
Z × 1000
```

这是因为 Back-projection 中：

```text
X ∝ Z
Y ∝ Z
Z ∝ Z
```

Geometry Function 不会自动发现这个错误，因为：

```text
1500 > 0
```

在纯数学上仍然是合法输入。

当前 Geometry Contract 检查的是：

```text
fx > 0
fy > 0
depth_m > 0
```

它并不知道 Raw Sensor Unit 或 `depth_scale`。

---

## 38.9 Q9 — Syntax / Runtime / Regression

**PASS**

### py_compile PASS

说明：

```text
Python Syntax 合法
```

但不能保证：

```text
运行时变量一定存在
逻辑一定正确
历史功能没有被破坏
```

### Runtime PASS

说明：

```text
当前输入和当前执行路径
能够实际运行
```

但不能保证所有情况正确。

### Regression PASS

说明：

```text
代码修改以后，
之前已经验证过的行为仍然通过已有 Tests / Experiments。
```

因此：

```text
py_compile PASS
!=
Runtime PASS
!=
Regression PASS
```

---

## 38.10 Q10 — Engineering Boundaries

**PASS**

职责划分如下：

| Check / Operation | Responsible Layer |
|---|---|
| `fx <= 0` | Geometry Layer |
| `fy <= 0` | Geometry Layer |
| `depth_m <= 0` | Geometry Layer |
| `(u,v,Z) → (X,Y,Z)` | Geometry Layer |
| Raw Depth × Depth Scale | Sensor / Adapter Layer |
| Pixel 是否超出当前 Image Bounds | Sensor / Adapter Layer |
| Depth 是否超过 Sensor 有效量程 | Sensor / Adapter Layer |
| Application-specific Threshold | Business Layer |

核心原则：

```text
Geometry Layer
→ 数学几何与基本输入契约

Sensor / Adapter Layer
→ Sensor-specific Unit / Metadata / Range

Business Layer
→ Application-specific Rules
```

不应该把所有逻辑全部塞入：

```text
pixel_to_camera()
```

否则会产生：

```text
设备耦合
业务耦合
复用困难
测试困难
隐藏假设增加
```

---

## 38.11 Q11 — Depth Map to Point Cloud

**PASS**

对于一张：

```text
H × W
```

Depth Map：

对每个有效 Pixel：

```text
(u, v)
```

执行：

```text
1. 读取 Raw Depth

2. 根据 Sensor Depth Scale
   转换为 meter

3. 得到 Camera Z-depth

4. 检查 Depth 是否有效

5. 使用 Camera Intrinsics：

   X = (u - cx) * Z / fx
   Y = (v - cy) * Z / fy

6. 得到：

   (X, Y, Z)

7. 将所有有效 3D Points 收集起来
```

无效 Depth，例如：

```text
0
NaN
Inf
Negative Value
Sensor-invalid Value
```

应该：

```text
跳过
或显式标记 Invalid
```

而不是直接参与 Back-projection。

因此：

```text
Depth Map
↓
Valid Pixel + Z-depth
↓
Back-projection
↓
3D Points
↓
Point Cloud
```

如果 RGB 与 Depth 已正确对齐，还可以：

```text
Pixel RGB
↓
附加到对应 3D Point
↓
Colored Point Cloud
```

---

# 39. Acceptance Result

最终能力验收：

```text
Explain                      PASS
Predict                      PASS
Calculate                    PASS
Derive                       PASS
Implement                    PASS
Modify                       PASS
Debug                        PASS
Design                       PASS
Transfer                     PASS
```

因此：

```text
EXP026 Acceptance Test       PASS
```

---

# 40. Precision Corrections

本次 Acceptance Test 有两个表述需要保持严格。

## 40.1 Z Is Not Ray Euclidean Distance

不要说：

```text
乘以 Z，就是沿 Ray 走 Z 米
```

更准确：

```text
P_camera = Z * K^-1 * p
```

中的 `Z` 是 Camera Z-depth。

它把：

```text
K^-1 p
```

按比例缩放，使最终 Point 的第三维：

```text
Z_coordinate = Z
```

只有使用单位 Ray Vector 时，乘上的标量才直接对应 Euclidean Distance。

---

## 40.2 Current Geometry Validation

当前：

```text
pixel_to_camera()
```

检查：

```text
fx > 0
fy > 0
depth_m > 0
```

而不是只检查：

```text
fx != 0
depth > 0
```

API Contract 应与实际 Implementation 保持一致。

---

# 41. EXP026 Final Learning Status

当前：

```text
Minimum Theory               PASS
Prediction Round 1           PASS
Minimal Implementation       PASS
Minimal Experiment           PASS
Parameter Prediction         PASS
Active Modification          PASS
Prediction vs Verification   PASS
Depth Unit Debug             PASS
Engineering Guardrails       PASS
API Hardening                PASS
API Refactor Debug           PASS
Regression Test              PASS
Invalid Input Test           PASS
Contract Test                PASS
Notes                        DONE
Acceptance Test              PASS
```

尚未完成：

```text
Final Verification           PASS
Git Commit                   PENDING
GitHub Push                  PENDING
```

因此：

```text
EXP026                       CONTINUE
Level 2                      CONTINUE
```

---

# 42. Remaining Closure Steps

EXP026 剩余闭环：

```text
Final Verification
↓
Git Status Review
↓
Git Diff Check
↓
Git Commit
↓
GitHub Push
↓
Remote Verification
↓
EXP026 COMPLETE
```








---

# 43. Closure Record

EXP026 engineering closure completed.

```text
Final Verification           PASS

Git Commit                   PASS
GitHub Push                  PASS
Remote Verification          PASS