# Level 2 Final｜故障案例审计索引

> 建议存放路径：`failure_cases/Level2_final_failures.md`  
> 对应配置：`configs/level2_final_cases.json`  
> 完整机读案例记录：`outputs/level2_final/failure_cases.json`  
> 环境：Windows / Python 3.11.17 / Open3D 0.20.0 / pytest 9.1.1  
> 证据：2026-10-29 本机实测 `Cases: 7`；2 个诊断、5 个拒绝；全部符合预期。

| Case | 故障注入 | 预期/实测状态 | 下游行为 |
| --- | --- | --- | --- |
| L2F2_001 | 正常同帧合成示例 | `DIAGNOSTIC_ONLY` | 可以生成演示几何；不发布 |
| L2F2_002 | 用帧 B 的 UID 消费帧 A 的验证快照 | `FRAME_UID_MISMATCH` | 拒绝后续坐标计算 |
| L2F2_003 | UID 不变但深度被改变 | `INPUT_DATA_HASH_MISMATCH` | 拒绝内容不一致的输入 |
| L2F2_004 | 票据时效超过截止时间 | `STALE_VALIDATION_TICKET` | 拒绝过期快照 |
| L2F2_005 | 目标/背景各交换一个点，点数不变 | `UPSTREAM_REJECTED` | 拒绝不可信目标归属 |
| L2F2_006 | 把逆向外参用于正向变换 | `FRAME_MISMATCH` | 拒绝变换方向不匹配 |
| L2F2_007 | 仅把标定声明设为 True | `DIAGNOSTIC_ONLY` | 仍无机器人发布授权 |

**全局安全不变量**：七项案例 `publishable=false`、`position_m=null`、`physical_accuracy_claimed=false`；拒绝案例中世界演示质心/AABB 不得提供可消费数值。

**复现（仓库根目录 PowerShell）**：

```powershell
python -m src.pipeline.os01_final2_failure_audit
python -m pytest tests/test_os01_final2_failure_audit.py -v
Get-Content .\outputs\level2_final\failure_cases.json -Encoding UTF8
```

**解释边界**：`All cases passed=True` 表示每个注入场景获得了期望的安全结果，不是所有输入都可定位。`input_sha256` 只能提供当前同进程示例的数据完整性检测，不能证明票据颁发者可信。没有真实采集、独立物理真值、标定签名或机器人执行许可。

**来源**：该索引由最终 Boss 验收阶段整理；配置及完整机器输出均需以本机执行的 JSON 为准，本文不凭空指定哈希值或实际捕获时间戳。
