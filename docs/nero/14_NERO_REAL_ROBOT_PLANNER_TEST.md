# NERO 真机 Planner 测试交接说明

## 当前测试目标

本轮不使用 learned policy 直接控制真机。当前目标是：

1. 从实验电脑读取当前 NERO 关节角。
2. 使用相机/RGBD 给出的目标点。
3. 在仿真侧用传统 `plan_tcp_motion()` 做 IK / staged planner。
4. 先验证 standoff 轨迹，再考虑低速真机执行。

## 最新交换数据

远端分支：

```text
origin/planner-exchange-2026-05-11
```

最新文件：

```text
calibration/results/nero_sim_current_to_bottle_exchange_2026-05-11.json
```

关键字段：

```json
{
  "coordinate_frame": "robot_base",
  "initial_state": {
    "q_deg": [-15.035, -15.774, 2.132, 85.265, -2.178, 1.918, 51.456],
    "q_rad": [-0.26241025, -0.27530824, 0.03721042, 1.48815499, -0.03801327, 0.03347542, 0.89807662],
    "flange_pose_m_rad": [-0.309167, 0.07723, 0.441148, 1.6103454876, -0.5383293545, -0.2916096114]
  },
  "target": {
    "semantic": "outer_bottle_cap",
    "xyz_m": [-0.3634054438, 0.08940719, 0.2846228182],
    "recommended_grasp_z_m": 0.2746228182
  },
  "table_z_m": 0.25,
  "planner_constraints": {
    "safe_z_m": 0.42,
    "table_clearance_min_m": 0.03
  }
}
```

## 本地仿真规划结果

使用当前本地 `nero.planning.plan_tcp_motion()`，从 `initial_state.q_rad` 规划到瓶盖目标点：

```text
current_tcp_xyz_m = [-0.193335, 0.383853, 0.168190]
target cap xyz    = [-0.363405, 0.089407, 0.284623]

plan ok           = True
final error       = 2.238 mm
path waypoints    = 26
goal_q_rad        = [-0.830083, 1.036197, 0.298190, 1.850549, -0.093877, -0.177993, 0.291727]
```

规划到安全 standoff：

```text
target_standoff_xyz_m = [-0.363405, 0.089407, 0.42]

plan ok               = True
final error           = 2.226 mm
path waypoints        = 26
goal_q_rad            = [0.197549, 0.320241, 0.666955, 1.640139, -0.003175, -0.254427, 0.183548]
```

## 关键安全发现

当前 FK/TCP 与真机场景还没有完全对齐：

```text
table_z_m       = 0.25
FK current TCP z = 0.168190
```

这表示按当前仿真 TCP 模型，末端 TCP 已经低于桌面约 81.8 mm。结合现场图片判断，这不符合真实物理场景，所以不能直接用当前 TCP z 做真机安全判据。

高概率原因：

- 仿真里的 TCP offset / 方向不适用于当前相机 + 夹爪装配。
- 当前 FK TCP 与实验电脑 SDK 读到的 flange pose 不是同一个点。
- `table_z_m` 是桌面高度，但当前 FK 模型下的 TCP 点可能落在工具几何之外或方向错误。

因此，真机前必须先确认：

1. FK flange 是否能匹配 SDK 读到的 flange pose。
2. 如果 flange 匹配，再重新标定 `flange -> actual TCP/tool tip`。
3. 不能把 `observed_flange_pose_m_rad` 当作相机观测 TCP。

## 当前允许测试范围

在 TCP/table 对齐完成前，只建议做：

```text
current_q -> target_standoff_xyz_m
```

其中：

```text
target_standoff_xyz_m = [target.x, target.y, 0.42]
```

不要直接下降到：

```text
target.xyz_m
recommended_grasp_z_m
```

原因：

```text
recommended_grasp_z_m = 0.2746228182
table_z_m + 0.03     = 0.28
```

推荐抓取高度比“桌面 + 30 mm clearance”低约 5.4 mm，不满足当前保守安全规则。

## 实验电脑建议流程

1. 读取当前关节角和 SDK flange pose。
2. 用同一组 `q_rad` 在仿真里计算 FK flange / TCP。
3. 对比 SDK flange 与 FK flange。
4. 如果 flange 误差大，先修 DH / 关节零点 / 关节方向。
5. 如果 flange 误差小但 TCP 不对，修 `flange -> TCP` offset。
6. 只规划到 standoff 点，先不下降。
7. 真机执行时低速、分段、人工确认。

## 禁止事项

- 不要运行 learned direct-control policy 上真机。
- 不要在 TCP/table 对齐前执行下降抓取。
- 不要把 `recommended_grasp_z_m` 当作立即可执行的真机目标。
- 不要跳过 FK/SDK flange sanity check。

## 需要实验电脑回传的数据

下一轮请回传：

```json
{
  "current_q_rad": [0, 0, 0, 0, 0, 0, 0],
  "sdk_flange_pose_m_rad": [0, 0, 0, 0, 0, 0],
  "observed_tool_tip_xyz_m": [0, 0, 0],
  "target_xyz_m": [0, 0, 0],
  "table_z_m": 0.25,
  "notes": ""
}
```

如果暂时无法识别 tool tip，也请至少回传 SDK flange pose 和当前关节角，先做 flange 级别对齐。
