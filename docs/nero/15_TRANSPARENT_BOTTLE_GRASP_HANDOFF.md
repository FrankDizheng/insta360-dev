# NERO 透明瓶抓取协作交接

本文记录 2026-05-19 至 2026-05-20 的真机抓取进展，供远程协作者继续定位和优化。

## 当前结论

- 机械臂仿真已切换到官方 NERO URDF 链路，不再使用旧 DH 近似。
- FK 输出的 `flange_T` 已按实测 J7 sweep 对齐到 SDK `get_flange_pose()` 的 flange frame，即官方 URDF 的 gripper joint origin。
- 历史 16 个 FK/SDK 样本用官方 URDF 重新对齐后，SDK flange RMS 已降到毫米级，可用于本地仿真和关节空间候选生成。
- 透明瓶深度不可靠。当前策略是：`Z` 固定使用成功高度先验 `260 mm`，`XY` 由 RGB 关键点/瓶身轴线投影到固定 Z 平面推算。
- 成功抓取经验表明，抓取前必须先重新扫描，不能复用旧目标；瓶子被移动后旧 XY 会直接导致夹空。

## 已固化代码

### URDF FK / SDK flange 对齐

文件：`nero/kinematics.py`

关键点：

- 使用 AgileX 官方 NERO URDF joint chain。
- 暴露 `link7_T`、`gripper_flange_T`、`gripper_base_T`、`flange_T` 等 debug frame。
- `flange_T` 明确表示 SDK flange frame，而不是旧 TCP 或旧 DH flange。
- `DEFAULT_TOOL_TCP_OFFSET_X_M = 0.0`，上层抓取如需 fingertip/contact offset，必须显式传入。

测试：

- `tests/test_nero_kinematics.py`
- 覆盖 zero-pose SDK flange 位置、J7 sweep 半径、debug frame 暴露。

### 透明瓶定位和抓取先验

文件：`calibration/scripts/pick_place_client.py`

当前默认：

- `PICK_PLACE_PICK_LOCALIZATION_MODE=red-cap-model`
- `PICK_PLACE_PICK_XY_MODE=fixed-z-plane`
- `PICK_PLACE_PICK_GRASP_Z_MODE=successful-prior`
- `PICK_PLACE_SUCCESSFUL_GRASP_Z_PRIOR_MM=260.0`

定位语义：

1. VLM/RGB 找红盖点与瓶身方向关键点。
2. 从 `cap_center` 沿 `cap -> body_tail/body_center` 方向偏移 `BOTTLE_GRASP_FROM_CAP_MM`。
3. 将该像素射线投影到 base-frame `Z=260 mm` 平面。
4. 得到瓶身中段的 `XY`，而不是红盖点本身的 `XY`。

注意：如果 VLM 输出异常（例如 normalized 坐标为负数，或把 `cap_center` 和 `body_tail_center` 返回成同一点），必须拒绝该结果或改用轮廓/人工校验，不应继续真机抓取。

## 当前真机经验

### 成功经验

- 第一次成功抓取的物理高度约 `266 mm`，后续将固定高度先验下调到 `260 mm` 做测试。
- 夹爪闭合后宽度约 `50-53 mm` 通常表示夹爪被瓶身撑住，疑似夹住瓶体。
- `Z=260 mm` 不应单独判定成功或失败；真正关键是 `XY` 是否落在瓶身中段。

### 当前已确认的问题

1. 透明瓶深度会污染 `XY`

   旧逻辑用透明瓶深度反投影 `px_to_base(u, v, d)`。当深度错误时，`X/Y/Z` 都会错。固定 `Z` 后仍不能继续使用透明瓶深度计算 `XY`。

2. 红盖点不能直接作为抓取点

   失败案例中目标点仍在红盖/瓶盖附近，导致夹爪夹到瓶盖上方。当前应使用瓶身轴线偏移后的瓶身中段点。

3. J7 姿态必须在预停留阶段完成

   成功案例使用接近 V2 的 wrist 姿态，尤其 `J5/J6/J7` 近似固定：

   ```text
   J5/J6/J7 ~= [-1.003, -12.612, 76.193] deg
   ```

   失败案例中，预停留位置 `J7` 未到位，随后在靠近瓶子时才从约 `51 deg` 转到 `76 deg`。这会让夹爪横向扫过瓶子，导致接触点从瓶身偏到瓶盖。

   执行门禁：

   - 预停留位置必须确认 `J5/J6/J7` 到位。
   - `J7` 未到位时必须重发预停留或中止。
   - 不允许在 `move_to_pregrasp.ok=false` 后继续下降抓取。
   - 最后接近瓶子时，应保持 wrist 姿态，只让本地 URDF 规划出的路点逐步降低 Z。

4. 接近瓶子时不能提前触碰

   本地仿真允许 `J1-J4` 求解，但 `J5/J6/J7` 应固定在成功抓取姿态附近。预停留点需要保证夹爪已在正确 wrist 姿态下位于瓶子旁/上方，并且不会碰到瓶子；之后再沿锁腕路点下降到 `260 mm`。

## 当前相机姿态

当前默认扫描姿态已更新为现场认为更合适的相机角度：

```text
SCAN_POSE_DEG = [7.277, -36.26, -14.662, 71.399, -8.695, 0.318, 97.698]
```

记录文件：

```text
calibration/results/current_default_scan_pose_2026-05-20_1605.json
```

最近一张当前姿态图：

```text
calibration/results/current_camera_view_2026-05-20_160348/scan_color.jpg
```

离线人工校验标注：

```text
calibration/results/current_camera_view_2026-05-20_160348/manual_fixed_z260_scene_analysis.jpg
```

该人工校验认为绿色 `PICK Z260` 点适合夹取，但 VLM 对该图曾返回负 normalized 坐标，说明自动 VLM 关键点仍需增强或增加轮廓 fallback。

## 红色盒子 / 白色凹槽

当前代码支持 VLM 返回 `empty_slot.corners`，并用四角求槽中心线：

- `top_left`
- `top_right`
- `bottom_right`
- `bottom_left`

在最近人工分析中，白色凹槽仍有一定偏差，不能直接认为已达到可放置精度。建议远程协作者继续：

- 增强白色凹槽四角识别提示。
- 增加基于白色区域/红盒几何的 CV fallback。
- 将 slot centerline 与瓶身轴线方向统一到 base frame 后，再做放置规划。

## 推荐下一步

1. 在 `pick_place_client.py` 中给 VLM 输出增加强校验：
   - normalized 坐标必须在 `[0,1]`。
   - `cap_center` 与 `body_tail_center` 距离必须超过阈值。
   - `body_center` 不得落在 cap 附近。
2. 增加黑色背景下的瓶身轮廓 fallback，减少对 VLM body keypoints 的依赖。
3. 将“锁定 `J5/J6/J7`，只优化 `J1-J4`，生成多 Z 路点”的逻辑从临时实验脚本固化为可复用函数。
4. 真机执行前强制门禁：
   - 每次抓取前重新扫描。
   - 预停留 wrist 到位。
   - 预停留不会接触瓶子。
   - 下降路径的每个路点都由本地 URDF 验证。

