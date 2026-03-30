# 标定现状总结

## 1. 当前结论

- 手眼标定已经完成，当前结果可以用于"看到标定板后输出目标点在 `base` 坐标系下的位置"。
- TCP 标定已经完成数值求解，并于 2026-03-29 完成了物理接触闭环验证，接触误差约 `0.24 mm`。
- 标定全流程（手眼 + TCP）可以认为已经收口。

## 2. 手眼标定状态

当前已保存的核心结果：

- `handeye_result.json`

从现有结果看：

- 总采样数 `15`
- 有效视觉检测样本 `14`
- 实际用于手眼求解样本 `14`
- 相机重投影误差约 `0.5687`

当前判断：

- 已经有可复现的 `flange_T_camera`
- 已经做过成功的纯视觉验证
- 可以稳定输出标定板原点和指定板上点在 `base` 下的位置

因此，按当前项目阶段的目标，手眼标定可以认为已经完成。

## 3. TCP 标定状态

当前推荐使用的 TCP 结果：

- `gripper_tcp_left_front_tip_samples_004_006.json`

该结果特征：

- 工具点定义：`gripper_left_front_tip`
- 使用样本数：`3`
- 样本：`tcp_sample_004` 到 `tcp_sample_006`
- 残差 RMSE 约 `1.27 mm`

历史结果不再推荐：

- `gripper_tcp_left_front_tip.json`，RMSE 约 `8.21 mm`
- `gripper_tcp.json`，RMSE 约 `8.44 mm`

### TCP 物理接触验证（2026-03-29 完成）

验证方式：

- 固定板位姿
- 先抬高 `100 mm` 脱离
- `move_p` 到 `5 mm` approach 位姿
- `move_l` 直线下探到 `3 mm` pretouch
- `move_l` 直线下探到板面目标点（`board_xy 0.075 0.06`）

验证结果（`touch_plan_20260329_144552.json`）：

- 模式：`target_executed`
- 位置误差 X：`-0.13 mm`
- 位置误差 Y：`+0.07 mm`
- 位置误差 Z：`+0.19 mm`
- **总位置误差：`0.24 mm`**

结论：TCP 物理验证通过，接触精度远优于标定残差（`1.27 mm`）。

### 深度融合定位验证（2026-03-29 完成）

发现纯 PnP 在不同相机高度下 Z 方向估计会漂移数毫米，导致近距离接触时出现挤压。
引入 Gemini 335 深度流进行 Z 修正后，定位精度显著提升：

验证方式：

- 从不同起始高度（相机 300-340mm）检测板面
- PnP 提供姿态和 XY，深度流对 Z 做中值修正（典型修正量 2-3mm）
- `move_p` 到 50mm 中间安全点，`move_l` 到 10mm 最终悬停

验证结果（两次独立测试）：

| 测试 | 相机高度 | 深度修正 | 10mm 悬停实测误差 |
|------|---------|---------|-----------------|
| #1 | 338mm | +2.3mm | 0.1mm |
| #2 | 298mm | +1.8mm | 0.1mm |

结论：深度融合策略稳定、可重复，悬停精度 0.1mm，可直接用于后续抓取任务。

## 4. 近距离接近实验暴露的问题（历史记录）

早期接近测试中，确认的主要问题有：

- 板位姿变化后，之前测出来的 `4-6 mm` 安全边界不能直接沿用
- `move_p` 是点到点运动，终点看似安全，不代表中间轨迹不会先碰到板
- `3-5 mm` 的间隙已经接近系统总误差预算，不适合作为默认安全距离
- 纯 PnP 在不同视角下 Z 估计存在 2-5mm 漂移

最终验证采用了推荐方案：先 lift、再 `move_p` approach、最后 `move_l` 沿法向直线下探，并引入深度流 Z 修正，成功解决了上述问题。

## 5. 当前产物位置

标定脚本和结果文件已归档到本仓库：

脚本目录：`calibration/scripts/`

- `handeye_board_runtime.py` — 核心运行时：board 检测、坐标变换、相机管理、机器人连接、深度融合定位
- `solve_handeye_charuco.py` — 手眼求解
- `nero_handeye_capture.py` — 手眼采样采集
- `validate_handeye_charuco.py` — 手眼验证
- `locate_board_target.py` — 板上目标点定位
- `calibrate_gripper_tcp.py` — TCP 标定（采样 + 求解）
- `touch_board_target.py` — TCP 物理接触验证
- `pi_run_touch_3mm_serial.py` — 批量 touch 运行器
- `pi_camera_preview_server.py` — 相机预览服务
- `test_depth_fused_approach.py` — 深度融合定位 + 安全悬停验证

结果目录：`calibration/results/session1/`

- `handeye_result.json` — 手眼标定结果
- `gripper_tcp_left_front_tip_samples_004_006.json` — 推荐 TCP 结果
- `touch_plan_20260329_144552.json` — TCP 物理验证通过记录
- `safe_reverse_5mm_pose.json` — 安全姿态参考
- `gripper_tcp_left_front_tip.json` — 历史 TCP 结果（不推荐）
- `gripper_tcp.json` — 历史 TCP 结果（不推荐）
