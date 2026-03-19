# NERO 控制层设计

## 1. 控制层目标

给平台提供一个统一的 NERO 执行接口，屏蔽底层到底是 Python SDK 还是 ROS2。

上层只关心这些动作：

- connect
- get_status
- enable
- disable
- move_home
- move_j
- move_p
- open_gripper
- close_gripper
- stop

## 2. 为什么先做控制抽象

本仓库已经有一个假的 `RobotController` 骨架，在 [hardware_agent.py](/Users/mico/Projects/insta360-dev/hardware_agent.py) 里现在只是模拟执行。

MVP 最应该做的不是重写整个平台，而是把这个抽象替换成 NERO 真机控制器。

## 3. 推荐的实现结构

建议新增一个独立目录，例如：

- `nero/driver.py`
- `nero/controller.py`
- `nero/gripper.py`
- `nero/types.py`

### driver.py

职责：

- 封装 `pyAgxArm` 或 ROS2 调用
- 管理连接、断连、错误状态

### controller.py

职责：

- 提供平台统一动作接口
- 做参数检查、安全边界和动作完成判定

### gripper.py

职责：

- 夹爪开口宽度与夹持力配置
- 把平台动作映射成官方接口

### types.py

职责：

- 定义状态类型、位姿类型、动作结果类型

## 4. 推荐优先级

### 第一优先级

- `connect()`
- `get_joint_states()`
- `enable()`
- `move_home()`
- `move_j()`
- `stop()`

### 第二优先级

- `move_p()`
- `get_tcp_pose()`
- `open_gripper()`
- `close_gripper()`

### 第三优先级

- 轨迹
- 示例教模式集成
- 更复杂状态订阅

## 5. 与官方接口的映射

根据官方 `agx_arm_ros`：

- 反馈话题：`/feedback/joint_states`、`/feedback/tcp_pose`、`/feedback/arm_status`
- 控制话题：`/control/move_j`、`/control/move_p`、`/control/joint_states`
- 服务：`/enable_agx_arm`、`/move_home`

根据官方 `pyAgxArm`：

- NERO 支持通过 CAN 建立配置并读取关节状态
- 官方仓库提供 NERO API 文档和首次使用 CAN 指南

## 6. MVP 的动作策略

MVP 阶段不要让 VLM 直接输出连续轨迹。

应该让控制层执行固定模板动作：

1. `move_p(pre_grasp_pose)`
2. `move_p(grasp_pose)`
3. `close_gripper(force)`
4. `move_p(lift_pose)`
5. `move_p(pre_place_pose)`
6. `move_p(place_pose)`
7. `open_gripper()`
8. `move_p(safe_pose)`

这样一来：

- VLM 只管告诉你抓哪个、放哪
- 安全与运动逻辑在本地可控

## 7. 安全要求

根据手册和官方驱动，控制层必须带这些保护：

- 启动前检查使能状态
- 动作前检查错误状态
- 动作前检查安装姿态配置是否正确
- 设置安全位与 home 位
- 动作超时自动 stop
- 发生碰撞 / 无解 / 超限时立即中止后续流程

## 8. 与现有仓库的对接建议

优先改造：

- [hardware_agent.py](/Users/mico/Projects/insta360-dev/hardware_agent.py)

做法：

- 保留主循环与服务调用方式
- 把内部模拟 `RobotController` 替换成 `NeroController`

这样可以最大化复用你们现有的 demo 骨架。
