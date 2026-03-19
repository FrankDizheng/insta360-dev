# NERO 编码任务清单

## P0: 到货前就可以开始

当前默认前提：

- 用 MacBook 开发
- 今天不依赖官方 SDK 本地落地
- 先把抽象层和服务链路打通

另外默认把“人工代替大模型”视为正式工作流的一部分，而不是临时凑合：

- 人工先做决策/纠偏
- 系统负责记录 case
- 后续再把这些 case 批量喂给真正的大模型或策略模块

### 1. 把硬件代理从模拟控制改造成可插拔控制器

目标文件：

- [hardware_agent.py](/Users/mico/Projects/insta360-dev/hardware_agent.py)

要做的事：

- 提取 `BaseRobotController` 接口
- 新增 `MockRobotController`
- 预留 `NeroRobotController`

### 2. 把 VLM 服务从“排序 demo”改成通用动作决策服务

目标文件：

- [api_server.py](/Users/mico/Projects/insta360-dev/api_server.py)

要做的事：

- 去掉 sorting 语义写死
- 改成 task-driven prompt
- 统一 action schema
- 增加原图和响应落盘

### 3. 把模型调用层改成支持远程后端

目标文件：

- [bridge/model_feed.py](/Users/mico/Projects/insta360-dev/bridge/model_feed.py)

要做的事：

- 支持配置 base URL
- 支持云端 HTTP 接口
- 支持远程 OpenAI-compatible / Ollama-compatible 后端

### 4. 建立 MVP 配置文件

建议新增：

- `config/nero/dev.yaml`
- `config/nero/demo.yaml`

配置内容：

- 相机源
- 机械臂连接参数
- VLM 服务地址
- 安全位 / home 位 / 预抓取位 / 放置位

### 4.1 增加 MacBook 开发模式

目标：

- 让代码在没有 NERO SDK、没有真机的情况下也能运行主流程
- 支持本地图片目录或普通摄像头输入
- 支持远程 VLM 地址配置

建议增加：

- `mock_robot: true`
- `camera_mode: image_dir | webcam | vendor_camera`
- `vlm_backend: ollama | openai_compatible | custom_http`

### 4.2 增加 case 采集与人工纠偏能力

目标：

- 每次图像输入、模型输出、执行结果都可落盘
- 支持人工后修正 action
- 为未来 batch analysis 和策略沉淀做准备

建议增加：

- `data/cases/<session_id>/case_xxx/`
- `log.json`
- `correction.json`

## P1: 到货当天做

### 5. 接入 NERO Python SDK

目标：

- 建立最小 `NeroRobotController`
- 打通 connect / enable / move_home / move_j / move_p

这一项可以明确后置，不是今天的阻塞项。

### 6. 完成夹爪控制

目标：

- open / close
- 力度与开口宽度参数可调

### 7. 增加状态可视化

目标：

- 当前关节状态
- 当前 TCP
- 当前错误状态
- 当前执行阶段

## P2: 到货后 1-2 天内完成

### 8. 相机适配器

目标：

- USB 相机抓图
- 单帧保存
- 工作区裁剪

### 9. 图像坐标到平面坐标映射

目标：

- 最少 4 点标定
- 输出桌面平面坐标

### 10. 固定模板抓放

目标：

- 不接 VLM 也能跑一次稳定抓放

## P3: MVP 串联

### 11. 任务编排器

目标：

- 输入任务描述
- 抓图
- 调 VLM
- 执行动作模板
- 记录日志

### 11.1 人工接管模式

目标：

- 当没有最终可用的大模型时，由人工工具链参与决策
- 人工可以查看 case 并给出修正动作
- 系统继续沿用同一套动作协议与日志机制

这一步不是权宜之计，而是未来自学习闭环的前置基础设施。

### 12. Demo 录制与复盘

目标：

- 自动存原图
- 自动存模型输出
- 自动存动作日志
- 自动生成一份 session summary

## 第一批最值得马上动手的代码

如果现在就开始写，我建议顺序是：

1. 重构 [hardware_agent.py](/Users/mico/Projects/insta360-dev/hardware_agent.py)
2. 重构 [api_server.py](/Users/mico/Projects/insta360-dev/api_server.py)
3. 改造 [bridge/model_feed.py](/Users/mico/Projects/insta360-dev/bridge/model_feed.py)
4. 新增 `nero/` 控制目录

这里的重点是：前 3 项今天都不需要依赖官方 SDK 仓库，完全可以先在 MacBook 上推进。

## 当前结论

你们现在最缺的不是算力，而是这三样：

- 真机控制抽象
- 通用动作协议
- 固定模板 demo 编排

这三样一旦补齐，周五硬件一到就不是“开始想怎么做”，而是直接进入联调。
