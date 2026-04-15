# 视觉与 VLM 层

## 1. 这一层解决什么问题

视觉与 VLM 层只负责三件事：

1. 取图
2. 看懂当前场景
3. 产出足够小、足够稳定的结构化结果给控制层

MVP 阶段不要让它直接负责轨迹规划。

## 2. 相机策略

当前最重要的工程判断：

标准 NERO 手册没有把相机列入标准装箱清单。

所以这一层必须按“相机可替换”设计。

建议抽象为：

- `capture_frame()`
- `save_frame()`
- `get_calibration()`
- `crop_workspace()`

无论到货的是：

- USB 相机
- 深度相机
- 组合套件相机

上层接口都不变。

## 2.1 当前采购约束下的相机选型原则

既然你们现在还在采购相机，选型标准应该直接服务于 MVP，而不是先追求最强配置。

MVP 阶段优先满足这几条：

- 能在 MacBook 上稳定取流
- 有成熟的 OpenCV 或厂商 SDK 接入路径
- 能覆盖完整工作区
- 支架容易固定，视角不容易漂移
- 在目标工作距离下，目标物体边缘足够清楚

建议优先级：

1. 稳定 RGB 相机
2. RGB-D 相机
3. 多相机方案

原因不是功能强弱，而是联调成本。MVP 最怕的是相机链路不稳。

## 2.2 视觉层当前的默认开发模式

今天先按下面的方式开发：

- 本地图片或普通相机作为输入源
- 不要求今天就拿到最终采购相机
- 视觉层先做统一适配接口

建议增加的抽象接口：

- `CameraAdapter`
- `capture_frame()`
- `get_frame_size()`
- `health_check()`
- `close()`

## 3. 你们当前最适合的 VLM 架构

你刚才的判断可以直接转化成工程方案：

### 方案 A: 云端 VLM

- 笔记本抓图
- 笔记本把图片发给云端服务
- 云端返回 JSON

优点：最省现场设备。

缺点：慢一些，但 MVP 完全能接受。

### 方案 B: 远程 4090 VLM

- 4090 主机放在更方便的位置
- 笔记本通过 HTTP / OpenAI-compatible API 调用

优点：延迟更低，可控性更高。

### 方案 C: 笔记本本地轻量模型

- 只在需要离线演示时考虑

这不是首选，因为你们已经有更好的外部算力路径。

## 3.1 2026-04-15 学校 API 实测结论

截至 2026-04-15，已完成一轮面向 MVP 的学校 OpenAI-compatible API 实测，当前建议如下：

- 当前可靠可用的 VL 入口是 `https://gpt-api.hkust-gz.edu.cn/v1/chat/completions`
- 推荐模型别名：
  - `gpt-4`，实际返回后端模型为 `gpt-4o-2024-08-06`
  - `gpt-4-turbo`，实际返回后端模型为 `gpt-4o-2024-11-20`
- 这两个模型都已通过受控图像验证：
  - 纯红色方块能正确识别为 `Red`
  - Python logo 能正确识别为 Python 图标
- 图像输入推荐使用 `base64` 的 `data:image/...` 方式，不要优先依赖外链 URL
- 原因是学校网关服务端对外链图片拉取不稳定，已有 `Failed to download image` 的实测返回

当前不建议作为机器人视觉主链路的入口：

- `https://aigc-api.hkust-gz.edu.cn/v1/chat/completions` 上的 `Qwen`
- `Qwen-VL`、`Qwen2-VL-*`、`Qwen2.5-VL-*`

原因：

- `Qwen` 文本对话可用，但图像输入测试结果不稳定
- 受控测试中，纯红色方块被误答为 `Black`
- 已知 logo 图像也出现明显误识别
- 因此它目前不能算作可靠的机器人 VLM 感知后端

## 3.2 相机视频流如何接入这类 VLM

当前这套可靠 VL 不是“原生视频流模型”，而是“单张图像推理模型”。

- 不要把 `25/30fps` 连续视频直接发给学校 VL API
- 推荐做法是：本地或远程算力机持续取流，然后按节拍或事件触发抽帧
- 每次只发送一张当前工作区图像给 `gpt-4` 或 `gpt-4-turbo`
- VLM 返回结构化结果，例如目标物、2D 抓点、放置区和置信度
- 树莓派仍只负责相机 / SDK / CAN / 执行桥接，不负责推理

推荐的最小链路：

1. `CameraAdapter.capture_frame()`
2. `crop_workspace()`
3. 编码为 `base64`
4. 调用远端 VL
5. 将结果转成统一 JSON
6. 再交给控制层执行

这意味着当前 VLM 适合：

- 扫描当前场景
- 选取抓取目标
- 放置前复核
- 状态切换时做低频视觉判断

这意味着当前 VLM 不适合：

- 高频实时跟踪
- 毫秒级视觉伺服
- 直接输出连续轨迹控制
- 把整段视频作为一个实时多模态会话输入

## 4. MVP 的 VLM 输出协议

MVP 阶段建议把协议限制到这几个字段：

```json
{
  "task": "pick_and_place",
  "target_object": "red bottle",
  "target_pick_point": [0.42, 0.37],
  "target_place_zone": "right_bin",
  "confidence": 0.88,
  "reason": "The red bottle is clearly visible on the left side of the workspace."
}
```

说明：

- `target_pick_point` 先允许是 2D 图像坐标
- `target_place_zone` MVP 阶段可以是固定放置区 ID
- 不要求 VLM 直接输出 6DoF 位姿

这个协议也是为了适配“相机后定、SDK 后接”的现实约束。只要输入能稳定拿到 2D 图像，主流程就能先跑。

## 5. 为什么先不要做 2D 到 3D 自动闭环

你们仓库里已经有 VLM + ManiSkill3 + 2D 到 3D 的思路，这是好的中长期方向。

但真机 MVP 里，最稳的做法不是一上来就做精确 2D 到 3D，而是：

- 固定视角
- 固定工作区
- 固定抓取模板
- 通过少量标定点，把图像区域映射到桌面平面

这会比完整 RGB-D 闭环快很多。

## 6. 视觉层的两阶段路线

### 阶段 1: 固定视角平面抓取

- 相机固定
- 工作区平面固定
- 用简单标定把 2D 点映射到桌面 XY

### 阶段 2: 深度 / 多视角 / 更复杂场景

- 深度补偿
- 目标姿态判断
- 失败重试

MVP 只做阶段 1。

## 7. 与现有代码的对接

可复用模块：

- [api_server.py](/Users/mico/Projects/insta360-dev/api_server.py)
- [hardware_agent.py](/Users/mico/Projects/insta360-dev/hardware_agent.py)
- [bridge/model_feed.py](/Users/mico/Projects/insta360-dev/bridge/model_feed.py)

建议改法：

- 把 `api_server.py` 从“排序任务控制器”收敛成通用 `vision_decision_service`
- 把 `hardware_agent.py` 作为现场执行代理继续使用
- 把 `bridge/model_feed.py` 改成支持远程模型地址和多后端

## 8. 这层最需要现在就编码的点

- 统一图片输入格式
- 统一 VLM 输出 JSON 协议
- VLM 超时与重试
- 结果缓存
- 每次推理保留原图与结果文本
- 相机适配器抽象，避免后续采购型号变化导致上层重写
