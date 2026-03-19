# 智能机械臂项目 — 硬件软件协作文档

## 项目概述

基于 **SO-ARM100/101** 开源机械臂 + **Hugging Face LeRobot** 框架 + **NVIDIA Jetson (reComputer)** 平台，构建一套可通过示教学习自主执行任务、并支持语音指令调度的智能机械臂系统。

### 你拥有什么

| 组件 | 具体型号 | 角色 |
|------|----------|------|
| 机械臂 | SO-ARM100/101 (TheRobotStudio) | Leader 臂（示教）+ Follower 臂（执行） |
| 舵机 | Feetech STS3215 × 12 | 6关节 × 2臂，UART 串口总线控制 |
| 舵机驱动板 | Serial Bus Servo Drive Board × 2 | USB 连接，`/dev/ttyACM*` |
| 计算平台 | Seeed reComputer (Jetson Orin NX) | 70-100 TOPS AI 算力，可本地训练和推理 |
| 备用计算 | Raspberry Pi 5 + Camera | 轻量部署或辅助视觉 |
| 麦克风 | ReSpeaker 4-Mic Array | 语音交互（后期） |
| 3D 打印机 | 可用 | 打印臂体、定制夹爪 |
| 电源 | 5V 4A / 12V 2A (取决于版本) | 舵机供电 |

### 核心框架：LeRobot

[LeRobot](https://github.com/huggingface/lerobot) 是 Hugging Face 的开源机器人学习框架，提供完整的 **模仿学习** 流水线：

```
示教 (Leader 臂) → 数据采集 (摄像头+关节) → 训练策略 (ACT/Pi0/SmolVLA) → 自主执行 (Follower 臂)
```

这不是传统的"写代码控制机械臂"，而是 **教机器人做事，它学会后自己做**。

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 1: 模仿学习                            │
│                                                                  │
│   ┌──────────┐     ┌──────────┐     ┌────────────────────────┐  │
│   │ Leader臂 │────→│ 数据采集  │────→│  训练 (Jetson/GPU)     │  │
│   │ (人示教)  │     │ (摄像头+  │     │  ACT / Pi0 / SmolVLA  │  │
│   └──────────┘     │  关节数据) │     └──────────┬───────────┘  │
│                     └──────────┘                 │              │
│                                                   ▼              │
│                                          ┌──────────────┐       │
│                                          │ 训练好的策略   │       │
│                                          │ (神经网络模型) │       │
│                                          └──────┬───────┘       │
│                                                  │              │
│                                                  ▼              │
│                                          ┌──────────────┐       │
│                                          │ Follower 臂   │       │
│                                          │ (自主执行)     │       │
│                                          └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Phase 2: 语音调度                             │
│                                                                  │
│   ┌──────────┐     ┌──────────┐     ┌────────────────────────┐  │
│   │ ReSpeaker │────→│ 语音识别  │────→│  LLM / OpenClaw        │  │
│   │ (语音)    │     │ (讯飞/    │     │  理解指令，选择技能     │  │
│   └──────────┘     │  本地)    │     └──────────┬───────────┘  │
│                     └──────────┘                 │              │
│                                                   ▼              │
│                                          ┌──────────────┐       │
│                                          │ 调用对应的      │       │
│                                          │ 训练好的策略    │       │
│                                          └──────┬───────┘       │
│                                                  │              │
│                                                  ▼              │
│                                          ┌──────────────┐       │
│                                          │ Follower 臂   │       │
│                                          │ (精准执行)     │       │
│                                          └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么是两个阶段

| | Phase 1: 模仿学习 | Phase 2: 语音调度 |
|---|---|---|
| **核心** | 教机器人学会具体动作 | 用语音选择已学会的动作 |
| **技术** | LeRobot + ACT/Pi0/SmolVLA | ReSpeaker + LLM + 策略调度 |
| **内容价值** | "我教了机器人10分钟，它就学会了" | "我对着机器人说话，它就去做" |
| **依赖** | 硬件组装 + 数据采集 + 训练 | Phase 1 必须先完成 |

---

## 分工

| 角色 | 负责人 | 职责范围 |
|------|--------|----------|
| **硬件** | _（姓名）_ | 机械臂组装、舵机校准、3D打印、布线供电、摄像头安装、工作台搭建 |
| **软件** | _（姓名）_ | LeRobot 环境配置、数据采集脚本、模型训练、策略部署、语音交互集成 |

---

## 硬件方交付清单

> 以下每一项完成后，请在对应的状态栏打 ✅ 并填写交付信息。

### H-1. 计算平台基础环境

> 主计算平台使用 reComputer (Jetson Orin NX)。Pi 5 可作为辅助。

| 项目 | 交付内容 | 状态 |
|------|----------|------|
| 设备型号 | reComputer J______（J301x / J401x） | ⬜ |
| Jetson 模块 | Orin NX ____GB (8GB / 16GB) | ⬜ |
| JetPack 版本 | `cat /etc/nv_tegra_release` 输出：______ （需 6.0 或 6.1） | ⬜ |
| SSH 访问 | IP：______ 用户名：______ 密码：______ | ⬜ |
| 网络连接 | WiFi / 有线，网络名称：______ | ⬜ |
| Python 版本 | `python3 --version` 输出：______ （需 3.10） | ⬜ |
| CUDA 版本 | `nvcc --version` 输出：______ | ⬜ |
| 存储 | 可用空间：______ GB（建议 NVMe SSD，训练数据集较大） | ⬜ |
| 散热 | 风扇/散热片已安装，满载无过热 | ⬜ |

**验收标准**：
1. 软件方可通过 SSH 远程登录
2. `python3 -c "import torch; print(torch.cuda.is_available())"` 输出 `True`

---

### H-2. 机械臂组装与校准

> 参考教程：
> - B站视频：[从零搭建具身智能机械臂1：组装和调试](https://www.bilibili.com/video/BV1uP6JY5EH3/)
> - 中文 Wiki：https://wiki.seeedstudio.com/cn/lerobot_so100m/
> - 英文 Wiki：https://wiki.seeedstudio.com/lerobot_so100m/

#### 基本信息

| 项目 | 交付内容 | 状态 |
|------|----------|------|
| 机械臂版本 | SO-ARM100 / SO-ARM101（请圈选） | ⬜ |
| 套件版本 | 标准版 (5V) / 专业版 Pro (12V follower) | ⬜ |
| 3D 打印件 | 全部打印完成，PLA+，填充 15% | ⬜ |

#### Leader 臂

| 项目 | 交付内容 | 状态 |
|------|----------|------|
| 舵机 ID 校准 | L1-L6 全部校准完成（`lerobot-setup-motors`） | ⬜ |
| 物理组装 | 按 Wiki 步骤全部组装完成 | ⬜ |
| USB 接口 | 串口设备路径：______ （例：`/dev/ttyACM1`） | ⬜ |
| 电源 | 5V 4A 电源已连接 | ⬜ |

#### Follower 臂

| 项目 | 交付内容 | 状态 |
|------|----------|------|
| 舵机 ID 校准 | F1-F6 全部校准完成（`lerobot-setup-motors`） | ⬜ |
| 物理组装 | 按 Wiki 步骤全部组装完成 | ⬜ |
| USB 接口 | 串口设备路径：______ （例：`/dev/ttyACM0`） | ⬜ |
| 电源 | 标准版 5V 4A / Pro版 12V 2A 已连接 | ⬜ |

#### 校准

| 项目 | 交付内容 | 状态 |
|------|----------|------|
| Follower 校准 | `lerobot-calibrate --robot.type=so101_follower` 完成 | ⬜ |
| Leader 校准 | `lerobot-calibrate --teleop.type=so101_leader` 完成 | ⬜ |
| 遥操作测试 | `lerobot-teleoperate` 运行正常，Leader 动 Follower 跟随 | ⬜ |

**验收标准**：运行 `lerobot-teleoperate` 命令后，用手移动 Leader 臂，Follower 臂同步跟随，动作流畅无卡顿。

---

### H-3. 摄像头

| 项目 | 交付内容 | 状态 |
|------|----------|------|
| 摄像头类型 | USB 摄像头 / Pi Camera / Orbbec 深度相机 | ⬜ |
| 数量 | ______ 个 | ⬜ |
| 安装位置 | 正前方 (front) / 侧面 (side) / 顶部 (top) | ⬜ |
| 设备索引 | `lerobot-find-cameras opencv` 输出：______ | ⬜ |
| 固定安装 | 摄像头固定在支架上，不会晃动 | ⬜ |
| 分辨率 | ______ × ______ （建议 640×480） | ⬜ |

**验收标准**：运行带摄像头的遥操作命令，画面实时显示且流畅：

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader \
    --display_data=true
```

---

### H-4. ReSpeaker 4-Mic Array（Phase 2）

| 项目 | 交付内容 | 状态 |
|------|----------|------|
| 连接方式 | USB / GPIO HAT | ⬜ |
| 连接到 | reComputer / Pi 5 | ⬜ |
| 驱动安装 | `arecord -l` 可识别设备 | ⬜ |
| ALSA 设备名 | 例：`plughw:2,0`，实际值：______ | ⬜ |
| 录音测试 | 3 秒录音播放可清晰听到人声 | ⬜ |

> 此项为 Phase 2 内容，优先级低于机械臂和摄像头。

---

### H-5. 工作台布局

| 项目 | 交付内容 | 状态 |
|------|----------|------|
| Leader 臂固定 | 用桌夹固定在桌面 | ⬜ |
| Follower 臂固定 | 用桌夹固定在桌面 | ⬜ |
| 两臂间距 | ______ cm | ⬜ |
| 工作台照片（俯视） | 附件 | ⬜ |
| 工作台照片（侧视） | 附件 | ⬜ |
| 工作区域尺寸 | 长 ______ × 宽 ______ cm | ⬜ |
| 光照条件 | 自然光 / LED灯 / 可调 | ⬜ |
| 操作物品 | 准备哪些物品用于示教（例：积木、杯子、笔等） | ⬜ |

---

## 软件方交付清单

### S-1. LeRobot 环境搭建

| 任务 | 说明 | 依赖硬件 |
|------|------|----------|
| 安装 Miniconda | Python 3.10 环境 | H-1 |
| 安装 LeRobot | `pip install -e ".[feetech]"` | H-1 |
| 验证 PyTorch CUDA | `torch.cuda.is_available() == True` | H-1 |
| 安装 ffmpeg | 视频编码依赖 | H-1 |

### S-2. 数据采集与训练

| 任务 | 说明 | 依赖硬件 |
|------|------|----------|
| 数据采集脚本 | `lerobot-record` 配置 | H-2 + H-3 |
| 第一个技能数据集 | 例："抓起积木放入盒子"，采集 50 组 | H-2 + H-3 + H-5 |
| 训练策略 | ACT / SmolVLA 训练 | H-1 (GPU) |
| 部署评估 | `lerobot-record --policy.path=...` | H-2 |

### S-3. 语音交互集成（Phase 2）

| 任务 | 说明 | 依赖硬件 |
|------|------|----------|
| 唤醒词检测 | 本地唤醒词 → 触发录音 | H-4 |
| 语音识别 | 讯飞 API 或本地 Whisper | H-4 |
| 技能调度 | 语音 → 识别意图 → 选择对应训练好的策略 → 执行 | S-2 完成 |

---

## 时间线

| 阶段 | 硬件方任务 | 软件方任务 | 预计时间 |
|------|-----------|-----------|----------|
| **第1周** | 3D 打印所有部件，校准舵机 ID | 在 Jetson 上安装 LeRobot + PyTorch CUDA 环境 | 并行 |
| **第2周** | 组装 Leader + Follower 臂，接线通电 | 熟悉 LeRobot 命令，阅读文档和教程 | 并行 |
| **第3周** | 校准两臂，测试遥操作，安装摄像头 | 连接到硬件，验证遥操作 + 摄像头画面 | 协作 |
| **第4周** | 搭建工作台，准备操作物品，优化光照 | 采集第一个技能数据集（50 组示教数据） | 协作 |
| **第5周** | 配合调试，调整物理安装 | 训练 ACT 策略，首次自主执行测试 | 协作 |
| **第6-7周** | 采集更多技能数据（不同任务） | 训练多个技能策略，优化成功率 | 协作 |
| **第8周** | 安装 ReSpeaker，准备拍摄场景 | 集成语音调度，多技能切换 | 协作 |
| **第9周+** | 外观优化，拍摄辅助 | 录制演示视频 | 协作 |

### 里程碑

| 里程碑 | 标志 | 预计 |
|--------|------|------|
| **M1: 动了** | 遥操作正常，Leader 动 Follower 跟 | 第3周 |
| **M2: 采了** | 完成 50 组示教数据采集，可视化确认数据质量 | 第4周 |
| **M3: 学了** | ACT 策略训练完成，Loss 收敛 | 第5周 |
| **M4: 会了** | Follower 臂自主执行任务，成功率 > 60% | 第5-6周 |
| **M5: 稳了** | 成功率 > 80%，可重复演示 | 第7周 |
| **M6: 听了** | 语音说"拿起积木"，机器人自主执行 | 第8周 |
| **M7: 拍了** | 录制完整演示视频，可发布 | 第9周 |

---

## 内容策略

### 可产出的视频内容（按时间线）

| 时间 | 视频主题 | 平台 |
|------|----------|------|
| 第1-2周 | 开箱 + 3D打印 + 组装过程 | B站 (长视频) |
| 第3周 | 遥操作演示：我动它就动 | B站 + 抖音 |
| 第4周 | 示教过程：教机器人做事 | B站 |
| 第5-6周 | **核心爆点：它学会了！** 自主执行 vs 示教对比 | B站 + 抖音 + 小红书 |
| 第7周 | 挑战赛：教它越来越难的任务 | 抖音 (系列) |
| 第8周 | **终极演示：对着它说话就能做** | 全平台 |
| 持续 | 失败合集、幕后花絮 | 抖音 + 小红书 |

### 内容差异化

你的竞争优势：
1. **在中国做 LeRobot 的先行者**——目前 B站 上相关内容极少
2. **完整记录从零到一的过程**——比只展示成果更有价值
3. **开源全部代码和数据集**——获得社区信任和传播

---

## 技术细节备忘

### 机械臂通信

- 舵机型号：Feetech STS3215 (多种齿轮比变体)
- 通信协议：UART 串行总线
- 连接方式：舵机 → 舵机驱动板 → USB-C → 计算平台
- 设备路径：`/dev/ttyACM0`（Follower）, `/dev/ttyACM1`（Leader）
- LeRobot 内置 Feetech 驱动：`pip install -e ".[feetech]"`

### 关节映射

| 关节 | ID | 功能 |
|------|-----|------|
| J1 | 1 | 底座旋转 (shoulder_pan) |
| J2 | 2 | 肩部俯仰 (shoulder_lift) |
| J3 | 3 | 肘部 (elbow_flex) |
| J4 | 4 | 腕部旋转 (wrist_flex) |
| J5 | 5 | 腕部俯仰 (wrist_roll) |
| J6 | 6 | 夹爪 (gripper) |

### LeRobot 支持的训练策略

| 策略 | 特点 | 训练时间 (参考) |
|------|------|----------------|
| **ACT** | 入门推荐，成熟稳定 | ~6h (RTX 3060), ~2-3h (RTX 4090) |
| **Diffusion Policy** | 更强的泛化能力 | 较长 |
| **Pi0 / Pi0.5** | 大模型微调，需预训练权重 | 中等 |
| **SmolVLA** | 轻量级视觉-语言-动作模型，450M参数 | ~4h (A100) |
| **GR00T N1.5** | NVIDIA 官方模型 | 需确认 Jetson 兼容性 |

### 推荐入门路线

1. **首选 ACT**：最成熟，社区案例最多，出问题容易排查
2. 待 ACT 跑通后，尝试 **SmolVLA**：视觉-语言-动作一体，更适合后期语音指令场景

---

## 安全注意事项

1. **机械臂运行时，手不要进入 Follower 臂工作范围**
2. Leader 臂供电始终使用 **5V**，Pro 版 Follower 臂使用 **12V**，**绝对不能接反**
3. 校准舵机时确认电压匹配，接错电压可能 **烧毁舵机**
4. 首次自主执行时低速运行，人员准备急停
5. USB 线和电源线均需插好，USB 不供电，两者必须同时连接
6. 摄像头不要通过 USB Hub 连接，直连设备以保证传输速度

---

## 沟通机制

| 事项 | 约定 |
|------|------|
| 日常沟通 | 微信群 |
| 文档更新 | 本文档，更新后通知对方 |
| 问题升级 | 遇到阻塞性问题，当天沟通，不过夜 |
| 交付确认 | 每项交付后，对方验收并在状态栏打 ✅ |

---

## 参考资料

| 资源 | 链接 |
|------|------|
| SO-ARM100 GitHub | https://github.com/TheRobotStudio/SO-ARM100 |
| LeRobot GitHub | https://github.com/huggingface/lerobot |
| Seeed LeRobot 英文 Wiki | https://wiki.seeedstudio.com/lerobot_so100m/ |
| Seeed LeRobot 中文 Wiki | https://wiki.seeedstudio.com/cn/lerobot_so100m/ |
| B站组装教程 | https://www.bilibili.com/video/BV1uP6JY5EH3/ |
| Seeed 淘宝购买 | https://s.click.taobao.com/j6b5cCs |
| STS3215 舵机数据表 | https://drive.weixin.qq.com/s?k=AGEAZwfLABEnnGawbMAT8AawY5AOc |
| 电源适配器数据表 | https://drive.weixin.qq.com/s?k=AGEAZwfLABEowXXaYyAT8AawY5AOc |
| 舵机驱动板数据表 | https://drive.weixin.qq.com/s?k=AGEAZwfLABE0CUwdz1AT8AawY5AOc |
| reComputer J301x 数据表 | https://files.seeedstudio.com/products/NVIDIA/reComputer-J301x-datasheet.pdf |
| reComputer J401x 数据表 | https://files.seeedstudio.com/products/NVIDIA/reComputer-J401x-datasheet.pdf |
| LeRobot Discord | https://discord.gg/8TnwDdjFGU |
| 科大讯飞语音 API | https://www.xfyun.cn/ |

---

## 常见问题速查

| 问题 | 解决方案 |
|------|----------|
| `Motor 'gripper' was not found` | 检查通信线缆和电源电压是否正确 |
| `Could not connect on port /dev/ttyACM0` | `sudo chmod 666 /dev/ttyACM*` |
| `Magnitude exceeds 2047` | 断电重启机械臂后重新校准 |
| `ConnectionError: Failed to sync read` | 检查对应端口的臂是否通电，总线线缆是否松动 |
| PyTorch CUDA 不可用 | LeRobot pip 安装会覆盖 GPU 版 PyTorch，需重新安装 GPU 版 |
| `rerun` 报错 | `pip3 install rerun-sdk==0.23` |
| 摄像头读不到数据 | 摄像头必须直连设备，不通过 USB Hub |
