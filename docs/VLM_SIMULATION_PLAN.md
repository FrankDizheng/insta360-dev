# VLM + ManiSkill3 仿真集成计划

## 目标

在 ManiSkill3 仿真环境中，用 Qwen VLM 作为"大脑"，实现：
- **输入**：仿真相机画面 → VLM
- **输出**：VLM → 结构化动作指令 → 机械臂执行

最终形成一个闭环：**看 → 想 → 做 → 看** 的循环。

---

## 已有基础

| 模块 | 状态 | 说明 |
|------|------|------|
| Qwen3.5 VLM (Ollama) | ✅ 已部署 | WSL + RTX 3060, ~3-4s/帧推理 |
| 视觉理解 | ✅ 已验证 | 场景描述、物体识别 |
| Visual Grounding | ✅ 已验证 | 输出 bounding box (0-1000 归一化坐标) |
| ManiSkill3 仿真 | ✅ 已跑通 | Windows + Vulkan, PickCube-v1 |
| SAPIEN 相机 | ✅ 已跑通 | 手动添加相机获取 RGB 图像 |
| IK 求解 | ✅ 已跑通 | pytorch_kinematics 做逆运动学 |
| 抓取流程 | ✅ 已跑通 | 8步 pick-and-place 脚本式控制 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    主控循环 (Python)                      │
│                                                         │
│  ┌─────────┐    ┌──────────┐    ┌────────────────────┐  │
│  │ ManiSkill│───>│ 相机截图  │───>│ Qwen VLM (Ollama) │  │
│  │ 仿真环境  │    │ RGB 图像  │    │                    │  │
│  │         │    └──────────┘    │  输入: 图像 + 提示词  │  │
│  │         │                    │  输出: JSON 动作指令  │  │
│  │         │    ┌──────────┐    │                    │  │
│  │         │<───│ 动作执行器 │<───│  {action, target,  │  │
│  │         │    │ IK + 控制 │    │   gripper, ...}   │  │
│  └─────────┘    └──────────┘    └────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 分阶段计划

### Phase 1：仿真画面 → VLM 理解（1-2天）

**目标**：VLM 能正确理解仿真场景中的物体和状态

- [ ] 1.1 从仿真中截取高质量 RGB 图像（多角度）
- [ ] 1.2 发送给 Qwen VLM，验证能否识别：
  - 机械臂的位置和姿态
  - 红色方块的位置
  - 绿色目标点的位置
  - 桌面和背景
- [ ] 1.3 测试 Visual Grounding：让 VLM 输出物体的 bounding box
- [ ] 1.4 对比 VLM 输出的坐标与仿真真实坐标的误差

**交付物**：`sim_vlm_perception.py` — 仿真截图 + VLM 分析模块

---

### Phase 2：定义动作指令协议（0.5天）

**目标**：制定 VLM 输出和机器人动作之间的标准接口

```json
{
  "action": "pick",          // pick | place | move | look | wait
  "target": "red cube",      // 物体名称
  "position": [0.1, -0.05],  // 2D 图像坐标（归一化）或 None
  "gripper": "close",        // open | close | unchanged
  "confidence": 0.85         // VLM 对自己判断的置信度
}
```

- [ ] 2.1 定义动作类型枚举（pick, place, move_to, look_around, wait）
- [ ] 2.2 设计 prompt 模板，让 VLM 严格输出 JSON 格式
- [ ] 2.3 实现 JSON 解析器 + 容错处理（VLM 输出不稳定）

**交付物**：`action_protocol.py` — 动作协议定义和解析

---

### Phase 3：2D→3D 坐标转换（1天）

**目标**：将 VLM 输出的 2D 图像坐标转换为 3D 世界坐标

- [ ] 3.1 利用 SAPIEN 相机的内参/外参矩阵做反投影
- [ ] 3.2 结合深度图（RGBD）将 2D 点转为 3D 世界坐标
- [ ] 3.3 验证转换精度：VLM 标注的位置 vs 仿真真实位置

```python
# 核心逻辑
def pixel_to_world(u, v, depth_map, camera):
    """将 VLM 输出的 2D 像素坐标转为 3D 世界坐标"""
    z = depth_map[v, u]
    K_inv = np.linalg.inv(camera.get_intrinsic_matrix())
    cam_point = K_inv @ np.array([u * z, v * z, z])
    world_point = camera.get_model_matrix() @ np.append(cam_point, 1)
    return world_point[:3]
```

**交付物**：`coord_transform.py` — 2D→3D 坐标转换模块

---

### Phase 4：VLM 驱动的闭环控制（2-3天）

**目标**：VLM 看一帧 → 输出动作 → 机械臂执行 → 再看一帧

- [ ] 4.1 实现主控循环：
  ```
  while not done:
      image = capture_sim_frame()
      action = vlm_analyze(image, task_prompt)
      world_pos = pixel_to_world(action.position, depth)
      joint_targets = solve_ik(world_pos)
      execute_action(env, joint_targets, action.gripper)
  ```
- [ ] 4.2 添加状态机管理（idle → reaching → grasping → lifting → placing → done）
- [ ] 4.3 实现多轮对话：VLM 记住之前的动作和结果
- [ ] 4.4 添加失败检测和重试逻辑

**交付物**：`vlm_controller.py` — VLM 闭环控制器

---

### Phase 5：自然语言任务指令（1天）

**目标**：用自然语言下达任务，VLM 自主规划和执行

示例：
- "把红色方块放到绿色标记处"
- "检查桌上有什么物品"
- "把所有物品按颜色分类摆放"

- [ ] 5.1 设计任务描述 prompt（中英文）
- [ ] 5.2 实现任务分解：高层指令 → 多步子动作
- [ ] 5.3 添加执行进度反馈（VLM 自己判断当前进展）

**交付物**：`task_planner.py` — 自然语言任务规划器

---

### Phase 6：场景扩展 — 玻璃瓶场景（2-3天）

**目标**：切换到贴近客户需求的玻璃瓶 pick-and-place 场景

- [ ] 6.1 在 ManiSkill3 中创建自定义环境：
  - 加载玻璃瓶 3D 模型（透明材质）
  - 设置传送带或托盘
  - 配置多个瓶子的随机摆放
- [ ] 6.2 测试 VLM 对透明物体的识别能力
- [ ] 6.3 处理多朝向瓶子的抓取策略
- [ ] 6.4 性能基准测试（抓取成功率、周期时间）

**交付物**：自定义仿真环境 + 完整 demo

---

## 技术风险和解决方案

| 风险 | 影响 | 应对 |
|------|------|------|
| VLM 推理慢 (~3-4s) | 控制频率低 | 仿真中可接受；生产用 YOLO 做高频检测，VLM 做低频决策 |
| VLM 输出格式不稳定 | 解析失败 | 多重 JSON 解析 + 正则兜底 + 重试机制 |
| 2D→3D 坐标误差 | 抓取偏移 | 用深度图修正 + 多次微调逼近 |
| 透明物体识别难 | 漏检/误检 | VLM 天然优势；补充偏振光/结构光方案 |
| IK 求解失败 | 无法到达 | 工作空间检查 + 多起始点重试 |

---

## 硬件需求

| 阶段 | 硬件 | 用途 |
|------|------|------|
| Phase 1-5 | 本地 RTX 3060 12GB | 仿真渲染 + VLM 推理 |
| Phase 6 | 本地 或 云 GPU | 复杂场景可能需要更多显存 |
| 生产部署 | 工控机 + 独立 GPU | VLM 服务 + 相机 + 机械臂通信 |

---

## 时间线

```
Week 1:  Phase 1 (感知验证) + Phase 2 (协议定义)
Week 2:  Phase 3 (坐标转换) + Phase 4 (闭环控制)
Week 3:  Phase 5 (自然语言) + Phase 6 (场景扩展)
```

---

## 下一步行动

**立即开始 Phase 1**：
1. 在 ManiSkill3 中截取仿真画面
2. 发送给 Ollama Qwen3.5 做场景分析
3. 验证 VLM 能否理解仿真中的机械臂和物体
