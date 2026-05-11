# NERO 仿真与 DAgger 进度说明

## 当前目标

这条线的目标是让 NERO 在仿真里学习从 A 点移动到 B 点的关节轨迹能力。

现实场景中，相机/VLM 可以给出物体或目标的三维坐标，但机械臂仍需要知道如何从当前初始姿态移动到目标点。当前工作聚焦在这一层：用 `sim-nero` 数字孪生生成专家轨迹，再通过 BC/DAgger 训练一个关节动作策略。

## 已完成内容

- 建立了 NERO 7 轴近似 FK/IK、TCP 和 staged motion planner。
- 接入了用户确认的 NERO 关节限位。
- 使用 132 mm 夹爪 TCP 近似值作为当前仿真 TCP。
- 从官方 STEP 模型提取夹爪几何和 frame/table 对齐信息。
- 增加了 DAgger 数据生成、失败分析、专家/策略 rollout 对比可视化。
- 增加了 DAgger teacher profile，用于观察每个 episode 的耗时、teacher 调用次数和失败分布。

## 关键问题

最初的 staged-teacher DAgger 生成 64 episodes 曾耗时 30 小时以上，无法用于迭代。

根因不是神经网络训练慢，而是 teacher 生成阶段慢：

- `generate_dagger_corrections.py` 在 rollout 每一步调用 teacher。
- teacher 进入 `plan_tcp_motion()`。
- planner 内部执行多阶段 waypoint planning。
- 每阶段再执行多 seed 数值 IK。
- 数值 IK 每轮会重复 FK、有限差分 Jacobian、路径 cost 和几何 envelope penalty。

这是一条单线程 Python CPU 循环，GPU 基本没有参与。

## 已验证的优化结果

当前最可靠的可用路径是：保留 full-staged teacher 质量，但使用新的 profile/参数透传生成器。

对比结果：

| 方案 | 64 episodes 生成时间 | 质量结论 |
| --- | ---: | --- |
| 旧 staged DAgger | 30 小时以上 | 质量可用，但时间不可接受 |
| cached/fast teacher | 12-33 分钟 | 速度快，但 label 分布污染训练 |
| consistency-gated fast teacher | 约 51.6 分钟 | label 质量接近，但 rollout 仍低于旧 full teacher |
| current full-staged teacher | 约 70.8 分钟 | 当前推荐，可接受时间内保持 teacher 质量 |

当前推荐数据和模型：

- DAgger 数据：`experiments/nero_sim/outputs/dagger_full_current64.jsonl`
- 训练输出：`experiments/nero_sim/outputs/reach_policy_staged_full_current64_match_old/`

这些输出文件暂未提交到 Git，因为属于训练产物/大文件。

## Rollout 对比

同一批 `train_staged512.jsonl` 前 64 episodes，`success_tol=12mm`，`max_steps=60`：

| 模型 | A 成功数 | B 成功数 | 备注 |
| --- | ---: | ---: | --- |
| `reach_policy_staged_bc512` | 28/64 | 21/64 | staged BC 基线 |
| 旧 `reach_policy_staged_dagger64` | 21/64 | 29/64 | 旧 full DAgger，B 段最好 |
| `reach_policy_staged_consistency64_match_old` | 19/64 | 23/64 | consistency fast 版，不推荐作为最终 |
| `reach_policy_staged_full_current64_match_old` | 23/64 | 27/64 | 当前推荐分支 |

结论：当前 full-staged 生成器已经把 64 episodes 压到约 70 分钟，并且质量最接近旧 full DAgger。

## 合作开发者重点看哪里

主要代码入口：

- `experiments/nero_sim/generate_dagger_corrections.py`
  - DAgger 数据生成入口。
  - 已加入 `teacher-mode`、`planner-preset`、profile、consistency gate、`--workers` episode 级多进程。
- `nero/planning.py`
  - FK/IK planner、staged planner、planner preset。
  - 当前 teacher 质量和耗时主要受这里影响。
- `experiments/nero_sim/train_reach_policy.py`
  - BC/DAgger 混合训练入口。
- `experiments/nero_sim/eval_reach_policy.py`
  - 单 episode rollout 验证入口。
- `experiments/nero_sim/analyze_reach_failures.py`
  - 批量失败样本分析和可视化入口。
- `nero/kinematics.py`
  - NERO DH FK 和 132 mm TCP 定义。
- `nero/geometry.py`
  - 几何 envelope、table 对齐和 STEP-derived gripper 配置。

## 推荐命令

当前推荐生成 DAgger 的命令：

```powershell
python experiments/nero_sim/generate_dagger_corrections.py `
  --dataset experiments/nero_sim/outputs/train_staged512.jsonl `
  --checkpoint experiments/nero_sim/outputs/reach_policy_staged_bc512/reach_policy.pt `
  --output experiments/nero_sim/outputs/dagger_full_current64.jsonl `
  --episodes 64 `
  --max-steps 60 `
  --success-tol-mm 12 `
  --teacher-mode every_step_staged `
  --planner-preset full_staged `
  --min-record-error-mm 8 `
  --keep-every-n-success 4 `
  --max-teacher-delta-rad 0.08 `
  --workers 8
```

当前推荐训练命令：

```powershell
python experiments/nero_sim/train_reach_policy.py `
  --dataset experiments/nero_sim/outputs/train_staged512.jsonl `
  --extra-dataset experiments/nero_sim/outputs/dagger_full_current64.jsonl `
  --output-dir experiments/nero_sim/outputs/reach_policy_staged_full_current64_match_old `
  --epochs 20 `
  --batch-size 128 `
  --hidden-dim 192 `
  --feature-mode legacy `
  --point-b-weight 1.0 `
  --seed 141
```

CUDA PyTorch 检查命令：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Windows + RTX 40 系列当前推荐安装命令：

```powershell
python -m pip install --force-reinstall "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl"
```

## 下一步建议

1. 先不要继续使用 fast/cached teacher 作为正式训练数据来源，除非新增更强的 IK 分支一致性约束。
2. 已加入 episode-level 多进程并行；下一步用 `--workers 8` 跑 64 episodes，确认当前机器的最佳 worker 数。
3. 已确认旧环境 PyTorch 是 CPU 版；安装 CUDA wheel 后，训练脚本会自动使用 `cuda`。
4. GPU 加速 planner 需要先把 FK/IK 改成 batched tensor 形式，直接把当前 Python 小循环搬到 GPU 不会有效。
5. 仿真几何仍是近似 envelope，后续需要补齐完整机械臂 3D 包络体。

## 当前结论

仿真训练方向是对的，已经能支持“相机/VLM 给目标点，策略学习关节轨迹”的主线。

当前最大瓶颈从“不可接受的 30 小时以上”降到了“约 70 分钟一轮 64 episodes”。下一阶段应优先做 CPU 多进程并行和 CUDA 版 PyTorch 环境，而不是牺牲 teacher 质量换 fast label。
