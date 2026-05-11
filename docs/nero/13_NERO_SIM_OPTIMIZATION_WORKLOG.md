# NERO 仿真优化工作记录

## 背景

本轮目标是把 NERO sim DAgger 从单进程 full-staged teacher 生成优化到可迭代状态，并确认下一步训练方向。

真实应用假设已更新：相机可以看到当前末端位置和目标点，因此后续应更关注“当前末端局部移动到目标”的 local reach，而不是固定 `HOME -> point_a` 的大范围全局移动。

## 已完成

### CUDA PyTorch

当前 Windows + RTX 4080 SUPER 环境已验证 CUDA 可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

验证结果：

- `torch 2.11.0+cu128`
- `torch.cuda.is_available() == True`
- CUDA runtime: `12.8`
- GPU: `NVIDIA GeForce RTX 4080 SUPER`

官方 PyTorch index 下载曾卡住。当前记录的可用安装方式是直接使用阿里云 mirror wheel：

```powershell
python -m pip install --force-reinstall "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl"
```

### Episode-Level 多进程 DAgger

`experiments/nero_sim/generate_dagger_corrections.py` 已新增：

- `--workers`
- `--worker-device`
- `--worker-torch-threads`

多进程实现按 episode 并行，主进程按 episode index 顺序写 JSONL，避免输出乱序。

推荐 full-staged DAgger 命令：

```powershell
python experiments/nero_sim/generate_dagger_corrections.py `
  --dataset experiments/nero_sim/outputs/train_staged512.jsonl `
  --checkpoint experiments/nero_sim/outputs/reach_policy_staged_bc512/reach_policy.pt `
  --output experiments/nero_sim/outputs/dagger_full_workers64.jsonl `
  --episodes 64 `
  --max-steps 60 `
  --success-tol-mm 12 `
  --teacher-mode every_step_staged `
  --planner-preset full_staged `
  --min-record-error-mm 8 `
  --keep-every-n-success 4 `
  --max-teacher-delta-rad 0.08 `
  --workers 8 `
  --worker-device cpu `
  --worker-torch-threads 1
```

本地实测结果：

- 输出：`experiments/nero_sim/outputs/dagger_full_workers64.jsonl`
- 耗时：约 15.9 分钟
- records：6093
- teacher source：全部 `full`
- average expert error：约 2.352 mm
- max expert error：约 3.657 mm

这把 64 episodes full-staged teacher 生成从约 70 分钟降到约 16 分钟，同时保持 full teacher label 质量。

## 训练与评估记录

### 当前最佳整体策略

训练数据：

- `train_staged512.jsonl`
- `dagger_full_workers64.jsonl`

输出：

- `experiments/nero_sim/outputs/reach_policy_staged_full_workers64/`

结果：

- best val loss：约 `0.000149`
- A rollout：18/64
- B rollout：33/64
- A 平均误差：约 102 mm
- B 平均误差：约 33 mm

结论：当前整体模型仍不足以上真机自主测试，但它是目前最稳定的 checkpoint。

### A-Failure 定向 DAgger

对 `reach_policy_staged_full_workers64` 的 A 段失败样本做定向 DAgger：

- A 失败样本：46/64
- 输出：`dagger_full_workers64_a_failures46.jsonl`
- records：4553
- teacher source：全部 `full`
- average expert error：约 2.387 mm

再训练：

- 输出：`reach_policy_staged_full_workers64_a_focus/`
- best val loss：约 `0.000143`

但 rollout 变差：

- A：15/64
- B：10/64

结论：简单把 A 失败 episode 加入 extra dataset 会污染闭环策略，不能继续沿这个方向堆数据。

### Feature Mode 对比

同一份高质量数据 `train_staged512 + dagger_full_workers64` 做公平对比：

| 模型 | A 成功 | B 成功 | A 平均误差 | B 平均误差 |
| --- | ---: | ---: | ---: | ---: |
| `legacy` | 18/64 | 33/64 | 102.34 mm | 32.93 mm |
| `segment` | 1/64 | 5/64 | 175.79 mm | 185.17 mm |
| `segment_margin` | 1/64 | 1/64 | 238.68 mm | 388.36 mm |

结论：当前不要切到 `segment` / `segment_margin`。loss 不能代表闭环成功率。

### Local Reach 方向

根据真实应用，后续更应该训练“当前可见末端位置 -> 目标点”的 local reach。

已生成两版 local 数据并训练：

1. `train_local_reach_full_workers64.jsonl`
   - expert path local samples + DAgger rollout local samples
   - error range: 25-220 mm
   - records：17034
   - 模型：`reach_policy_local_full_workers64`
   - 局部评估：755/1828，约 41.3%

2. `train_local_short_expert_staged512.jsonl`
   - expert path only
   - error range: 25-120 mm
   - records：7394
   - 模型：`reach_policy_local_short_expert`
   - 短距离局部评估：436/794，约 54.9%

action clamp 不能显著改善第一版 local policy，说明问题不只是动作过大。

## 当前判断

1. `--workers 8` full teacher 生成已经可用，后续 DAgger 迭代不再受 70 分钟瓶颈限制。
2. 当前 one-step MLP 对闭环 rollout 不稳定，loss 与成功率已经明显脱钩。
3. 固定 `HOME -> point_a` 不是最终应用的真实任务，应转向 local reach。
4. 直接堆失败样本、切 `segment`、切 `segment_margin` 都不是当前有效路径。

## 建议下一步

优先做 local reach 的策略形式调整，而不是继续堆同类型数据：

- 尝试预测 TCP-space 小步方向，再通过 IK/servo 转成关节动作。
- 或预测短 horizon waypoint，而不是单步关节 delta。
- 评估必须使用闭环 rollout success，不能只看 train/val loss。
- 真机前必须加动作限幅、关节限位、速度限制、workspace/table envelope 检查。

当前可交接 checkpoint：

- 整体策略：`experiments/nero_sim/outputs/reach_policy_staged_full_workers64/reach_policy.pt`
- local 诊断策略：`experiments/nero_sim/outputs/reach_policy_local_short_expert/reach_policy.pt`

注意：`experiments/nero_sim/outputs/` 是训练产物目录，当前不建议提交到 Git。远程合作者可按本文命令复现关键产物。
