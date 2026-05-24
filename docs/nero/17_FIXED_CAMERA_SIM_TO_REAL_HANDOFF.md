# 固定机位凹槽 → 仿真 → 真机测试交接

本文记录 2026-05-24 完成的固定机位凹槽识别 + SimNERO 全流程仿真验证，供下次上真机使用。

## 同事快速上手（pull 之后）

```powershell
cd insta360-dev
git pull origin main
python -m pip install -r python/requirements.txt   # 若环境未装
```

### A. 只看结果、准备上真机

1. 打开交接 JSON（目标点、验证记录）  
   `calibration/results/fixed_camera_sim_handoff_2026-05-24/fixed_camera_sim_handoff.json`
2. 看标注图确认凹槽检测  
   `calibration/results/fixed_camera_slot_detect_2026-05-24/all_moves_slot_overlay_grid.jpg`
3. 按本文 **「真机首日流程」** 执行；**首日只做 standoff（Z=0.42 m）**

### B. 新拍一张固定机位图后重新检测

```powershell
python calibration/scripts/fixed_camera_slot_detect.py path/to/fixed_camera.jpg `
  --out-dir calibration/results/fixed_camera_slot_detect_2026-05-24
```

### C. 重新跑仿真 + 更新交接 JSON

```powershell
python experiments/nero_sim/run_fixed_camera_pick_place_sim.py `
  --image data/cases/dual_camera_bottle_moves_2026-05-23_170330/move_01/fixed_camera.jpg
```

输出覆盖：`calibration/results/fixed_camera_sim_handoff_2026-05-24/fixed_camera_sim_handoff.json`

### D. 跑单元测试

```powershell
python -m pytest tests/test_fixed_camera_slot_detect.py -q
```

### 关键脚本

| 脚本 | 作用 |
|------|------|
| `calibration/scripts/fixed_camera_slot_detect.py` | 固定机位图 → 凹槽像素 + base 坐标 |
| `calibration/scripts/top_camera_plane_project.py` | 单点像素 → base XY |
| `experiments/nero_sim/run_fixed_camera_pick_place_sim.py` | 检测 + 规划验证 + SimNERO 全流程 + 交接 JSON |
| `experiments/nero_sim/plan_to_point.py` | 单点仿真调试 |

## 结论

| 项目 | 状态 |
|------|------|
| 固定机位凹槽检测（mask 质心） | ✅ |
| Homography → base XY | ✅ |
| 8 项关节空间规划（home / scan 起点） | ✅ 误差 < 2 mm |
| 2 项 staged TCP 规划（scan 起点） | ✅ |
| SimNERO 完整 pick→place（from home） | ✅ |
| SimNERO 完整 pick→place（from scan pose） | ✅ |
| **真机首次动作** | ⚠️ **仅 standoff（Z=0.42 m）**，确认后再 lower |

固化交接文件：

```text
calibration/results/fixed_camera_sim_handoff_2026-05-24/fixed_camera_sim_handoff.json
```

## 一键复现仿真

```powershell
cd d:\DevProjects\insta360-dev
python experiments/nero_sim/run_fixed_camera_pick_place_sim.py
```

仅规划、不跑 SimNERO 动作：

```powershell
python experiments/nero_sim/run_fixed_camera_pick_place_sim.py --dry-plan-only
```

## 固化目标点（robot base, 单位 m）

| 语义 | XYZ |
|------|-----|
| 固定机位凹槽中心（支撑面） | `[-0.549, 0.248, 0.248]` |
| 放置点（槽面 + 30 mm） | `[-0.549, 0.248, 0.278]` |
| 放置 standoff | `[-0.549, 0.248, 0.420]` |
| 抓取点（腕部相机历史） | `[-0.363, 0.089, 0.275]` |
| 抓取 standoff | `[-0.363, 0.089, 0.420]` |

抓取点来自 `calibration/results/nero_sim_current_to_bottle_exchange_2026-05-11.json`（腕部 RGB-D）。放置点来自固定机位 homography。

## 真机首日流程（必须按顺序）

### 1. 上电前

- 确认顶部相机、黑垫、红盒、桌高未动（否则重标 homography）。
- 树莓派 `robot_server` / CAN 正常。

### 2. 移到 scan 姿态

```text
q_deg = [7.817, -51.497, -10.701, 101.596, -9.619, 5.546, 66.204]
```

### 3. 读 SDK flange，与仿真 FK 对比

仿真 FK（scan 姿态）参考：

```text
flange_xyz_mm ≈ [-266.6, 52.8, 299.7]
```

若 SDK flange 与上述相差 > 10 mm，先修 DH/TCP，**不要下降**。

### 4. 固定机位拍图 + 检测

```powershell
python calibration/scripts/fixed_camera_slot_detect.py `
  /path/to/fixed_camera.jpg `
  --out-dir calibration/results/fixed_camera_slot_detect_2026-05-24
```

确认 overlay 凹槽框正确。

### 5. 真机只做 standoff（两次）

先 **pick standoff**，再 **place standoff**，人工确认路径无碰撞、高度合理：

```text
pick  standoff: [-0.363, 0.089, 0.420]
place standoff: [-0.549, 0.248, 0.420]
```

可用 `pi_pick_place_bridge` 或 `robot_server` 分段移动；低速、可急停。

### 6. 对齐检查通过后再做抓取/放置

```text
pick  grasp: [-0.363, 0.089, 0.275]
place:       [-0.549, 0.248, 0.278]
```

## 已知风险（上真机前必读）

### 固定机位 vs 腕部相机 XY 差约 163 mm

| 来源 | 槽位 XY (m) |
|------|-------------|
| 腕部 RGB-D（2026-05-11） | `[-0.389, 0.279]` |
| 固定机位 homography（2026-05-24） | `[-0.549, 0.248]` |

仿真两条链路都能规划成功，但 **不能假设两者在真机上等价**。建议：

1. 首日 place standoff 用 **固定机位坐标**；
2. 用腕部相机再拍一张空槽，对比 base XY；
3. 若差异 > 20 mm，先统一坐标系再 full place。

### Z 是先验

凹槽 Z 使用支撑面 `0.248 m`，非深度测量。放置 `0.278 m` = 槽面 + 30 mm（与 `place_plan.json` 一致）。

### TCP / 桌面

见 `14_NERO_REAL_ROBOT_PLANNER_TEST.md`：TCP 未完全对齐前，禁止直接用 grasp Z 做首次下降。

## 禁止事项

- 不要跳过 standoff 直接 `lower` 到 grasp/place Z。
- 不要运行 learned policy 上真机。
- 不要在未对比 SDK flange 与 sim FK 时执行全行程。

## 相关文件

| 文件 | 说明 |
|------|------|
| `calibration/scripts/fixed_camera_slot_detect.py` | 固定机位凹槽检测 |
| `experiments/nero_sim/run_fixed_camera_pick_place_sim.py` | 仿真验证 + 生成交接 JSON |
| `calibration/results/fixed_camera_sim_handoff_2026-05-24/` | 交接 JSON + slot_detect 副本 |
| `calibration/results/fixed_camera_slot_detect_2026-05-24/` | 标注图 |
| `docs/nero/16_TOP_CAMERA_PLANE_LOCALIZATION_HANDOFF.md` | Homography 标定说明 |
| `docs/nero/14_NERO_REAL_ROBOT_PLANNER_TEST.md` | 真机 planner 安全规则 |
