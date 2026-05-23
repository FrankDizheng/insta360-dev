# NERO 顶部相机平面定位交接

本文记录 2026-05-23 固化的顶部相机支撑平面 homography，用于远端协作者接入透明红盖瓶的主 `XY` 定位链路。

## 当前结论

- 顶部相机应作为透明瓶抓取的主 `XY` 定位来源。
- 腕部 RGBD 相机后续只做近场验证，不再直接决定透明瓶最终 `XY`。
- 透明瓶抓取 `Z` 仍使用成功高度先验 `260 mm`，不要使用透明瓶深度图直接决定 `Z`。
- 只要顶部相机、黑色垫子、工作台支撑平面相对 robot base 没有移动，当前 homography 可以复用。

## 固化文件

当前代码应优先读取：

```text
calibration/results/top_camera_plane_homography_current.json
```

同一次工作图和固化副本保存在：

```text
calibration/results/top_camera_workspace_2026-05-23_223801/
```

完整触碰点、残差和 ChArUco 中间结果保存在：

```text
calibration/results/top_camera_plane_touch_2026-05-23_204617/
```

## 标定配置

- 标定板：ChArUco
- 字典：`DICT_5X5_100`
- 棋盘：`12 x 9`
- 方格边长：`0.015 m`
- 顶部相机名称：`top_camera`
- 坐标约定：图像像素 `[u, v]` 映射到 robot base 支撑平面 `[x, y]`，单位为 metre。

推荐 homography 来源：

```text
corner_plus_intersections_soft_fit_p01_p02_p03_p04_p06_p07
```

推荐拟合点：

- `p01` 到 `p04`：ChArUco 板四个外角。
- `p06`：左上 `(0,0)` 基准下的网格交点 `(4,6)`。
- `p07`：左上 `(0,0)` 基准下的网格交点 `(7,8)`。

`p05` 是格子中心触碰点，不纳入推荐拟合。它只作为记录保留，避免混用“交点”和“格子中心”定义。

## 残差

推荐拟合残差：

```text
RMS   = 0.7406 mm
Mean  = 0.5441 mm
Max   = 1.4106 mm
```

这说明顶部相机支撑平面 `XY` 标定已达到毫米级，可以用于后续红盖瓶平面定位。

## 使用方式

新增脚本：

```text
calibration/scripts/top_camera_plane_project.py
```

示例：

```bash
python calibration/scripts/top_camera_plane_project.py 900 220
```

输出：

```json
{
  "pixel_uv": [900.0, 220.0],
  "base_xy_m": [-0.0, 0.0],
  "homography": "calibration/results/top_camera_plane_homography_current.json"
}
```

实际值以当前 homography 计算结果为准。

## 后续接入建议

透明瓶抓取定位链路建议改成：

```text
top camera image
  -> red cap / bottle axis detection
  -> pixel UV to base XY via top-camera homography
  -> offset from red cap toward bottle body
  -> fixed Z = 260 mm
  -> local URDF / joint-space grasp planning
  -> wrist camera verification only
```

优先做传统 CV 红盖检测和瓶身轴线估计。VLM 可用于语义确认“哪个是盒外红盖瓶”，但不应直接输出最终几何坐标。

## 重新标定条件

以下任一情况发生时必须重新标定：

- 顶部相机被移动、旋转或重新固定。
- 黑色垫子或红盒所在工作平面整体移动。
- 工作台高度或支撑平面变化。
- robot base 与工作台相对位置变化。
- 后续验证发现顶部相机投影点与实际触碰点偏差超过约 `5 mm`。
