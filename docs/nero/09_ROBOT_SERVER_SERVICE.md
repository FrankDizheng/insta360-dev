# Raspberry Pi 常驻 `robot_server`

## 1. 目标

把树莓派侧的启动成本变成一次性的：

- bring up `can0`
- `connect_robot()`
- 初始化夹爪
- 打开 Orbbec RGB-D pipeline

之后本机只通过 HTTP 调用动作接口，不再每次 SSH 启脚本。

## 2. 代码位置

- 服务入口：`insta360-dev/calibration/scripts/robot_server.py`
- 兼容底层实现：`insta360-dev/calibration/scripts/pi_pick_place_bridge.py`
- `systemd` 模板：`insta360-dev/calibration/scripts/robot_server.service.template`
- 安装脚本：`insta360-dev/calibration/scripts/install_robot_server_service.sh`

## 3. 默认行为

`robot_server.py` 默认启用：

- `--eager-open`
- `--keep-hardware-alive`
- `can_activate.sh can0 1000000 2-1:1.0`

这意味着：

- 服务启动时就会尽量拉起 `can0`
- 服务启动时就会预热机器人和相机
- 客户端调用 `/session/close` 时，不会把硬件连接关掉

这样本机多次运行 `pick_place_session.py` 时，不再重复支付相机 / SDK / CAN 启动成本。

补充说明：

- 如果 Orbbec 初次打开较慢，服务端口可能要等预热完成后才开始监听
- `GET /status` 里出现 `hardware_ready=true` 且 `session_open=false` 是正常状态，表示硬件已就绪，但当前没有客户端会话占用

## 4. 安装到树莓派

先把脚本同步到树莓派对应目录，然后在 Pi 上执行：

```bash
chmod +x install_robot_server_service.sh
./install_robot_server_service.sh
```

如果你需要自定义参数，可以这样：

```bash
SERVICE_NAME=robot_server \
PYTHON_BIN=/usr/bin/python3 \
RUN_USER=pi \
HOST=0.0.0.0 \
PORT=8765 \
./install_robot_server_service.sh
```

## 5. 常用命令

```bash
sudo systemctl status robot_server
sudo systemctl restart robot_server
sudo systemctl stop robot_server
sudo journalctl -u robot_server -f
```

## 6. 本机调用边界

本机继续负责：

- `scan` 后图像分析
- 像素选点 / 自动建议
- 坐标换算
- 任务编排

树莓派服务负责：

- 相机采集
- 机器人执行
- 夹爪执行
- 状态查询

## 7. 当前接口

- `GET /health`
- `GET /status`
- `POST /session/open`
- `POST /session/close`
- `POST /scan_pose`
- `POST /scan`
- `POST /move_above`
- `POST /grasp`
- `POST /place`
- `POST /home`
