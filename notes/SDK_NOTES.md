# Insta360 X5 SDK 学习笔记

## OSC API

X5 支持 Open Spherical Camera (OSC) API，在 WiFi 模式下可通过 HTTP 访问。

### 默认地址

- IP: `192.168.42.1`
- 端口: `80`

### 基础接口

```
GET  /osc/info           # 相机信息
POST /osc/state          # 相机状态
POST /osc/commands/execute   # 执行命令
POST /osc/commands/status    # 查询命令状态
```

### 常用命令

| 命令 | 说明 |
|------|------|
| `camera.takePicture` | 拍照 |
| `camera.startCapture` | 开始录像 |
| `camera.stopCapture` | 停止录像 |
| `camera.listFiles` | 列出文件 |
| `camera.delete` | 删除文件 |
| `camera.getOptions` | 获取设置 |
| `camera.setOptions` | 修改设置 |

## X5 特有功能

- PureShot 模式拍照
- 8K 全景视频
- 5.7K 60fps
- 延时摄影（最低 0.5 秒间隔）

## 待探索

- [ ] USB 连接方式
- [ ] 实时预览流获取
- [ ] 全景拼接流程
- [ ] 陀螺仪数据获取
