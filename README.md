# Insta360 X5 开发项目

使用 Insta360 X5 相机进行开发的项目模板。

## 快速开始

### 1. 连接相机

**WiFi 连接（推荐用于 API 开发）：**
1. 开启 X5 相机
2. Mac 连接到 X5 的 WiFi 热点
   - 名称类似：`Insta360 X5.xxxx`
   - 默认密码：`88888888`

**USB 连接：**
1. 用 USB-C 线连接 X5 到 Mac
2. X5 屏幕选择 "USB 模式"（不是存储模式）

### 2. 安装依赖

```bash
cd python
pip install -r requirements.txt
```

### 3. 测试连接

```bash
cd python/examples
python 01_connect.py
```

成功输出示例：
```
==================================================
Insta360 X5 连接测试
==================================================
[INFO] 正在连接到 http://192.168.42.1...

==================================================
相机信息
==================================================
  型号: Insta360 X5
  固件版本: v1.x.x
  
==================================================
相机状态
==================================================
  电池电量: 85%
```

## 项目结构

```
insta360-dev/
├── sdk/                    # Insta360 官方 SDK（需手动下载）
│   ├── CameraSDK/         # 相机控制 SDK
│   └── MediaSDK/          # 媒体处理 SDK
├── python/
│   ├── requirements.txt
│   └── examples/
│       └── 01_connect.py  # 连接测试脚本
├── samples/               # 测试素材
│   ├── photos/
│   └── videos/
└── notes/                 # 学习笔记
```

## SDK 下载

1. 访问 https://www.insta360.com/sdk/record
2. 登录后下载：
   - Camera SDK for Linux/macOS
   - Media SDK for Linux/macOS
3. 解压到 `sdk/` 目录

## API 参考

X5 在 WiFi 模式下支持 OSC (Open Spherical Camera) API：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/osc/info` | GET | 获取相机信息 |
| `/osc/state` | POST | 获取相机状态 |
| `/osc/commands/execute` | POST | 执行命令 |

常用命令：
- `camera.takePicture` - 拍照
- `camera.startCapture` - 开始录像
- `camera.stopCapture` - 停止录像
- `camera.listFiles` - 列出文件

## 参考链接

- [Insta360 SDK 文档](https://github.com/Insta360Develop)
- [OSC API 规范](https://developers.google.com/streetview/open-spherical-camera/)
