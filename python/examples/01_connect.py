#!/usr/bin/env python3
"""
Insta360 X5 连接测试脚本

使用方法：
1. 开启 X5 相机
2. Mac 连接到 X5 的 WiFi 热点（名称类似 Insta360 X5.xxxx，密码 88888888）
3. 运行此脚本：python 01_connect.py

OSC API 文档：https://developers.google.com/streetview/open-spherical-camera/
"""

import requests
import json
from typing import Optional, Dict, Any

# X5 WiFi 模式下的默认地址
X5_HOST = "http://192.168.42.1"
TIMEOUT = 5


def get_camera_info() -> Optional[Dict[str, Any]]:
    """获取相机基本信息"""
    try:
        url = f"{X5_HOST}/osc/info"
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 获取相机信息失败: {e}")
        return None


def get_camera_state() -> Optional[Dict[str, Any]]:
    """获取相机状态（电量、存储等）"""
    try:
        url = f"{X5_HOST}/osc/state"
        response = requests.post(url, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 获取相机状态失败: {e}")
        return None


def execute_command(command: str, parameters: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """执行相机命令"""
    try:
        url = f"{X5_HOST}/osc/commands/execute"
        payload = {"name": command}
        if parameters:
            payload["parameters"] = parameters
        
        response = requests.post(url, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 执行命令 {command} 失败: {e}")
        return None


def take_picture() -> Optional[Dict[str, Any]]:
    """拍照"""
    print("[INFO] 正在拍照...")
    result = execute_command("camera.takePicture")
    if result:
        print(f"[SUCCESS] 拍照成功!")
    return result


def start_capture() -> Optional[Dict[str, Any]]:
    """开始录像"""
    print("[INFO] 开始录像...")
    result = execute_command("camera.startCapture")
    if result:
        print(f"[SUCCESS] 录像已开始!")
    return result


def stop_capture() -> Optional[Dict[str, Any]]:
    """停止录像"""
    print("[INFO] 停止录像...")
    result = execute_command("camera.stopCapture")
    if result:
        print(f"[SUCCESS] 录像已停止!")
    return result


def list_files(entry_count: int = 10) -> Optional[Dict[str, Any]]:
    """列出相机中的文件"""
    return execute_command("camera.listFiles", {
        "fileType": "all",
        "entryCount": entry_count,
        "maxThumbSize": 0
    })


def print_camera_info(info: Dict[str, Any]):
    """打印相机信息"""
    print("\n" + "=" * 50)
    print("相机信息")
    print("=" * 50)
    print(f"  型号: {info.get('model', 'Unknown')}")
    print(f"  制造商: {info.get('manufacturer', 'Unknown')}")
    print(f"  序列号: {info.get('serialNumber', 'Unknown')}")
    print(f"  固件版本: {info.get('firmwareVersion', 'Unknown')}")
    print(f"  API 版本: {info.get('apiLevel', 'Unknown')}")
    

def print_camera_state(state: Dict[str, Any]):
    """打印相机状态"""
    print("\n" + "=" * 50)
    print("相机状态")
    print("=" * 50)
    
    state_data = state.get("state", {})
    print(f"  电池电量: {state_data.get('batteryLevel', 'Unknown')}%")
    print(f"  存储空间: {state_data.get('storageUri', 'Unknown')}")
    
    # 尝试获取更多状态信息
    if "_captureStatus" in state_data:
        print(f"  拍摄状态: {state_data.get('_captureStatus', 'Unknown')}")


def main():
    print("\n" + "=" * 50)
    print("Insta360 X5 连接测试")
    print("=" * 50)
    print(f"\n[INFO] 正在连接到 {X5_HOST}...")
    
    # 获取相机信息
    info = get_camera_info()
    if info:
        print_camera_info(info)
    else:
        print("\n[ERROR] 无法连接到相机!")
        print("\n请检查:")
        print("  1. X5 相机是否已开机")
        print("  2. Mac 是否已连接到 X5 的 WiFi 热点")
        print("  3. WiFi 热点名称类似: Insta360 X5.xxxx")
        print("  4. 默认密码: 88888888")
        return
    
    # 获取相机状态
    state = get_camera_state()
    if state:
        print_camera_state(state)
    
    print("\n" + "=" * 50)
    print("连接成功! 可用命令:")
    print("=" * 50)
    print("  - take_picture()  # 拍照")
    print("  - start_capture() # 开始录像")
    print("  - stop_capture()  # 停止录像")
    print("  - list_files()    # 列出文件")
    print("\n")


if __name__ == "__main__":
    main()
