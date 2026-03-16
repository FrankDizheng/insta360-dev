"""
Hardware Agent — runs on the Edge PC next to the robot arm and camera.
Captures images, sends to VLM API server, receives actions, controls robot.

Usage:
    pip install requests opencv-python
    python hardware_agent.py --server http://<server-ip>:8100
"""
import argparse, time, json, sys
import requests

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv not installed, using dummy camera")


class CameraCapture:
    def __init__(self, device=0):
        if HAS_CV2:
            self.cap = cv2.VideoCapture(device)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera device {device}")
        else:
            self.cap = None

    def capture(self, path="/tmp/frame.jpg"):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(path, frame)
                return path
        import numpy as np
        from PIL import Image
        img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        img.save(path)
        return path

    def release(self):
        if self.cap:
            self.cap.release()


class RobotController:
    """
    Base class for robot arm control.
    Subclass this for your specific robot SDK.
    """
    def __init__(self):
        self.connected = False

    def connect(self):
        print("[Robot] Connecting to robot arm...")
        self.connected = True
        print("[Robot] Connected (simulation mode)")

    def move_above(self, target_name, target_pos=None):
        print(f"[Robot] Moving above {target_name}")
        time.sleep(0.5)
        return True

    def lower(self, target_name, target_pos=None):
        print(f"[Robot] Lowering to {target_name}")
        time.sleep(0.5)
        return True

    def grasp(self):
        print("[Robot] Closing gripper")
        time.sleep(0.3)
        return True

    def lift(self):
        print("[Robot] Lifting")
        time.sleep(0.5)
        return True

    def release(self):
        print("[Robot] Opening gripper")
        time.sleep(0.3)
        return True


def run_agent(server_url, camera_device=0, context=""):
    cam = CameraCapture(camera_device)
    robot = RobotController()
    robot.connect()

    print(f"\nHardware Agent started")
    print(f"  Server: {server_url}")
    print(f"  Camera: device {camera_device}")
    print(f"  Press Ctrl+C to stop\n")

    requests.post(f"{server_url}/reset")

    step = 0
    while True:
        frame_path = cam.capture("/tmp/hw_frame.jpg")
        print(f"\n--- Step {step} ---")
        print(f"  Captured frame: {frame_path}")

        with open(frame_path, "rb") as f:
            try:
                resp = requests.post(
                    f"{server_url}/decide",
                    files={"image": ("frame.jpg", f, "image/jpeg")},
                    data={"context": context, "use_cache": "true"},
                    timeout=30,
                )
                resp.raise_for_status()
            except requests.exceptions.ConnectionError:
                print(f"  ERROR: Cannot connect to server at {server_url}")
                print(f"  Retrying in 5s...")
                time.sleep(5)
                continue
            except requests.exceptions.Timeout:
                print(f"  ERROR: Server timeout, retrying...")
                continue

        result = resp.json()
        action = result["action"]
        target = result.get("target", "")
        source = result.get("source", "vlm")
        latency = result.get("latency_ms", 0)

        tag = "VLM" if source == "vlm" else "CACHE"
        print(f"  Decision: [{tag}] {action} {target} ({latency:.0f}ms)")
        print(f"  Reason: {result.get('reason', '')}")

        if action == "done":
            print("\n  Task complete!")
            break

        success = False
        if action == "move_above":
            success = robot.move_above(target)
        elif action == "lower":
            success = robot.lower(target)
        elif action == "grasp":
            success = robot.grasp()
        elif action == "lift":
            success = robot.lift()
        elif action == "release":
            success = robot.release()

        requests.post(f"{server_url}/feedback", json={
            "action": action,
            "target": target,
            "success": success,
        })

        print(f"  Execution: {'OK' if success else 'FAILED'}")
        step += 1

    # Print final status
    status = requests.get(f"{server_url}/status").json()
    print(f"\n{'='*50}")
    print(f"  Session Summary")
    print(f"{'='*50}")
    print(f"  VLM calls:    {status['total_vlm_calls']}")
    print(f"  Cache hits:   {status['total_cache_hits']}")
    print(f"  Objects sorted: {status['sorted_objects']}")
    print(f"{'='*50}")

    cam.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hardware Agent for VLM Sorting")
    parser.add_argument("--server", default="http://localhost:8100",
                       help="VLM API server URL")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device index")
    parser.add_argument("--context", default="",
                       help="Scene context for VLM")
    args = parser.parse_args()
    run_agent(args.server, args.camera, args.context)
