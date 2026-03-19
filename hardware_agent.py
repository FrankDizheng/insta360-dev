"""Hardware agent for mock-first robot integration and local simulation.

Captures images, sends them to the decision service, receives actions,
and executes them through a pluggable robot controller.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import requests

from nero import ActionDecision, dispatch_action, get_robot_controller
from nero.perception import get_camera, CameraInterface


class ImageDirectoryCapture:
    def __init__(self, image_dir: str | Path):
        self.image_dir = Path(image_dir)
        self.images = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self.index = 0

    def capture(self, path=None):
        selected = self.images[self.index % len(self.images)]
        self.index += 1
        return str(selected)

    def release(self):
        return None


def run_agent(
    server_url,
    context="",
    task_description="",
    robot_mode="mock",
    image_dir="",
    use_cache=True,
    camera_mode="mock",
):
    if image_dir:
        cam = ImageDirectoryCapture(image_dir)
    else:
        cam = get_camera(camera_mode)
        cam.connect()
    robot = get_robot_controller(robot_mode)
    robot.connect()
    session_id = f"agent_{uuid4().hex[:8]}"

    print(f"\nHardware Agent started")
    print(f"  Server: {server_url}")
    print(f"  Robot mode: {robot_mode}")
    print(f"  Session ID: {session_id}")
    if image_dir:
        print(f"  Camera: image-dir {image_dir}")
    else:
        print(f"  Camera: {camera_mode}")
    print(f"  Press Ctrl+C to stop\n")

    requests.post(f"{server_url}/reset")

    step = 0
    while True:
        temp_path = Path(tempfile.gettempdir()) / "hw_frame.jpg"
        if isinstance(cam, ImageDirectoryCapture):
            frame_path = cam.capture(str(temp_path))
        else:
            frame_path = cam.save_rgb(str(temp_path))
        print(f"\n--- Step {step} ---")
        print(f"  Captured frame: {frame_path}")

        with open(frame_path, "rb") as f:
            try:
                resp = requests.post(
                    f"{server_url}/decide",
                    files={"image": ("frame.jpg", f, "image/jpeg")},
                    data={
                        "context": context,
                        "task_description": task_description or context,
                        "robot_state": json.dumps(robot.get_status(), ensure_ascii=False),
                        "spatial_context": json.dumps({"camera_mode": "image_dir" if image_dir else "live_camera"}),
                        "metadata": json.dumps({"agent": "hardware_agent", "robot_mode": robot_mode}),
                        "use_cache": str(use_cache).lower(),
                        "session_id": session_id,
                    },
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

        decision = ActionDecision.from_dict(resp.json())
        action = decision.action
        target = decision.target
        source = decision.source
        latency = decision.latency_ms

        tag = source.upper()
        print(f"  Decision: [{tag}] {action} {target} ({latency:.0f}ms)")
        print(f"  Reason: {decision.reason}")
        if decision.case_id:
            print(f"  Case ID: {decision.case_id}")

        if action == "done":
            print("\n  Task complete!")
            break

        try:
            success = dispatch_action(robot, action, target)
        except ValueError as exc:
            print(f"  ERROR: {exc}")
            success = False

        requests.post(f"{server_url}/feedback", json={
            "action": action,
            "target": target,
            "success": success,
            "case_id": decision.case_id,
            "session_id": session_id,
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
    print(f"  Cases logged: {status.get('total_cases', 0)}")
    print(f"  Completed targets: {status.get('completed_targets', [])}")
    print(f"{'='*50}")

    if isinstance(cam, ImageDirectoryCapture):
        cam.release()
    else:
        cam.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hardware Agent for generic robot MVP")
    parser.add_argument("--server", default="http://localhost:8100",
                       help="Decision service URL")
    parser.add_argument("--camera-mode", default="mock",
                       help="Camera mode: mock or orbbec")
    parser.add_argument("--context", default="",
                       help="Extra scene context for the decision service")
    parser.add_argument("--task", default="",
                       help="Task description for the current run")
    parser.add_argument("--robot-mode", default="mock",
                       help="Robot controller mode: mock or nero")
    parser.add_argument("--image-dir", default="",
                       help="Replay images from a directory instead of a live camera")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable decision cache reuse")
    args = parser.parse_args()
    run_agent(
        args.server,
        context=args.context,
        task_description=args.task,
        robot_mode=args.robot_mode,
        image_dir=args.image_dir,
        use_cache=not args.no_cache,
        camera_mode=args.camera_mode,
    )
