import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pyAgxArm import AgxArmFactory, create_agx_arm_config


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    """Convert XYZ roll/pitch/yaw to a 3x3 rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])

    return (rz @ ry @ rx).tolist()


def pose_to_transform(pose: list[float]) -> list[list[float]]:
    x, y, z, roll, pitch, yaw = pose
    rot = rpy_to_matrix(roll, pitch, yaw)
    return [
        [rot[0][0], rot[0][1], rot[0][2], x],
        [rot[1][0], rot[1][1], rot[1][2], y],
        [rot[2][0], rot[2][1], rot[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def open_camera(camera_source: str, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {camera_source}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_fresh_frame(cap: cv2.VideoCapture, flush_count: int = 5) -> np.ndarray:
    frame = None
    ok = False
    for _ in range(flush_count):
        ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read a camera frame")
    return frame


def wait_for_robot_feedback(robot, timeout_s: float = 3.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pose = robot.get_flange_pose()
        joints = robot.get_joint_angles()
        if pose is not None and joints is not None:
            return
        time.sleep(0.05)
    raise RuntimeError("Robot feedback timeout: flange pose or joint angles unavailable")


def ensure_robot_feedback(robot) -> None:
    """Try current mode first, then switch to normal mode to enable CAN push."""
    try:
        wait_for_robot_feedback(robot, timeout_s=2.0)
        return
    except RuntimeError:
        pass

    print("No robot feedback yet, switching to normal mode...")
    robot.set_normal_mode()
    time.sleep(0.5)
    wait_for_robot_feedback(robot, timeout_s=4.0)


def next_sample_index(output_dir: Path) -> int:
    existing = sorted(output_dir.glob("sample_*.json"))
    if not existing:
        return 1
    return max(int(p.stem.split("_")[-1]) for p in existing) + 1


def save_sample(output_dir: Path, sample_idx: int, frame: np.ndarray, board: dict, args, robot) -> bool:
    flange_pose_msg = robot.get_flange_pose()
    joint_angles_msg = robot.get_joint_angles()
    arm_status_msg = robot.get_arm_status()

    if flange_pose_msg is None or joint_angles_msg is None:
        print("Robot pose unavailable, sample skipped.")
        return False

    flange_pose = list(flange_pose_msg.msg)
    joint_angles = list(joint_angles_msg.msg)

    image_name = f"sample_{sample_idx:03d}.png"
    json_name = f"sample_{sample_idx:03d}.json"
    image_path = output_dir / image_name
    json_path = output_dir / json_name

    cv2.imwrite(str(image_path), frame)

    sample = {
        "sample_index": sample_idx,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "image_path": image_name,
        "image_shape": list(frame.shape),
        "camera": {
            "device": args.camera_device,
            "width": args.width,
            "height": args.height,
        },
        "board": board,
        "robot": {
            "frame": "flange",
            "channel": args.robot_channel,
            "flange_pose_xyzrpy_m_rad": flange_pose,
            "base_T_flange": pose_to_transform(flange_pose),
            "joint_angles_rad": joint_angles,
            "arm_status": str(arm_status_msg.msg) if arm_status_msg is not None else None,
        },
    }

    json_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"Saved {image_name} and {json_name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect eye-in-hand hand-eye calibration samples.")
    parser.add_argument("--output-dir", default="", help="Dataset directory. Defaults to timestamped folder.")
    parser.add_argument("--camera-device", default="auto", help="V4L2 color camera device path or 'auto'.")
    parser.add_argument("--width", type=int, default=1920, help="Color capture width.")
    parser.add_argument("--height", type=int, default=1080, help="Color capture height.")
    parser.add_argument("--robot-channel", default="can0", help="Robot CAN channel.")
    parser.add_argument("--single", action="store_true", help="Save a single sample and exit.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path.home() / "handeye_dataset" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    board = {
        "type": "charuco",
        "squares_x": 12,
        "squares_y": 9,
        "square_length_m": 0.015,
        "marker_length_m": 0.01125,
        "board_outer_size_m": [0.2, 0.2],
    }

    print(f"Saving dataset to: {output_dir}")
    print("Connecting robot...")
    robot_cfg = create_agx_arm_config(
        robot="nero",
        comm="can",
        channel=args.robot_channel,
        interface="socketcan",
    )
    robot = AgxArmFactory.create_arm(robot_cfg)
    robot.connect()
    ensure_robot_feedback(robot)

    print("Opening color camera...")
    cap = open_camera(args.camera_device, args.width, args.height)
    time.sleep(0.5)

    print()
    print("Hand-eye capture ready.")
    print("Move the robot to a new pose where the full board is visible.")
    print("Press Enter to save one sample, or type q then Enter to quit.")
    print()

    sample_idx = next_sample_index(output_dir)
    try:
        if args.single:
            frame = read_fresh_frame(cap)
            save_sample(output_dir, sample_idx, frame, board, args, robot)
            return

        while True:
            cmd = input(f"[sample {sample_idx:03d}] Enter=save, q=quit: ").strip().lower()
            if cmd == "q":
                break

            frame = read_fresh_frame(cap)
            if save_sample(output_dir, sample_idx, frame, board, args, robot):
                sample_idx += 1
    finally:
        cap.release()


if __name__ == "__main__":
    main()
