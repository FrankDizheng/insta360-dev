#!/usr/bin/env python3
"""Step 4: Move to placement location, release object, and return home."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/pi")
from handeye_board_runtime import connect_robot, load_handeye, load_tcp_offset
from safe_motion import (
    check_target_safe,
    get_current_pose,
    safe_lift,
    safe_move_to,
    wait_move_done,
)


def main():
    parser = argparse.ArgumentParser(description="Place object at destination")
    parser.add_argument("--objects", required=True, help="Path to objects.json")
    parser.add_argument("--destination", required=True,
                        help="Destination object key in objects.json (e.g. 'blue_board')")
    parser.add_argument("--handeye", required=True, help="Path to handeye_result.json")
    parser.add_argument("--tcp", default=None, help="Path to TCP offset JSON")
    parser.add_argument("--place-z-mm", type=float, default=50,
                        help="Absolute Z height in base frame (mm) for placement")
    parser.add_argument("--z-safe-mm", type=float, default=300,
                        help="Transit height in mm for horizontal moves (default 300 mm)")
    parser.add_argument("--speed", type=int, default=10, help="Robot speed percent")
    args = parser.parse_args()

    print(f"[step4] Loading objects from {args.objects}")
    objects_data = json.loads(Path(args.objects).read_text(encoding="utf-8"))
    if args.destination not in objects_data["objects"]:
        available = list(objects_data["objects"].keys())
        raise RuntimeError(f"Destination '{args.destination}' not found. Available: {available}")

    dest_xyz = objects_data["objects"][args.destination]["base_xyz_m"]
    print(f"[step4] Destination '{args.destination}' at base "
          f"[{dest_xyz[0]:.4f}, {dest_xyz[1]:.4f}, {dest_xyz[2]:.4f}] m")

    load_handeye(args.handeye)

    print("[step4] Connecting to robot ...")
    robot = connect_robot()
    robot.enable()
    robot.set_speed_percent(args.speed)
    time.sleep(0.1)

    tcp_offset = None
    if args.tcp:
        tcp_offset = load_tcp_offset(args.tcp)
        robot.set_tcp_offset(tcp_offset)
        print(f"[step4] TCP offset set: {[round(v, 5) for v in tcp_offset]}")
        time.sleep(0.2)

    print("[step4] Initializing gripper ...")
    effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

    if tcp_offset:
        current_rpy = list(robot.get_tcp_pose().msg[3:6])
    else:
        current_rpy = get_current_pose(robot)[3:6].tolist()

    z_safe = args.z_safe_mm / 1000.0
    above_pose = [dest_xyz[0], dest_xyz[1], z_safe, *current_rpy]
    if tcp_offset:
        flange_above = robot.get_tcp2flange_pose(above_pose)
    else:
        flange_above = above_pose

    check_target_safe(flange_above)

    print(f"[step4] Moving above destination (speed={args.speed}%, z_safe={z_safe:.3f} m) ...")
    safe_move_to(robot, flange_above, z_safe_m=z_safe, speed_pct=args.speed)

    if tcp_offset:
        current_rpy = list(robot.get_tcp_pose().msg[3:6])
    else:
        current_rpy = get_current_pose(robot)[3:6].tolist()

    place_z = args.place_z_mm / 1000.0
    place_pose = [dest_xyz[0], dest_xyz[1], place_z, *current_rpy]
    if tcp_offset:
        flange_place = robot.get_tcp2flange_pose(place_pose)
    else:
        flange_place = place_pose

    check_target_safe(flange_place, z_min_m=0.02)

    print(f"[step4] Lowering to place height Z={place_z:.4f} m (move_l) ...")
    robot.set_speed_percent(args.speed)
    time.sleep(0.05)
    robot.move_l(flange_place)
    wait_move_done(robot, flange_place[:3])

    print("[step4] Opening gripper to release ...")
    effector.move_gripper(width=0.06, force=1.0)
    time.sleep(1.5)   # wait for gripper to open
    time.sleep(0.8)   # extra settle: let arm stabilize after load change

    print("[step4] Lifting away ...")
    safe_lift(robot, height_m=0.25)

    print("[step4] Moving to home position ...")
    robot.set_speed_percent(args.speed)
    time.sleep(0.05)
    robot.move_j([0, 0, 0, 0, 0, 0, 0])

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        ja = robot.get_joint_angles()
        if ja is not None and ja.msg is not None:
            angles = np.array(ja.msg[:7], dtype=np.float64)
            if np.max(np.abs(angles)) < 0.02:
                time.sleep(0.5)
                break
        time.sleep(0.1)
    else:
        print("[step4] WARNING: Joint move to home timed out")

    print("[step4] Pick-and-place complete!")


if __name__ == "__main__":
    main()
