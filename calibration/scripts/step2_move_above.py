#!/usr/bin/env python3
"""Step 2: Move robot arm to standoff position above a target object."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/pi")
from handeye_board_runtime import connect_robot, load_handeye, load_tcp_offset
from safe_motion import check_target_safe, get_current_pose, safe_move_to


def main():
    parser = argparse.ArgumentParser(description="Move arm above target object")
    parser.add_argument("--objects", required=True, help="Path to objects.json")
    parser.add_argument("--object-name", required=True, help="Target object key in objects.json")
    parser.add_argument("--handeye", required=True, help="Path to handeye_result.json")
    parser.add_argument("--tcp", default=None, help="Path to TCP offset JSON")
    parser.add_argument("--standoff-mm", type=float, default=100, help="Height above object in mm")
    parser.add_argument("--speed", type=int, default=10, help="Robot speed percent")
    args = parser.parse_args()

    print(f"[step2] Loading objects from {args.objects}")
    objects_data = json.loads(Path(args.objects).read_text(encoding="utf-8"))
    if args.object_name not in objects_data["objects"]:
        available = list(objects_data["objects"].keys())
        raise RuntimeError(f"Object '{args.object_name}' not found. Available: {available}")

    obj_xyz = objects_data["objects"][args.object_name]["base_xyz_m"]
    print(f"[step2] Target '{args.object_name}' at base "
          f"[{obj_xyz[0]:.4f}, {obj_xyz[1]:.4f}, {obj_xyz[2]:.4f}] m")

    load_handeye(args.handeye)
    print(f"[step2] Handeye calibration validated: {args.handeye}")

    print("[step2] Connecting to robot ...")
    robot = connect_robot()
    robot.enable()
    robot.set_speed_percent(args.speed)
    time.sleep(0.1)

    tcp_offset = None
    if args.tcp:
        tcp_offset = load_tcp_offset(args.tcp)
        robot.set_tcp_offset(tcp_offset)
        print(f"[step2] TCP offset set: {[round(v, 5) for v in tcp_offset]}")
        time.sleep(0.2)

    if tcp_offset:
        current_rpy = list(robot.get_tcp_pose().msg[3:6])
    else:
        current_rpy = get_current_pose(robot)[3:6].tolist()

    standoff_z = obj_xyz[2] + args.standoff_mm / 1000.0
    standoff_pose = [obj_xyz[0], obj_xyz[1], standoff_z, *current_rpy]
    print(f"[step2] Standoff pose: "
          f"[{standoff_pose[0]:.4f}, {standoff_pose[1]:.4f}, {standoff_pose[2]:.4f}] m "
          f"({args.standoff_mm:.0f} mm above object)")

    if tcp_offset:
        flange_target = robot.get_tcp2flange_pose(standoff_pose)
        print(f"[step2] Flange target: "
              f"[{flange_target[0]:.4f}, {flange_target[1]:.4f}, {flange_target[2]:.4f}] m")
    else:
        flange_target = standoff_pose

    check_target_safe(flange_target)

    print(f"[step2] Moving to standoff (speed={args.speed}%) ...")
    safe_move_to(robot, flange_target, z_safe_m=0.30, speed_pct=args.speed)

    final_flange = get_current_pose(robot)
    if tcp_offset:
        final_pos = np.array(robot.get_tcp_pose().msg[:3], dtype=np.float64)
        print(f"[step2] Final TCP position: "
              f"[{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}] m")
    else:
        final_pos = final_flange[:3]
        print(f"[step2] Final flange position: "
              f"[{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}] m")

    obj_xy = np.array(obj_xyz[:2], dtype=np.float64)
    dist_xy = np.linalg.norm(final_pos[:2] - obj_xy) * 1000.0
    dist_z = (final_pos[2] - obj_xyz[2]) * 1000.0
    print(f"[step2] Distance from object — XY: {dist_xy:.1f} mm, Z above: {dist_z:.1f} mm")
    print("[step2] Ready for step3 (grasp)")


if __name__ == "__main__":
    main()
