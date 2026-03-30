#!/usr/bin/env python3
"""Step 3: Lower arm from standoff, grasp object, and lift back up."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/pi")
from handeye_board_runtime import connect_robot, load_handeye, load_tcp_offset
from safe_motion import check_target_safe, get_current_pose, safe_lift, wait_move_done


def main():
    parser = argparse.ArgumentParser(description="Grasp target object")
    parser.add_argument("--objects", required=True, help="Path to objects.json")
    parser.add_argument("--object-name", required=True, help="Target object key in objects.json")
    parser.add_argument("--handeye", required=True, help="Path to handeye_result.json")
    parser.add_argument("--tcp", default=None, help="Path to TCP offset JSON")
    parser.add_argument("--grasp-z-mm", type=float, default=50,
                        help="Absolute Z height in base frame (mm) for grasping")
    parser.add_argument("--gripper-width", type=float, default=0.05,
                        help="Gripper open width before grasp (m)")
    parser.add_argument("--gripper-force", type=float, default=1.0, help="Gripper close force")
    parser.add_argument("--speed", type=int, default=8, help="Speed percent for descent")
    args = parser.parse_args()

    grasp_z = args.grasp_z_mm / 1000.0
    if grasp_z < 0.02:
        raise RuntimeError(f"[step3] grasp-z-mm={args.grasp_z_mm} results in "
                           f"Z={grasp_z:.4f} m, below 20 mm safety minimum")

    print(f"[step3] Loading objects from {args.objects}")
    objects_data = json.loads(Path(args.objects).read_text(encoding="utf-8"))
    obj = objects_data["objects"].get(args.object_name)
    if obj:
        print(f"[step3] Reference: '{args.object_name}' at base {obj['base_xyz_m']}")

    load_handeye(args.handeye)

    print("[step3] Connecting to robot ...")
    robot = connect_robot()
    robot.enable()
    robot.set_speed_percent(args.speed)
    time.sleep(0.1)

    tcp_offset = None
    if args.tcp:
        tcp_offset = load_tcp_offset(args.tcp)
        robot.set_tcp_offset(tcp_offset)
        print(f"[step3] TCP offset set: {[round(v, 5) for v in tcp_offset]}")
        time.sleep(0.2)

    print("[step3] Initializing gripper ...")
    effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

    print(f"[step3] Opening gripper (width={args.gripper_width} m) ...")
    effector.move_gripper(width=args.gripper_width, force=1.0)
    time.sleep(1.5)

    if tcp_offset:
        current = np.array(robot.get_tcp_pose().msg[:6], dtype=np.float64)
    else:
        current = get_current_pose(robot)

    descent_pose = [float(current[0]), float(current[1]), grasp_z,
                    float(current[3]), float(current[4]), float(current[5])]
    print(f"[step3] Descent target: "
          f"[{descent_pose[0]:.4f}, {descent_pose[1]:.4f}, {descent_pose[2]:.4f}] m "
          f"(grasp Z = {args.grasp_z_mm:.0f} mm)")

    if tcp_offset:
        flange_descent = robot.get_tcp2flange_pose(descent_pose)
    else:
        flange_descent = descent_pose

    check_target_safe(flange_descent, z_min_m=0.02)

    print(f"[step3] Lowering to grasp height (move_l, speed={args.speed}%) ...")
    robot.set_speed_percent(args.speed)
    time.sleep(0.05)
    robot.move_l(flange_descent)
    wait_move_done(robot, flange_descent[:3])

    print(f"[step3] Closing gripper (force={args.gripper_force}) ...")
    effector.move_gripper(width=0.0, force=args.gripper_force)
    time.sleep(2.0)

    status = effector.get_gripper_status()
    grip_width = status.msg.width
    grip_force = status.msg.force
    print(f"[step3] Gripper status — width: {grip_width:.4f} m, force: {grip_force:.2f}")

    print("[step3] Lifting with grasped object ...")
    safe_lift(robot, height_m=0.25)

    final = get_current_pose(robot)
    print(f"[step3] Lift complete — flange at "
          f"[{final[0]:.4f}, {final[1]:.4f}, {final[2]:.4f}] m")
    grasped = grip_width > 0.001
    print(f"[step3] Grasp {'likely successful' if grasped else 'may have missed'} "
          f"(gripper width={grip_width:.4f} m)")


if __name__ == "__main__":
    main()
