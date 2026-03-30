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
    safe_move_to,
    wait_move_done,
)


def lift_straight_up(robot, target_height_m: float, speed_pct: int, tcp_offset: list[float] | None) -> None:
    """Raise straight up in base Z after release.

    This avoids any horizontal retreat immediately after opening the gripper.
    """
    if tcp_offset:
        current_tcp = robot.get_tcp_pose()
        if current_tcp is None or current_tcp.msg is None:
            raise RuntimeError("TCP pose unavailable for post-release lift")
        lift_pose = list(current_tcp.msg[:6])
        current_z = float(lift_pose[2])
        if current_z >= target_height_m - 0.001:
            print(f"[step4] Post-release lift skipped: TCP Z={current_z:.4f} m >= {target_height_m:.4f} m")
            return
        lift_pose[2] = target_height_m
        flange_target = robot.get_tcp2flange_pose(lift_pose)
    else:
        current_flange = get_current_pose(robot)
        current_z = float(current_flange[2])
        if current_z >= target_height_m - 0.001:
            print(f"[step4] Post-release lift skipped: flange Z={current_z:.4f} m >= {target_height_m:.4f} m")
            return
        lift_pose = current_flange.copy()
        lift_pose[2] = target_height_m
        flange_target = lift_pose.tolist()

    check_target_safe(flange_target)

    print(f"[step4] Lifting straight up to Z={target_height_m:.4f} m (move_l) ...")
    robot.set_speed_percent(speed_pct)
    time.sleep(0.05)
    robot.move_l(flange_target)
    wait_move_done(robot, flange_target[:3])


def place_object(
    robot,
    effector,
    args,
    tcp_offset: list[float] | None,
    destination: str,
    place_z_mm: float,
    z_safe_mm: float,
) -> dict:
    print(f"[step4] Loading objects from {args.objects}")
    objects_data = json.loads(Path(args.objects).read_text(encoding="utf-8"))
    if destination not in objects_data["objects"]:
        available = list(objects_data["objects"].keys())
        raise RuntimeError(f"Destination '{destination}' not found. Available: {available}")

    dest_xyz = objects_data["objects"][destination]["base_xyz_m"]
    print(f"[step4] Destination '{destination}' at base "
          f"[{dest_xyz[0]:.4f}, {dest_xyz[1]:.4f}, {dest_xyz[2]:.4f}] m")

    if tcp_offset:
        current_rpy = list(robot.get_tcp_pose().msg[3:6])
    else:
        current_rpy = get_current_pose(robot)[3:6].tolist()

    z_safe = z_safe_mm / 1000.0
    if z_safe < 0.05:
        raise RuntimeError(f"[step4] z-safe-mm={z_safe_mm} results in "
                           f"Z={z_safe:.4f} m, below 50 mm safety minimum")
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

    place_z = place_z_mm / 1000.0
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

    # Ensure a meaningful vertical clearance before any joint-space retreat.
    post_release_z = max(z_safe, place_z + 0.05)
    print("[step4] Lifting away vertically after release ...")
    lift_straight_up(robot, target_height_m=post_release_z, speed_pct=args.speed, tcp_offset=tcp_offset)

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
    return {
        "destination": destination,
        "place_z_mm": place_z_mm,
        "z_safe_mm": z_safe_mm,
        "dest_xyz": dest_xyz,
    }


def print_persistent_help(default_destination: str, default_place_z_mm: float, default_z_safe_mm: float) -> None:
    print(
        "\n[persistent] Commands:\n"
        f"  <enter> or place                           Place at {default_destination} using Z={default_place_z_mm:.0f} mm\n"
        "  place <destination>                        Place at another destination using current heights\n"
        "  place <destination> <place-z-mm>          Place with a new drop height\n"
        "  place <destination> <place-z-mm> <z-safe-mm>\n"
        "                                            Place with a new drop and transit height\n"
        "  help                                       Show this help\n"
        "  quit                                       Exit persistent mode"
    )


def parse_place_command(
    raw: str,
    default_destination: str,
    default_place_z_mm: float,
    default_z_safe_mm: float,
) -> tuple[str, float, float]:
    if raw in {"", "place"}:
        return default_destination, default_place_z_mm, default_z_safe_mm

    parts = raw.split()
    if parts[0] != "place":
        raise ValueError("Unknown command")
    if len(parts) == 2:
        return parts[1], default_place_z_mm, default_z_safe_mm
    if len(parts) == 3:
        return parts[1], float(parts[2]), default_z_safe_mm
    if len(parts) == 4:
        return parts[1], float(parts[2]), float(parts[3])
    raise ValueError("Usage: place [destination] [place-z-mm] [z-safe-mm]")


def run_persistent_session(args) -> None:
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

    current_destination = args.destination
    current_place_z_mm = args.place_z_mm
    current_z_safe_mm = args.z_safe_mm
    print_persistent_help(current_destination, current_place_z_mm, current_z_safe_mm)

    while True:
        try:
            raw = input("\n[persistent] command> ").strip()
        except EOFError:
            print("\n[persistent] EOF received, exiting.")
            break

        if raw in {"quit", "q", "exit"}:
            print("[persistent] Exiting.")
            break
        if raw in {"help", "?"}:
            print_persistent_help(current_destination, current_place_z_mm, current_z_safe_mm)
            continue

        try:
            destination, place_z_mm, z_safe_mm = parse_place_command(
                raw,
                current_destination,
                current_place_z_mm,
                current_z_safe_mm,
            )
            place_object(robot, effector, args, tcp_offset, destination, place_z_mm, z_safe_mm)
            current_destination = destination
            current_place_z_mm = place_z_mm
            current_z_safe_mm = z_safe_mm
        except Exception as exc:
            print(f"[persistent] ERROR: {exc}")


def run_single(args) -> None:
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
    place_object(robot, effector, args, tcp_offset, args.destination, args.place_z_mm, args.z_safe_mm)


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
    parser.add_argument("--persistent", action="store_true",
                        help="Keep robot connection alive for repeated place commands")
    args = parser.parse_args()

    if args.persistent:
        run_persistent_session(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
