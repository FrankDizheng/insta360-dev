import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from handeye_board_runtime import (
    board_point_to_base,
    capture_live_board_result_with_robot,
    load_handeye,
    load_tcp_offset,
    resolve_board_target,
    save_json,
    connect_robot,
    wait_motion_done,
)


def ensure_enabled(robot, timeout_s: float = 6.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if robot.enable():
            return
        time.sleep(0.1)
    raise RuntimeError("Failed to enable robot")


def current_tcp_pose_or_raise(robot) -> list[float]:
    tcp = robot.get_tcp_pose()
    if tcp is None:
        raise RuntimeError("TCP pose unavailable")
    return list(tcp.msg)


def current_flange_pose_or_raise(robot) -> list[float]:
    flange = robot.get_flange_pose()
    if flange is None:
        raise RuntimeError("Flange pose unavailable")
    return list(flange.msg)


def log_step(name: str) -> None:
    print(f"[step] {name}", flush=True)


def load_locked_rpy(path: str | Path) -> tuple[list[float], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = [
        "approach_reached_tcp_pose_xyzrpy_m_rad",
        "approach_pose_xyzrpy_m_rad",
        "pretouch_reached_tcp_pose_xyzrpy_m_rad",
        "pretouch_pose_xyzrpy_m_rad",
        "target_reached_tcp_pose_xyzrpy_m_rad",
        "target_pose_xyzrpy_m_rad",
        "current_tcp_pose_xyzrpy_m_rad",
        "fixed_pose_xyzrpy_m_rad",
    ]
    for key in candidates:
        pose = data.get(key)
        if pose is not None and len(pose) >= 6:
            return list(pose[3:6]), key
    raise RuntimeError(f"No pose with rpy found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or execute a guarded board-target touch using a calibrated TCP.")
    parser.add_argument("--handeye", required=True, help="Path to handeye_result.json")
    parser.add_argument("--tcp", required=True, help="Path to gripper TCP JSON")
    parser.add_argument("--output-dir", default="", help="Directory for touch plans and logs")
    parser.add_argument("--camera-device", default="auto", help="V4L2 color camera device path or 'auto'")
    parser.add_argument("--robot-channel", default="can0", help="Robot CAN channel")
    parser.add_argument("--target", choices=["origin", "center", "board_xy"], default="origin", help="Board target")
    parser.add_argument("--board-xy", type=float, nargs=2, metavar=("X_M", "Y_M"), help="Board X/Y in meters")
    parser.add_argument("--locked-rpy-json", default="", help="Use roll/pitch/yaw from a known-safe JSON result instead of current TCP pose")
    parser.add_argument("--lift-first-mm", type=float, default=0.0, help="Lift upward in base Z before board localization and approach")
    parser.add_argument("--approach-mm", type=float, default=30.0, help="Stand-off distance from the board")
    parser.add_argument("--pretouch-mm", type=float, default=3.0, help="Final stand-off before touch")
    parser.add_argument("--approach-sign", choices=["positive", "negative"], default="positive", help="Board normal side")
    parser.add_argument("--speed-percent", type=int, default=10, help="Robot speed percent during execution")
    parser.add_argument("--execute", action="store_true", help="Actually move to approach and pre-touch poses")
    parser.add_argument("--approach-only", action="store_true", help="Stop after reaching the safe approach pose")
    parser.add_argument("--execute-touch", action="store_true", help="Allow the final move to the visual target point")
    args = parser.parse_args()

    if args.pretouch_mm < 0.0 or args.approach_mm <= args.pretouch_mm:
        raise RuntimeError("Need approach-mm > pretouch-mm >= 0")
    if args.execute_touch and not args.execute:
        raise RuntimeError("--execute-touch requires --execute")
    if args.execute_touch and args.approach_only:
        raise RuntimeError("--execute-touch cannot be used with --approach-only")

    handeye_path = Path(args.handeye)
    handeye = load_handeye(handeye_path)
    tcp_offset = load_tcp_offset(args.tcp)
    output_dir = Path(args.output_dir) if args.output_dir else handeye_path.parent / "touch_board_target"
    output_dir.mkdir(parents=True, exist_ok=True)

    robot = connect_robot(robot_channel=args.robot_channel)
    log_step("robot_connected")
    robot.set_tcp_offset(tcp_offset)
    current_tcp_pose = current_tcp_pose_or_raise(robot)

    lift_start_flange_pose = None
    lift_target_flange_pose = None
    lift_final_flange_pose = None
    lift_delta_xyz = None
    lift_motion_reached = None

    if args.execute and args.lift_first_mm > 0.0:
        log_step("lift_start")
        ensure_enabled(robot)
        lift_start_flange_pose = current_flange_pose_or_raise(robot)
        lift_target_flange_pose = [
            lift_start_flange_pose[0],
            lift_start_flange_pose[1],
            lift_start_flange_pose[2] + args.lift_first_mm / 1000.0,
            lift_start_flange_pose[3],
            lift_start_flange_pose[4],
            lift_start_flange_pose[5],
        ]
        robot.set_speed_percent(args.speed_percent)
        robot.move_p(lift_target_flange_pose)
        lift_motion_reached = wait_motion_done(robot, timeout_s=20.0)
        if not lift_motion_reached:
            raise RuntimeError("Timeout while moving to lift-first pose")
        lift_final_flange_pose = current_flange_pose_or_raise(robot)
        lift_delta_xyz = [
            lift_final_flange_pose[0] - lift_start_flange_pose[0],
            lift_final_flange_pose[1] - lift_start_flange_pose[1],
            lift_final_flange_pose[2] - lift_start_flange_pose[2],
        ]
        log_step("lift_done")

    log_step("capture_board_start")
    live_result, _, _ = capture_live_board_result_with_robot(
        handeye=handeye,
        robot=robot,
        camera_device=args.camera_device,
    )
    log_step("capture_board_done")
    target_board_xyz, target_name = resolve_board_target(handeye["board"], args.target, args.board_xy)
    target_base_xyz = board_point_to_base(live_result["base_T_board"], target_board_xyz)
    board_normal = np.array(live_result["base_T_board"][:3, 2], dtype=np.float64)
    board_normal /= np.linalg.norm(board_normal)
    sign = 1.0 if args.approach_sign == "positive" else -1.0
    approach_vec = sign * board_normal

    if args.locked_rpy_json:
        target_rpy, locked_rpy_source = load_locked_rpy(args.locked_rpy_json)
    else:
        target_rpy = current_tcp_pose[3:6]
        locked_rpy_source = "current_tcp_pose_xyzrpy_m_rad"

    approach_pose = [
        float(target_base_xyz[0] + approach_vec[0] * args.approach_mm / 1000.0),
        float(target_base_xyz[1] + approach_vec[1] * args.approach_mm / 1000.0),
        float(target_base_xyz[2] + approach_vec[2] * args.approach_mm / 1000.0),
        *target_rpy,
    ]
    pretouch_pose = [
        float(target_base_xyz[0] + approach_vec[0] * args.pretouch_mm / 1000.0),
        float(target_base_xyz[1] + approach_vec[1] * args.pretouch_mm / 1000.0),
        float(target_base_xyz[2] + approach_vec[2] * args.pretouch_mm / 1000.0),
        *target_rpy,
    ]
    target_pose = [float(target_base_xyz[0]), float(target_base_xyz[1]), float(target_base_xyz[2]), *target_rpy]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"touch_plan_{stamp}.json"
    result = {
        "timestamp": stamp,
        "mode": "dry_run",
        "touch_mode": "commanded_pose_no_force_feedback",
        "target_name": target_name,
        "target_point_in_board_xyz_m": target_board_xyz,
        "target_point_in_base_xyz_m": target_base_xyz.tolist(),
        "board_normal_in_base_xyz": board_normal.tolist(),
        "tcp_offset_xyzrpy_m_rad": tcp_offset,
        "current_tcp_pose_xyzrpy_m_rad": current_tcp_pose,
        "lift_first_mm": args.lift_first_mm,
        "lift_motion_reached": lift_motion_reached,
        "lift_start_flange_pose_xyzrpy_m_rad": lift_start_flange_pose,
        "lift_target_flange_pose_xyzrpy_m_rad": lift_target_flange_pose,
        "lift_final_flange_pose_xyzrpy_m_rad": lift_final_flange_pose,
        "lift_delta_xyz_m": lift_delta_xyz,
        "locked_rpy_source": locked_rpy_source,
        "locked_rpy_json": args.locked_rpy_json if args.locked_rpy_json else None,
        "target_rpy_m_rad": target_rpy,
        "approach_pose_xyzrpy_m_rad": approach_pose,
        "pretouch_pose_xyzrpy_m_rad": pretouch_pose,
        "target_pose_xyzrpy_m_rad": target_pose,
        "speed_percent": args.speed_percent,
    }

    try:
        if args.execute:
            log_step("approach_start")
            ensure_enabled(robot)
            robot.set_speed_percent(args.speed_percent)
            robot.move_p(robot.get_tcp2flange_pose(approach_pose))
            if not wait_motion_done(robot, timeout_s=15.0):
                raise RuntimeError("Timeout while moving to approach pose")

            result["mode"] = "approach_executed"
            result["approach_reached_tcp_pose_xyzrpy_m_rad"] = current_tcp_pose_or_raise(robot)
            log_step("approach_done")

            if not args.approach_only:
                log_step("pretouch_start")
                robot.move_l(robot.get_tcp2flange_pose(pretouch_pose))
                if not wait_motion_done(robot, timeout_s=15.0):
                    raise RuntimeError("Timeout while moving to pre-touch pose")

                result["mode"] = "pretouch_executed"
                result["pretouch_reached_tcp_pose_xyzrpy_m_rad"] = current_tcp_pose_or_raise(robot)
                log_step("pretouch_done")

            if args.execute_touch:
                log_step("touch_confirm_wait")
                confirm = input("Type TOUCH to execute the final straight move to the target point: ").strip()
                if confirm == "TOUCH":
                    log_step("touch_start")
                    robot.move_l(robot.get_tcp2flange_pose(target_pose))
                    if not wait_motion_done(robot, timeout_s=10.0):
                        raise RuntimeError("Timeout while moving to target pose")
                    result["mode"] = "target_executed"
                    result["target_reached_tcp_pose_xyzrpy_m_rad"] = current_tcp_pose_or_raise(robot)
                    reached = np.array(result["target_reached_tcp_pose_xyzrpy_m_rad"][:3], dtype=np.float64)
                    result["commanded_target_error_xyz_m"] = (reached - target_base_xyz).tolist()
                    result["commanded_target_error_norm_m"] = float(np.linalg.norm(reached - target_base_xyz))
                    log_step("touch_done")
                else:
                    result["final_touch_cancelled"] = True
    except Exception as exc:
        result["mode"] = "failed"
        result["error"] = str(exc)
        save_json(out_path, result)
        print(f"saved_touch_plan_json: {out_path}")
        print("error:", result["error"])
        raise

    save_json(out_path, result)

    print(f"saved_touch_plan_json: {out_path}")
    print("target_point_in_base_xyz_m:", result["target_point_in_base_xyz_m"])
    print("approach_pose_xyzrpy_m_rad:", result["approach_pose_xyzrpy_m_rad"])
    print("pretouch_pose_xyzrpy_m_rad:", result["pretouch_pose_xyzrpy_m_rad"])
    if "target_pose_xyzrpy_m_rad" in result:
        print("target_pose_xyzrpy_m_rad:", result["target_pose_xyzrpy_m_rad"])


if __name__ == "__main__":
    main()
