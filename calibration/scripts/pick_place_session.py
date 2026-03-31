#!/usr/bin/env python3
"""Single-process pick-and-place session.

Connects to the robot and camera ONCE, then runs the full
scan → locate → move_above → grasp → place → home pipeline
in one resident process — eliminating repeated reconnect overhead.

Raspberry Pi role: hardware I/O hub (CAN / camera / SDK).
All coordinate math runs here but is lightweight (NumPy only).
Heavy inference (VLM) must NOT run on Pi; invoke from a laptop
over the network before calling this script.

Typical usage:
    python3 pick_place_session.py \\
        --handeye  /home/pi/calibration/results/session1/handeye_result.json \\
        --tcp      /home/pi/calibration/results/session1/gripper_tcp_left_front_tip_samples_004_006.json \\
        --scan-pose "-19.4,10.7,4.6,63.0,7.1,1.4,56.6" \\
        --output-dir /home/pi/session
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/pi")
from handeye_board_runtime import (
    PersistentAlignedRGBDCapture,
    connect_robot,
    load_handeye,
    load_tcp_offset,
    pose_to_transform,
)
from safe_motion import (
    check_arm_error,
    check_target_safe,
    get_current_pose,
    safe_lift,
    safe_move_to,
    wait_move_done,
)

_TAG = "[session]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confirm(prompt: str) -> bool:
    """Ask user y/n; return True for yes."""
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


def _parse_scan_pose(raw: str) -> list[float]:
    parts = [s.strip() for s in raw.split(",")]
    if len(parts) != 7:
        raise ValueError(f"scan-pose must have 7 comma-separated values, got {len(parts)}")
    return [float(p) for p in parts]


def _parse_pixel_coords(raw: str) -> dict[str, tuple[int, int]]:
    result = {}
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected 'name:u:v', got '{entry.strip()}'")
        result[parts[0].strip()] = (int(parts[1]), int(parts[2]))
    return result


def _depth_at_pixel(depth_u16: np.ndarray, u: int, v: int,
                    depth_scale: float, window: int = 7) -> float | None:
    h, w = depth_u16.shape
    half = window // 2
    roi = depth_u16[max(0, v - half):min(h, v + half + 1),
                    max(0, u - half):min(w, u + half + 1)]
    valid = roi[roi > 0]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) * depth_scale / 1000.0


def _pixel_to_base(u: int, v: int, depth_m: float,
                   intrinsics: dict, base_T_camera: np.ndarray) -> np.ndarray:
    x = (u - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    p_cam_h = np.array([x, y, depth_m, 1.0])
    return (base_T_camera @ p_cam_h)[:3]


def _depth_colormap(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    depth_mm = depth_u16.astype(np.float32) * depth_scale
    valid = depth_mm > 0
    if not np.any(valid):
        return np.zeros((*depth_u16.shape, 3), dtype=np.uint8)
    max_mm = float(np.percentile(depth_mm[valid], 98))
    gray = (np.clip(depth_mm / max(max_mm, 1.0), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase_move_to_scan_pose(robot, scan_pose_deg: list[float], speed_pct: int) -> None:
    """Move joints to the fixed scan pose."""
    scan_pose_rad = [math.radians(d) for d in scan_pose_deg]
    print(f"{_TAG} Moving to scan pose: {[round(d, 1) for d in scan_pose_deg]}°")
    robot.set_speed_percent(speed_pct)
    time.sleep(0.05)
    robot.move_j(scan_pose_rad)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        check_arm_error(robot)
        ja = robot.get_joint_angles()
        if ja is not None and ja.msg is not None:
            current = np.array(ja.msg[:7], dtype=np.float64)
            target = np.array(scan_pose_rad, dtype=np.float64)
            if np.max(np.abs(current - target)) < 0.03:
                time.sleep(0.3)
                print(f"{_TAG} Scan pose reached.")
                return
        time.sleep(0.1)
    raise RuntimeError(f"{_TAG} Timed out moving to scan pose")


def phase_scan(robot, handeye: dict, camera: PersistentAlignedRGBDCapture,
               out_dir: Path, flush_frames: int) -> dict:
    """Capture RGBD, save images, return scan data dict."""
    fp = robot.get_flange_pose()
    if fp is None or fp.msg is None:
        raise RuntimeError(f"{_TAG} Flange pose unavailable")
    flange_pose = np.array(fp.msg[:6], dtype=np.float64)
    base_T_camera = pose_to_transform(flange_pose) @ handeye["flange_T_camera_np"]

    print(f"{_TAG} Capturing RGBD ...")
    color_bgr, depth_u16, depth_scale, intrinsics = camera.capture(flush_frames=flush_frames)

    out_dir.mkdir(parents=True, exist_ok=True)
    color_path = out_dir / "scan_color.jpg"
    depth_path = out_dir / "scan_depth.jpg"
    cv2.imwrite(str(color_path), color_bgr)
    cv2.imwrite(str(depth_path), _depth_colormap(depth_u16, depth_scale))
    print(f"{_TAG} Saved {color_path}")
    print(f"{_TAG} Saved {depth_path}")
    print(f"{_TAG} Camera height: {base_T_camera[2, 3] * 1000:.1f} mm above base")

    return {
        "flange_pose": flange_pose,
        "base_T_camera": base_T_camera,
        "color_bgr": color_bgr,
        "depth_u16": depth_u16,
        "depth_scale": depth_scale,
        "intrinsics": intrinsics,
        "color_path": color_path,
    }


def phase_locate(scan: dict, pixel_coords_raw: str,
                 out_dir: Path) -> dict[str, dict]:
    """Convert pixel coords to 3D base-frame positions. Returns objects dict."""
    coords = _parse_pixel_coords(pixel_coords_raw)
    objects: dict[str, dict] = {}

    for name, (u, v) in coords.items():
        depth_m = _depth_at_pixel(scan["depth_u16"], u, v, scan["depth_scale"])
        if depth_m is None or depth_m <= 0:
            print(f"{_TAG}   {name}: pixel ({u},{v}) — no valid depth, skipping")
            continue
        p_base = _pixel_to_base(u, v, depth_m, scan["intrinsics"], scan["base_T_camera"])
        depth_mm = round(depth_m * 1000.0, 1)
        objects[name] = {
            "pixel_uv": [u, v],
            "depth_mm": depth_mm,
            "base_xyz_m": [round(float(c), 5) for c in p_base],
        }
        print(f"{_TAG}   {name}: depth {depth_mm:.1f} mm  "
              f"base [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}] m")

    objects_path = out_dir / "objects.json"
    objects_path.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scan_pose_flange": scan["flange_pose"].tolist(),
        "objects": objects,
    }, indent=2), encoding="utf-8")
    print(f"{_TAG} Saved {objects_path}")
    return objects


def phase_move_above(robot, tcp_offset, obj_xyz: list[float],
                     standoff_mm: float, z_safe_m: float,
                     speed_pct: int, tol_mm: float) -> None:
    """Move gripper to standoff height directly above target object."""
    if tcp_offset:
        current_rpy = list(robot.get_tcp_pose().msg[3:6])
    else:
        current_rpy = get_current_pose(robot)[3:6].tolist()

    standoff_z = obj_xyz[2] + standoff_mm / 1000.0
    target_pose = [obj_xyz[0], obj_xyz[1], standoff_z, *current_rpy]

    if tcp_offset:
        flange_target = robot.get_tcp2flange_pose(target_pose)
    else:
        flange_target = target_pose

    check_target_safe(flange_target)
    print(f"{_TAG} Moving to standoff {standoff_mm:.0f} mm above object "
          f"(Z={standoff_z*1000:.0f} mm, tol={tol_mm:.0f} mm) ...")
    robot.set_speed_percent(speed_pct)
    safe_move_to(robot, flange_target, z_safe_m=z_safe_m,
                 speed_pct=speed_pct, tol_mm=tol_mm)


def phase_grasp(robot, effector, tcp_offset,
                grasp_z_mm: float, lift_z_mm: float,
                gripper_width: float, gripper_force: float,
                speed_pct: int,
                retry_step_mm: float = 15.0, retry_count: int = 2) -> float:
    """Lower, close gripper, lift. Returns actual grasp Z used."""
    print(f"{_TAG} Opening gripper (width={gripper_width} m) ...")
    effector.move_gripper(width=gripper_width, force=1.0)
    time.sleep(1.5)

    for attempt in range(retry_count + 1):
        z_try = grasp_z_mm + attempt * retry_step_mm
        if attempt > 0:
            print(f"{_TAG} Retrying grasp at Z={z_try:.0f} mm (attempt {attempt + 1})")

        grasp_z = z_try / 1000.0
        if grasp_z < 0.02:
            raise RuntimeError(f"{_TAG} grasp Z={grasp_z:.4f} m below 20 mm safety minimum")

        if tcp_offset:
            current = np.array(robot.get_tcp_pose().msg[:6], dtype=np.float64)
        else:
            current = get_current_pose(robot)

        descent = [float(current[0]), float(current[1]), grasp_z,
                   float(current[3]), float(current[4]), float(current[5])]
        flange_descent = robot.get_tcp2flange_pose(descent) if tcp_offset else descent
        check_target_safe(flange_descent, z_min_m=0.02)

        print(f"{_TAG} Descending to Z={z_try:.0f} mm (move_l, speed={speed_pct}%) ...")
        robot.set_speed_percent(speed_pct)
        time.sleep(0.05)
        robot.move_l(flange_descent)

        try:
            wait_move_done(robot, flange_descent[:3], tol_mm=3.0)
        except RuntimeError as exc:
            if "no_ik_solution" in str(exc) and attempt < retry_count:
                print(f"{_TAG} IK failed at Z={z_try:.0f} mm, will retry higher")
                continue
            raise

        print(f"{_TAG} Closing gripper (force={gripper_force}) ...")
        effector.move_gripper(width=0.0, force=gripper_force)
        time.sleep(2.0)

        st = effector.get_gripper_status()
        grip_width = st.msg.width
        print(f"{_TAG} Grip width: {grip_width:.4f} m  "
              f"({'likely grasped' if grip_width > 0.001 else 'may have missed'})")

        print(f"{_TAG} Lifting to Z={lift_z_mm:.0f} mm ...")
        safe_lift(robot, height_m=lift_z_mm / 1000.0)
        return z_try

    raise RuntimeError(f"{_TAG} Grasp failed after {retry_count + 1} attempts")


def phase_place(robot, effector, tcp_offset,
                dest_xyz: list[float],
                place_z_mm: float, z_safe_m: float,
                speed_pct: int) -> None:
    """Transit to destination, lower, release, lift away."""
    if tcp_offset:
        current_rpy = list(robot.get_tcp_pose().msg[3:6])
    else:
        current_rpy = get_current_pose(robot)[3:6].tolist()

    above_pose = [dest_xyz[0], dest_xyz[1], z_safe_m, *current_rpy]
    flange_above = robot.get_tcp2flange_pose(above_pose) if tcp_offset else above_pose
    check_target_safe(flange_above)

    print(f"{_TAG} Moving above destination (Z_safe={z_safe_m*1000:.0f} mm) ...")
    robot.set_speed_percent(speed_pct)
    safe_move_to(robot, flange_above, z_safe_m=z_safe_m,
                 speed_pct=speed_pct, tol_mm=15.0)

    if tcp_offset:
        current_rpy = list(robot.get_tcp_pose().msg[3:6])
    else:
        current_rpy = get_current_pose(robot)[3:6].tolist()

    place_z = place_z_mm / 1000.0
    place_pose = [dest_xyz[0], dest_xyz[1], place_z, *current_rpy]
    flange_place = robot.get_tcp2flange_pose(place_pose) if tcp_offset else place_pose
    check_target_safe(flange_place, z_min_m=0.02)

    print(f"{_TAG} Lowering to place Z={place_z_mm:.0f} mm (move_l) ...")
    robot.set_speed_percent(speed_pct)
    time.sleep(0.05)
    robot.move_l(flange_place)
    wait_move_done(robot, flange_place[:3], tol_mm=3.0)

    print(f"{_TAG} Releasing gripper ...")
    effector.move_gripper(width=0.06, force=1.0)
    time.sleep(1.5)   # gripper open
    time.sleep(0.8)   # arm settle after load release

    print(f"{_TAG} Lifting away after release ...")
    safe_lift(robot, height_m=z_safe_m)


def phase_home(robot, speed_pct: int) -> None:
    """Return to zero joint configuration."""
    print(f"{_TAG} Returning to home (all-zero joints) ...")
    robot.set_speed_percent(speed_pct)
    time.sleep(0.05)
    robot.move_j([0.0] * 7)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        check_arm_error(robot)
        ja = robot.get_joint_angles()
        if ja is not None and ja.msg is not None:
            if np.max(np.abs(np.array(ja.msg[:7], dtype=np.float64))) < 0.02:
                time.sleep(0.3)
                print(f"{_TAG} Home reached.")
                return
        time.sleep(0.1)
    print(f"{_TAG} WARNING: home move timed out (arm may still be moving)")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def print_help() -> None:
    print("""
[session] Commands:
  scan                   Move to scan pose and capture RGBD image
  capture                Capture RGBD without moving (arm already at scan pose)
  pixels name:u:v,...    Locate objects from last scan (reuse frame, instant)
  rescan name:u:v,...    Re-capture and locate in one step
  load [path]            Load objects.json directly — skip scan entirely
                           (use when objects haven't moved since last run)
  objects                Show currently loaded object positions
  pick  <object>         Move above → grasp → lift  (<object> key in objects)
  place <destination>    Transit → lower → release → lift away
  home                   Return to all-zero joint configuration
  status                 Show current flange pose and arm_status
  help                   Show this help
  quit                   Exit session (arm stays in current position)
""")


def cmd_status(robot) -> None:
    try:
        fp = robot.get_flange_pose()
        if fp and fp.msg:
            p = fp.msg[:6]
            print(f"  Flange: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m  "
                  f"rpy=[{p[3]:.3f}, {p[4]:.3f}, {p[5]:.3f}] rad")
        st = robot.get_arm_status()
        code = int(getattr(st.msg, "arm_status", -1)) if st and st.msg else -1
        labels = {0:"normal", 1:"emergency_stop", 2:"no_ik_solution",
                  3:"singularity", 4:"target_over_limit", 7:"collision"}
        print(f"  arm_status: {code} ({labels.get(code, 'unknown')})")
    except Exception as exc:
        print(f"  status error: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Single-process pick-and-place session")
    parser.add_argument("--handeye", required=True, help="Path to handeye_result.json")
    parser.add_argument("--tcp", default=None, help="Path to TCP offset JSON")
    parser.add_argument("--output-dir", default="/home/pi/session",
                        help="Directory for scan images and objects.json")
    parser.add_argument("--scan-pose", default="-19.4,10.7,4.6,63.0,7.1,1.4,56.6",
                        help="7 joint angles (deg) for the fixed scan position")
    parser.add_argument("--standoff-mm", type=float, default=80,
                        help="Height above object for standoff (mm)")
    parser.add_argument("--grasp-z-mm", type=float, default=300,
                        help="Starting grasp Z in base frame (mm); auto-retries upward on IK fail")
    parser.add_argument("--grasp-retry-step-mm", type=float, default=15.0,
                        help="Z step (mm) for each IK retry attempt")
    parser.add_argument("--grasp-retry-count", type=int, default=2,
                        help="Max retry attempts on IK failure (default 2: tries +15, +30 mm)")
    parser.add_argument("--lift-z-mm", type=float, default=400,
                        help="Z height to lift to after grasp (mm)")
    parser.add_argument("--place-z-mm", type=float, default=295,
                        help="Z height for placing object (mm)")
    parser.add_argument("--z-safe-mm", type=float, default=400,
                        help="Transit height for horizontal moves (mm)")
    parser.add_argument("--gripper-width", type=float, default=0.06,
                        help="Gripper open width before grasp (m)")
    parser.add_argument("--gripper-force", type=float, default=1.0,
                        help="Gripper close force")
    parser.add_argument("--speed", type=int, default=10,
                        help="Robot speed percent (1-100)")
    parser.add_argument("--flush-frames", type=int, default=2,
                        help="Extra RGBD frames to discard before each capture")
    args = parser.parse_args()

    scan_pose_deg = _parse_scan_pose(args.scan_pose)
    z_safe_m = args.z_safe_mm / 1000.0
    out_dir = Path(args.output_dir)

    print(f"{_TAG} Loading calibration ...")
    handeye = load_handeye(args.handeye)

    print(f"{_TAG} Connecting to robot (CAN) ...")
    robot = connect_robot()
    robot.enable()
    robot.set_speed_percent(args.speed)
    time.sleep(0.2)

    tcp_offset = None
    if args.tcp:
        tcp_offset = load_tcp_offset(args.tcp)
        robot.set_tcp_offset(tcp_offset)
        print(f"{_TAG} TCP offset set: {[round(v, 5) for v in tcp_offset]}")
        time.sleep(0.2)

    print(f"{_TAG} Initializing gripper ...")
    effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

    print(f"{_TAG} Opening Orbbec camera (keeping pipeline alive) ...")
    camera = PersistentAlignedRGBDCapture(flush_frames=args.flush_frames)
    camera.start()

    last_scan: dict | None = None
    objects: dict[str, dict] = {}

    print(f"\n{_TAG} Session ready. Type 'help' to see commands.\n")
    print_help()

    try:
        while True:
            try:
                raw = input("\n[session]> ").strip()
            except EOFError:
                print(f"\n{_TAG} EOF — exiting.")
                break

            if not raw:
                continue
            if raw in {"quit", "q", "exit"}:
                print(f"{_TAG} Exiting session.")
                break
            if raw in {"help", "?"}:
                print_help()
                continue

            try:
                # ---- scan ------------------------------------------------
                if raw == "scan":
                    phase_move_to_scan_pose(robot, scan_pose_deg, args.speed)
                    last_scan = phase_scan(robot, handeye, camera, out_dir, args.flush_frames)
                    print(f"{_TAG} Inspect {last_scan['color_path']} then run:\n"
                          f"  pixels bottle:U:V,blue_board:U2:V2")

                # ---- capture (no robot movement) -------------------------
                elif raw == "capture":
                    last_scan = phase_scan(robot, handeye, camera, out_dir, args.flush_frames)
                    print(f"{_TAG} Captured (arm not moved). Inspect {last_scan['color_path']} then run:\n"
                          f"  pixels bottle:U:V,blue_board:U2:V2")

                # ---- pixels (reuse last frame) ----------------------------
                elif raw.startswith("pixels "):
                    if last_scan is None:
                        print(f"{_TAG} No scan yet. Run 'scan' or 'capture' first.")
                        continue
                    pixel_raw = raw[len("pixels "):].strip()
                    objects = phase_locate(last_scan, pixel_raw, out_dir)

                # ---- rescan (capture + locate) ----------------------------
                elif raw.startswith("rescan "):
                    pixel_raw = raw[len("rescan "):].strip()
                    phase_move_to_scan_pose(robot, scan_pose_deg, args.speed)
                    last_scan = phase_scan(robot, handeye, camera, out_dir, args.flush_frames)
                    objects = phase_locate(last_scan, pixel_raw, out_dir)

                # ---- load objects.json directly --------------------------
                elif raw.startswith("load"):
                    parts = raw.split(maxsplit=1)
                    obj_path = Path(parts[1].strip()) if len(parts) > 1 else out_dir / "objects.json"
                    if not obj_path.exists():
                        print(f"{_TAG} File not found: {obj_path}")
                        continue
                    data = json.loads(obj_path.read_text(encoding="utf-8"))
                    objects = data.get("objects", {})
                    ts = data.get("timestamp", "unknown")
                    print(f"{_TAG} Loaded {len(objects)} object(s) from {obj_path} (timestamp: {ts})")
                    for name, obj in objects.items():
                        xyz = obj["base_xyz_m"]
                        print(f"  {name}: [{xyz[0]*1000:.1f}, {xyz[1]*1000:.1f}, {xyz[2]*1000:.1f}] mm")

                # ---- objects (show current) ------------------------------
                elif raw == "objects":
                    if not objects:
                        print(f"{_TAG} No objects loaded. Run 'scan'+'pixels' or 'load'.")
                    else:
                        for name, obj in objects.items():
                            xyz = obj["base_xyz_m"]
                            print(f"  {name}: [{xyz[0]*1000:.1f}, {xyz[1]*1000:.1f}, {xyz[2]*1000:.1f}] mm")

                # ---- pick -------------------------------------------------
                elif raw.startswith("pick"):
                    parts = raw.split()
                    if len(parts) < 2:
                        print(f"{_TAG} Usage: pick <object-name>")
                        continue
                    obj_name = parts[1]
                    if obj_name not in objects:
                        print(f"{_TAG} '{obj_name}' not in objects. Available: {list(objects)}")
                        continue
                    obj_xyz = objects[obj_name]["base_xyz_m"]
                    print(f"{_TAG} Will pick '{obj_name}' at {[round(v*1000,1) for v in obj_xyz]} mm")
                    if not _confirm(f"{_TAG} Move above '{obj_name}' and grasp?"):
                        continue
                    phase_move_above(robot, tcp_offset, obj_xyz,
                                     args.standoff_mm, z_safe_m, args.speed, tol_mm=15.0)
                    phase_grasp(robot, effector, tcp_offset,
                                args.grasp_z_mm, args.lift_z_mm,
                                args.gripper_width, args.gripper_force, args.speed,
                                retry_step_mm=args.grasp_retry_step_mm,
                                retry_count=args.grasp_retry_count)
                    print(f"{_TAG} Pick complete.")

                # ---- place ------------------------------------------------
                elif raw.startswith("place"):
                    parts = raw.split()
                    if len(parts) < 2:
                        print(f"{_TAG} Usage: place <destination-name>")
                        continue
                    dest_name = parts[1]
                    if dest_name not in objects:
                        print(f"{_TAG} '{dest_name}' not in objects. Available: {list(objects)}")
                        continue
                    dest_xyz = objects[dest_name]["base_xyz_m"]
                    print(f"{_TAG} Will place at '{dest_name}' at {[round(v*1000,1) for v in dest_xyz]} mm")
                    if not _confirm(f"{_TAG} Transit to '{dest_name}' and release?"):
                        continue
                    phase_place(robot, effector, tcp_offset,
                                dest_xyz, args.place_z_mm, z_safe_m, args.speed)
                    print(f"{_TAG} Place complete.")

                # ---- home -------------------------------------------------
                elif raw == "home":
                    if not _confirm(f"{_TAG} Move to home (all joints zero)?"):
                        continue
                    phase_home(robot, args.speed)

                # ---- status -----------------------------------------------
                elif raw == "status":
                    cmd_status(robot)

                else:
                    print(f"{_TAG} Unknown command '{raw}'. Type 'help'.")

            except RuntimeError as exc:
                print(f"{_TAG} ERROR: {exc}")
                print(f"{_TAG} Arm halted. Check status before continuing.")

    finally:
        print(f"{_TAG} Closing camera pipeline ...")
        camera.stop()
        print(f"{_TAG} Session ended.")


if __name__ == "__main__":
    main()
