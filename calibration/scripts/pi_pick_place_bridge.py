#!/usr/bin/env python3
"""Raspberry Pi pick-place bridge.

Keeps robot and RGB-D camera connections alive on the Pi and exposes
lightweight HTTP endpoints so the session controller can run on a laptop.

Pi responsibility:
- CAN / SDK connection
- RGB-D capture
- Motion execution

Laptop responsibility:
- Session flow
- Pixel selection / coordinate math
- Any heavy inference
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from handeye_board_runtime import PersistentAlignedRGBDCapture, connect_robot

_TAG = "[bridge]"
SERVICE_NAME = "pi_pick_place_bridge"
_FATAL_ARM_STATUS_CODES = {1, 2, 3, 4, 7}
_ARM_STATUS_LABELS = {
    0: "normal",
    1: "emergency_stop",
    2: "no_ik_solution",
    3: "singularity",
    4: "target_over_limit",
    7: "collision",
}


def _b64_jpeg(image_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _b64_npy(array: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, array)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_tcp_offset_json(tcp_path: str | Path) -> list[float]:
    data = json.loads(Path(tcp_path).read_text(encoding="utf-8"))
    if "tcp_offset_xyzrpy_m_rad" not in data:
        raise RuntimeError(f"tcp_offset_xyzrpy_m_rad missing in {tcp_path}")
    return list(data["tcp_offset_xyzrpy_m_rad"])


def read_can_operstate(channel: str) -> str | None:
    path = Path(f"/sys/class/net/{channel}/operstate")
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def maybe_activate_can(command: list[str] | None, channel: str) -> None:
    if not command:
        return

    operstate = read_can_operstate(channel)
    if operstate == "up":
        print(f"{_TAG} {channel} already up, skip CAN activation")
        return

    print(f"{_TAG} Activating {channel}: {' '.join(command)}")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"{_TAG} CAN activation failed with exit_code={result.returncode}")


def arm_status_label(code: int) -> str:
    return _ARM_STATUS_LABELS.get(code, "unknown")


def get_current_pose(robot) -> np.ndarray:
    fp = robot.get_flange_pose()
    if fp is None or fp.msg is None:
        raise RuntimeError(f"{_TAG} Flange pose unavailable")
    return np.array(fp.msg[:6], dtype=np.float64)


def check_target_safe(target_pose, z_min_m: float = 0.05) -> None:
    z = float(target_pose[2])
    if z < z_min_m:
        raise RuntimeError(
            f"{_TAG} Target Z={z:.4f} m is below minimum {z_min_m:.4f} m - aborting to prevent collision"
        )


def check_arm_error(robot) -> None:
    try:
        st = robot.get_arm_status()
    except Exception:
        return
    if st is None or st.msg is None:
        return
    code = int(getattr(st.msg, "arm_status", 0))
    if code in _FATAL_ARM_STATUS_CODES:
        raise RuntimeError(f"{_TAG} arm_status={code} ({arm_status_label(code)}) - aborting motion")


def wait_move_done(robot, target_flange_xyz, tol_mm: float = 1.0, timeout_s: float = 20.0) -> None:
    time.sleep(0.3)
    target = np.array(target_flange_xyz[:3], dtype=np.float64)
    deadline = time.monotonic() + timeout_s
    err_mm = float("inf")
    while time.monotonic() < deadline:
        check_arm_error(robot)
        fp = robot.get_flange_pose()
        if fp is None or fp.msg is None:
            time.sleep(0.1)
            continue
        pos = np.array(fp.msg[:3], dtype=np.float64)
        err_mm = float(np.linalg.norm(pos - target) * 1000.0)
        if err_mm < tol_mm:
            time.sleep(0.2)
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"{_TAG} Motion timeout after {timeout_s:.1f}s - "
        f"target={target.tolist()}, last error={err_mm:.1f} mm"
    )


def safe_lift(robot, height_m: float = 0.30) -> np.ndarray:
    pose = get_current_pose(robot)
    current_z = pose[2]

    if current_z >= height_m - 0.001:
        print(f"{_TAG} Already at Z={current_z:.4f} m >= {height_m:.4f} m, skip lift")
        return pose

    lift_pose = pose.copy()
    lift_pose[2] = height_m
    check_target_safe(lift_pose)

    delta_mm = (height_m - current_z) * 1000.0
    print(f"{_TAG} Lifting Z: {current_z:.4f} -> {height_m:.4f} m (delta={delta_mm:.1f} mm)")

    robot.set_speed_percent(10)
    time.sleep(0.05)
    robot.move_l(lift_pose.tolist())

    wait_move_done(robot, lift_pose[:3])
    return get_current_pose(robot)


def _read_current_rpy(robot, tcp_offset: list[float] | None) -> list[float]:
    if tcp_offset:
        tcp = robot.get_tcp_pose()
        if tcp is None or tcp.msg is None:
            raise RuntimeError(f"{_TAG} TCP pose unavailable")
        return list(tcp.msg[3:6])
    return get_current_pose(robot)[3:6].tolist()


def _safe_move_to(robot, target_pose, z_safe_m: float = 0.30,
                  z_min_m: float = 0.05, speed_pct: int = 10,
                  tol_mm: float = 1.0) -> np.ndarray:
    target = np.array(target_pose[:6], dtype=np.float64)
    check_target_safe(target, z_min_m=z_min_m)

    current = get_current_pose(robot)
    print(
        f"{_TAG} safe_move_to: "
        f"[{current[0]:.4f}, {current[1]:.4f}, {current[2]:.4f}] -> "
        f"[{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]"
    )

    if current[2] < z_safe_m - 0.001:
        print(f"{_TAG} Phase 1: lift to Z={z_safe_m:.4f} m")
        robot.set_speed_percent(speed_pct)
        time.sleep(0.05)
        lift_pose = current.copy()
        lift_pose[2] = z_safe_m
        check_target_safe(lift_pose, z_min_m=z_min_m)

        delta_mm = (z_safe_m - current[2]) * 1000.0
        if delta_mm < 50.0:
            robot.move_l(lift_pose.tolist())
        else:
            # Keep the pre-transit lift strictly vertical to avoid dipping into
            # nearby objects during a joint-space arc.
            robot.move_l(lift_pose.tolist())
        wait_move_done(robot, lift_pose[:3])
    else:
        print(f"{_TAG} Phase 1: already at Z={current[2]:.4f} m >= {z_safe_m:.4f} m")

    transit_pose = target.copy()
    transit_pose[2] = max(z_safe_m, target[2])
    check_target_safe(transit_pose, z_min_m=z_min_m)

    print(
        f"{_TAG} Phase 2: transit to "
        f"XY=[{transit_pose[0]:.4f}, {transit_pose[1]:.4f}] at Z={transit_pose[2]:.4f} m"
    )
    robot.set_speed_percent(speed_pct)
    time.sleep(0.05)
    # Keep the hover transit on a Cartesian line at the safe height instead of
    # a joint-space interpolation, which can dip toward the object mid-path.
    robot.move_l(transit_pose.tolist())
    wait_move_done(robot, transit_pose[:3], tol_mm=tol_mm)

    if abs(transit_pose[2] - target[2]) > 0.001:
        print(f"{_TAG} Phase 3: lower to Z={target[2]:.4f} m (move_l)")
        robot.set_speed_percent(speed_pct)
        time.sleep(0.05)
        robot.move_l(target.tolist())
        wait_move_done(robot, target[:3], tol_mm=tol_mm)
    else:
        print(f"{_TAG} Phase 3: already at target Z, skipping")

    final = get_current_pose(robot)
    print(f"{_TAG} Arrived at [{final[0]:.4f}, {final[1]:.4f}, {final[2]:.4f}]")
    return final


def _status_payload(robot, effector=None) -> dict:
    payload: dict = {"ok": True}

    fp = robot.get_flange_pose() if robot else None
    if fp is not None and fp.msg is not None:
        payload["flange_pose"] = [float(v) for v in fp.msg[:6]]

    try:
        tcp = robot.get_tcp_pose() if robot else None
    except Exception:
        tcp = None
    if tcp is not None and tcp.msg is not None:
        payload["tcp_pose"] = [float(v) for v in tcp.msg[:6]]

    st = robot.get_arm_status() if robot else None
    if st is not None and st.msg is not None:
        payload["arm_status_code"] = int(getattr(st.msg, "arm_status", -1))
        payload["motion_status_code"] = int(getattr(st.msg, "motion_status", -1))

    ja = robot.get_joint_angles() if robot else None
    if ja is not None and ja.msg is not None:
        payload["joint_angles_deg"] = [round(math.degrees(float(v)), 3) for v in ja.msg[:7]]

    if effector is not None:
        try:
            gs = effector.get_gripper_status()
        except Exception:
            gs = None
        if gs is not None and gs.msg is not None:
            payload["gripper_width"] = float(gs.msg.width)
            payload["gripper_force"] = float(gs.msg.force)

    return payload


def phase_move_to_scan_pose(robot, scan_pose_deg: list[float], speed_pct: int) -> None:
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


def phase_capture_scan(robot, camera: PersistentAlignedRGBDCapture, flush_frames: int) -> dict:
    fp = robot.get_flange_pose()
    if fp is None or fp.msg is None:
        raise RuntimeError(f"{_TAG} Flange pose unavailable")
    flange_pose = [float(v) for v in fp.msg[:6]]

    print(f"{_TAG} Capturing RGBD ...")
    color_bgr, depth_u16, depth_scale, intrinsics = camera.capture(flush_frames=flush_frames)
    intrinsics_json = {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
        "width": int(intrinsics["width"]),
        "height": int(intrinsics["height"]),
    }

    return {
        "timestamp": time.time(),
        "flange_pose": flange_pose,
        "intrinsics": intrinsics_json,
        "depth_scale": float(depth_scale),
        "color_jpeg_b64": _b64_jpeg(color_bgr),
        "depth_u16_npy_b64": _b64_npy(depth_u16),
    }


def phase_move_above(robot, tcp_offset, obj_xyz: list[float],
                     standoff_mm: float, z_safe_m: float,
                     speed_pct: int, tol_mm: float,
                     min_hover_z_m: float | None = None) -> dict:
    current_rpy = _read_current_rpy(robot, tcp_offset)
    standoff_z = obj_xyz[2] + standoff_mm / 1000.0
    if min_hover_z_m is not None and standoff_z < min_hover_z_m:
        print(f"{_TAG} Raising hover Z from {standoff_z*1000:.0f} mm to "
              f"{min_hover_z_m*1000:.0f} mm to avoid low transparent-depth hover")
        standoff_z = min_hover_z_m
    target_pose = [obj_xyz[0], obj_xyz[1], standoff_z, *current_rpy]
    flange_target = robot.get_tcp2flange_pose(target_pose) if tcp_offset else target_pose

    check_target_safe(flange_target)
    print(f"{_TAG} Moving to standoff {standoff_mm:.0f} mm above object "
          f"(Z={standoff_z*1000:.0f} mm, tol={tol_mm:.0f} mm) ...")
    robot.set_speed_percent(speed_pct)
    _safe_move_to(robot, flange_target, z_safe_m=z_safe_m, speed_pct=speed_pct, tol_mm=tol_mm)
    return _status_payload(robot)


def _move_to_grasp_target(robot, tcp_offset,
                          target_xyz: list[float],
                          standoff_mm: float | None,
                          z_safe_m: float,
                          speed_pct: int,
                          tol_mm: float) -> None:
    current_rpy = _read_current_rpy(robot, tcp_offset)
    if tcp_offset:
        tcp = robot.get_tcp_pose()
        if tcp is None or tcp.msg is None:
            raise RuntimeError(f"{_TAG} TCP pose unavailable")
        current_pose = np.array(tcp.msg[:6], dtype=np.float64)
    else:
        current_pose = get_current_pose(robot)

    delta_xy_mm = float(np.linalg.norm(current_pose[:2] - np.array(target_xyz[:2], dtype=np.float64)) * 1000.0)
    if delta_xy_mm < 0.5:
        return

    align_z = max(float(current_pose[2]), z_safe_m)
    if standoff_mm is not None:
        align_z = max(align_z, float(target_xyz[2]) + standoff_mm / 1000.0)

    target_pose = [float(target_xyz[0]), float(target_xyz[1]), align_z, *current_rpy]
    flange_target = robot.get_tcp2flange_pose(target_pose) if tcp_offset else target_pose
    check_target_safe(flange_target)

    print(f"{_TAG} Repositioning to final grasp XY at Z={align_z*1000:.0f} mm "
          f"(delta XY={delta_xy_mm:.1f} mm, tol={tol_mm:.0f} mm) ...")
    robot.set_speed_percent(speed_pct)
    _safe_move_to(robot, flange_target, z_safe_m=align_z, speed_pct=speed_pct, tol_mm=tol_mm)


def phase_grasp(robot, effector, tcp_offset,
                grasp_z_mm: float, lift_z_mm: float,
                gripper_width: float, gripper_force: float,
                speed_pct: int,
                retry_step_mm: float = 15.0, retry_count: int = 2,
                target_xyz: list[float] | None = None,
                standoff_mm: float | None = None,
                z_safe_m: float = 0.40,
                xy_tol_mm: float = 15.0) -> dict:
    if target_xyz is not None:
        _move_to_grasp_target(
            robot,
            tcp_offset,
            target_xyz,
            standoff_mm=standoff_mm,
            z_safe_m=z_safe_m,
            speed_pct=speed_pct,
            tol_mm=xy_tol_mm,
        )

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
            tcp = robot.get_tcp_pose()
            if tcp is None or tcp.msg is None:
                raise RuntimeError(f"{_TAG} TCP pose unavailable")
            current = np.array(tcp.msg[:6], dtype=np.float64)
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
            msg = str(exc)
            if ("no_ik_solution" in msg or "NO_SOLUTION" in msg) and attempt < retry_count:
                print(f"{_TAG} IK failed at Z={z_try:.0f} mm, retrying higher")
                continue
            raise

        print(f"{_TAG} Closing gripper (force={gripper_force}) ...")
        effector.move_gripper(width=0.0, force=gripper_force)
        time.sleep(2.0)

        st = effector.get_gripper_status()
        grip_width = float(st.msg.width)
        print(f"{_TAG} Grip width: {grip_width:.4f} m  "
              f"({'likely grasped' if grip_width > 0.001 else 'may have missed'})")

        print(f"{_TAG} Lifting to Z={lift_z_mm:.0f} mm ...")
        safe_lift(robot, height_m=lift_z_mm / 1000.0)
        payload = _status_payload(robot, effector)
        payload["actual_grasp_z_mm"] = float(z_try)
        if target_xyz is not None:
            payload["grasp_target_xyz"] = [float(v) for v in target_xyz]
        return payload

    raise RuntimeError(f"{_TAG} Grasp failed after {retry_count + 1} attempts")


def _lift_straight_up(robot, target_height_m: float, speed_pct: int, tcp_offset: list[float] | None) -> None:
    if tcp_offset:
        current_tcp = robot.get_tcp_pose()
        if current_tcp is None or current_tcp.msg is None:
            raise RuntimeError(f"{_TAG} TCP pose unavailable for post-release lift")
        lift_pose = list(current_tcp.msg[:6])
        current_z = float(lift_pose[2])
        if current_z >= target_height_m - 0.001:
            return
        lift_pose[2] = target_height_m
        flange_target = robot.get_tcp2flange_pose(lift_pose)
    else:
        current_flange = get_current_pose(robot)
        current_z = float(current_flange[2])
        if current_z >= target_height_m - 0.001:
            return
        lift_pose = current_flange.copy()
        lift_pose[2] = target_height_m
        flange_target = lift_pose.tolist()

    check_target_safe(flange_target)
    robot.set_speed_percent(speed_pct)
    time.sleep(0.05)
    robot.move_l(flange_target)
    wait_move_done(robot, flange_target[:3], tol_mm=3.0)


def phase_place(robot, effector, tcp_offset,
                dest_xyz: list[float], place_z_mm: float,
                z_safe_m: float, speed_pct: int) -> dict:
    current_rpy = _read_current_rpy(robot, tcp_offset)
    above_pose = [dest_xyz[0], dest_xyz[1], z_safe_m, *current_rpy]
    flange_above = robot.get_tcp2flange_pose(above_pose) if tcp_offset else above_pose
    check_target_safe(flange_above)

    print(f"{_TAG} Moving above destination (Z_safe={z_safe_m*1000:.0f} mm) ...")
    robot.set_speed_percent(speed_pct)
    _safe_move_to(robot, flange_above, z_safe_m=z_safe_m, speed_pct=speed_pct, tol_mm=15.0)

    current_rpy = _read_current_rpy(robot, tcp_offset)
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
    time.sleep(1.5)
    time.sleep(0.8)

    print(f"{_TAG} Lifting away after release ...")
    _lift_straight_up(robot, target_height_m=max(z_safe_m, place_z + 0.05),
                      speed_pct=speed_pct, tcp_offset=tcp_offset)
    return _status_payload(robot, effector)


def phase_home(robot, speed_pct: int) -> dict:
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
                return _status_payload(robot)
        time.sleep(0.1)
    raise RuntimeError(f"{_TAG} Home move timed out")


class BridgeState:
    def __init__(self):
        self.lock = threading.RLock()
        self.robot = None
        self.effector = None
        self.camera = None
        self.tcp_offset = None
        self.flush_frames = 2
        self.speed_pct = 10
        self.hardware_ready = False
        self.client_session_open = False
        self.keep_hardware_alive = False
        self.can_channel = "can0"
        self.can_activate_command: list[str] | None = None

    def configure(self, *, keep_hardware_alive: bool, speed_pct: int, flush_frames: int,
                  can_channel: str, can_activate_command: list[str] | None) -> None:
        with self.lock:
            self.keep_hardware_alive = bool(keep_hardware_alive)
            self.speed_pct = int(speed_pct)
            self.flush_frames = int(flush_frames)
            self.can_channel = can_channel
            self.can_activate_command = list(can_activate_command) if can_activate_command else None

    def _apply_runtime_settings(self, tcp_offset: list[float] | None, speed_pct: int, flush_frames: int) -> None:
        self.flush_frames = int(flush_frames)
        self.speed_pct = int(speed_pct)
        self.robot.set_speed_percent(self.speed_pct)
        time.sleep(0.05)

        if tcp_offset:
            self.tcp_offset = list(tcp_offset)
            self.robot.set_tcp_offset(self.tcp_offset)
            time.sleep(0.2)

    def ensure_hardware_ready(self, tcp_offset: list[float] | None = None,
                              speed_pct: int | None = None, flush_frames: int | None = None) -> dict:
        with self.lock:
            maybe_activate_can(self.can_activate_command, self.can_channel)

            if self.robot is None:
                print(f"{_TAG} Connecting robot ...")
                self.robot = connect_robot()

            # Re-enter normal mode on each open so latched IK/controller
            # states do not block the next client session after a failed move.
            self.robot.set_normal_mode()
            time.sleep(0.5)
            self.robot.enable()
            time.sleep(0.2)

            self._apply_runtime_settings(
                tcp_offset=tcp_offset,
                speed_pct=self.speed_pct if speed_pct is None else int(speed_pct),
                flush_frames=self.flush_frames if flush_frames is None else int(flush_frames),
            )

            if self.effector is None:
                print(f"{_TAG} Initializing gripper ...")
                self.effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

            if self.camera is None:
                print(f"{_TAG} Opening RGB-D camera ...")
                self.camera = PersistentAlignedRGBDCapture(flush_frames=self.flush_frames)
                self.camera.start()

            self.hardware_ready = True
            payload = _status_payload(self.robot, self.effector)
            payload["hardware_ready"] = True
            payload["session_open"] = self.client_session_open
            return payload

    def open(self, tcp_offset: list[float] | None, speed_pct: int, flush_frames: int) -> dict:
        with self.lock:
            payload = self.ensure_hardware_ready(
                tcp_offset=tcp_offset,
                speed_pct=speed_pct,
                flush_frames=flush_frames,
            )
            self.client_session_open = True
            payload["session_open"] = True
            return payload

    def close(self) -> dict:
        with self.lock:
            self.client_session_open = False
            if self.keep_hardware_alive:
                return {"ok": True, "session_open": False, "hardware_ready": self.hardware_ready}
            return self.shutdown()

    def shutdown(self) -> dict:
        with self.lock:
            self.client_session_open = False
            if self.camera is not None:
                print(f"{_TAG} Closing RGB-D camera ...")
                self.camera.stop()
                self.camera = None
            self.effector = None
            self.robot = None
            self.tcp_offset = None
            self.hardware_ready = False
            return {"ok": True, "session_open": False, "hardware_ready": False}

    def reset_camera(self):
        with self.lock:
            if self.camera is not None:
                print(f"{_TAG} Resetting RGB-D camera ...")
                try:
                    self.camera.stop()
                finally:
                    self.camera = None
            self.camera = PersistentAlignedRGBDCapture(flush_frames=self.flush_frames)
            self.camera.start()
            self.hardware_ready = True
            return self.camera

    def require_open(self):
        if self.robot is None or self.camera is None or self.effector is None:
            raise RuntimeError(f"{_TAG} Session not open. Call /session/open first.")
        return self.robot, self.effector, self.camera


STATE = BridgeState()


class Handler(BaseHTTPRequestHandler):
    server_version = "PiPickPlaceBridge/1.0"

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, fn):
        try:
            payload = fn()
            self._send_json(payload, 200)
        except Exception as exc:
            self._send_json({
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }, 500)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"ok": True, "service": SERVICE_NAME})
            return

        if self.path == "/status":
            def fn():
                with STATE.lock:
                    if STATE.robot is None:
                        return {"ok": True, "session_open": False, "hardware_ready": False}
                    payload = _status_payload(STATE.robot, STATE.effector)
                    payload["session_open"] = STATE.client_session_open
                    payload["hardware_ready"] = STATE.hardware_ready
                    return payload
            self._handle(fn)
            return

        self.send_error(404)

    def do_POST(self):
        if self.path == "/session/open":
            self._handle(self._session_open)
        elif self.path == "/session/close":
            self._handle(self._session_close)
        elif self.path == "/scan_pose":
            self._handle(self._scan_pose)
        elif self.path == "/scan":
            self._handle(self._scan)
        elif self.path == "/move_above":
            self._handle(self._move_above)
        elif self.path == "/grasp":
            self._handle(self._grasp)
        elif self.path == "/place":
            self._handle(self._place)
        elif self.path == "/home":
            self._handle(self._home)
        else:
            self.send_error(404)

    def _session_open(self):
        req = self._read_json()
        return STATE.open(
            tcp_offset=req.get("tcp_offset"),
            speed_pct=int(req.get("speed_pct", 10)),
            flush_frames=int(req.get("flush_frames", 2)),
        )

    def _session_close(self):
        return STATE.close()

    def _scan_pose(self):
        req = self._read_json()
        scan_pose_deg = [float(v) for v in req["scan_pose_deg"]]
        speed_pct = int(req.get("speed_pct", 10))
        with STATE.lock:
            robot, _effector, _camera = STATE.require_open()
            phase_move_to_scan_pose(robot, scan_pose_deg, speed_pct)
            payload = _status_payload(robot)
            payload["session_open"] = STATE.client_session_open
            payload["hardware_ready"] = STATE.hardware_ready
            return payload

    def _scan(self):
        req = self._read_json()
        flush_frames = int(req.get("flush_frames", STATE.flush_frames))
        with STATE.lock:
            robot, _effector, camera = STATE.require_open()
            try:
                payload = phase_capture_scan(robot, camera, flush_frames)
            except Exception as exc:
                print(f"{_TAG} Scan capture failed: {exc}. Retrying after camera reset ...")
                camera = STATE.reset_camera()
                payload = phase_capture_scan(robot, camera, flush_frames)
            payload["session_open"] = STATE.client_session_open
            payload["hardware_ready"] = STATE.hardware_ready
            return payload

    def _move_above(self):
        req = self._read_json()
        with STATE.lock:
            robot, _effector, _camera = STATE.require_open()
            payload = phase_move_above(
                robot,
                STATE.tcp_offset,
                [float(v) for v in req["target_xyz"]],
                float(req.get("standoff_mm", 80)),
                float(req.get("z_safe_mm", 400)) / 1000.0,
                int(req.get("speed_pct", 10)),
                float(req.get("tol_mm", 15.0)),
                min_hover_z_m=(float(req["min_hover_z_mm"]) / 1000.0) if "min_hover_z_mm" in req else None,
            )
            payload["session_open"] = STATE.client_session_open
            payload["hardware_ready"] = STATE.hardware_ready
            return payload

    def _grasp(self):
        req = self._read_json()
        with STATE.lock:
            robot, effector, _camera = STATE.require_open()
            payload = phase_grasp(
                robot,
                effector,
                STATE.tcp_offset,
                float(req["grasp_z_mm"]),
                float(req["lift_z_mm"]),
                float(req.get("gripper_width", 0.06)),
                float(req.get("gripper_force", 1.0)),
                int(req.get("speed_pct", 10)),
                retry_step_mm=float(req.get("retry_step_mm", 15.0)),
                retry_count=int(req.get("retry_count", 2)),
                target_xyz=[float(v) for v in req["target_xyz"]] if "target_xyz" in req else None,
                standoff_mm=float(req["standoff_mm"]) if "standoff_mm" in req else None,
                z_safe_m=float(req.get("z_safe_mm", 400)) / 1000.0,
                xy_tol_mm=float(req.get("xy_tol_mm", 15.0)),
            )
            payload["session_open"] = STATE.client_session_open
            payload["hardware_ready"] = STATE.hardware_ready
            return payload

    def _place(self):
        req = self._read_json()
        with STATE.lock:
            robot, effector, _camera = STATE.require_open()
            payload = phase_place(
                robot,
                effector,
                STATE.tcp_offset,
                [float(v) for v in req["dest_xyz"]],
                float(req["place_z_mm"]),
                float(req.get("z_safe_mm", 400)) / 1000.0,
                int(req.get("speed_pct", 10)),
            )
            payload["session_open"] = STATE.client_session_open
            payload["hardware_ready"] = STATE.hardware_ready
            return payload

    def _home(self):
        req = self._read_json()
        with STATE.lock:
            robot, _effector, _camera = STATE.require_open()
            payload = phase_home(robot, int(req.get("speed_pct", 10)))
            payload["session_open"] = STATE.client_session_open
            payload["hardware_ready"] = STATE.hardware_ready
            return payload

    def log_message(self, fmt, *args):
        return


def serve(*, host: str, port: int, service_name: str, keep_hardware_alive: bool,
          eager_open: bool, speed_pct: int, flush_frames: int, can_channel: str,
          can_activate_command: list[str] | None, tcp_offset: list[float] | None) -> None:
    global SERVICE_NAME
    SERVICE_NAME = service_name
    STATE.configure(
        keep_hardware_alive=keep_hardware_alive,
        speed_pct=speed_pct,
        flush_frames=flush_frames,
        can_channel=can_channel,
        can_activate_command=can_activate_command,
    )
    if eager_open:
        print(f"{_TAG} Eager hardware warm-up enabled", flush=True)
        STATE.ensure_hardware_ready(
            tcp_offset=tcp_offset,
            speed_pct=speed_pct,
            flush_frames=flush_frames,
        )
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"{_TAG} listening on http://{host}:{port} ({SERVICE_NAME})", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        STATE.shutdown()


def main(*, description: str = "Persistent Pi-side pick-place bridge",
         service_name: str = "pi_pick_place_bridge",
         default_keep_hardware_alive: bool = False,
         default_eager_open: bool = False,
         default_can_activate_script: str | None = None,
         default_can_channel: str = "can0",
         default_can_bitrate: int = 1000000,
         default_can_usb_address: str | None = None,
         default_tcp_json: str | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--speed-pct", type=int, default=10, help="Default robot speed percent")
    parser.add_argument("--flush-frames", type=int, default=2, help="Default extra RGB-D frames to discard")
    parser.add_argument("--keep-hardware-alive", action="store_true", default=default_keep_hardware_alive,
                        help="Do not tear down robot/camera on /session/close")
    parser.add_argument("--eager-open", action="store_true", default=default_eager_open,
                        help="Bring up CAN and open robot/camera during server startup")
    parser.add_argument("--can-channel", default=default_can_channel, help="CAN network interface name")
    parser.add_argument("--can-bitrate", type=int, default=default_can_bitrate, help="CAN bitrate for activation")
    parser.add_argument("--can-usb-address", default=default_can_usb_address, help="USB hardware address for can_activate.sh")
    parser.add_argument("--activate-can-script", default=default_can_activate_script,
                        help="Optional can_activate.sh path to run before connecting")
    parser.add_argument("--tcp-json", default=default_tcp_json,
                        help="Optional TCP JSON to preload during eager startup")
    args = parser.parse_args()

    can_activate_command = None
    if args.activate_can_script:
        can_activate_command = [
            args.activate_can_script,
            args.can_channel,
            str(args.can_bitrate),
        ]
        if args.can_usb_address:
            can_activate_command.append(args.can_usb_address)

    tcp_offset = load_tcp_offset_json(args.tcp_json) if args.tcp_json else None

    serve(
        host=args.host,
        port=args.port,
        service_name=service_name,
        keep_hardware_alive=args.keep_hardware_alive,
        eager_open=args.eager_open,
        speed_pct=args.speed_pct,
        flush_frames=args.flush_frames,
        can_channel=args.can_channel,
        can_activate_command=can_activate_command,
        tcp_offset=tcp_offset,
    )


if __name__ == "__main__":
    main()
