#!/usr/bin/env python3
"""Persistent robot HTTP server for NERO arm.

Runs on Raspberry Pi as a long-lived service. Initialises CAN, robot
connection, Orbbec camera and gripper ONCE at startup, then exposes a
simple HTTP API so any client (Mac agent, VLM pipeline, scripts) can
drive the arm without paying per-call reconnect costs.

Start:
    python3 robot_server.py \
        --handeye  /home/pi/calibration/results/session1/handeye_result.json \
        --tcp      /home/pi/calibration/results/session1/gripper_tcp_left_front_tip_samples_004_006.json \
        --scan-pose "-19.4,10.7,4.6,63.0,7.1,1.4,56.6" \
        --port 8001

API (all JSON):
    GET  /status                     → arm status, flange pose, objects loaded
    GET  /image/scan                 → latest scan_color.jpg as JPEG bytes
    POST /scan                       → move to scan pose, capture RGBD, return image+meta
    POST /locate   {pixels: "bottle:u:v,board:u:v"}  → 3D positions
    POST /move_above {object: "bottle"}
    POST /grasp    {object: "bottle"}
    POST /place    {destination: "blue_board"}
    POST /home
    POST /stop                       → electronic emergency stop
    POST /move_j   {angles_deg: [j1..j7]}
"""

import base64
import json
import logging
import math
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, "/home/pi")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("fastapi/pydantic not installed. Run: pip install fastapi uvicorn")

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

log = logging.getLogger("robot_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Global session state (initialised at startup)
# ---------------------------------------------------------------------------

class RobotSession:
    robot: Any = None
    effector: Any = None
    camera: PersistentAlignedRGBDCapture | None = None
    handeye: dict = {}
    tcp_offset: list[float] | None = None
    scan_pose_deg: list[float] = [0.0] * 7
    last_scan: dict | None = None
    objects: dict[str, dict] = {}
    out_dir: Path = Path("/home/pi/session")
    speed_pct: int = 10
    standoff_mm: float = 80.0
    grasp_z_mm: float = 300.0
    grasp_retry_step_mm: float = 15.0
    grasp_retry_count: int = 2
    lift_z_mm: float = 400.0
    place_z_mm: float = 295.0
    z_safe_mm: float = 400.0
    gripper_width: float = 0.06
    gripper_force: float = 1.0
    flush_frames: int = 2

SESSION = RobotSession()


# ---------------------------------------------------------------------------
# Helpers (mirrors pick_place_session.py logic)
# ---------------------------------------------------------------------------

def _parse_scan_pose(raw: str) -> list[float]:
    parts = [s.strip() for s in raw.split(",")]
    if len(parts) != 7:
        raise ValueError(f"scan-pose needs 7 values, got {len(parts)}")
    return [float(p) for p in parts]


def _parse_pixel_coords(raw: str) -> dict[str, tuple[int, int]]:
    result = {}
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected 'name:u:v', got '{entry}'")
        result[parts[0].strip()] = (int(parts[1]), int(parts[2]))
    return result


def _depth_at_pixel(depth_u16: np.ndarray, u: int, v: int,
                    depth_scale: float, window: int = 7) -> float | None:
    h, w = depth_u16.shape
    half = window // 2
    roi = depth_u16[max(0, v - half):min(h, v + half + 1),
                    max(0, u - half):min(w, u + half + 1)]
    valid = roi[roi > 0]
    return float(np.median(valid)) * depth_scale / 1000.0 if len(valid) > 0 else None


def _pixel_to_base(u, v, depth_m, intrinsics, base_T_camera):
    x = (u - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    return (base_T_camera @ np.array([x, y, depth_m, 1.0]))[:3]


def _depth_colormap(depth_u16, depth_scale):
    dm = depth_u16.astype(np.float32) * depth_scale
    valid = dm > 0
    if not np.any(valid):
        return np.zeros((*depth_u16.shape, 3), dtype=np.uint8)
    max_mm = float(np.percentile(dm[valid], 98))
    gray = (np.clip(dm / max(max_mm, 1.0), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _ensure_can() -> None:
    """Bring up can0 if it is not already UP."""
    result = subprocess.run(
        ["ip", "link", "show", "can0"],
        capture_output=True, text=True
    )
    if "state UP" not in result.stdout:
        log.info("can0 is DOWN — bringing up ...")
        subprocess.run(
            ["sudo", "ip", "link", "set", "can0", "up",
             "type", "can", "bitrate", "1000000"],
            check=True
        )
        time.sleep(0.5)
        log.info("can0 is UP")
    else:
        log.info("can0 already UP")


def _do_scan() -> dict:
    s = SESSION
    fp = s.robot.get_flange_pose()
    if fp is None or fp.msg is None:
        raise RuntimeError("Flange pose unavailable")
    flange_pose = np.array(fp.msg[:6], dtype=np.float64)
    base_T_camera = pose_to_transform(flange_pose) @ s.handeye["flange_T_camera_np"]

    color_bgr, depth_u16, depth_scale, intrinsics = s.camera.capture(
        flush_frames=s.flush_frames
    )

    s.out_dir.mkdir(parents=True, exist_ok=True)
    color_path = s.out_dir / "scan_color.jpg"
    depth_path = s.out_dir / "scan_depth.jpg"
    cv2.imwrite(str(color_path), color_bgr)
    cv2.imwrite(str(depth_path), _depth_colormap(depth_u16, depth_scale))

    camera_z_mm = float(base_T_camera[2, 3]) * 1000.0
    log.info(f"Scan captured. Camera height: {camera_z_mm:.1f} mm")

    return {
        "flange_pose": flange_pose,
        "base_T_camera": base_T_camera,
        "camera_height_mm": round(camera_z_mm, 1),
        "color_bgr": color_bgr,
        "depth_u16": depth_u16,
        "depth_scale": depth_scale,
        "intrinsics": intrinsics,
        "color_path": color_path,
    }


def _do_locate(pixel_coords_raw: str) -> dict[str, dict]:
    s = SESSION
    if s.last_scan is None:
        raise RuntimeError("No scan available. Call /scan first.")
    coords = _parse_pixel_coords(pixel_coords_raw)
    objects: dict[str, dict] = {}
    for name, (u, v) in coords.items():
        depth_m = _depth_at_pixel(
            s.last_scan["depth_u16"], u, v, s.last_scan["depth_scale"]
        )
        if depth_m is None or depth_m <= 0:
            log.warning(f"  {name}: no valid depth at ({u},{v}), skipping")
            continue
        p_base = _pixel_to_base(
            u, v, depth_m, s.last_scan["intrinsics"], s.last_scan["base_T_camera"]
        )
        objects[name] = {
            "pixel_uv": [u, v],
            "depth_mm": round(depth_m * 1000.0, 1),
            "base_xyz_m": [round(float(c), 5) for c in p_base],
        }
        log.info(f"  {name}: depth {depth_m*1000:.1f} mm  "
                 f"base [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}] m")

    obj_path = s.out_dir / "objects.json"
    obj_path.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "objects": objects,
    }, indent=2), encoding="utf-8")
    s.objects = objects
    return objects


def _move_to_scan_pose() -> None:
    s = SESSION
    scan_rad = [math.radians(d) for d in s.scan_pose_deg]
    log.info(f"Moving to scan pose: {[round(d,1) for d in s.scan_pose_deg]}°")
    s.robot.set_speed_percent(s.speed_pct)
    time.sleep(0.05)
    s.robot.move_j(scan_rad)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        check_arm_error(s.robot)
        ja = s.robot.get_joint_angles()
        if ja and ja.msg is not None:
            if np.max(np.abs(np.array(ja.msg[:7]) - np.array(scan_rad))) < 0.03:
                time.sleep(0.3)
                log.info("Scan pose reached")
                return
        time.sleep(0.1)
    raise RuntimeError("Timed out moving to scan pose")


# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: bring up hardware. Shutdown: stop camera."""
    s = SESSION
    log.info("=== Robot server starting ===")

    _ensure_can()

    log.info("Connecting to robot ...")
    s.robot = connect_robot()
    s.robot.enable()
    s.robot.set_speed_percent(s.speed_pct)
    time.sleep(0.2)

    if s.tcp_offset:
        s.robot.set_tcp_offset(s.tcp_offset)
        log.info(f"TCP offset set: {[round(v,5) for v in s.tcp_offset]}")
        time.sleep(0.2)

    log.info("Initialising gripper ...")
    s.effector = s.robot.init_effector(s.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

    log.info("Starting Orbbec camera pipeline ...")
    s.camera = PersistentAlignedRGBDCapture(
        warmup_frames=5, settle_s=1.0, flush_frames=s.flush_frames
    )
    s.camera.start()

    log.info("=== Robot server ready ===")
    yield

    log.info("Stopping camera ...")
    if s.camera:
        s.camera.stop()
    log.info("=== Robot server stopped ===")


app = FastAPI(title="NERO Robot Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LocateRequest(BaseModel):
    pixels: str           # "bottle:260:360,blue_board:810:400"

class MoveAboveRequest(BaseModel):
    object: str
    standoff_mm: float | None = None
    z_safe_mm: float | None = None

class GraspRequest(BaseModel):
    object: str
    grasp_z_mm: float | None = None
    lift_z_mm: float | None = None

class PlaceRequest(BaseModel):
    destination: str
    place_z_mm: float | None = None
    z_safe_mm: float | None = None

class MoveJRequest(BaseModel):
    angles_deg: list[float]   # 7 values


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
def get_status():
    s = SESSION
    flange = None
    arm_code = -1
    try:
        fp = s.robot.get_flange_pose()
        if fp and fp.msg:
            p = list(fp.msg[:6])
            flange = {"xyz_m": [round(v, 5) for v in p[:3]],
                      "rpy_rad": [round(v, 5) for v in p[3:]]}
        st = s.robot.get_arm_status()
        arm_code = int(getattr(st.msg, "arm_status", -1)) if st and st.msg else -1
    except Exception as exc:
        log.warning(f"status read error: {exc}")
    return {
        "arm_status_code": arm_code,
        "arm_status_label": {
            0:"normal", 1:"emergency_stop", 2:"no_ik_solution",
            3:"singularity", 4:"target_over_limit", 7:"collision",
        }.get(arm_code, "unknown"),
        "flange": flange,
        "objects_loaded": list(s.objects.keys()),
        "camera_ready": s.camera is not None and s.camera._started,
    }


@app.get("/image/scan")
def get_scan_image():
    """Return latest scan_color.jpg as JPEG bytes."""
    color_path = SESSION.out_dir / "scan_color.jpg"
    if not color_path.exists():
        raise HTTPException(404, "No scan image yet. Call POST /scan first.")
    return Response(content=color_path.read_bytes(), media_type="image/jpeg")


@app.post("/scan")
def post_scan():
    """Move to scan pose, capture RGBD. Returns scan metadata + base64 image."""
    try:
        _move_to_scan_pose()
        SESSION.last_scan = _do_scan()
        scan = SESSION.last_scan
        _, buf = cv2.imencode(".jpg", scan["color_bgr"],
                              [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return {
            "camera_height_mm": scan["camera_height_mm"],
            "image_base64": b64,
            "image_url": "/image/scan",
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/locate")
def post_locate(req: LocateRequest):
    """Compute 3D base-frame positions from pixel coords using last scan."""
    try:
        objects = _do_locate(req.pixels)
        return {"objects": objects}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/move_above")
def post_move_above(req: MoveAboveRequest):
    s = SESSION
    obj_name = req.object
    if obj_name not in s.objects:
        raise HTTPException(404, f"'{obj_name}' not in objects. Available: {list(s.objects)}")
    obj_xyz = s.objects[obj_name]["base_xyz_m"]
    standoff_mm = req.standoff_mm or s.standoff_mm
    z_safe_m = (req.z_safe_mm or s.z_safe_mm) / 1000.0

    try:
        if s.tcp_offset:
            current_rpy = list(s.robot.get_tcp_pose().msg[3:6])
        else:
            current_rpy = get_current_pose(s.robot)[3:6].tolist()

        standoff_z = obj_xyz[2] + standoff_mm / 1000.0
        target_pose = [obj_xyz[0], obj_xyz[1], standoff_z, *current_rpy]
        flange_target = (s.robot.get_tcp2flange_pose(target_pose)
                         if s.tcp_offset else target_pose)
        check_target_safe(flange_target)
        s.robot.set_speed_percent(s.speed_pct)
        safe_move_to(s.robot, flange_target, z_safe_m=z_safe_m,
                     speed_pct=s.speed_pct, tol_mm=15.0)
        final = get_current_pose(s.robot)
        return {
            "status": "ok",
            "standoff_mm": standoff_mm,
            "flange_xyz_m": [round(float(v), 5) for v in final[:3]],
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/grasp")
def post_grasp(req: GraspRequest):
    s = SESSION
    grasp_z_mm = req.grasp_z_mm or s.grasp_z_mm
    lift_z_mm  = req.lift_z_mm  or s.lift_z_mm

    try:
        s.effector.move_gripper(width=s.gripper_width, force=1.0)
        time.sleep(1.5)

        actual_z = None
        for attempt in range(s.grasp_retry_count + 1):
            z_try = grasp_z_mm + attempt * s.grasp_retry_step_mm
            if attempt > 0:
                log.info(f"IK retry at Z={z_try:.0f} mm")
            grasp_z = z_try / 1000.0
            if grasp_z < 0.02:
                raise RuntimeError(f"grasp Z={grasp_z:.4f} m below 20 mm safety minimum")

            current = (np.array(s.robot.get_tcp_pose().msg[:6])
                       if s.tcp_offset else get_current_pose(s.robot))
            descent = [float(current[0]), float(current[1]), grasp_z,
                       float(current[3]), float(current[4]), float(current[5])]
            flange_d = s.robot.get_tcp2flange_pose(descent) if s.tcp_offset else descent
            check_target_safe(flange_d, z_min_m=0.02)

            s.robot.set_speed_percent(s.speed_pct)
            time.sleep(0.05)
            s.robot.move_l(flange_d)
            try:
                wait_move_done(s.robot, flange_d[:3], tol_mm=3.0)
                actual_z = z_try
                break
            except RuntimeError as exc:
                if "no_ik_solution" in str(exc) and attempt < s.grasp_retry_count:
                    continue
                raise

        s.effector.move_gripper(width=0.0, force=s.gripper_force)
        time.sleep(2.0)
        st = s.effector.get_gripper_status()
        grip_width = float(st.msg.width)
        safe_lift(s.robot, height_m=lift_z_mm / 1000.0)

        return {
            "status": "ok",
            "grasp_z_mm": actual_z,
            "grip_width_m": round(grip_width, 5),
            "grasped": grip_width > 0.001,
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/place")
def post_place(req: PlaceRequest):
    s = SESSION
    dest_name = req.destination
    if dest_name not in s.objects:
        raise HTTPException(404, f"'{dest_name}' not found. Available: {list(s.objects)}")
    dest_xyz  = s.objects[dest_name]["base_xyz_m"]
    place_z_m = (req.place_z_mm or s.place_z_mm) / 1000.0
    z_safe_m  = (req.z_safe_mm  or s.z_safe_mm)  / 1000.0

    try:
        current_rpy = (list(s.robot.get_tcp_pose().msg[3:6])
                       if s.tcp_offset else get_current_pose(s.robot)[3:6].tolist())
        above_pose = [dest_xyz[0], dest_xyz[1], z_safe_m, *current_rpy]
        flange_above = (s.robot.get_tcp2flange_pose(above_pose)
                        if s.tcp_offset else above_pose)
        check_target_safe(flange_above)
        s.robot.set_speed_percent(s.speed_pct)
        safe_move_to(s.robot, flange_above, z_safe_m=z_safe_m,
                     speed_pct=s.speed_pct, tol_mm=15.0)

        current_rpy = (list(s.robot.get_tcp_pose().msg[3:6])
                       if s.tcp_offset else get_current_pose(s.robot)[3:6].tolist())
        place_pose = [dest_xyz[0], dest_xyz[1], place_z_m, *current_rpy]
        flange_place = (s.robot.get_tcp2flange_pose(place_pose)
                        if s.tcp_offset else place_pose)
        check_target_safe(flange_place, z_min_m=0.02)

        s.robot.set_speed_percent(s.speed_pct)
        time.sleep(0.05)
        s.robot.move_l(flange_place)
        wait_move_done(s.robot, flange_place[:3], tol_mm=3.0)

        s.effector.move_gripper(width=0.06, force=1.0)
        time.sleep(1.5)
        time.sleep(0.8)   # arm settle after load release

        safe_lift(s.robot, height_m=z_safe_m)
        return {"status": "ok", "place_z_mm": place_z_m * 1000}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/home")
def post_home():
    s = SESSION
    try:
        s.robot.set_speed_percent(s.speed_pct)
        time.sleep(0.05)
        s.robot.move_j([0.0] * 7)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            check_arm_error(s.robot)
            ja = s.robot.get_joint_angles()
            if ja and ja.msg is not None:
                if np.max(np.abs(np.array(ja.msg[:7], dtype=np.float64))) < 0.02:
                    time.sleep(0.3)
                    return {"status": "ok"}
            time.sleep(0.1)
        return {"status": "timeout"}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/stop")
def post_stop():
    try:
        SESSION.robot.electronic_emergency_stop()
        return {"status": "emergency_stop_sent"}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/move_j")
def post_move_j(req: MoveJRequest):
    if len(req.angles_deg) != 7:
        raise HTTPException(400, "angles_deg must have exactly 7 values")
    s = SESSION
    try:
        rad = [math.radians(d) for d in req.angles_deg]
        s.robot.set_speed_percent(s.speed_pct)
        time.sleep(0.05)
        s.robot.move_j(rad)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            check_arm_error(s.robot)
            ja = s.robot.get_joint_angles()
            if ja and ja.msg is not None:
                if np.max(np.abs(np.array(ja.msg[:7]) - np.array(rad))) < 0.03:
                    time.sleep(0.3)
                    return {"status": "ok", "angles_deg": req.angles_deg}
            time.sleep(0.1)
        return {"status": "timeout"}
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="NERO persistent robot HTTP server")
    parser.add_argument("--handeye", required=True)
    parser.add_argument("--tcp", default=None)
    parser.add_argument("--scan-pose", default="-19.4,10.7,4.6,63.0,7.1,1.4,56.6")
    parser.add_argument("--output-dir", default="/home/pi/session")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--standoff-mm", type=float, default=80.0)
    parser.add_argument("--grasp-z-mm", type=float, default=300.0)
    parser.add_argument("--lift-z-mm", type=float, default=400.0)
    parser.add_argument("--place-z-mm", type=float, default=295.0)
    parser.add_argument("--z-safe-mm", type=float, default=400.0)
    parser.add_argument("--flush-frames", type=int, default=2)
    args = parser.parse_args()

    # Populate session config before app starts
    s = SESSION
    s.handeye     = load_handeye(args.handeye)
    s.tcp_offset  = load_tcp_offset(args.tcp) if args.tcp else None
    s.scan_pose_deg = _parse_scan_pose(args.scan_pose)
    s.out_dir     = Path(args.output_dir)
    s.speed_pct   = args.speed
    s.standoff_mm = args.standoff_mm
    s.grasp_z_mm  = args.grasp_z_mm
    s.lift_z_mm   = args.lift_z_mm
    s.place_z_mm  = args.place_z_mm
    s.z_safe_mm   = args.z_safe_mm
    s.flush_frames = args.flush_frames

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
