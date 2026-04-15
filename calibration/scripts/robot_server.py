#!/usr/bin/env python3
"""Persistent robot HTTP server for NERO arm — Pi-side I/O only.

Architecture (Plan A):
  - Pi:  hardware I/O hub — CAN signals, camera capture, SDK execution
  - Mac: all computation — coordinate transforms, VLM inference, motion planning

Pi never does coordinate math or inference. Mac sends already-computed
3D target coordinates; Pi executes and returns sensor data.

Start:
    python3 robot_server.py --port 8001

API:
    GET  /status
         → arm_status, flange pose, can0 state

    POST /scan
         → captures RGBD, returns:
             image_base64    (JPEG)
             depth_base64    (uint16 PNG, aligned to color)
             depth_scale     (multiply uint16 → mm)
             intrinsics      {fx, fy, cx, cy, width, height}
             flange_pose     [x,y,z,roll,pitch,yaw] in metres/rad
           Mac uses flange_pose + handeye + intrinsics to compute 3D positions.

    POST /move_above  {"xyz_m": [x,y,z], "standoff_mm": 80, "z_safe_mm": 400}
         → moves gripper to xyz + standoff above, returns final flange pose

    POST /grasp       {"grasp_z_mm": 330, "lift_z_mm": 400}
         → opens gripper, descends to grasp_z (auto-retries on IK fail), closes, lifts

    POST /place       {"xyz_m": [x,y,z], "place_z_mm": 295, "z_safe_mm": 400}
         → transits above xyz, lowers to place_z, releases, lifts

    POST /home        → move_j to all-zero joints

    POST /move_j      {"angles_deg": [j1..j7]}

    POST /stop        → electronic emergency stop

    GET  /image/scan  → latest scan_color.jpg as raw JPEG bytes
"""

import base64
import logging
import math
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, "/home/pi")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("fastapi/pydantic not installed. Run: pip install fastapi uvicorn")

from handeye_board_runtime import (
    PersistentAlignedRGBDCapture,
    connect_robot,
    load_tcp_offset,
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
# Global hardware state — initialised once at startup
# ---------------------------------------------------------------------------

class HardwareSession:
    robot: Any = None
    effector: Any = None
    camera: PersistentAlignedRGBDCapture | None = None
    tcp_offset: list[float] | None = None
    out_dir: Path = Path("/home/pi/session")
    speed_pct: int = 30
    grasp_retry_step_mm: float = 15.0
    grasp_retry_count: int = 2
    gripper_width: float = 0.06
    gripper_force: float = 1.0
    flush_frames: int = 2

HW = HardwareSession()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_can() -> None:
    result = subprocess.run(["ip", "link", "show", "can0"],
                            capture_output=True, text=True)
    if "state UP" not in result.stdout:
        log.info("can0 DOWN — bringing up ...")
        subprocess.run(
            ["sudo", "ip", "link", "set", "can0", "up",
             "type", "can", "bitrate", "1000000"],
            check=True,
        )
        time.sleep(0.5)
    log.info("can0 UP")


def _depth_colormap(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    dm = depth_u16.astype(np.float32) * depth_scale
    valid = dm > 0
    if not np.any(valid):
        return np.zeros((*depth_u16.shape, 3), dtype=np.uint8)
    max_mm = float(np.percentile(dm[valid], 98))
    gray = (np.clip(dm / max(max_mm, 1.0), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


# ---------------------------------------------------------------------------
# FastAPI lifespan: hardware init / teardown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== Robot server starting ===")
    _ensure_can()

    log.info("Connecting to robot ...")
    HW.robot = connect_robot()
    HW.robot.enable()
    HW.robot.set_speed_percent(HW.speed_pct)
    time.sleep(0.2)

    if HW.tcp_offset:
        HW.robot.set_tcp_offset(HW.tcp_offset)
        log.info(f"TCP offset set: {[round(v, 5) for v in HW.tcp_offset]}")
        time.sleep(0.2)

    log.info("Initialising gripper ...")
    HW.effector = HW.robot.init_effector(HW.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

    log.info("Starting Orbbec camera pipeline ...")
    HW.camera = PersistentAlignedRGBDCapture(
        warmup_frames=5, settle_s=1.0, flush_frames=HW.flush_frames
    )
    HW.camera.start()

    log.info("=== Robot server ready ===")
    yield

    log.info("Stopping camera ...")
    if HW.camera:
        HW.camera.stop()
    log.info("=== Robot server stopped ===")


app = FastAPI(title="NERO Robot Server (Pi I/O)", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class MoveAboveRequest(BaseModel):
    xyz_m: list[float]          # [x, y, z] in base frame — computed on Mac
    standoff_mm: float = 80.0
    z_safe_mm: float = 400.0

class GraspRequest(BaseModel):
    grasp_z_mm: float = 300.0   # absolute Z in base frame — computed on Mac
    lift_z_mm: float = 400.0

class PlaceRequest(BaseModel):
    xyz_m: list[float]          # destination [x, y, z] — computed on Mac
    place_z_mm: float = 295.0
    z_safe_mm: float = 400.0

class MoveJRequest(BaseModel):
    angles_deg: list[float]     # 7 values


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
def get_status():
    flange = None
    arm_code = -1
    can_state = "unknown"
    try:
        res = subprocess.run(["ip", "link", "show", "can0"],
                             capture_output=True, text=True)
        can_state = "UP" if "state UP" in res.stdout else "DOWN"
        fp = HW.robot.get_flange_pose()
        if fp and fp.msg:
            p = list(fp.msg[:6])
            flange = {"xyz_m": [round(v, 5) for v in p[:3]],
                      "rpy_rad": [round(v, 5) for v in p[3:]]}
        st = HW.robot.get_arm_status()
        arm_code = int(getattr(st.msg, "arm_status", -1)) if st and st.msg else -1
    except Exception as exc:
        log.warning(f"status error: {exc}")

    labels = {0:"normal", 1:"emergency_stop", 2:"no_ik_solution",
              3:"singularity", 4:"target_over_limit", 7:"collision"}
    return {
        "can0": can_state,
        "arm_status_code": arm_code,
        "arm_status": labels.get(arm_code, "unknown"),
        "flange": flange,
    }


@app.get("/image/scan")
def get_scan_image():
    path = HW.out_dir / "scan_color.jpg"
    if not path.exists():
        raise HTTPException(404, "No scan yet. Call POST /scan first.")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


@app.post("/scan")
def post_scan():
    """Capture RGBD. Returns image + depth + intrinsics + flange pose.

    Mac uses this data to:
      1. Send image to VLM → get pixel coords
      2. Use intrinsics + depth + handeye + flange_pose → compute base_xyz_m
      3. Call /move_above or /place with the computed xyz_m
    """
    try:
        fp = HW.robot.get_flange_pose()
        if fp is None or fp.msg is None:
            raise RuntimeError("Flange pose unavailable")
        flange_pose = list(fp.msg[:6])

        color_bgr, depth_u16, depth_scale, intrinsics = HW.camera.capture(
            flush_frames=HW.flush_frames
        )

        HW.out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(HW.out_dir / "scan_color.jpg"), color_bgr)
        cv2.imwrite(str(HW.out_dir / "scan_depth.jpg"),
                    _depth_colormap(depth_u16, depth_scale))

        # Encode color as JPEG base64
        _, color_buf = cv2.imencode(".jpg", color_bgr,
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        color_b64 = base64.b64encode(color_buf.tobytes()).decode()

        # Encode depth as PNG base64 (lossless uint16)
        _, depth_buf = cv2.imencode(".png", depth_u16)
        depth_b64 = base64.b64encode(depth_buf.tobytes()).decode()

        return {
            "image_base64": color_b64,        # JPEG, for VLM
            "depth_base64": depth_b64,        # uint16 PNG, for 3D calc on Mac
            "depth_scale": depth_scale,       # multiply uint16 → mm
            "intrinsics": intrinsics,         # {fx, fy, cx, cy, width, height}
            "flange_pose": flange_pose,       # [x,y,z,roll,pitch,yaw] m/rad
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/move_above")
def post_move_above(req: MoveAboveRequest):
    """Move gripper to standoff above a 3D point computed on Mac."""
    if len(req.xyz_m) != 3:
        raise HTTPException(400, "xyz_m must have 3 values")
    try:
        z_safe_m = req.z_safe_mm / 1000.0
        current_rpy = (list(HW.robot.get_tcp_pose().msg[3:6])
                       if HW.tcp_offset else get_current_pose(HW.robot)[3:6].tolist())
        standoff_z = req.xyz_m[2] + req.standoff_mm / 1000.0
        target = [req.xyz_m[0], req.xyz_m[1], standoff_z, *current_rpy]
        flange_t = HW.robot.get_tcp2flange_pose(target) if HW.tcp_offset else target
        check_target_safe(flange_t)
        HW.robot.set_speed_percent(HW.speed_pct)
        safe_move_to(HW.robot, flange_t, z_safe_m=z_safe_m,
                     speed_pct=HW.speed_pct, tol_mm=15.0)
        final = get_current_pose(HW.robot)
        return {"status": "ok",
                "flange_xyz_m": [round(float(v), 5) for v in final[:3]]}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/grasp")
def post_grasp(req: GraspRequest):
    """Descend to grasp_z, close gripper, lift to lift_z. Auto-retries on IK fail."""
    try:
        HW.effector.move_gripper(width=HW.gripper_width, force=1.0)
        time.sleep(0.8)

        actual_z = None
        for attempt in range(HW.grasp_retry_count + 1):
            z_try = req.grasp_z_mm + attempt * HW.grasp_retry_step_mm
            if attempt > 0:
                log.info(f"IK retry grasp at Z={z_try:.0f} mm")
            grasp_z = z_try / 1000.0
            if grasp_z < 0.02:
                raise RuntimeError(f"grasp Z={grasp_z:.4f} m below 20 mm safety minimum")

            current = (np.array(HW.robot.get_tcp_pose().msg[:6])
                       if HW.tcp_offset else get_current_pose(HW.robot))
            descent = [float(current[0]), float(current[1]), grasp_z,
                       float(current[3]), float(current[4]), float(current[5])]
            flange_d = HW.robot.get_tcp2flange_pose(descent) if HW.tcp_offset else descent
            check_target_safe(flange_d, z_min_m=0.02)

            HW.robot.set_speed_percent(min(HW.speed_pct, 15))
            time.sleep(0.05)
            HW.robot.move_l(flange_d)
            try:
                wait_move_done(HW.robot, flange_d[:3], tol_mm=3.0)
                actual_z = z_try
                break
            except RuntimeError as exc:
                if "no_ik_solution" in str(exc) and attempt < HW.grasp_retry_count:
                    continue
                raise

        HW.effector.move_gripper(width=0.0, force=HW.gripper_force)
        time.sleep(1.0)
        st = HW.effector.get_gripper_status()
        grip_width = float(st.msg.width)
        safe_lift(HW.robot, height_m=req.lift_z_mm / 1000.0)

        return {"status": "ok",
                "grasp_z_mm": actual_z,
                "grip_width_m": round(grip_width, 5),
                "grasped": grip_width > 0.001}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/place")
def post_place(req: PlaceRequest):
    """Transit to xyz, lower to place_z, release, lift. xyz computed on Mac."""
    if len(req.xyz_m) != 3:
        raise HTTPException(400, "xyz_m must have 3 values")
    try:
        z_safe_m = req.z_safe_mm / 1000.0
        place_z = req.place_z_mm / 1000.0

        current_rpy = (list(HW.robot.get_tcp_pose().msg[3:6])
                       if HW.tcp_offset else get_current_pose(HW.robot)[3:6].tolist())
        above = [req.xyz_m[0], req.xyz_m[1], z_safe_m, *current_rpy]
        flange_above = HW.robot.get_tcp2flange_pose(above) if HW.tcp_offset else above
        check_target_safe(flange_above)
        HW.robot.set_speed_percent(HW.speed_pct)
        safe_move_to(HW.robot, flange_above, z_safe_m=z_safe_m,
                     speed_pct=HW.speed_pct, tol_mm=15.0)

        current_rpy = (list(HW.robot.get_tcp_pose().msg[3:6])
                       if HW.tcp_offset else get_current_pose(HW.robot)[3:6].tolist())
        place_pose = [req.xyz_m[0], req.xyz_m[1], place_z, *current_rpy]
        flange_p = HW.robot.get_tcp2flange_pose(place_pose) if HW.tcp_offset else place_pose
        check_target_safe(flange_p, z_min_m=0.02)

        HW.robot.set_speed_percent(min(HW.speed_pct, 15))
        time.sleep(0.05)
        HW.robot.move_l(flange_p)
        wait_move_done(HW.robot, flange_p[:3], tol_mm=3.0)

        HW.effector.move_gripper(width=0.06, force=1.0)
        time.sleep(1.0)

        safe_lift(HW.robot, height_m=z_safe_m)
        return {"status": "ok", "place_z_mm": req.place_z_mm}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/home")
def post_home():
    try:
        HW.robot.set_speed_percent(HW.speed_pct)
        time.sleep(0.05)
        HW.robot.move_j([0.0] * 7)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            check_arm_error(HW.robot)
            ja = HW.robot.get_joint_angles()
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
        HW.robot.electronic_emergency_stop()
        return {"status": "emergency_stop_sent"}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/move_j")
def post_move_j(req: MoveJRequest):
    if len(req.angles_deg) != 7:
        raise HTTPException(400, "angles_deg must have exactly 7 values")
    try:
        rad = [math.radians(d) for d in req.angles_deg]
        HW.robot.set_speed_percent(HW.speed_pct)
        time.sleep(0.05)
        HW.robot.move_j(rad)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            check_arm_error(HW.robot)
            ja = HW.robot.get_joint_angles()
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

    parser = argparse.ArgumentParser(description="NERO Pi robot server (I/O only)")
    parser.add_argument("--tcp", default=None, help="Path to TCP offset JSON")
    parser.add_argument("--output-dir", default="/home/pi/session")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--speed", type=int, default=30)
    parser.add_argument("--flush-frames", type=int, default=2)
    args = parser.parse_args()

    HW.tcp_offset  = load_tcp_offset(args.tcp) if args.tcp else None
    HW.out_dir     = Path(args.output_dir)
    HW.speed_pct   = args.speed
    HW.flush_frames = args.flush_frames

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
