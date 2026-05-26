#!/usr/bin/env python3
"""Validate fixed-camera slot detection through SimNERO pick+place planning.

Reads (or runs) fixed-camera slot JSON, combines with wrist-camera pick prior,
plans all waypoints in simulation, runs the high-level pick-place sequence, and
writes a real-robot handoff exchange file.

Usage:
    python experiments/nero_sim/run_fixed_camera_pick_place_sim.py
    python experiments/nero_sim/run_fixed_camera_pick_place_sim.py --dry-plan-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "calibration" / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fixed_camera_slot_detect import analyze_image  # noqa: E402
from gripper_align_place import load_p5_place_standoff_target  # noqa: E402
from nero import Position3D, get_robot_controller  # noqa: E402
from nero.kinematics import forward_kinematics, tcp_position  # noqa: E402
from nero.planning import deg_to_rad, plan_joint_motion, plan_tcp_motion, rad_to_deg  # noqa: E402
from nero.types import HOME_ANGLES_DEG  # noqa: E402

DEFAULT_SLOT_IMAGE = (
    REPO_ROOT
    / "data"
    / "cases"
    / "dual_camera_bottle_moves_2026-05-23_170330"
    / "move_01"
    / "fixed_camera.jpg"
)
DEFAULT_HOMOGRAPHY = REPO_ROOT / "calibration/results/top_camera_plane_homography_current.json"
DEFAULT_PICK_EXCHANGE = (
    REPO_ROOT / "calibration/results/nero_sim_current_to_bottle_exchange_2026-05-11.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "calibration/results/fixed_camera_sim_handoff_2026-05-24"

SCAN_POSE_DEG = [7.817, -51.497, -10.701, 101.596, -9.619, 5.546, 66.204]
SAFE_STANDOFF_Z_M = 0.42
TABLE_Z_M = 0.25
TABLE_CLEARANCE_M = 0.03
PLACE_Z_OFFSET_M = 0.030
PICK_GRASP_Z_M = 0.275


@dataclass
class WaypointCheck:
    name: str
    xyz_m: list[float]
    start_mode: str
    plan_ok: bool
    reason: str
    position_error_mm: float
    path_waypoints: int


def _load_pick_xyz(exchange_path: Path) -> list[float]:
    payload = json.loads(exchange_path.read_text(encoding="utf-8-sig"))
    target = payload.get("target") or {}
    xyz = target.get("xyz_m")
    if xyz and len(xyz) == 3:
        return [float(x) for x in xyz]
    raise ValueError(f"target.xyz_m missing in {exchange_path}")


def _slot_targets(slot_analysis: dict, *, use_p5_place_standoff: bool = True) -> dict[str, list[float]]:
    slot_xyz = list(slot_analysis["slot_base"]["center_base_xyz_m"])
    slot_surface_z = float(slot_xyz[2])
    place_xyz = [float(slot_xyz[0]), float(slot_xyz[1]), slot_surface_z + PLACE_Z_OFFSET_M]
    if use_p5_place_standoff:
        p5 = load_p5_place_standoff_target()
        place_standoff_xyz = list(p5["flange_xyz_m"])
        place_standoff_method = p5["method"]
    else:
        place_standoff_xyz = [place_xyz[0], place_xyz[1], SAFE_STANDOFF_Z_M]
        place_standoff_method = "homography_slot_center_xy"
    return {
        "slot_surface_xyz_m": slot_xyz,
        "place_xyz_m": place_xyz,
        "place_standoff_xyz_m": place_standoff_xyz,
        "place_standoff_method": place_standoff_method,
        "pick_standoff_xyz_m": None,
    }


def _plan_check(
    name: str,
    xyz_m: list[float],
    start_rad: list[float],
    *,
    start_mode: str,
) -> WaypointCheck:
    plan = plan_joint_motion(xyz_m, start_rad)
    return WaypointCheck(
        name=name,
        xyz_m=xyz_m,
        start_mode=start_mode,
        plan_ok=bool(plan.ok),
        reason=str(plan.reason),
        position_error_mm=float(plan.position_error_m) * 1000.0,
        path_waypoints=len(plan.path_rad),
    )


def _plan_tcp_check(
    name: str,
    xyz_m: list[float],
    start_rad: list[float],
    *,
    start_mode: str,
) -> WaypointCheck:
    plan = plan_tcp_motion(xyz_m, start_rad)
    return WaypointCheck(
        name=name,
        xyz_m=xyz_m,
        start_mode=start_mode,
        plan_ok=bool(plan.ok),
        reason=str(plan.reason),
        position_error_mm=float(plan.position_error_m) * 1000.0,
        path_waypoints=len(plan.path_rad),
    )


def _fk_flange_mm(q_deg: list[float]) -> list[float]:
    fk = forward_kinematics(deg_to_rad(q_deg))
    pos = fk["flange_T"][:3, 3]
    return [float(pos[0]) * 1000.0, float(pos[1]) * 1000.0, float(pos[2]) * 1000.0]


def _run_sim_sequence(
    pick_xyz: list[float],
    place_xyz: list[float],
    *,
    start_from_scan_pose: bool,
) -> dict[str, Any]:
    robot = get_robot_controller("sim-nero")
    robot.connect()
    steps_log: list[dict[str, Any]] = []

    def record(step: str, ok: bool) -> None:
        state = robot.get_state()
        flange = state.flange_pose
        steps_log.append(
            {
                "step": step,
                "ok": bool(ok),
                "joint_angles_deg": list(state.joint_angles_deg),
                "flange_xyz_m": [flange.x, flange.y, flange.z],
            }
        )

    if start_from_scan_pose:
        ok_scan = robot.move_joints(SCAN_POSE_DEG, settle_s=0.01)
        record("move_to_scan_pose", ok_scan)
        if not ok_scan:
            robot.disconnect()
            return {"ok": False, "steps": steps_log, "reason": "scan_pose_unreachable"}

    pick = Position3D(*pick_xyz)
    place = Position3D(*place_xyz)
    sequence = [
        ("move_above_pick", lambda: robot.move_above("pick", pick)),
        ("lower_pick", lambda: robot.lower("pick", pick)),
        ("grasp", robot.grasp),
        ("lift", robot.lift),
        ("move_above_place", lambda: robot.move_above("place", place)),
        ("lower_place", lambda: robot.lower("place", place)),
        ("release", robot.release),
        ("home", robot.home),
    ]

    all_ok = True
    for name, fn in sequence:
        ok = bool(fn())
        record(name, ok)
        if not ok:
            all_ok = False
            break

    robot.disconnect()
    return {"ok": all_ok, "steps": steps_log, "reason": "ok" if all_ok else "step_failed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-camera slot sim validation + handoff")
    parser.add_argument("--image", type=Path, default=DEFAULT_SLOT_IMAGE)
    parser.add_argument("--homography", type=Path, default=DEFAULT_HOMOGRAPHY)
    parser.add_argument("--pick-exchange", type=Path, default=DEFAULT_PICK_EXCHANGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--pick-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override pick XYZ (metres); default from exchange JSON",
    )
    parser.add_argument("--dry-plan-only", action="store_true", help="Only run planner checks, no SimNERO motion")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H%M%S")

    slot_analysis = analyze_image(args.image, homography_path=args.homography)
    slot_json_path = args.out_dir / "slot_detect.json"
    slot_json_path.write_text(json.dumps(slot_analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    pick_xyz = list(args.pick_xyz) if args.pick_xyz is not None else _load_pick_xyz(args.pick_exchange)
    pick_xyz[2] = PICK_GRASP_Z_M
    targets = _slot_targets(slot_analysis)
    targets["pick_grasp_xyz_m"] = pick_xyz
    targets["pick_standoff_xyz_m"] = [pick_xyz[0], pick_xyz[1], SAFE_STANDOFF_Z_M]

    home_rad = deg_to_rad(HOME_ANGLES_DEG)
    scan_rad = deg_to_rad(SCAN_POSE_DEG)

    planner_checks: list[dict[str, Any]] = []
    for name, xyz, start_rad, mode in [
        ("pick_standoff_from_home", targets["pick_standoff_xyz_m"], home_rad, "home"),
        ("place_standoff_from_home", targets["place_standoff_xyz_m"], home_rad, "home"),
        ("pick_grasp_from_home", pick_xyz, home_rad, "home"),
        ("place_from_home", targets["place_xyz_m"], home_rad, "home"),
        ("pick_standoff_from_scan", targets["pick_standoff_xyz_m"], scan_rad, "scan_pose"),
        ("place_standoff_from_scan", targets["place_standoff_xyz_m"], scan_rad, "scan_pose"),
        ("pick_grasp_from_scan", pick_xyz, scan_rad, "scan_pose"),
        ("place_from_scan", targets["place_xyz_m"], scan_rad, "scan_pose"),
    ]:
        check = _plan_check(name, xyz, start_rad, start_mode=mode)
        planner_checks.append(asdict(check))

    tcp_checks: list[dict[str, Any]] = []
    for name, xyz, start_rad, mode in [
        ("tcp_pick_from_scan", pick_xyz, scan_rad, "scan_pose"),
        ("tcp_place_from_scan", targets["place_xyz_m"], scan_rad, "scan_pose"),
    ]:
        check = _plan_tcp_check(name, xyz, start_rad, start_mode=mode)
        tcp_checks.append(asdict(check))

    sim_from_home = {"skipped": True}
    sim_from_scan = {"skipped": True}
    if not args.dry_plan_only:
        sim_from_home = _run_sim_sequence(
            pick_xyz, targets["place_xyz_m"], start_from_scan_pose=False
        )
        sim_from_scan = _run_sim_sequence(
            pick_xyz, targets["place_xyz_m"], start_from_scan_pose=True
        )

    all_plan_ok = all(c["plan_ok"] for c in planner_checks)
    sim_ok = (
        (sim_from_home.get("ok") is True if not sim_from_home.get("skipped") else True)
        and (sim_from_scan.get("ok") is True if not sim_from_scan.get("skipped") else True)
    )

  # Wrist vs fixed-camera place XY gap (informational)
    wrist_slot = [-0.38942663085572354, 0.2789906363966242, 0.2595467642065272]
    fixed_slot = targets["slot_surface_xyz_m"]
    place_xy_gap_mm = math.hypot(fixed_slot[0] - wrist_slot[0], fixed_slot[1] - wrist_slot[1]) * 1000.0

    handoff = {
        "schema": "nero.fixed_camera_sim_handoff.v1",
        "created_at_local": timestamp,
        "status": "ready_for_real_robot_standoff_test" if all_plan_ok and sim_ok else "sim_validation_incomplete",
        "source": {
            "slot_image": str(args.image.relative_to(REPO_ROOT)).replace("\\", "/"),
            "homography": str(args.homography.relative_to(REPO_ROOT)).replace("\\", "/"),
            "pick_exchange": str(args.pick_exchange.relative_to(REPO_ROOT)).replace("\\", "/"),
            "slot_detect_json": str(slot_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "coordinate_frame": "robot_base",
        "perception": {
            "slot_detection_mode": slot_analysis.get("detection_mode"),
            "slot_center_px": slot_analysis["slot_pixels"]["center_px"],
            "slot_center_mode": slot_analysis["slot_pixels"].get("center_mode", "unknown"),
        },
        "targets_m": targets,
        "safety": {
            "table_z_m": TABLE_Z_M,
            "table_clearance_min_m": TABLE_CLEARANCE_M,
            "safe_standoff_z_m": SAFE_STANDOFF_Z_M,
            "first_real_robot_motion": "standoff_only",
            "do_not_lower_until": [
                "SDK flange pose matches sim FK flange at scan pose",
                "TCP/tool tip offset confirmed",
                "fixed_camera vs wrist_camera place XY gap reviewed",
            ],
        },
        "initial_states": {
            "home_q_deg": list(HOME_ANGLES_DEG),
            "scan_q_deg": list(SCAN_POSE_DEG),
            "scan_fk_flange_mm": _fk_flange_mm(SCAN_POSE_DEG),
        },
        "consistency_check": {
            "wrist_camera_slot_xyz_m": wrist_slot,
            "fixed_camera_slot_xyz_m": fixed_slot,
            "place_xy_gap_mm": place_xy_gap_mm,
            "note": "Large XY gap expected until homography and wrist-depth frames are unified.",
        },
        "planner_validation": {
            "all_plan_ok": all_plan_ok,
            "joint_plans": planner_checks,
            "tcp_plans": tcp_checks,
        },
        "sim_execution": {
            "from_home": sim_from_home,
            "from_scan_pose": sim_from_scan,
        },
        "real_robot_commands": {
            "pi_scan_pose_deg": SCAN_POSE_DEG,
            "standoff_pick": {
                "xyz_m": targets["pick_standoff_xyz_m"],
                "note": "Use pi_pick_place_bridge or robot_server; confirm before lower.",
            },
            "standoff_place": {
                "xyz_m": targets["place_standoff_xyz_m"],
                "note": "Fixed-camera place target; confirm XY vs wrist prior.",
            },
            "grasp_after_checks": {
                "pick_xyz_m": pick_xyz,
                "place_xyz_m": targets["place_xyz_m"],
            },
        },
    }

    exchange_path = args.out_dir / "fixed_camera_sim_handoff.json"
    exchange_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(handoff, ensure_ascii=False, indent=2))
    print(f"\nWrote {exchange_path}", file=sys.stderr)

    if not all_plan_ok or not sim_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
