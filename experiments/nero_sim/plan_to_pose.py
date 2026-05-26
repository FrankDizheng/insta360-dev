"""Validate an SDK-flange XYZ+RPY target with local NERO pose IK.

This is an offline safety gate. It does not command the real robot. Use the
exported ``goal_q_deg`` only after reviewing geometry details and path preview.

By default ``--rpy`` is interpreted as SDK-reported RPY. The script aligns that
RPY convention to the local FK rotation matrix using a reference pose.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nero.geometry import envelope_penalty  # noqa: E402
from nero.kinematics import forward_kinematics  # noqa: E402
from nero.planning import (  # noqa: E402
    deg_to_rad,
    matrix_to_rpy,
    plan_pose_motion,
    plan_relaxed_pose_motion,
    rad_to_deg,
    rpy_to_matrix,
)


DEFAULT_SCAN_POSE_DEG = [7.817, -51.497, -10.701, 101.596, -9.619, 5.546, 66.204]
DEFAULT_SCAN_SDK_RPY = [1.4849086743042557, -0.4781853084614064, -0.12283627275536092]


def _parse_csv_floats(raw: str, expected_len: int, label: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated floats") from exc
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(f"{label} expected {expected_len} values, got {len(values)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NERO XYZ+RPY pose IK locally")
    parser.add_argument("--xyz", required=True, type=lambda raw: _parse_csv_floats(raw, 3, "xyz"))
    parser.add_argument("--rpy", required=True, type=lambda raw: _parse_csv_floats(raw, 3, "rpy"))
    parser.add_argument(
        "--rpy-frame",
        default="sdk",
        choices=["sdk", "fk"],
        help="Interpret --rpy as SDK-reported RPY or local FK RPY. Default: sdk.",
    )
    parser.add_argument(
        "--start-deg",
        default=",".join(str(v) for v in DEFAULT_SCAN_POSE_DEG),
        type=lambda raw: _parse_csv_floats(raw, 7, "start-deg"),
    )
    parser.add_argument(
        "--reference-sdk-rpy",
        default=",".join(str(v) for v in DEFAULT_SCAN_SDK_RPY),
        type=lambda raw: _parse_csv_floats(raw, 3, "reference-sdk-rpy"),
        help="SDK RPY observed at --start-deg, used to align SDK and local FK frames.",
    )
    parser.add_argument(
        "--relax-axis",
        default="none",
        choices=["none", "tool_x", "tool_y", "tool_z", "world_x", "world_y", "world_z"],
        help="Release one rotation axis instead of solving a fully fixed RPY.",
    )
    parser.add_argument("--relax-sweep-deg", type=float, default=180.0)
    parser.add_argument("--relax-step-deg", type=float, default=15.0)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    start_rad = deg_to_rad(args.start_deg)
    planning_rpy = list(args.rpy)
    frame_alignment = None
    if args.rpy_frame == "sdk":
        ref_fk = forward_kinematics(start_rad, clamp=False)["flange_T"]
        ref_fk_rot = ref_fk[:3, :3]
        ref_sdk_rot = rpy_to_matrix(*args.reference_sdk_rpy)
        fk_to_sdk_rot = ref_fk_rot.T @ ref_sdk_rot
        target_sdk_rot = rpy_to_matrix(*args.rpy)
        target_fk_rot = target_sdk_rot @ fk_to_sdk_rot.T
        planning_rpy = matrix_to_rpy(target_fk_rot)
        frame_alignment = {
            "reference_start_q_deg": [float(v) for v in args.start_deg],
            "reference_sdk_rpy": [float(v) for v in args.reference_sdk_rpy],
            "reference_fk_rpy": matrix_to_rpy(ref_fk_rot),
            "fk_to_sdk_rotation": fk_to_sdk_rot.tolist(),
        }

    if args.relax_axis == "none":
        result = plan_pose_motion(args.xyz, planning_rpy, start_rad)
        planner_mode = "fixed_rpy"
    else:
        result = plan_relaxed_pose_motion(
            args.xyz,
            planning_rpy,
            start_rad,
            free_axis=args.relax_axis,
            sweep_rad=math.radians(args.relax_sweep_deg),
            step_rad=math.radians(args.relax_step_deg),
        )
        planner_mode = "relaxed_rpy"
    fk_goal = forward_kinematics(result.goal_rad, clamp=False)["flange_T"]
    goal_penalty, goal_details = envelope_penalty(result.goal_rad)
    reported_details = dict(goal_details)
    reported_details.update(result.geometry_details)

    payload = {
        "schema": "nero.pose_ik_validation.v1",
        "target": {
            "xyz_m": [float(v) for v in args.xyz],
            "rpy_rad": [float(v) for v in args.rpy],
            "rpy_frame": args.rpy_frame,
            "planning_rpy_rad": [float(v) for v in planning_rpy],
            "selected_planning_rpy_rad": [float(v) for v in result.target_rpy_rad],
        },
        "planner_mode": planner_mode,
        "relaxation": {
            "axis": args.relax_axis,
            "sweep_deg": float(args.relax_sweep_deg),
            "step_deg": float(args.relax_step_deg),
        },
        "frame_alignment": frame_alignment,
        "start_q_deg": [float(v) for v in args.start_deg],
        "ok": bool(result.ok),
        "reason": result.reason,
        "goal_q_deg": [float(v) for v in rad_to_deg(result.goal_rad)],
        "position_error_mm": float(result.position_error_m) * 1000.0,
        "rotation_error_deg": float(result.rotation_error_rad) * 180.0 / 3.141592653589793,
        "path_waypoints": len(result.path_rad),
        "cost": float(result.cost),
        "goal_flange_xyz_m": [float(v) for v in fk_goal[:3, 3]],
        "geometry_penalty": float(goal_penalty),
        "geometry_details": reported_details,
        "path_q_deg": [[float(v) for v in rad_to_deg(q)] for q in result.path_rad],
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
