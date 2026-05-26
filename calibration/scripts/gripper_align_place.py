"""Plan A: P5 gripper-alignment place standoff targets.

Uses the measured P5 slot_center contact-aligned flange pose directly,
instead of projecting slot pixels through the support-plane homography.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO_ROOT / "calibration/results/red_box_gripper_align_calib_2026-05-26"
DEFAULT_P5_PLACE_TARGET = CALIB_DIR / "place_standoff_p5_target.json"
DEFAULT_P5_PLAN_FROM_SCAN = CALIB_DIR / "place_standoff_p5_plan_from_scan.json"
DEFAULT_SCAN_POSE_DEG = [7.817, -51.497, -10.701, 101.596, -9.619, 5.546, 66.204]
DEFAULT_SCAN_SDK_RPY = [1.4849086743042557, -0.4781853084614064, -0.12283627275536092]
PLACE_STAGE_NAME = "place_standoff_slot_centerline"


def load_p5_place_standoff_target(
    path: Path = DEFAULT_P5_PLACE_TARGET,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    xyz = payload["place_standoff_flange_xyz_m"]
    rpy = payload["place_standoff_flange_rpy_rad"]
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError(f"Invalid place standoff pose in {path}")
    return {
        "flange_xyz_m": [float(v) for v in xyz],
        "flange_rpy_rad": [float(v) for v in rpy],
        "slot_pixel_uv": payload.get("slot_pixel_uv_scan_pose"),
        "method": payload.get("method", "plan_a_direct_p5_measurement"),
        "source": str(path),
    }


def load_p5_place_plan(path: Path = DEFAULT_P5_PLAN_FROM_SCAN) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    goal_q = payload.get("goal_q_deg")
    if not goal_q or len(goal_q) != 7:
        raise ValueError(f"goal_q_deg missing in {path}")
    return payload


def sdk_rpy_to_planning_rpy(
    target_rpy_sdk: list[float],
    *,
    start_q_deg: list[float] | None = None,
    reference_sdk_rpy: list[float] | None = None,
) -> list[float]:
    from nero.kinematics import forward_kinematics
    from nero.planning import deg_to_rad, matrix_to_rpy, rpy_to_matrix

    start = deg_to_rad(start_q_deg or DEFAULT_SCAN_POSE_DEG)
    ref_sdk = reference_sdk_rpy or DEFAULT_SCAN_SDK_RPY
    ref_fk_rot = forward_kinematics(start, clamp=False)["flange_T"][:3, :3]
    ref_sdk_rot = rpy_to_matrix(*ref_sdk)
    fk_to_sdk_rot = ref_fk_rot.T @ ref_sdk_rot
    target_sdk_rot = rpy_to_matrix(*target_rpy_sdk)
    return list(matrix_to_rpy(target_sdk_rot @ fk_to_sdk_rot.T))


def plan_p5_place_standoff_from_q(
    start_q_deg: list[float],
    *,
    fixed_rpy: bool = True,
) -> dict[str, Any]:
    """Plan joint target for P5 place standoff from an arbitrary start configuration."""
    from nero.geometry import envelope_penalty
    from nero.kinematics import forward_kinematics
    from nero.planning import deg_to_rad, plan_pose_motion, plan_relaxed_pose_motion, rad_to_deg

    target = load_p5_place_standoff_target()
    xyz = target["flange_xyz_m"]
    rpy_sdk = target["flange_rpy_rad"]
    start_rad = deg_to_rad(start_q_deg)
    planning_rpy = sdk_rpy_to_planning_rpy(rpy_sdk, start_q_deg=start_q_deg)

    if fixed_rpy:
        result = plan_pose_motion(xyz, planning_rpy, start_rad)
        planner_mode = "fixed_p5_rpy"
    else:
        result = plan_relaxed_pose_motion(
            xyz,
            planning_rpy,
            start_rad,
            free_axis="tool_z",
            sweep_rad=math.radians(180.0),
            step_rad=math.radians(30.0),
        )
        planner_mode = "relaxed_tool_z"

    fk_goal = forward_kinematics(result.goal_rad, clamp=False)["flange_T"]
    penalty, details = envelope_penalty(result.goal_rad)
    return {
        "schema": "place_standoff_p5_plan.v1",
        "method": "plan_a_direct_p5_measurement",
        "planner_mode": planner_mode,
        "start_q_deg": [float(v) for v in start_q_deg],
        "target_xyz_m": xyz,
        "target_rpy_sdk_rad": rpy_sdk,
        "ok": bool(result.ok),
        "reason": result.reason,
        "goal_q_deg": [float(v) for v in rad_to_deg(result.goal_rad)],
        "position_error_mm": float(result.position_error_m) * 1000.0,
        "rotation_error_deg": float(result.rotation_error_rad) * 180.0 / math.pi,
        "goal_flange_xyz_m": [float(v) for v in fk_goal[:3, 3]],
        "geometry_penalty": float(penalty),
        "geometry_details": details,
    }


def patch_plan_place_standoff(plan: dict[str, Any], p5_plan: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *plan* with the place standoff stage replaced by P5 calibration."""
    out = copy.deepcopy(plan)
    stages = out.get("stages") or []
    patched = False
    for idx, stage in enumerate(stages):
        if PLACE_STAGE_NAME not in stage.get("name", ""):
            continue
        stages[idx] = {
            **stage,
            "name": "place_standoff_p5_calib",
            "method": "plan_a_direct_p5_measurement",
            "target_xyz_m": p5_plan["target_xyz_m"],
            "target_rpy_sdk_rad": p5_plan.get("target_rpy_sdk_rad"),
            "goal_q_deg": p5_plan["goal_q_deg"],
            "position_error_mm": p5_plan.get("position_error_mm"),
            "rotation_error_deg": p5_plan.get("rotation_error_deg"),
            "goal_flange_xyz_m": p5_plan.get("goal_flange_xyz_m"),
            "geometry_penalty": p5_plan.get("geometry_penalty"),
            "geometry_details": p5_plan.get("geometry_details"),
            "replaced_stage": stage.get("name"),
            "replaced_target_xyz_m": stage.get("target_xyz_m"),
        }
        patched = True
        break
    if not patched:
        raise KeyError(f"No stage matching {PLACE_STAGE_NAME!r} in plan")
    perception = out.setdefault("perception", {})
    p5_target = load_p5_place_standoff_target()
    perception["place_standoff_flange_xyz_m"] = p5_target["flange_xyz_m"]
    perception["place_standoff_method"] = p5_target["method"]
    perception["deprecated_slot_center_base_xyz_m"] = perception.get("slot_center_base_xyz_m")
    out["place_standoff_policy"] = "plan_a_direct_p5_measurement"
    return out


def attach_p5_place_target(result: dict[str, Any]) -> dict[str, Any]:
    """Attach Plan A place standoff metadata to a slot-detect or localization dict."""
    p5 = load_p5_place_standoff_target()
    out = copy.deepcopy(result)
    out["place_standoff"] = {
        "method": p5["method"],
        "flange_xyz_m": p5["flange_xyz_m"],
        "flange_rpy_rad": p5["flange_rpy_rad"],
        "slot_pixel_uv": p5.get("slot_pixel_uv"),
        "source": p5["source"],
        "note": "Use flange_xyz_m as move_j / move_above target, not slot_base.center_base_xyz_m.",
    }
    slot_base = out.get("slot_base") or {}
    if slot_base.get("center_base_xyz_m"):
        slot_xy = slot_base["center_base_xyz_m"][:2]
        flange_xy = p5["flange_xyz_m"][:2]
        out["place_standoff"]["delta_mm_vs_homography_slot_center"] = [
            round((flange_xy[i] - slot_xy[i]) * 1000.0, 2) for i in range(2)
        ]
    return out
