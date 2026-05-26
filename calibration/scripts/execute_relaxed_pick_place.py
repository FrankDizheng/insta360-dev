#!/usr/bin/env python3
"""Execute key relaxed pick-place plan stages on the real robot.

Plan A (default): replace homography-based place standoff with P5 calibration
via move_j to the contact-aligned flange pose.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gripper_align_place import (  # noqa: E402
    load_p5_place_plan,
    patch_plan_place_standoff,
    plan_p5_place_standoff_from_q,
)

DEFAULT_PI_URL = os.getenv("PICK_PLACE_PI_URL", "http://10.13.167.212:8765")
DEFAULT_PLAN = (
    REPO_ROOT
    / "calibration/results/live_rescan_relaxed_plan_2026-05-26_123215"
    / "relaxed_centerline_offline_plan_fixed_grasp.json"
)
PICK_STAGE = "pick_standoff_fixed_grasp"
PLACE_STAGE = "place_standoff_slot_centerline"


def _get_status(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/status", timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(base_url: str, path: str, payload: dict, *, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{base_url}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _move_j(base_url: str, q_deg: list[float], *, retries: int = 2) -> dict:
    last: dict = {"status": "not_sent"}
    for attempt in range(retries + 1):
        last = _post_json(base_url, "/move_j", {"angles_deg": q_deg}, timeout=120.0)
        time.sleep(1.0 if last.get("status") == "timeout" else 0.4)
        status = _get_status(base_url)
        joints = status.get("joint_angles_deg") or []
        if joints and max(abs(a - b) for a, b in zip(joints, q_deg)) < 1.5:
            return {"move_j": last, "status": status, "converged": True}
        if attempt < retries:
            continue
    return {"move_j": last, "status": _get_status(base_url), "converged": False}


def _find_stage(plan: dict, name: str) -> dict:
    for stage in plan.get("stages") or []:
        if stage.get("name") == name:
            return stage
    raise KeyError(f"Stage {name!r} not found in plan")


def _resolve_place_stage(plan: dict, *, place_mode: str, start_q_deg: list[float] | None) -> dict:
    if place_mode == "plan":
        return _find_stage(plan, PLACE_STAGE)
    if start_q_deg is None:
        p5_plan = load_p5_place_plan()
    else:
        p5_plan = plan_p5_place_standoff_from_q(start_q_deg, fixed_rpy=True)
    if not p5_plan.get("ok"):
        raise RuntimeError(f"P5 place standoff planning failed: {p5_plan.get('reason')}")
    patched = patch_plan_place_standoff(plan, p5_plan)
    for stage in patched["stages"]:
        if stage.get("name") == "place_standoff_p5_calib":
            return stage
    raise RuntimeError("Patched plan missing place_standoff_p5_calib stage")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute relaxed pick-place key stages")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--pi-url", default=DEFAULT_PI_URL)
    parser.add_argument(
        "--stage",
        choices=["pick_standoff", "place_standoff", "both"],
        default="both",
        help="Which key stage(s) to execute",
    )
    parser.add_argument(
        "--place-mode",
        choices=["p5", "plan"],
        default="p5",
        help="Place standoff target source (default: P5 calibration)",
    )
    parser.add_argument(
        "--stop-before-place",
        action="store_true",
        help="After pick standoff, stop without moving to place standoff",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve targets only; do not command the robot",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional execution log JSON path")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    log: dict = {
        "schema": "relaxed_pick_place_execution.v1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "plan": str(args.plan).replace("\\", "/"),
        "pi_url": args.pi_url,
        "stage": args.stage,
        "place_mode": args.place_mode,
        "dry_run": args.dry_run,
        "steps": [],
    }

    status_before = None if args.dry_run else _get_status(args.pi_url)
    log["status_before"] = status_before

    pick_stage = _find_stage(plan, PICK_STAGE) if args.stage in ("pick_standoff", "both") else None
    place_stage = None
    if args.stage in ("place_standoff", "both") and not args.stop_before_place:
        start_q = None
        if pick_stage is not None:
            start_q = pick_stage["goal_q_deg"]
        elif status_before and status_before.get("joint_angles_deg"):
            start_q = status_before["joint_angles_deg"]
        place_stage = _resolve_place_stage(plan, place_mode=args.place_mode, start_q_deg=start_q)

    if pick_stage is not None and args.stage in ("pick_standoff", "both"):
        step = {
            "name": pick_stage["name"],
            "goal_q_deg": pick_stage["goal_q_deg"],
            "target_xyz_m": pick_stage.get("target_xyz_m"),
        }
        if not args.dry_run:
            step.update(_move_j(args.pi_url, pick_stage["goal_q_deg"]))
        log["steps"].append(step)

    if place_stage is not None and args.stage in ("place_standoff", "both") and not args.stop_before_place:
        step = {
            "name": place_stage["name"],
            "goal_q_deg": place_stage["goal_q_deg"],
            "target_xyz_m": place_stage.get("target_xyz_m"),
            "method": place_stage.get("method", args.place_mode),
        }
        if not args.dry_run:
            step.update(_move_j(args.pi_url, place_stage["goal_q_deg"]))
        log["steps"].append(step)

    if not args.dry_run:
        log["status_after"] = _get_status(args.pi_url)
    log["stopped_before_descent"] = True

    if args.out is None:
        out_dir = args.plan.parent / "execution_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.out = out_dir / f"execute_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
