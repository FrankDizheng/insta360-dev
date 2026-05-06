"""Lightweight motion planning helpers for the NERO 7-DOF arm.

This module intentionally stays dependency-light: it uses the local FK model
from ``nero.kinematics`` plus damped least-squares IK and joint-space
interpolation.  It is designed for early simulation / planning work where we
need a repeatable baseline before introducing a full physics engine or a
learned policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np

from nero.geometry import ACTIVE_FRAME_ALIGNMENT, envelope_penalty
from nero.kinematics import tcp_position
from nero.types import HOME_ANGLES_DEG, JOINT_LIMITS_RAD, NUM_JOINTS, clamp_joints_rad


TABLE_Z_M = ACTIVE_FRAME_ALIGNMENT.table_z_m
DEFAULT_MAX_IK_ITERS = 160
DEFAULT_MAX_STEP_RAD = 0.08


@dataclass
class WorkspaceSamplerConfig:
    x_range_m: tuple[float, float] = (-0.18, 0.18)
    y_range_m: tuple[float, float] = (0.28, 0.52)
    z_range_m: tuple[float, float] = (0.16, 0.30)
    tolerance_m: float = 0.004
    max_tries: int = 200


@dataclass
class MotionPlanConfig:
    tolerance_m: float = 0.003
    max_ik_iters: int = DEFAULT_MAX_IK_ITERS
    ik_damping: float = 0.03
    max_step_rad: float = DEFAULT_MAX_STEP_RAD
    safe_lift_m: float = 0.08
    min_transit_z_m: float = 0.20
    direct_move_threshold_m: float = 0.04
    planner_mode: str = "staged"


@dataclass
class PlannerResult:
    ok: bool
    reason: str
    target_xyz_m: tuple[float, float, float]
    start_rad: list[float]
    goal_rad: list[float]
    path_rad: list[list[float]]
    position_error_m: float
    cost: float
    planner_mode: str = "direct"
    waypoint_targets_xyz_m: list[tuple[float, float, float]] = field(default_factory=list)
    stage_reasons: list[str] = field(default_factory=list)


@dataclass
class ReachSample:
    xyz_m: tuple[float, float, float]
    plan: PlannerResult
    attempts: int


def deg_to_rad(angles_deg: Sequence[float]) -> list[float]:
    return [math.radians(float(v)) for v in angles_deg[:NUM_JOINTS]]


def rad_to_deg(angles_rad: Sequence[float]) -> list[float]:
    return [math.degrees(float(v)) for v in angles_rad[:NUM_JOINTS]]


def _normalize_joint_vector(values: Sequence[float]) -> np.ndarray:
    if len(values) < NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} joints, got {len(values)}")
    clamped = clamp_joints_rad([float(v) for v in values[:NUM_JOINTS]])
    return np.asarray(clamped, dtype=np.float64)


def _target_vec(target_xyz_m: Sequence[float]) -> np.ndarray:
    if len(target_xyz_m) < 3:
        raise ValueError(f"Need xyz target, got {target_xyz_m!r}")
    return np.asarray(target_xyz_m[:3], dtype=np.float64)


def numerical_position_jacobian(
    joint_angles_rad: Sequence[float],
    *,
    eps: float = 1e-4,
) -> np.ndarray:
    """Finite-difference Jacobian of tool TCP XYZ wrt joint angles."""
    q = _normalize_joint_vector(joint_angles_rad)
    base = tcp_position(q.tolist(), clamp=False)
    jac = np.zeros((3, NUM_JOINTS), dtype=np.float64)
    for idx in range(NUM_JOINTS):
        qp = q.copy()
        qp[idx] += eps
        qm = q.copy()
        qm[idx] -= eps
        fp = tcp_position(qp.tolist(), clamp=True)
        fm = tcp_position(qm.tolist(), clamp=True)
        jac[:, idx] = (fp - fm) / (2.0 * eps)
    if not np.all(np.isfinite(base)):
        raise RuntimeError("FK returned a non-finite flange position")
    return jac


def solve_ik_position(
    target_xyz_m: Sequence[float],
    initial_guess_rad: Sequence[float],
    *,
    max_iters: int = DEFAULT_MAX_IK_ITERS,
    damping: float = 0.03,
    tolerance_m: float = 0.003,
) -> tuple[list[float], float]:
    """Solve position-only IK for the approximate stock-gripper TCP."""
    target = _target_vec(target_xyz_m)
    q = _normalize_joint_vector(initial_guess_rad)
    best_q = q.copy()
    best_err = float("inf")

    for _ in range(max_iters):
        pos = tcp_position(q.tolist(), clamp=False)
        err_vec = target - pos
        err = float(np.linalg.norm(err_vec))
        if err < best_err:
            best_err = err
            best_q = q.copy()
        if err <= tolerance_m:
            break

        jac = numerical_position_jacobian(q.tolist())
        jjt = jac @ jac.T + (damping ** 2) * np.eye(3, dtype=np.float64)
        dq = jac.T @ np.linalg.solve(jjt, err_vec)

        # Conservative step size helps keep the numeric IK stable.
        q = q + 0.45 * dq
        q = _normalize_joint_vector(q.tolist())

    return best_q.tolist(), best_err


def default_seed_bank(current_rad: Sequence[float]) -> list[list[float]]:
    current = _normalize_joint_vector(current_rad).tolist()
    home = deg_to_rad(HOME_ANGLES_DEG)
    mirrored = list(current)
    mirrored[0] = -mirrored[0]
    lifted = list(current)
    lifted[1] = max(JOINT_LIMITS_RAD[1][0], lifted[1] - math.radians(15.0))
    relaxed = list(home)
    relaxed[1] = math.radians(40.0)
    relaxed[3] = math.radians(45.0)
    elbow_up = list(current)
    elbow_up[3] = min(JOINT_LIMITS_RAD[3][1], elbow_up[3] + math.radians(20.0))
    elbow_down = list(current)
    elbow_down[3] = max(JOINT_LIMITS_RAD[3][0], elbow_down[3] - math.radians(20.0))
    wrist_roll_pos = list(current)
    wrist_roll_pos[5] = min(JOINT_LIMITS_RAD[5][1], wrist_roll_pos[5] + math.radians(12.0))
    wrist_roll_neg = list(current)
    wrist_roll_neg[5] = max(JOINT_LIMITS_RAD[5][0], wrist_roll_neg[5] - math.radians(12.0))
    seeds = [current, home, mirrored, lifted, relaxed, elbow_up, elbow_down, wrist_roll_pos, wrist_roll_neg]
    unique: list[list[float]] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        key = tuple(int(round(v * 10000.0)) for v in seed)
        if key not in seen:
            unique.append(seed)
            seen.add(key)
    return unique


def interpolate_joint_path(
    start_rad: Sequence[float],
    goal_rad: Sequence[float],
    *,
    max_step_rad: float = DEFAULT_MAX_STEP_RAD,
) -> list[list[float]]:
    start = _normalize_joint_vector(start_rad)
    goal = _normalize_joint_vector(goal_rad)
    delta = goal - start
    max_delta = float(np.max(np.abs(delta)))
    steps = max(2, int(math.ceil(max_delta / max_step_rad)) + 1)

    path: list[list[float]] = []
    for idx in range(steps):
        t = idx / (steps - 1)
        smooth_t = 0.5 * (1.0 - math.cos(math.pi * t))
        q = start + smooth_t * delta
        path.append(_normalize_joint_vector(q.tolist()).tolist())
    return path


def stitch_joint_paths(paths_rad: Sequence[Sequence[Sequence[float]]]) -> list[list[float]]:
    stitched: list[list[float]] = []
    for path in paths_rad:
        if not path:
            continue
        if not stitched:
            stitched.extend([list(waypoint) for waypoint in path])
            continue
        if np.allclose(np.asarray(stitched[-1]), np.asarray(path[0]), atol=1e-9):
            stitched.extend([list(waypoint) for waypoint in path[1:]])
        else:
            stitched.extend([list(waypoint) for waypoint in path])
    return stitched


def joint_path_cost(path_rad: Sequence[Sequence[float]]) -> float:
    """Score a path using travel distance + safety penalties."""
    if not path_rad:
        return float("inf")
    total = 0.0
    margin_penalty = 0.0
    geometry_penalty = 0.0
    prev = np.asarray(path_rad[0], dtype=np.float64)
    for waypoint in path_rad:
        g_penalty, _details = envelope_penalty(waypoint)
        geometry_penalty += g_penalty
    for waypoint in path_rad[1:]:
        q = np.asarray(waypoint, dtype=np.float64)
        total += float(np.linalg.norm(q - prev, ord=1))
        prev = q
        for joint_idx, value in enumerate(q):
            lo, hi = JOINT_LIMITS_RAD[joint_idx]
            span = hi - lo
            near = min(value - lo, hi - value)
            if near < 0.1 * span:
                margin_penalty += (0.1 * span - near) / max(span, 1e-6)
    return total + 0.35 * margin_penalty + geometry_penalty


def plan_joint_motion(
    target_xyz_m: Sequence[float],
    start_rad: Sequence[float],
    *,
    seed_candidates_rad: Iterable[Sequence[float]] | None = None,
    tolerance_m: float = 0.003,
) -> PlannerResult:
    """Plan a direct joint path to reach a Cartesian XYZ goal."""
    target = _target_vec(target_xyz_m)
    if target[2] < TABLE_Z_M - 1e-6:
        return PlannerResult(
            ok=False,
            reason="target_below_table",
            target_xyz_m=tuple(float(v) for v in target),
            start_rad=_normalize_joint_vector(start_rad).tolist(),
            goal_rad=_normalize_joint_vector(start_rad).tolist(),
            path_rad=[],
            position_error_m=float("inf"),
            cost=float("inf"),
            planner_mode="direct",
            waypoint_targets_xyz_m=[tuple(float(v) for v in target)],
            stage_reasons=["target_below_table"],
        )

    start = _normalize_joint_vector(start_rad).tolist()
    seeds = list(seed_candidates_rad or default_seed_bank(start))
    best_goal = start
    best_err = float("inf")
    best_cost = float("inf")
    best_path: list[list[float]] = []

    for seed in seeds:
        candidate_goal, err = solve_ik_position(
            target,
            seed,
            tolerance_m=tolerance_m,
        )
        candidate_path = interpolate_joint_path(start, candidate_goal)
        candidate_cost = joint_path_cost(candidate_path)
        if err < best_err or (math.isclose(err, best_err, abs_tol=1e-6) and candidate_cost < best_cost):
            best_goal = candidate_goal
            best_err = err
            best_cost = candidate_cost
            best_path = candidate_path

    return PlannerResult(
        ok=best_err <= tolerance_m,
        reason="ok" if best_err <= tolerance_m else "ik_tolerance_not_met",
        target_xyz_m=tuple(float(v) for v in target),
        start_rad=start,
        goal_rad=best_goal,
        path_rad=best_path,
        position_error_m=best_err,
        cost=best_cost,
        planner_mode="direct",
        waypoint_targets_xyz_m=[tuple(float(v) for v in target)],
        stage_reasons=["ok" if best_err <= tolerance_m else "ik_tolerance_not_met"],
    )


def _planner_seed_bank(
    current_rad: Sequence[float],
    target_xyz_m: Sequence[float],
) -> list[list[float]]:
    seeds = default_seed_bank(current_rad)
    current = _normalize_joint_vector(current_rad).tolist()
    target = _target_vec(target_xyz_m)
    forward_yaw = math.atan2(float(target[0]), max(float(target[1]), 1e-6))
    facing_seed = list(current)
    facing_seed[0] = max(JOINT_LIMITS_RAD[0][0], min(JOINT_LIMITS_RAD[0][1], forward_yaw))
    seeds.append(facing_seed)
    shoulder_seed = list(current)
    shoulder_seed[1] = max(JOINT_LIMITS_RAD[1][0], min(JOINT_LIMITS_RAD[1][1], math.radians(20.0)))
    seeds.append(shoulder_seed)
    deduped: list[list[float]] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        key = tuple(int(round(v * 10000.0)) for v in seed)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(seed)
    return deduped


def plan_tcp_motion(
    target_xyz_m: Sequence[float],
    start_rad: Sequence[float],
    *,
    config: MotionPlanConfig | None = None,
) -> PlannerResult:
    """Plan a staged TCP motion resembling a basic motion-planning pipeline.

    The planner breaks a move into:
      1. optional lift to a safe transit height
      2. optional XY transit at safe height
      3. final descent to the target
    """
    cfg = config or MotionPlanConfig()
    target = _target_vec(target_xyz_m)
    start = _normalize_joint_vector(start_rad).tolist()
    start_tcp = tcp_position(start, clamp=False)

    direct_dist = float(np.linalg.norm(target - start_tcp))
    if direct_dist <= cfg.direct_move_threshold_m:
        direct = plan_joint_motion(
            target,
            start,
            seed_candidates_rad=_planner_seed_bank(start, target),
            tolerance_m=cfg.tolerance_m,
        )
        direct.planner_mode = "direct_short"
        direct.waypoint_targets_xyz_m = [tuple(float(v) for v in target)]
        return direct

    safe_z = max(cfg.min_transit_z_m, float(start_tcp[2]), float(target[2])) + cfg.safe_lift_m
    waypoints: list[tuple[float, float, float]] = []
    if abs(float(start_tcp[2]) - safe_z) > 0.01:
        waypoints.append((float(start_tcp[0]), float(start_tcp[1]), safe_z))
    if (
        not waypoints
        or abs(float(target[0]) - waypoints[-1][0]) > 0.01
        or abs(float(target[1]) - waypoints[-1][1]) > 0.01
    ):
        waypoints.append((float(target[0]), float(target[1]), safe_z))
    waypoints.append((float(target[0]), float(target[1]), float(target[2])))

    stage_paths: list[list[list[float]]] = []
    stage_reasons: list[str] = []
    current_q = list(start)
    final_goal = list(start)
    final_err = float("inf")
    final_cost = 0.0

    for waypoint in waypoints:
        result = plan_joint_motion(
            waypoint,
            current_q,
            seed_candidates_rad=_planner_seed_bank(current_q, waypoint),
            tolerance_m=cfg.tolerance_m,
        )
        stage_reasons.append(result.reason)
        if not result.path_rad:
            return PlannerResult(
                ok=False,
                reason=f"stage_failed:{result.reason}",
                target_xyz_m=tuple(float(v) for v in target),
                start_rad=start,
                goal_rad=list(current_q),
                path_rad=stitch_joint_paths(stage_paths),
                position_error_m=result.position_error_m,
                cost=float("inf"),
                planner_mode=cfg.planner_mode,
                waypoint_targets_xyz_m=list(waypoints),
                stage_reasons=stage_reasons,
            )
        stage_paths.append(result.path_rad)
        current_q = list(result.goal_rad)
        final_goal = list(result.goal_rad)
        final_err = float(result.position_error_m)
        final_cost += float(result.cost)

    stitched = stitch_joint_paths(stage_paths)
    final_cost += joint_path_cost(stitched)
    return PlannerResult(
        ok=final_err <= cfg.tolerance_m,
        reason="ok" if final_err <= cfg.tolerance_m else "ik_tolerance_not_met",
        target_xyz_m=tuple(float(v) for v in target),
        start_rad=start,
        goal_rad=final_goal,
        path_rad=stitched,
        position_error_m=final_err,
        cost=final_cost,
        planner_mode=cfg.planner_mode,
        waypoint_targets_xyz_m=list(waypoints),
        stage_reasons=stage_reasons,
    )


def planner_result_to_dict(result: PlannerResult) -> dict[str, object]:
    return asdict(result)


def sample_workspace_target(
    rng: np.random.Generator,
    config: WorkspaceSamplerConfig | None = None,
) -> tuple[float, float, float]:
    cfg = config or WorkspaceSamplerConfig()
    return (
        float(rng.uniform(*cfg.x_range_m)),
        float(rng.uniform(*cfg.y_range_m)),
        float(rng.uniform(*cfg.z_range_m)),
    )


def sample_reachable_target(
    rng: np.random.Generator,
    start_rad: Sequence[float],
    *,
    config: WorkspaceSamplerConfig | None = None,
) -> ReachSample:
    """Randomly sample workspace points until the planner finds a valid path."""
    cfg = config or WorkspaceSamplerConfig()
    for attempt in range(1, cfg.max_tries + 1):
        xyz = sample_workspace_target(rng, cfg)
        plan = plan_tcp_motion(
            xyz,
            start_rad,
            config=MotionPlanConfig(tolerance_m=cfg.tolerance_m),
        )
        if plan.ok:
            return ReachSample(xyz_m=xyz, plan=plan, attempts=attempt)
    raise RuntimeError(
        "Failed to sample a reachable target within the configured workspace "
        f"after {cfg.max_tries} attempts"
    )
