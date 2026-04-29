"""Lightweight motion planning helpers for the NERO 7-DOF arm.

This module intentionally stays dependency-light: it uses the local FK model
from ``nero.kinematics`` plus damped least-squares IK and joint-space
interpolation.  It is designed for early simulation / planning work where we
need a repeatable baseline before introducing a full physics engine or a
learned policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
class PlannerResult:
    ok: bool
    reason: str
    target_xyz_m: tuple[float, float, float]
    start_rad: list[float]
    goal_rad: list[float]
    path_rad: list[list[float]]
    position_error_m: float
    cost: float


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
    seeds = [current, home, mirrored, lifted]
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
    """Plan a simple joint path to reach a Cartesian XYZ goal."""
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
        plan = plan_joint_motion(xyz, start_rad, tolerance_m=cfg.tolerance_m)
        if plan.ok:
            return ReachSample(xyz_m=xyz, plan=plan, attempts=attempt)
    raise RuntimeError(
        "Failed to sample a reachable target within the configured workspace "
        f"after {cfg.max_tries} attempts"
    )
