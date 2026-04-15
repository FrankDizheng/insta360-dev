"""Safety wrapper for NERO robot arm motion.

Prevents table collisions by enforcing minimum Z heights and using
a lift-transit-lower movement pattern instead of direct move_p,
whose intermediate trajectory is unpredictable.
"""

import time

import numpy as np

from nero.types import FATAL_ARM_STATUS_CODES, arm_status_label

_TAG = "[safe_motion]"


def get_current_pose(robot) -> np.ndarray:
    """Return current flange pose as a 6-element numpy array [x,y,z,roll,pitch,yaw]."""
    fp = robot.get_flange_pose()
    if fp is None or fp.msg is None:
        raise RuntimeError(f"{_TAG} Flange pose unavailable")
    return np.array(fp.msg[:6], dtype=np.float64)


def check_target_safe(target_pose, z_min_m: float = 0.05) -> None:
    """Raise RuntimeError if target Z is below the safety floor."""
    z = float(target_pose[2])
    if z < z_min_m:
        raise RuntimeError(
            f"{_TAG} Target Z={z:.4f} m is below minimum {z_min_m:.4f} m — "
            f"aborting to prevent table collision"
        )


def check_arm_error(robot) -> None:
    """Raise RuntimeError if controller reports fatal arm_status (IK, collision, etc.)."""
    try:
        st = robot.get_arm_status()
    except Exception:
        return
    if st is None or st.msg is None:
        return
    code = int(getattr(st.msg, "arm_status", 0))
    if code in FATAL_ARM_STATUS_CODES:
        raise RuntimeError(
            f"{_TAG} arm_status={code} ({arm_status_label(code)}) — aborting motion"
        )


def wait_move_done(
    robot,
    target_flange_xyz,
    tol_mm: float = 1.0,
    timeout_s: float = 20.0,
) -> None:
    """Poll flange position until within tolerance of target, or timeout.

    Fails fast on fatal arm_status (no IK solution, collision, over-limit, etc.)
    instead of waiting for a full position timeout.
    """
    time.sleep(0.1)
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
        f"{_TAG} Motion timeout after {timeout_s:.1f}s — "
        f"target={target.tolist()}, last error={err_mm:.1f} mm"
    )


def safe_lift(robot, height_m: float = 0.30) -> np.ndarray:
    """Lift the arm straight up to the given Z height.

    Uses move_l for small lifts (<50 mm) to stay predictable,
    move_p for larger lifts where linear interpolation is unnecessary.
    Returns the final flange pose.
    """
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

    robot.set_speed_percent(30)
    time.sleep(0.05)

    if delta_mm < 50.0:
        robot.move_l(lift_pose.tolist())
    else:
        robot.move_p(lift_pose.tolist())

    wait_move_done(robot, lift_pose[:3])
    return get_current_pose(robot)


def safe_move_to(
    robot,
    target_pose,
    z_safe_m: float = 0.30,
    z_min_m: float = 0.05,
    speed_pct: int = 10,
    tol_mm: float = 1.0,
) -> np.ndarray:
    """Three-phase safe move: lift -> transit horizontally -> lower.

    tol_mm: position tolerance for completion check. Raise this (e.g. 15 mm)
    for standoff/approach moves where sub-millimetre accuracy is unnecessary.

    Returns the final flange pose.
    """
    target = np.array(target_pose[:6], dtype=np.float64)
    check_target_safe(target, z_min_m=z_min_m)

    current = get_current_pose(robot)
    print(
        f"{_TAG} safe_move_to: "
        f"[{current[0]:.4f}, {current[1]:.4f}, {current[2]:.4f}] -> "
        f"[{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]"
    )

    # Phase 1: lift to safe height
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
            robot.move_p(lift_pose.tolist())
        wait_move_done(robot, lift_pose[:3])
    else:
        print(f"{_TAG} Phase 1: already at Z={current[2]:.4f} m >= {z_safe_m:.4f} m")

    # Phase 2: horizontal transit to target XY + orientation at safe Z
    transit_pose = target.copy()
    transit_pose[2] = max(z_safe_m, target[2])
    check_target_safe(transit_pose, z_min_m=z_min_m)

    print(
        f"{_TAG} Phase 2: transit to "
        f"XY=[{transit_pose[0]:.4f}, {transit_pose[1]:.4f}] at Z={transit_pose[2]:.4f} m"
    )
    robot.set_speed_percent(speed_pct)
    time.sleep(0.05)
    robot.move_p(transit_pose.tolist())
    wait_move_done(robot, transit_pose[:3], tol_mm=tol_mm)

    # Phase 3: lower to target Z with linear move
    if abs(transit_pose[2] - target[2]) > 0.001:
        print(f"{_TAG} Phase 3: lower to Z={target[2]:.4f} m (move_l)")
        robot.set_speed_percent(speed_pct)
        time.sleep(0.05)
        robot.move_l(target.tolist())
        wait_move_done(robot, target[:3], tol_mm=tol_mm)
    else:
        print(f"{_TAG} Phase 3: already at target Z, skipping")

    final = get_current_pose(robot)
    print(
        f"{_TAG} Arrived at "
        f"[{final[0]:.4f}, {final[1]:.4f}, {final[2]:.4f}]"
    )
    return final


def safe_move_j_then_to(
    robot,
    target_pose,
    z_safe_m: float = 0.30,
    z_min_m: float = 0.05,
    speed_pct: int = 10,
) -> np.ndarray:
    """For large reconfigurations: move joints to zero, then safe_move_to.

    Returns the final flange pose.
    """
    target = np.array(target_pose[:6], dtype=np.float64)
    check_target_safe(target, z_min_m=z_min_m)

    print(f"{_TAG} Phase 0: lifting before joint-space reconfiguration")
    safe_lift(robot, height_m=z_safe_m)

    print(f"{_TAG} Phase 0: move_j to zero joint configuration")
    zero_joints = [0.0] * 7
    robot.set_speed_percent(speed_pct)
    time.sleep(0.05)
    robot.move_j(zero_joints)

    # Wait for joint move by polling until joints are near zero
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        check_arm_error(robot)
        ja = robot.get_joint_angles()
        if ja is not None and ja.msg is not None:
            angles = np.array(ja.msg[:7], dtype=np.float64)
            if np.max(np.abs(angles)) < 0.02:  # ~1 degree
                time.sleep(0.5)
                break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"{_TAG} Joint move to zero timed out")

    print(f"{_TAG} Joint reconfiguration done, proceeding with safe_move_to")
    return safe_move_to(
        robot, target, z_safe_m=z_safe_m, z_min_m=z_min_m, speed_pct=speed_pct,
    )
