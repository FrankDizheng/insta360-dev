"""Local forward kinematics for NERO (7-DOF) — preview / simulation only.

The real flange pose and IK are computed in the arm firmware; this module uses
the Denavit–Hartenberg parameters from the NERO user manual for offline
visualization and rough reach checks.

DH convention (standard Craig): each row is (a, alpha, d, theta_offset_rad).
Joint angle q_i is added to theta_offset for link i.

For the simulation baseline we also attach a simple tool TCP model for the
stock two-finger gripper.  The true geometry can be refined later from the
official 3D model / CAD, but for now we approximate the TCP as the top-front
centre of the standard gripper, 132 mm away from the flange along the local
tool forward axis.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from nero.types import NUM_JOINTS, clamp_joints_rad

# NERO manual: j1..j7 — a (m), alpha (rad), d (m), theta offset (rad)
_NERO_DH_ROWS: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.138, 0.0),
    (0.0, math.pi / 2, 0.0, math.pi),
    (0.0, math.pi / 2, 0.31, math.pi),
    (0.0, math.pi / 2, 0.0, math.pi),
    (0.0, math.pi / 2, 0.27001, math.pi / 2),
    (0.0, math.pi / 2, 0.0, math.pi / 2),
    (0.0, math.pi / 2, 0.0235, 0.0),
]

# Approximate flange -> stock gripper TCP offset used in simulation.
# The offset is modelled along flange +X, which matches the outward direction
# of the arm in the home pose under the current DH chain.
DEFAULT_TOOL_TCP_OFFSET_X_M = 0.132


def _tool_tcp_transform(offset_x_m: float = DEFAULT_TOOL_TCP_OFFSET_X_M) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[0, 3] = float(offset_x_m)
    return t


def _dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def forward_kinematics(
    joint_angles_rad: Sequence[float],
    *,
    clamp: bool = True,
) -> dict[str, np.ndarray]:
    """Compute link origins plus flange / TCP transforms in base frame.

    Parameters
    ----------
    joint_angles_rad
        Length-7 joint vector (radians), J1..J7.
    clamp
        If True, clamp to `JOINT_LIMITS_RAD` before FK.

    Returns
    -------
    dict with keys:
        ``link_positions`` — shape (8, 3): base origin + after each of 7 joints.
        ``flange_T`` — 4x4 homogeneous transform base_T_flange.
        ``tcp_T`` — 4x4 homogeneous transform base_T_tcp (tool-tip approximation).
        ``tcp_position`` — shape (3,), tool-tip XYZ in base frame.
    """
    if len(joint_angles_rad) < NUM_JOINTS:
        raise ValueError(f"Need {NUM_JOINTS} joint angles, got {len(joint_angles_rad)}")
    q = list(float(x) for x in joint_angles_rad[:NUM_JOINTS])
    if clamp:
        q = clamp_joints_rad(q)

    t_acc = np.eye(4, dtype=np.float64)
    positions: list[np.ndarray] = [t_acc[:3, 3].copy()]

    for i, (a, alpha, d, th_off) in enumerate(_NERO_DH_ROWS):
        theta = q[i] + th_off
        a_i = _dh_matrix(theta, d, a, alpha)
        t_acc = t_acc @ a_i
        positions.append(t_acc[:3, 3].copy())

    tcp_t = t_acc @ _tool_tcp_transform()
    return {
        "link_positions": np.stack(positions, axis=0),
        "flange_T": t_acc,
        "tcp_T": tcp_t,
        "tcp_position": tcp_t[:3, 3].copy(),
    }


def flange_position(joint_angles_rad: Sequence[float], *, clamp: bool = True) -> np.ndarray:
    """Flange XYZ in base frame (metres)."""
    fk = forward_kinematics(joint_angles_rad, clamp=clamp)
    return fk["flange_T"][:3, 3].copy()


def tcp_position(joint_angles_rad: Sequence[float], *, clamp: bool = True) -> np.ndarray:
    """Approximate stock-gripper TCP XYZ in base frame (metres)."""
    fk = forward_kinematics(joint_angles_rad, clamp=clamp)
    return fk["tcp_position"].copy()


def approximate_reach_m() -> float:
    """Rough max radial reach from DH link lengths (upper bound, not exact workspace)."""
    # Sum of |d| and |a| along chain as a crude ballpark
    total = 0.0
    for a, _alpha, d, _off in _NERO_DH_ROWS:
        total += abs(a) + abs(d)
    total += DEFAULT_TOOL_TCP_OFFSET_X_M
    return float(total)
