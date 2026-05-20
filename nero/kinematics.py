"""Local forward kinematics for NERO (7-DOF).

The arm chain follows AgileX's official NERO URDF rather than the older
manual-DH approximation.  The returned ``flange_T`` is intentionally the SDK
flange frame: a J7 sweep on the real robot showed SDK ``get_flange_pose()``
matches the official gripper joint origin, not ``link7`` or ``gripper_flange``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from nero.types import NUM_JOINTS, clamp_joints_rad

# Official NERO URDF commit used during calibration:
# agilexrobotics/agx_arm_urdf@1ee5659d02b33b9379fd647a2e5647800d67f0f4
# Each revolute joint is represented as parent_T_joint_origin, then a rotation
# about the joint-local axis.  Units are metres/radians.
_OFFICIAL_URDF_JOINTS: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = [
    ((0.0, 0.0, 0.138), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (math.pi / 2.0, math.pi, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, -0.31, 0.0), (-math.pi / 2.0, 0.0, math.pi), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, math.pi), (0.0, 0.0, 1.0)),
    ((0.0, -0.27001, 0.0), (math.pi / 2.0, -math.pi / 2.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (math.pi / 2.0, -math.pi / 2.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, -0.0235, 0.0), (math.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
]

# Official gripper xacro fixed chain:
# link7 -> gripper_flange -> gripper_base -> gripper_joint{1,2}_origin.
# J7-only real-robot sweep matched this point within ~1.7 mm.
_LINK7_T_GRIPPER_FLANGE_XYZ_RPY = ((0.032, 0.0, -0.0235), (-1.5708, 0.0, -1.5708))
_GRIPPER_FLANGE_T_BASE_XYZ_RPY = ((0.0, 0.0, 0.0055), (0.0, 0.0, 0.0))
_GRIPPER_BASE_T_JOINT_ORIGIN_XYZ_RPY = ((0.0, 0.0, 0.1358), (math.pi / 2.0, 0.0, math.pi))

# Keep the public constant for callers that import it.  The SDK flange is now
# already at the calibrated gripper joint origin, so no extra TCP offset is
# applied by default.  Higher-level grasp code can still add a task-specific
# local offset if it wants a fingertip/contact point.
DEFAULT_TOOL_TCP_OFFSET_X_M = 0.0


def _tool_tcp_transform(offset_x_m: float = DEFAULT_TOOL_TCP_OFFSET_X_M) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[0, 3] = float(offset_x_m)
    return t


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_transform(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = _rpy_matrix(*rpy)
    t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


def _axis_rotation(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.float64)
    axis_arr = axis_arr / np.linalg.norm(axis_arr)
    x, y, z = axis_arr.tolist()
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    one_c = 1.0 - c
    rot = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rot
    return t


def _link7_to_sdk_flange_transform() -> np.ndarray:
    return (
        _origin_transform(*_LINK7_T_GRIPPER_FLANGE_XYZ_RPY)
        @ _origin_transform(*_GRIPPER_FLANGE_T_BASE_XYZ_RPY)
        @ _origin_transform(*_GRIPPER_BASE_T_JOINT_ORIGIN_XYZ_RPY)
    )


def forward_kinematics(
    joint_angles_rad: Sequence[float],
    *,
    clamp: bool = True,
) -> dict[str, np.ndarray]:
    """Compute link origins plus SDK-flange / TCP transforms in base frame.

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
        ``link7_T`` — 4x4 homogeneous transform base_T_link7.
        ``flange_T`` — 4x4 homogeneous transform matching SDK get_flange_pose().
        ``tcp_T`` — 4x4 homogeneous transform base_T_tcp.
        ``tcp_position`` — shape (3,), tool-tip XYZ in base frame.
    """
    if len(joint_angles_rad) < NUM_JOINTS:
        raise ValueError(f"Need {NUM_JOINTS} joint angles, got {len(joint_angles_rad)}")
    q = list(float(x) for x in joint_angles_rad[:NUM_JOINTS])
    if clamp:
        q = clamp_joints_rad(q)

    base_t_link = np.eye(4, dtype=np.float64)
    positions: list[np.ndarray] = [base_t_link[:3, 3].copy()]

    for qi, (xyz, rpy, axis) in zip(q, _OFFICIAL_URDF_JOINTS, strict=True):
        base_t_link = base_t_link @ _origin_transform(xyz, rpy) @ _axis_rotation(axis, qi)
        positions.append(base_t_link[:3, 3].copy())

    link7_t = base_t_link
    gripper_flange_t = link7_t @ _origin_transform(*_LINK7_T_GRIPPER_FLANGE_XYZ_RPY)
    gripper_base_t = gripper_flange_t @ _origin_transform(*_GRIPPER_FLANGE_T_BASE_XYZ_RPY)
    sdk_flange_t = link7_t @ _link7_to_sdk_flange_transform()
    tcp_t = sdk_flange_t @ _tool_tcp_transform()
    return {
        "link_positions": np.stack(positions, axis=0),
        "link7_T": link7_t,
        "gripper_flange_T": gripper_flange_t,
        "gripper_base_T": gripper_base_t,
        "flange_T": sdk_flange_t,
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
    """Rough max reach from official link translations (upper bound)."""
    total = 0.0
    for xyz, _rpy, _axis in _OFFICIAL_URDF_JOINTS:
        total += float(np.linalg.norm(np.asarray(xyz, dtype=np.float64)))
    total += float(np.linalg.norm(_link7_to_sdk_flange_transform()[:3, 3]))
    total += DEFAULT_TOOL_TCP_OFFSET_X_M
    return float(total)
