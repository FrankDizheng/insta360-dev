"""Approximate geometry envelope for the official NERO arm + stock gripper.

This module is intentionally pragmatic: until the official 3D tarball is
available locally, it provides a dimension-based geometry skeleton that is
closer to the real hardware than the previous line-only FK preview.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from nero.kinematics import DEFAULT_TOOL_TCP_OFFSET_X_M, forward_kinematics


@dataclass(frozen=True)
class GripperEnvelope:
    flange_to_tcp_x_m: float = DEFAULT_TOOL_TCP_OFFSET_X_M
    body_length_x_m: float = 0.059
    body_width_y_m: float = 0.057
    body_height_z_m: float = 0.056
    jaw_length_x_m: float = 0.073
    jaw_inner_gap_y_m: float = 0.070
    jaw_outer_width_y_m: float = 0.145
    jaw_height_z_m: float = 0.020


@dataclass(frozen=True)
class NeroGeometryEnvelope:
    link_radii_m: tuple[float, ...] = (0.030, 0.028, 0.028, 0.026, 0.024, 0.022, 0.020)
    gripper: GripperEnvelope = GripperEnvelope()
    source_note: str = (
        "Approximated from official NERO user manual appendix dimension drawings "
        "and a 132 mm flange-to-tool TCP assumption for the stock gripper."
    )


OFFICIAL_NERO_GEOMETRY = NeroGeometryEnvelope()


@dataclass(frozen=True)
class StepFrameAlignment:
    table_z_m: float = -0.04
    base_to_table_offset_m: float = 0.04
    step_primary_axis: str = "z"
    step_width_axis: str = "x"
    step_height_axis: str = "y"
    sim_tool_axis_local: str = "x"
    sim_width_axis_local: str = "y"
    sim_height_axis_local: str = "z"
    source_note: str = (
        "Table height and STEP axis mapping aligned from official gripper STEP "
        "bbox/tool-end slab analysis."
    )


OFFICIAL_FRAME_ALIGNMENT = StepFrameAlignment()


@dataclass(frozen=True)
class WorkspaceSafetyEnvelope:
    """Site-specific static workspace limits in the robot base frame."""

    # The wall on the robot side reached by decreasing J2 is about 41 cm from
    # the zero/base line. In the current URDF/base convention, decreasing J2 at
    # zero pose moves the arm toward +X, so all geometry should stay at
    # x <= wall_x_max_m. If a site uses the opposite base sign, set wall_x_min_m
    # instead and leave wall_x_max_m as None.
    wall_x_max_m: float | None = 0.41
    wall_x_min_m: float | None = None
    wall_soft_margin_m: float = 0.03
    source_note: str = "Site observation: J2-decrease side wall is ~0.41 m from robot zero."


ACTIVE_WORKSPACE_SAFETY = WorkspaceSafetyEnvelope()


@dataclass(frozen=True)
class CapsulePrimitive:
    start_xyz_m: tuple[float, float, float]
    end_xyz_m: tuple[float, float, float]
    radius_m: float
    name: str


@dataclass(frozen=True)
class BoxPrimitive:
    transform: np.ndarray
    size_xyz_m: tuple[float, float, float]
    name: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_step_aligned_geometry() -> NeroGeometryEnvelope:
    """Load STEP-derived geometry overrides when available."""
    cfg_path = _repo_root() / "assets" / "nero_official_3d" / "step_alignment_gripper.json"
    if not cfg_path.exists():
        return OFFICIAL_NERO_GEOMETRY
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return OFFICIAL_NERO_GEOMETRY

    gripper = OFFICIAL_NERO_GEOMETRY.gripper
    slab = payload.get("tool_end_slab", {})
    bbox = payload.get("bbox", {})
    return NeroGeometryEnvelope(
        link_radii_m=OFFICIAL_NERO_GEOMETRY.link_radii_m,
        gripper=GripperEnvelope(
            flange_to_tcp_x_m=float(payload.get("tcp_offset_m", gripper.flange_to_tcp_x_m)),
            body_length_x_m=float(slab.get("length_m", gripper.body_length_x_m)),
            body_width_y_m=float(slab.get("width_m", gripper.body_width_y_m)),
            body_height_z_m=float(slab.get("height_m", gripper.body_height_z_m)),
            jaw_length_x_m=float(payload.get("jaw_length_m", gripper.jaw_length_x_m)),
            jaw_inner_gap_y_m=float(payload.get("jaw_inner_gap_m", gripper.jaw_inner_gap_y_m)),
            jaw_outer_width_y_m=float(slab.get("width_m", gripper.jaw_outer_width_y_m)),
            jaw_height_z_m=float(slab.get("height_m", gripper.jaw_height_z_m)),
        ),
        source_note=str(payload.get("source_note", OFFICIAL_NERO_GEOMETRY.source_note))
        + f" | bbox_m={bbox}" if bbox else str(payload.get("source_note", OFFICIAL_NERO_GEOMETRY.source_note)),
    )


ACTIVE_NERO_GEOMETRY = load_step_aligned_geometry()


def load_step_frame_alignment() -> StepFrameAlignment:
    cfg_path = _repo_root() / "assets" / "nero_official_3d" / "step_alignment_gripper.json"
    if not cfg_path.exists():
        return OFFICIAL_FRAME_ALIGNMENT
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return OFFICIAL_FRAME_ALIGNMENT

    bbox = payload.get("bbox", {})
    slab = payload.get("tool_end_slab", {})
    table_z = float(payload.get("table_z_m", bbox.get("zmin", OFFICIAL_FRAME_ALIGNMENT.table_z_m)))
    base_to_table = float(payload.get("base_to_table_offset_m", abs(table_z)))
    return StepFrameAlignment(
        table_z_m=table_z,
        base_to_table_offset_m=base_to_table,
        step_primary_axis=str(payload.get("primary_axis", OFFICIAL_FRAME_ALIGNMENT.step_primary_axis)),
        step_width_axis=str(slab.get("width_axis", OFFICIAL_FRAME_ALIGNMENT.step_width_axis)),
        step_height_axis=str(slab.get("height_axis", OFFICIAL_FRAME_ALIGNMENT.step_height_axis)),
        sim_tool_axis_local=OFFICIAL_FRAME_ALIGNMENT.sim_tool_axis_local,
        sim_width_axis_local=OFFICIAL_FRAME_ALIGNMENT.sim_width_axis_local,
        sim_height_axis_local=OFFICIAL_FRAME_ALIGNMENT.sim_height_axis_local,
        source_note=str(payload.get("source_note", OFFICIAL_FRAME_ALIGNMENT.source_note)),
    )


ACTIVE_FRAME_ALIGNMENT = load_step_frame_alignment()


def _transform_points(transform: np.ndarray, points_local: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points_local, np.ones((points_local.shape[0], 1), dtype=np.float64)], axis=1)
    world = (transform @ pts_h.T).T
    return world[:, :3]


def _distance_point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def _distance_segment_to_segment(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    # Based on the standard closest-point solution between two line segments.
    u = a1 - a0
    v = b1 - b0
    w0 = a0 - b0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w0))
    e = float(np.dot(v, w0))
    denom = a * c - b * b
    sc, s_num, s_den = 0.0, denom, denom
    tc, t_num, t_den = 0.0, denom, denom

    if denom < 1e-12:
        s_num = 0.0
        s_den = 1.0
        t_num = e
        t_den = c
    else:
        s_num = b * e - c * d
        t_num = a * e - b * d
        if s_num < 0.0:
            s_num = 0.0
            t_num = e
            t_den = c
        elif s_num > s_den:
            s_num = s_den
            t_num = e + b
            t_den = c

    if t_num < 0.0:
        t_num = 0.0
        if -d < 0.0:
            s_num = 0.0
        elif -d > a:
            s_num = s_den
        else:
            s_num = -d
            s_den = a
    elif t_num > t_den:
        t_num = t_den
        if (-d + b) < 0.0:
            s_num = 0.0
        elif (-d + b) > a:
            s_num = s_den
        else:
            s_num = -d + b
            s_den = a

    sc = 0.0 if abs(s_num) < 1e-12 else s_num / s_den
    tc = 0.0 if abs(t_num) < 1e-12 else t_num / t_den
    dp = w0 + sc * u - tc * v
    return float(np.linalg.norm(dp))


def make_box_points(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    return np.asarray(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        dtype=np.float64,
    )


def link_capsule_primitives(
    joint_angles_rad: list[float] | tuple[float, ...] | np.ndarray,
    *,
    geometry: NeroGeometryEnvelope = ACTIVE_NERO_GEOMETRY,
) -> list[CapsulePrimitive]:
    """Return capsule approximations for each arm link."""
    fk = forward_kinematics(joint_angles_rad)
    pts = fk["link_positions"]
    capsules: list[CapsulePrimitive] = []
    for idx in range(min(len(geometry.link_radii_m), len(pts) - 1)):
        start = tuple(float(v) for v in pts[idx])
        end = tuple(float(v) for v in pts[idx + 1])
        if np.linalg.norm(np.asarray(end) - np.asarray(start)) < 1e-9:
            continue
        capsules.append(
            CapsulePrimitive(
                start_xyz_m=start,
                end_xyz_m=end,
                radius_m=float(geometry.link_radii_m[idx]),
                name=f"link_{idx + 1}",
            )
        )
    return capsules


def gripper_box_primitives(
    flange_t: np.ndarray,
    *,
    geometry: NeroGeometryEnvelope = ACTIVE_NERO_GEOMETRY,
) -> list[BoxPrimitive]:
    """Return oriented box approximations for the stock gripper body and jaws."""
    g = geometry.gripper
    jaw_thickness = max((g.jaw_outer_width_y_m - g.jaw_inner_gap_y_m) * 0.5, 0.005)

    def _box_transform(center_xyz: tuple[float, float, float]) -> np.ndarray:
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = flange_t[:3, :3]
        t[:3, 3] = flange_t[:3, :3] @ np.asarray(center_xyz, dtype=np.float64) + flange_t[:3, 3]
        return t

    body_center = (g.body_length_x_m * 0.5, 0.0, 0.0)
    jaw_x_center = g.body_length_x_m + g.jaw_length_x_m * 0.5
    jaw_y_center = g.jaw_inner_gap_y_m * 0.5 + jaw_thickness * 0.5
    return [
        BoxPrimitive(
            transform=_box_transform(body_center),
            size_xyz_m=(g.body_length_x_m, g.body_width_y_m, g.body_height_z_m),
            name="gripper_body",
        ),
        BoxPrimitive(
            transform=_box_transform((jaw_x_center, jaw_y_center, 0.0)),
            size_xyz_m=(g.jaw_length_x_m, jaw_thickness, g.jaw_height_z_m),
            name="gripper_jaw_upper",
        ),
        BoxPrimitive(
            transform=_box_transform((jaw_x_center, -jaw_y_center, 0.0)),
            size_xyz_m=(g.jaw_length_x_m, jaw_thickness, g.jaw_height_z_m),
            name="gripper_jaw_lower",
        ),
    ]


def box_corners_world(box: BoxPrimitive) -> np.ndarray:
    sx, sy, sz = box.size_xyz_m
    local = make_box_points(
        x_min=-sx * 0.5,
        x_max=sx * 0.5,
        y_min=-sy * 0.5,
        y_max=sy * 0.5,
        z_min=-sz * 0.5,
        z_max=sz * 0.5,
    )
    return _transform_points(box.transform, local)


def envelope_penalty(
    joint_angles_rad: list[float] | tuple[float, ...] | np.ndarray,
    *,
    table_z_m: float | None = None,
    geometry: NeroGeometryEnvelope = ACTIVE_NERO_GEOMETRY,
    workspace: WorkspaceSafetyEnvelope = ACTIVE_WORKSPACE_SAFETY,
) -> tuple[float, dict[str, float]]:
    """Compute a soft safety penalty from table, wall, and self-nearness."""
    effective_table_z = ACTIVE_FRAME_ALIGNMENT.table_z_m if table_z_m is None else float(table_z_m)
    fk = forward_kinematics(joint_angles_rad)
    capsules = link_capsule_primitives(joint_angles_rad, geometry=geometry)
    boxes = gripper_box_primitives(fk["flange_T"], geometry=geometry)

    table_penalty = 0.0
    min_clearance = float("inf")
    for cap in capsules:
        clearance = min(cap.start_xyz_m[2], cap.end_xyz_m[2]) - effective_table_z - cap.radius_m
        min_clearance = min(min_clearance, clearance)
        if clearance < 0.0:
            table_penalty += abs(clearance) * 80.0
    for box in boxes:
        corners = box_corners_world(box)
        clearance = float(np.min(corners[:, 2]) - effective_table_z)
        min_clearance = min(min_clearance, clearance)
        if clearance < 0.0:
            table_penalty += abs(clearance) * 120.0

    wall_penalty = 0.0
    min_wall_clearance = float("inf")
    soft_margin = float(workspace.wall_soft_margin_m)
    wall_x_max = None if workspace.wall_x_max_m is None else float(workspace.wall_x_max_m)
    wall_x_min = None if workspace.wall_x_min_m is None else float(workspace.wall_x_min_m)

    if wall_x_max is not None:
        for cap in capsules:
            clearance = wall_x_max - max(cap.start_xyz_m[0], cap.end_xyz_m[0]) - cap.radius_m
            min_wall_clearance = min(min_wall_clearance, clearance)
            if clearance < 0.0:
                wall_penalty += abs(clearance) * 100.0
            elif clearance < soft_margin:
                wall_penalty += (soft_margin - clearance) * 10.0
        for box in boxes:
            corners = box_corners_world(box)
            clearance = float(wall_x_max - np.max(corners[:, 0]))
            min_wall_clearance = min(min_wall_clearance, clearance)
            if clearance < 0.0:
                wall_penalty += abs(clearance) * 140.0
            elif clearance < soft_margin:
                wall_penalty += (soft_margin - clearance) * 14.0

    if wall_x_min is not None:
        for cap in capsules:
            clearance = min(cap.start_xyz_m[0], cap.end_xyz_m[0]) - wall_x_min - cap.radius_m
            min_wall_clearance = min(min_wall_clearance, clearance)
            if clearance < 0.0:
                wall_penalty += abs(clearance) * 100.0
            elif clearance < soft_margin:
                wall_penalty += (soft_margin - clearance) * 10.0
        for box in boxes:
            corners = box_corners_world(box)
            clearance = float(np.min(corners[:, 0]) - wall_x_min)
            min_wall_clearance = min(min_wall_clearance, clearance)
            if clearance < 0.0:
                wall_penalty += abs(clearance) * 140.0
            elif clearance < soft_margin:
                wall_penalty += (soft_margin - clearance) * 14.0

    self_penalty = 0.0
    min_self_gap = float("inf")
    for idx, cap_a in enumerate(capsules):
        a0 = np.asarray(cap_a.start_xyz_m, dtype=np.float64)
        a1 = np.asarray(cap_a.end_xyz_m, dtype=np.float64)
        for jdx in range(idx + 2, len(capsules)):
            # Skip adjacent links, but check the rest.
            cap_b = capsules[jdx]
            b0 = np.asarray(cap_b.start_xyz_m, dtype=np.float64)
            b1 = np.asarray(cap_b.end_xyz_m, dtype=np.float64)
            gap = _distance_segment_to_segment(a0, a1, b0, b1) - (cap_a.radius_m + cap_b.radius_m)
            min_self_gap = min(min_self_gap, gap)
            if gap < 0.02:
                self_penalty += max(0.0, 0.02 - gap) * 12.0

    return (
        table_penalty + wall_penalty + self_penalty,
        {
            "table_penalty": table_penalty,
            "wall_penalty": wall_penalty,
            "self_penalty": self_penalty,
            "table_z_m": effective_table_z,
            "wall_x_max_m": float("nan") if wall_x_max is None else wall_x_max,
            "wall_x_min_m": float("nan") if wall_x_min is None else wall_x_min,
            "min_table_clearance_m": min_clearance,
            "min_wall_clearance_m": min_wall_clearance,
            "min_self_gap_m": min_self_gap,
        },
    )


def gripper_wireframe_segments(
    flange_t: np.ndarray,
    *,
    geometry: NeroGeometryEnvelope = ACTIVE_NERO_GEOMETRY,
) -> list[np.ndarray]:
    """Return polyline segments for an approximate stock gripper wireframe."""
    g = geometry.gripper
    jaw_thickness = max((g.jaw_outer_width_y_m - g.jaw_inner_gap_y_m) * 0.5, 0.005)

    body = make_box_points(
        x_min=0.0,
        x_max=g.body_length_x_m,
        y_min=-g.body_width_y_m * 0.5,
        y_max=g.body_width_y_m * 0.5,
        z_min=-g.body_height_z_m * 0.5,
        z_max=g.body_height_z_m * 0.5,
    )
    jaw_upper = make_box_points(
        x_min=g.body_length_x_m,
        x_max=g.body_length_x_m + g.jaw_length_x_m,
        y_min=g.jaw_inner_gap_y_m * 0.5,
        y_max=g.jaw_inner_gap_y_m * 0.5 + jaw_thickness,
        z_min=-g.jaw_height_z_m * 0.5,
        z_max=g.jaw_height_z_m * 0.5,
    )
    jaw_lower = make_box_points(
        x_min=g.body_length_x_m,
        x_max=g.body_length_x_m + g.jaw_length_x_m,
        y_min=-(g.jaw_inner_gap_y_m * 0.5 + jaw_thickness),
        y_max=-g.jaw_inner_gap_y_m * 0.5,
        z_min=-g.jaw_height_z_m * 0.5,
        z_max=g.jaw_height_z_m * 0.5,
    )

    def _box_edges(box: np.ndarray) -> list[np.ndarray]:
        idx_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        world = _transform_points(flange_t, box)
        return [world[[i0, i1]] for i0, i1 in idx_pairs]

    segments = []
    segments.extend(_box_edges(body))
    segments.extend(_box_edges(jaw_upper))
    segments.extend(_box_edges(jaw_lower))
    return segments
