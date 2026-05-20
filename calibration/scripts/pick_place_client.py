"""
pick_place_client.py — Mac-side pick-and-place orchestrator (v5).

Usage:
    python3 pick_place_client.py                           # full auto run
    python3 pick_place_client.py --dry-run                 # scan + detect + plan only
    python3 pick_place_client.py --skip-place              # pick only, no placement
    python3 pick_place_client.py --confirm                 # pause before each motion step
    python3 pick_place_client.py --dry-run --pi-url ...    # override Pi URL
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import time
from collections import deque
from pathlib import Path

import numpy as np
import requests


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid float for {name}: {raw}") from exc


def _parse_float_csv(raw: str, expected_len: int | None = None) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated floats, got: {raw}") from exc
    if expected_len is not None and len(values) != expected_len:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_len} values, got {len(values)}: {raw}"
        )
    return values


def _env_float_csv(name: str, default: list[float], expected_len: int | None = None) -> list[float]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return list(default)
    try:
        return _parse_float_csv(raw, expected_len=expected_len)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"Invalid {name}: {exc}") from exc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PI_URL = os.getenv("PICK_PLACE_PI_URL", "http://10.13.167.212:8765")
VLM_KEY = os.getenv("PICK_PLACE_VLM_KEY", "sk-ce34c80757f4459096adf73d53323c68")
VLM_BASE = os.getenv("PICK_PLACE_VLM_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
VLM_MODEL = os.getenv("PICK_PLACE_VLM_MODEL", "qwen3-vl-flash")

PICK_OBJECT_NAME = "outside_red_cap_bottle"
PLACE_OBJECT_NAME = "empty_slot"
PICK_LOCALIZATION_MODE = os.getenv("PICK_PLACE_PICK_LOCALIZATION_MODE", "red-cap-model")

SCAN_POSE_DEG = _env_float_csv(
    "PICK_PLACE_SCAN_POSE_DEG",
    [7.277, -36.26, -14.662, 71.399, -8.695, 0.318, 97.698],
    expected_len=7,
)

FLANGE_T_CAM = np.array([
    [0.01984594, 0.36872915, -0.92932500, 0.09973553],
    [-0.01531797, -0.92928683, -0.36904113, 0.06435725],
    [-0.99968570, 0.02155934, -0.01279439, -0.01483681],
    [0.0, 0.0, 0.0, 1.0],
])

STANDOFF_MM = 80
PICK_XY_ALIGN_ABOVE_GRASP_MM = _env_float("PICK_PLACE_PICK_XY_ALIGN_ABOVE_GRASP_MM", 30.0)
GRASP_BELOW_MM = 10
LIFT_Z_MM = 420
SUCCESSFUL_GRASP_Z_PRIOR_MM = _env_float("PICK_PLACE_SUCCESSFUL_GRASP_Z_PRIOR_MM", 260.0)
PICK_GRASP_Z_MODE = os.getenv("PICK_PLACE_PICK_GRASP_Z_MODE", "successful-prior")
BOTTLE_HEIGHT_MM = _env_float("PICK_PLACE_BOTTLE_HEIGHT_MM", 95.0)
BOTTLE_GRASP_FROM_CAP_MM = _env_float("PICK_PLACE_BOTTLE_GRASP_FROM_CAP_MM", 45.0)
BOTTLE_MODEL_AXIS_UV = _env_float_csv("PICK_PLACE_BOTTLE_MODEL_AXIS_UV", [0.0, -1.0], expected_len=2)
BOTTLE_MODEL_AXIS_SEARCH_DEG = _env_float("PICK_PLACE_BOTTLE_MODEL_AXIS_SEARCH_DEG", 20.0)
BOTTLE_MODEL_LENGTH_PX = _env_float("PICK_PLACE_BOTTLE_MODEL_LENGTH_PX", 165.0)
BOTTLE_MODEL_WIDTH_PX = _env_float("PICK_PLACE_BOTTLE_MODEL_WIDTH_PX", 70.0)
PLACE_CLEARANCE_MM = _env_float("PICK_PLACE_PLACE_CLEARANCE_MM", -3.0)
GRIPPER_OPEN_M = 0.06
GRIPPER_FORCE = 1.0
GRIP_CENTER_OFFSET_Y_MM = _env_float("PICK_PLACE_GRIP_CENTER_OFFSET_Y_MM", -15.0)
PICK_CONTACT_LOCAL_OFFSET_MM = _env_float_csv(
    "PICK_PLACE_PICK_CONTACT_LOCAL_OFFSET_MM",
    [0.0, 0.0, 0.0],
    expected_len=3,
)
PICK_XY_MODE = os.getenv("PICK_PLACE_PICK_XY_MODE", "fixed-z-plane")
GRASP_TEMPLATE_RPY_RAD = _env_float_csv(
    "PICK_PLACE_GRASP_TEMPLATE_RPY_RAD",
    [-0.90010, -1.46119, 2.48800],
    expected_len=3,
)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def pose_to_T(pose6) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_mat(*pose6[3:])
    T[:3, 3] = pose6[:3]
    return T


def _flange_target_from_contact(
    contact_xyz_m: np.ndarray,
    grasp_rpy_rad: list[float],
    contact_local_offset_mm: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a desired gripper contact point into an SDK flange target.

    The SDK flange frame is calibrated to the official gripper joint origin.
    Vision/refinement produces the object/contact point, so any fingertip or
    contact-point offset must be explicit at this layer:
    contact = flange + R_grasp @ local_offset.
    """
    offset_local_m = np.asarray(contact_local_offset_mm, dtype=np.float64) / 1000.0
    delta_base_m = rpy_to_mat(*grasp_rpy_rad) @ offset_local_m
    flange_xyz_m = np.asarray(contact_xyz_m, dtype=np.float64) - delta_base_m
    return flange_xyz_m, delta_base_m


def _normalize_pick_grasp_z_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("_", "-")
    aliases = {
        "depth": "depth",
        "vision-depth": "depth",
        "successful-prior": "successful-prior",
        "success-prior": "successful-prior",
        "prior": "successful-prior",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported pick grasp Z mode: {mode}. "
            "Use 'depth' or 'successful-prior'."
        )
    return aliases[normalized]


def _normalize_pick_xy_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("_", "-")
    aliases = {
        "depth": "depth",
        "vision-depth": "depth",
        "fixed-z-plane": "fixed-z-plane",
        "support-plane": "fixed-z-plane",
        "plane": "fixed-z-plane",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported pick XY mode: {mode}. "
            "Use 'depth' or 'fixed-z-plane'."
        )
    return aliases[normalized]


def px_to_base(u: int, v: int, d_mm: float, base_T_cam: np.ndarray, intr: dict) -> np.ndarray:
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    d = d_mm / 1000.0
    p_cam = np.array([(u - cx) * d / fx, (v - cy) * d / fy, d, 1.0])
    return (base_T_cam @ p_cam)[:3]


def px_ray_to_base_z(u: int, v: int, z_base_m: float, base_T_cam: np.ndarray, intr: dict) -> np.ndarray:
    """Project an image pixel onto a fixed base-frame Z plane.

    This is used for transparent bottles: RGB gives the pixel/contour, while
    the calibrated support/grasp plane supplies Z so bad depth pixels do not
    contaminate XY.
    """
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    origin = base_T_cam[:3, 3]
    direction = base_T_cam[:3, :3] @ ray_cam
    if abs(float(direction[2])) < 1e-9:
        raise RuntimeError("Pixel ray is parallel to the fixed Z plane")
    t = (float(z_base_m) - float(origin[2])) / float(direction[2])
    return origin + t * direction


def _norm_uv_to_px(uv: object, w: int, h: int) -> tuple[int, int] | None:
    if isinstance(uv, dict):
        raw = (uv.get("u"), uv.get("v"))
    elif isinstance(uv, (list, tuple)) and len(uv) == 2:
        raw = (uv[0], uv[1])
    else:
        return None
    try:
        u = float(raw[0])
        v = float(raw[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    return int(round(u * (w - 1))), int(round(v * (h - 1)))


def _parse_slot_corners(obj: dict, w: int, h: int) -> dict[str, tuple[int, int]] | None:
    corners = obj.get("corners")
    if not isinstance(corners, dict):
        return None

    aliases = {
        "top_left": ("top_left", "tl", "upper_left"),
        "top_right": ("top_right", "tr", "upper_right"),
        "bottom_right": ("bottom_right", "br", "lower_right"),
        "bottom_left": ("bottom_left", "bl", "lower_left"),
    }
    parsed: dict[str, tuple[int, int]] = {}
    for canonical, names in aliases.items():
        value = next((corners[name] for name in names if name in corners), None)
        px = _norm_uv_to_px(value, w, h)
        if px is None:
            return None
        parsed[canonical] = px
    return parsed


def _slot_geometry_from_corners(
    corners_px: dict[str, tuple[int, int]],
    depth_img: np.ndarray,
    ds: float,
    base_T_cam: np.ndarray,
    intr: dict,
) -> dict | None:
    ordered = [
        np.array(corners_px["top_left"], dtype=np.float64),
        np.array(corners_px["top_right"], dtype=np.float64),
        np.array(corners_px["bottom_right"], dtype=np.float64),
        np.array(corners_px["bottom_left"], dtype=np.float64),
    ]
    center = np.mean(np.stack(ordered), axis=0)
    center_px = (int(round(center[0])), int(round(center[1])))

    top_axis = ordered[1] - ordered[0]
    bottom_axis = ordered[2] - ordered[3]
    left_axis = ordered[3] - ordered[0]
    right_axis = ordered[2] - ordered[1]
    horizontal = (top_axis + bottom_axis) * 0.5
    vertical = (left_axis + right_axis) * 0.5
    long_axis = horizontal if np.linalg.norm(horizontal) >= np.linalg.norm(vertical) else vertical
    axis_norm = float(np.linalg.norm(long_axis))
    if axis_norm < 1.0:
        return None

    depth_mm = _local_valid_depth(depth_img, center_px[0], center_px[1], ds=ds, radius=18)
    if depth_mm is None:
        xs = [int(p[0]) for p in ordered]
        ys = [int(p[1]) for p in ordered]
        x1, x2 = max(0, min(xs)), min(depth_img.shape[1], max(xs) + 1)
        y1, y2 = max(0, min(ys)), min(depth_img.shape[0], max(ys) + 1)
        roi = depth_img[y1:y2, x1:x2].astype(float) * ds
        valid = roi[(roi > 100) & (roi < 2000)]
        depth_mm = float(np.median(valid)) if len(valid) > 0 else None
    if depth_mm is None:
        return None

    center_base = px_to_base(center_px[0], center_px[1], depth_mm, base_T_cam, intr)
    axis_unit_px = long_axis / axis_norm
    half = long_axis * 0.5
    p1_px = center - half
    p2_px = center + half
    p1_base = px_to_base(int(round(p1_px[0])), int(round(p1_px[1])), depth_mm, base_T_cam, intr)
    p2_base = px_to_base(int(round(p2_px[0])), int(round(p2_px[1])), depth_mm, base_T_cam, intr)
    axis_base = p2_base - p1_base
    axis_base_norm = float(np.linalg.norm(axis_base))
    if axis_base_norm > 1e-6:
        axis_base = axis_base / axis_base_norm

    width_px = min(np.linalg.norm(horizontal), np.linalg.norm(vertical))
    length_px = max(np.linalg.norm(horizontal), np.linalg.norm(vertical))
    return {
        "center_px": [center_px[0], center_px[1]],
        "depth_mm": float(depth_mm),
        "center_base_xyz_m": center_base.tolist(),
        "axis_uv": axis_unit_px.tolist(),
        "axis_base": axis_base.tolist(),
        "length_px": float(length_px),
        "width_px": float(width_px),
        "corners_px": {name: [int(px[0]), int(px[1])] for name, px in corners_px.items()},
    }


def _parse_bottle_body_points(obj: dict, w: int, h: int) -> dict[str, tuple[int, int]] | None:
    endpoints = obj.get("endpoints")
    width_points = obj.get("width_points")
    keypoints = obj.get("keypoints")
    if not isinstance(endpoints, dict) and isinstance(keypoints, dict):
        endpoints = {
            "cap_center": keypoints.get("cap_center"),
            "bottom_center": keypoints.get("body_tail_center") or keypoints.get("bottom_center"),
        }
    if not isinstance(width_points, dict) and isinstance(keypoints, dict):
        width_points = {
            "left_mid": keypoints.get("body_left_mid") or keypoints.get("left_mid"),
            "right_mid": keypoints.get("body_right_mid") or keypoints.get("right_mid"),
        }
    if not isinstance(endpoints, dict) or not isinstance(width_points, dict):
        return None

    aliases = {
        "cap_center": ("cap_center", "cap", "neck_center"),
        "bottom_center": ("body_tail_center", "bottom_center", "base_center", "tail_center"),
        "left_mid": ("body_left_mid", "left_mid", "mid_left"),
        "right_mid": ("body_right_mid", "right_mid", "mid_right"),
        "body_center": ("body_center", "center", "grasp_center"),
    }
    sources = {
        "cap_center": endpoints,
        "bottom_center": endpoints,
        "left_mid": width_points,
        "right_mid": width_points,
        "body_center": keypoints if isinstance(keypoints, dict) else obj,
    }

    parsed: dict[str, tuple[int, int]] = {}
    for canonical, names in aliases.items():
        source = sources[canonical]
        value = next((source[name] for name in names if name in source), None)
        px = _norm_uv_to_px(value, w, h)
        if px is None:
            return None
        parsed[canonical] = px
    return parsed


def _bottle_geometry_from_body_points(
    points_px: dict[str, tuple[int, int]],
    depth_img: np.ndarray,
    ds: float,
    base_T_cam: np.ndarray,
    intr: dict,
) -> dict | None:
    cap = np.array(points_px["cap_center"], dtype=np.float64)
    bottom = np.array(points_px["bottom_center"], dtype=np.float64)
    left_mid = np.array(points_px["left_mid"], dtype=np.float64)
    right_mid = np.array(points_px["right_mid"], dtype=np.float64)
    body_center = np.array(points_px["body_center"], dtype=np.float64)

    axis = bottom - cap
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 60.0:
        return None

    body_mid = (cap + bottom) * 0.5
    width_mid = (left_mid + right_mid) * 0.5
    width_px = float(np.linalg.norm(right_mid - left_mid))
    if width_px < 15.0:
        return None

    axis_unit = axis / axis_norm
    center_axis_t = float(np.dot(body_center - cap, axis_unit) / axis_norm)
    # Reject VLM outputs where the "body center" is still on the cap/neck area.
    if not (0.25 <= center_axis_t <= 0.75):
        return None

    center = (body_center + width_mid + body_mid) / 3.0
    center_px = (int(round(center[0])), int(round(center[1])))

    depth_candidates = []
    for p in (center, body_center, width_mid, body_mid, left_mid, right_mid, cap, bottom):
        d = _local_valid_depth(depth_img, int(round(p[0])), int(round(p[1])), ds=ds, radius=14)
        if d is not None:
            depth_candidates.append(d)
    if not depth_candidates:
        return None
    depth_mm = float(np.median(depth_candidates))

    center_base = px_to_base(center_px[0], center_px[1], depth_mm, base_T_cam, intr)
    cap_base = px_to_base(int(round(cap[0])), int(round(cap[1])), depth_mm, base_T_cam, intr)
    bottom_base = px_to_base(int(round(bottom[0])), int(round(bottom[1])), depth_mm, base_T_cam, intr)
    axis_base = bottom_base - cap_base
    axis_base_norm = float(np.linalg.norm(axis_base))
    if axis_base_norm > 1e-6:
        axis_base = axis_base / axis_base_norm

    return {
        "center_px": [center_px[0], center_px[1]],
        "depth_mm": depth_mm,
        "center_base_xyz_m": center_base.tolist(),
        "axis_uv": (axis / axis_norm).tolist(),
        "axis_base": axis_base.tolist(),
        "length_px": axis_norm,
        "width_px": width_px,
        "body_center_axis_fraction": center_axis_t,
        "points_px": {name: [int(px[0]), int(px[1])] for name, px in points_px.items()},
    }


def _unit_axis_uv(raw: object) -> np.ndarray | None:
    try:
        arr = np.array(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (2,):
        return None
    norm = float(np.linalg.norm(arr))
    if norm < 1e-6:
        return None
    return arr / norm


def _bottle_axis_prior_uv(
    points_px: dict[str, tuple[int, int]],
    default_axis_uv: list[float],
) -> tuple[np.ndarray, str]:
    cap = np.array(points_px["cap_center"], dtype=np.float64)
    candidates = [
        ("vlm_bottom_center", points_px.get("bottom_center"), 25.0),
        ("vlm_body_center", points_px.get("body_center"), 12.0),
    ]
    if points_px.get("left_mid") is not None and points_px.get("right_mid") is not None:
        width_mid = (
            np.array(points_px["left_mid"], dtype=np.float64)
            + np.array(points_px["right_mid"], dtype=np.float64)
        ) * 0.5
        candidates.append(("vlm_width_mid", (float(width_mid[0]), float(width_mid[1])), 12.0))

    for source, px, min_dist in candidates:
        if px is None:
            continue
        axis = np.array(px, dtype=np.float64) - cap
        norm = float(np.linalg.norm(axis))
        if norm >= min_dist:
            return axis / norm, source

    axis = _unit_axis_uv(default_axis_uv)
    if axis is None:
        axis = np.array([0.0, -1.0], dtype=np.float64)
    return axis, "configured_default"


def _axis_score_from_image(
    color_img: np.ndarray,
    anchor_px: np.ndarray,
    axis_uv: np.ndarray,
    model_length_px: float,
    model_width_px: float,
) -> float:
    h, w = color_img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rel_x = xx.astype(np.float64) - float(anchor_px[0])
    rel_y = yy.astype(np.float64) - float(anchor_px[1])
    perp = np.array([-axis_uv[1], axis_uv[0]], dtype=np.float64)
    along = rel_x * axis_uv[0] + rel_y * axis_uv[1]
    across = rel_x * perp[0] + rel_y * perp[1]
    corridor = (
        (along > model_width_px * 0.25)
        & (along < model_length_px)
        & (np.abs(across) < model_width_px * 0.5)
    )
    if not np.any(corridor):
        return 0.0

    img = color_img.astype(np.float64)
    gray = img.mean(axis=2)
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    rgb_range = img.max(axis=2) - img.min(axis=2)
    highlight = (gray > 95.0) & (rgb_range < 85.0)
    return float(np.mean(edge[corridor]) + 0.25 * np.mean(highlight[corridor]))


def _refine_bottle_axis_uv(
    color_img: np.ndarray,
    anchor_px: np.ndarray,
    prior_axis_uv: np.ndarray,
    axis_search_deg: float,
    model_length_px: float,
    model_width_px: float,
) -> np.ndarray:
    base_angle = float(math.atan2(prior_axis_uv[1], prior_axis_uv[0]))
    best_axis = prior_axis_uv
    best_score = -float("inf")
    for delta_deg in np.linspace(-axis_search_deg, axis_search_deg, 17):
        theta = base_angle + math.radians(float(delta_deg))
        axis = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        score = _axis_score_from_image(color_img, anchor_px, axis, model_length_px, model_width_px)
        # Keep image evidence as a small correction around the prior, not a free search.
        score -= abs(float(delta_deg)) * 0.02
        if score > best_score:
            best_score = score
            best_axis = axis
    return best_axis / max(float(np.linalg.norm(best_axis)), 1e-6)


def _image_model_box(
    center_px: np.ndarray,
    axis_uv: np.ndarray,
    length_px: float,
    width_px: float,
    image_w: int,
    image_h: int,
) -> list[list[int]]:
    perp = np.array([-axis_uv[1], axis_uv[0]], dtype=np.float64)
    corners = [
        center_px - axis_uv * length_px * 0.5 - perp * width_px * 0.5,
        center_px + axis_uv * length_px * 0.5 - perp * width_px * 0.5,
        center_px + axis_uv * length_px * 0.5 + perp * width_px * 0.5,
        center_px - axis_uv * length_px * 0.5 + perp * width_px * 0.5,
    ]
    return [
        [
            int(np.clip(round(float(p[0])), 0, image_w - 1)),
            int(np.clip(round(float(p[1])), 0, image_h - 1)),
        ]
        for p in corners
    ]


def _bottle_geometry_from_cap_model(
    color_img: np.ndarray,
    depth_img: np.ndarray,
    points_px: dict[str, tuple[int, int]],
    ds: float,
    base_T_cam: np.ndarray,
    intr: dict,
    grasp_from_cap_mm: float,
    default_axis_uv: list[float],
    axis_search_deg: float,
    model_length_px: float,
    model_width_px: float,
) -> dict | None:
    """Use the reliable red cap as anchor and VLM body points only for axis."""
    cap_px = points_px["cap_center"]
    cap_ref_x, cap_ref_y, cap_depth_mm = refine_red_cap(
        color_img,
        depth_img,
        cap_px[0],
        cap_px[1],
        ds=ds,
        radius=55,
    )
    if cap_depth_mm is None:
        return None

    axis_uv, axis_source = _bottle_axis_prior_uv(points_px, default_axis_uv)
    axis_uv = _refine_bottle_axis_uv(
        color_img,
        np.array([cap_ref_x, cap_ref_y], dtype=np.float64),
        axis_uv,
        axis_search_deg=axis_search_deg,
        model_length_px=model_length_px,
        model_width_px=model_width_px,
    )

    # Project a second point at the same local depth, then use the camera/base
    # transform to convert the image axis into a metric base-frame direction.
    sample_px = np.array([cap_ref_x, cap_ref_y], dtype=np.float64) + axis_uv * 80.0
    h, w = depth_img.shape[:2]
    sample_u = int(np.clip(round(sample_px[0]), 0, w - 1))
    sample_v = int(np.clip(round(sample_px[1]), 0, h - 1))
    cap_base = px_to_base(cap_ref_x, cap_ref_y, cap_depth_mm, base_T_cam, intr)
    sample_base = px_to_base(sample_u, sample_v, cap_depth_mm, base_T_cam, intr)
    axis_base = sample_base - cap_base
    axis_base_norm = float(np.linalg.norm(axis_base))
    if axis_base_norm < 1e-6:
        return None
    axis_base = axis_base / axis_base_norm

    center_base = cap_base + axis_base * (grasp_from_cap_mm / 1000.0)
    body_anchor_px = np.array([cap_ref_x, cap_ref_y], dtype=np.float64)
    model_center_px = body_anchor_px + axis_uv * (model_length_px * 0.5)
    grasp_px_f = body_anchor_px + axis_uv * min(model_length_px * 0.5, 80.0)
    model_box = _image_model_box(model_center_px, axis_uv, model_length_px, model_width_px, w, h)
    center_px = (
        int(np.clip(round(grasp_px_f[0]), 0, w - 1)),
        int(np.clip(round(grasp_px_f[1]), 0, h - 1)),
    )
    return {
        "mode": "red_cap_anchor_metric_model",
        "center_px": [center_px[0], center_px[1]],
        "depth_mm": cap_depth_mm,
        "center_base_xyz_m": center_base.tolist(),
        "cap_px": [int(cap_ref_x), int(cap_ref_y)],
        "cap_depth_mm": cap_depth_mm,
        "axis_uv": axis_uv.tolist(),
        "axis_source": axis_source,
        "axis_base": axis_base.tolist(),
        "grasp_from_cap_mm": float(grasp_from_cap_mm),
        "model_length_px": float(model_length_px),
        "model_width_px": float(model_width_px),
        "model_box_px": model_box,
        "points_px": {name: [int(px[0]), int(px[1])] for name, px in points_px.items()},
    }

# ---------------------------------------------------------------------------
# Fixed-Z bottle-axis model
# ---------------------------------------------------------------------------

def _bottle_geometry_from_cap_fixed_z_model(
    color_img: np.ndarray,
    points_px: dict[str, tuple[int, int]],
    base_T_cam: np.ndarray,
    intr: dict,
    fixed_z_m: float,
    grasp_from_cap_mm: float,
    default_axis_uv: list[float],
    axis_search_deg: float,
    model_length_px: float,
    model_width_px: float,
) -> dict | None:
    """Anchor on the cap pixel, move along bottle axis, then project to fixed Z."""
    cap_px = np.array(points_px["cap_center"], dtype=np.float64)
    axis_uv, axis_source = _bottle_axis_prior_uv(points_px, default_axis_uv)
    axis_uv = _refine_bottle_axis_uv(
        color_img,
        cap_px,
        axis_uv,
        axis_search_deg=axis_search_deg,
        model_length_px=model_length_px,
        model_width_px=model_width_px,
    )

    h, w = color_img.shape[:2]
    cap_base = px_ray_to_base_z(
        int(np.clip(round(float(cap_px[0])), 0, w - 1)),
        int(np.clip(round(float(cap_px[1])), 0, h - 1)),
        fixed_z_m,
        base_T_cam,
        intr,
    )

    target_offset_m = float(grasp_from_cap_mm) / 1000.0
    best_px = cap_px.copy()
    best_base = cap_base.copy()
    best_err = float("inf")
    max_offset_px = max(float(model_length_px), 80.0)
    for offset_px in np.linspace(0.0, max_offset_px, 80):
        sample_px = cap_px + axis_uv * float(offset_px)
        u = int(np.clip(round(float(sample_px[0])), 0, w - 1))
        v = int(np.clip(round(float(sample_px[1])), 0, h - 1))
        sample_base = px_ray_to_base_z(u, v, fixed_z_m, base_T_cam, intr)
        dist_m = float(np.linalg.norm(sample_base[:2] - cap_base[:2]))
        err = abs(dist_m - target_offset_m)
        if err < best_err:
            best_err = err
            best_px = np.array([u, v], dtype=np.float64)
            best_base = sample_base

    model_center_px = cap_px + axis_uv * (model_length_px * 0.5)
    model_box = _image_model_box(model_center_px, axis_uv, model_length_px, model_width_px, w, h)
    axis_base = best_base - cap_base
    axis_base[2] = 0.0
    axis_base_norm = float(np.linalg.norm(axis_base))
    if axis_base_norm > 1e-6:
        axis_base = axis_base / axis_base_norm

    return {
        "mode": "red_cap_anchor_fixed_z_axis_offset",
        "center_px": [int(best_px[0]), int(best_px[1])],
        "depth_mm": None,
        "center_base_xyz_m": best_base.tolist(),
        "cap_px": [int(round(float(cap_px[0]))), int(round(float(cap_px[1])))],
        "cap_base_xyz_m": cap_base.tolist(),
        "fixed_z_m": float(fixed_z_m),
        "axis_uv": axis_uv.tolist(),
        "axis_source": axis_source,
        "axis_base": axis_base.tolist(),
        "grasp_from_cap_mm": float(grasp_from_cap_mm),
        "grasp_offset_error_mm": float(best_err * 1000.0),
        "model_length_px": float(model_length_px),
        "model_width_px": float(model_width_px),
        "model_box_px": model_box,
        "points_px": {name: [int(px[0]), int(px[1])] for name, px in points_px.items()},
    }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_session = requests.Session()


def api(method: str, path: str, *, pi_url: str = PI_URL, **kwargs):
    r = getattr(_session, method)(f"{pi_url}{path}", **kwargs)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {path}: {r.text[:300]}")
    return r.json()


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_pick_localization_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("_", "-")
    aliases = {
        "single": "red-cap-single",
        "single-point": "red-cap-single",
        "red-cap": "red-cap-single",
        "red-cap-single-point": "red-cap-single",
        "model": "red-cap-model",
        "cap-model": "red-cap-model",
        "body": "body-points",
        "five-point": "body-points",
        "five-points": "body-points",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"red-cap-single", "red-cap-model", "body-points"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported pick localization mode: {mode}. Expected one of {sorted(allowed)}")
    return normalized


# ---------------------------------------------------------------------------
# Seeded connected-component refinement
# ---------------------------------------------------------------------------

def _find_nearest_mask_seed(mask: np.ndarray, seed_x: int, seed_y: int, max_radius: int) -> tuple[int, int] | None:
    h, w = mask.shape
    if h == 0 or w == 0 or not np.any(mask):
        return None

    seed_x = int(np.clip(seed_x, 0, w - 1))
    seed_y = int(np.clip(seed_y, 0, h - 1))
    if mask[seed_y, seed_x]:
        return seed_x, seed_y

    x1 = max(0, seed_x - max_radius)
    x2 = min(w, seed_x + max_radius + 1)
    y1 = max(0, seed_y - max_radius)
    y2 = min(h, seed_y + max_radius + 1)

    pts = np.argwhere(mask[y1:y2, x1:x2])
    if len(pts) == 0:
        return None

    pts[:, 0] += y1
    pts[:, 1] += x1
    d2 = (pts[:, 1] - seed_x) ** 2 + (pts[:, 0] - seed_y) ** 2
    idx = int(np.argmin(d2))
    return int(pts[idx, 1]), int(pts[idx, 0])


def _connected_component_from_seed(mask: np.ndarray, seed_x: int, seed_y: int) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = mask.shape
    if not (0 <= seed_x < w and 0 <= seed_y < h):
        return None
    if not mask[seed_y, seed_x]:
        return None

    visited = np.zeros_like(mask, dtype=bool)
    q = deque([(seed_x, seed_y)])
    visited[seed_y, seed_x] = True
    xs = []
    ys = []

    while q:
        x, y = q.popleft()
        xs.append(x)
        ys.append(y)
        for ny in range(max(0, y - 1), min(h, y + 2)):
            for nx in range(max(0, x - 1), min(w, x + 2)):
                if mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((nx, ny))

    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)


def _component_bbox_center_and_depth(
    depth_roi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ds: float = 1.0,
) -> tuple[int, int, float | None]:
    center_x = int(round((int(xs.min()) + int(xs.max())) / 2.0))
    center_y = int(round((int(ys.min()) + int(ys.max())) / 2.0))

    d_vals = depth_roi[ys, xs].astype(float) * ds
    d_valid = d_vals[(d_vals > 100) & (d_vals < 2000)]
    depth_mm = float(np.median(d_valid)) if len(d_valid) > 0 else None
    return center_x, center_y, depth_mm


def _local_valid_depth(
    depth_img: np.ndarray,
    px: int,
    py: int,
    ds: float = 1.0,
    radius: int = 12,
) -> float | None:
    h, w = depth_img.shape[:2]
    x1, x2 = max(0, px - radius), min(w, px + radius + 1)
    y1, y2 = max(0, py - radius), min(h, py + radius + 1)
    roi = depth_img[y1:y2, x1:x2].astype(float) * ds
    valid = roi[(roi > 100) & (roi < 2000)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def refine_red_cap(color_img, depth_img, vlm_px: int, vlm_py: int, ds: float = 1.0, radius: int = 55):
    h, w = color_img.shape[:2]
    y1, y2 = max(0, vlm_py - radius), min(h, vlm_py + radius)
    x1, x2 = max(0, vlm_px - radius), min(w, vlm_px + radius)
    crop = color_img[y1:y2, x1:x2].astype(float)

    r = crop[:, :, 0]
    g = crop[:, :, 1]
    b = crop[:, :, 2]
    brightness = crop.mean(axis=2)
    mask = (r > 70) & (r > g + 18) & (r > b + 12) & (brightness > 40)

    pts = np.argwhere(mask)
    if len(pts) < 20:
        return vlm_px, vlm_py, None

    cy, cx = pts.mean(axis=0)
    ref_px = int(x1 + cx)
    ref_py = int(y1 + cy)
    d_roi = depth_img[y1:y2, x1:x2].astype(float) * ds
    d_vals = d_roi[mask]
    d_valid = d_vals[(d_vals > 100) & (d_vals < 2000)]
    depth_mm = float(np.median(d_valid)) if len(d_valid) > 0 else None
    if depth_mm is None:
        depth_mm = _local_valid_depth(depth_img, ref_px, ref_py, ds=ds, radius=14)
    if depth_mm is None:
        depth_mm = _local_valid_depth(depth_img, vlm_px, vlm_py, ds=ds, radius=18)
    return ref_px, ref_py, depth_mm


def refine_empty_slot(color_img, depth_img, vlm_px: int, vlm_py: int, ds: float = 1.0, radius: int = 140):
    h, w = color_img.shape[:2]
    y1, y2 = max(0, vlm_py - radius), min(h, vlm_py + radius)
    x1, x2 = max(0, vlm_px - radius), min(w, vlm_px + radius)
    crop = color_img[y1:y2, x1:x2].astype(float)

    brightness = crop.mean(axis=2)
    sat = crop.max(axis=2) - crop.min(axis=2)
    mask = (brightness > 150) & (sat < 50)

    seed = _find_nearest_mask_seed(mask, vlm_px - x1, vlm_py - y1, max_radius=60)
    if seed is None:
        return vlm_px, vlm_py, None
    component = _connected_component_from_seed(mask, *seed)
    if component is None:
        return vlm_px, vlm_py, None
    xs, ys = component
    if len(xs) < 80:
        return vlm_px, vlm_py, None

    d_roi = depth_img[y1:y2, x1:x2]
    cx, cy, depth_mm = _component_bbox_center_and_depth(d_roi, xs, ys, ds=ds)
    return int(x1 + cx), int(y1 + cy), depth_mm


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.laps = []
        self._t0 = time.monotonic()
        self._lap_t = self._t0

    def lap(self, label: str):
        now = time.monotonic()
        dt = now - self._lap_t
        self._lap_t = now
        self.laps.append((label, dt))
        return dt

    def summary(self):
        total = time.monotonic() - self._t0
        lines = [f"  {'Step':<30s} {'Time':>6s}"]
        lines.append("  " + "-" * 38)
        for label, dt in self.laps:
            lines.append(f"  {label:<30s} {dt:>5.1f}s")
        lines.append("  " + "-" * 38)
        lines.append(f"  {'TOTAL':<30s} {total:>5.1f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    dry_run: bool = False,
    skip_place: bool = False,
    pregrasp_only: bool = False,
    confirm: bool = False,
    save_dir: str = "/tmp/pick_place",
    pi_url: str = PI_URL,
    vlm_key: str = VLM_KEY,
    vlm_base: str = VLM_BASE,
    vlm_model: str = VLM_MODEL,
    scan_pose_deg: list[float] = SCAN_POSE_DEG,
    bottle_height_mm: float = BOTTLE_HEIGHT_MM,
    place_clearance_mm: float = PLACE_CLEARANCE_MM,
    grip_center_offset_y_mm: float = GRIP_CENTER_OFFSET_Y_MM,
    pick_contact_local_offset_mm: list[float] = PICK_CONTACT_LOCAL_OFFSET_MM,
    grasp_template_rpy_rad: list[float] = GRASP_TEMPLATE_RPY_RAD,
    pick_localization_mode: str = PICK_LOCALIZATION_MODE,
    pick_grasp_z_mode: str = PICK_GRASP_Z_MODE,
    successful_grasp_z_prior_mm: float = SUCCESSFUL_GRASP_Z_PRIOR_MM,
    pick_xy_mode: str = PICK_XY_MODE,
):
    try:
        from openai import OpenAI
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install at least: openai Pillow"
        ) from exc

    if not vlm_key:
        raise RuntimeError("Missing VLM key. Pass --vlm-key or set PICK_PLACE_VLM_KEY.")

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    pick_localization_mode = _normalize_pick_localization_mode(pick_localization_mode)
    pick_grasp_z_mode = _normalize_pick_grasp_z_mode(pick_grasp_z_mode)
    pick_xy_mode = _normalize_pick_xy_mode(pick_xy_mode)

    def gate(msg: str):
        if confirm:
            input(f"\n[confirm] {msg} Press ENTER to continue, Ctrl+C to abort: ")

    print("=" * 55)
    print("1/6  Move to scan pose + capture RGBD")
    print("=" * 55)
    api("post", "/move_j", pi_url=pi_url, json={"angles_deg": scan_pose_deg}, timeout=30)
    timer.lap("move_j -> scan pose")

    scan = api("post", "/scan", pi_url=pi_url, timeout=30)
    fp6 = scan["flange_pose"]
    intr = scan["intrinsics"]
    ds = scan["depth_scale"]
    w, h = intr["width"], intr["height"]
    print(f"  Flange Z = {fp6[2] * 1000:.0f} mm")

    color_bytes = base64.b64decode(scan["image_base64"])
    depth_bytes = base64.b64decode(scan["depth_base64"])
    color_img = np.array(Image.open(io.BytesIO(color_bytes)))
    depth_img = np.array(Image.open(io.BytesIO(depth_bytes)))
    img_b64 = scan["image_base64"]

    base_T_cam = pose_to_T(fp6) @ FLANGE_T_CAM
    timer.lap("scan + decode")

    print("\n" + "=" * 55)
    print("2/6  VLM object detection")
    print("=" * 55)
    client = OpenAI(api_key=vlm_key, base_url=vlm_base)
    # Keep this protocol normalized (0-1). Qwen-VL has been reliable with
    # normalized u/v, while integer-pixel prompts can produce out-of-bounds
    # values that look like pixels but are actually scale-confused coordinates.
    if pick_localization_mode == "red-cap-single":
        pick_prompt = (
            f'{{"name":"{PICK_OBJECT_NAME}","u":0.XX,"v":0.XX,"confidence":0.0}}'
        )
        pick_instructions = (
            f"1. {PICK_OBJECT_NAME} - the center of the red cap on the single glass bottle "
            "that is OUTSIDE the red box. Return only the red cap center for this object; "
            "do not infer transparent bottle body points.\n"
        )
    else:
        pick_prompt = (
            f'{{"name":"{PICK_OBJECT_NAME}",'
            f'"keypoints":{{"cap_center":[0.XX,0.XX],"body_tail_center":[0.XX,0.XX],'
            f'"body_left_mid":[0.XX,0.XX],"body_right_mid":[0.XX,0.XX],'
            f'"body_center":[0.XX,0.XX]}},'
            f'"confidence":0.0}}'
        )
        pick_instructions = (
            f"1. {PICK_OBJECT_NAME} - the single glass bottle with the red cap "
            "that is OUTSIDE the red box. Ignore bottles already inside the red box. Return the red "
            "cap center accurately, plus rough body direction prior points for a fixed-size bottle "
            "model: far body end opposite the red cap, left/right body edges at the cylinder middle, "
            "and body center. The body points are for direction only; the far body end and body center "
            "must not be the same point as the red cap.\n"
        )

    prompt = (
        f"This is a {w}x{h} image from a robot camera looking down.\n"
        "Find:\n"
        f"{pick_instructions}"
        f"2. {PLACE_OBJECT_NAME} - the four inside corner points around the empty white foam slot "
        "inside the red box where the next bottle should be inserted. Use the groove/slot boundary, "
        "not the red box wall, and match the orientation of bottles already placed in the box.\n"
        f'Return ONLY JSON: {{"objects":[{pick_prompt},'
        f'{{"name":"{PLACE_OBJECT_NAME}","corners":{{"top_left":[0.XX,0.XX],'
        f'"top_right":[0.XX,0.XX],"bottom_right":[0.XX,0.XX],"bottom_left":[0.XX,0.XX]}},'
        f'"axis":"left-right","confidence":0.0}}]}}\n'
        "All u,v values must be normalized 0-1, top-left=(0,0). Do not return integer pixels."
    )
    resp = client.chat.completions.create(
        model=vlm_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        extra_body={"enable_thinking": False},
        temperature=0,
        max_tokens=400,
    )
    raw = resp.choices[0].message.content
    print(f"  VLM raw: {raw}")

    match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
    if not match:
        raise RuntimeError(f"Failed to parse VLM JSON from response: {raw}")
    parsed = json.loads(match.group())
    detected = parsed if isinstance(parsed, list) else parsed.get("objects", [parsed])

    vlm: dict[str, tuple[int, int]] = {}
    bottle_body_points_px: dict[str, tuple[int, int]] | None = None
    bottle_vlm_meta: dict | None = None
    slot_corners_px: dict[str, tuple[int, int]] | None = None
    slot_vlm_meta: dict | None = None
    for obj in detected:
        name = obj["name"]
        if name == PICK_OBJECT_NAME:
            bottle_body_points_px = _parse_bottle_body_points(obj, w, h)
            bottle_vlm_meta = {
                "mode": pick_localization_mode,
                "confidence": obj.get("confidence"),
                "points_px": None if bottle_body_points_px is None else {
                    key: [int(px[0]), int(px[1])] for key, px in bottle_body_points_px.items()
                },
            }
            if pick_localization_mode == "red-cap-single" and bottle_body_points_px is not None:
                vlm[name] = bottle_body_points_px["cap_center"]
                print(f"  {name}: cap point {vlm[name]} from keypoints")
                continue
            if pick_localization_mode == "red-cap-single":
                px = _norm_uv_to_px(obj, w, h)
                if px is not None:
                    vlm[name] = px
                    print(f"  {name}: red cap pixel {vlm[name]}")
                    continue
            if bottle_body_points_px is not None:
                center = np.mean(
                    np.array(
                        [
                            bottle_body_points_px["cap_center"],
                            bottle_body_points_px["bottom_center"],
                            bottle_body_points_px["left_mid"],
                            bottle_body_points_px["right_mid"],
                        ],
                        dtype=np.float64,
                    ),
                    axis=0,
                )
                vlm[name] = (int(round(center[0])), int(round(center[1])))
                print(f"  {name}: body points {bottle_vlm_meta['points_px']} center {vlm[name]}")
                continue

        if name == PLACE_OBJECT_NAME:
            slot_corners_px = _parse_slot_corners(obj, w, h)
            slot_vlm_meta = {
                "axis": obj.get("axis"),
                "confidence": obj.get("confidence"),
                "corners_px": None if slot_corners_px is None else {
                    key: [int(px[0]), int(px[1])] for key, px in slot_corners_px.items()
                },
            }
            if slot_corners_px is not None:
                center = np.mean(np.array(list(slot_corners_px.values()), dtype=np.float64), axis=0)
                vlm[name] = (int(round(center[0])), int(round(center[1])))
                print(f"  {name}: corners {slot_vlm_meta['corners_px']} center {vlm[name]}")
                continue

        px = _norm_uv_to_px(obj, w, h)
        if px is not None:
            vlm[name] = px
            print(f"  {name}: pixel {vlm[name]}")

    if PICK_OBJECT_NAME not in vlm:
        raise RuntimeError(f"VLM did not detect {PICK_OBJECT_NAME}")
    if PLACE_OBJECT_NAME not in vlm and not skip_place and not pregrasp_only:
        raise RuntimeError(f"VLM did not detect {PLACE_OBJECT_NAME}")
    timer.lap("VLM inference")

    print("\n" + "=" * 55)
    print("3/6  Seeded refinement + 3D localization")
    print("=" * 55)

    bottle_geometry = None
    if bottle_body_points_px is not None and pick_localization_mode in {"red-cap-model", "body-points"}:
        if pick_xy_mode == "fixed-z-plane":
            bottle_geometry = _bottle_geometry_from_cap_fixed_z_model(
                color_img,
                bottle_body_points_px,
                base_T_cam,
                intr,
                successful_grasp_z_prior_mm / 1000.0,
                BOTTLE_GRASP_FROM_CAP_MM,
                BOTTLE_MODEL_AXIS_UV,
                BOTTLE_MODEL_AXIS_SEARCH_DEG,
                BOTTLE_MODEL_LENGTH_PX,
                BOTTLE_MODEL_WIDTH_PX,
            )
        else:
            bottle_geometry = _bottle_geometry_from_cap_model(
                color_img,
                depth_img,
                bottle_body_points_px,
                ds,
                base_T_cam,
                intr,
                BOTTLE_GRASP_FROM_CAP_MM,
                BOTTLE_MODEL_AXIS_UV,
                BOTTLE_MODEL_AXIS_SEARCH_DEG,
                BOTTLE_MODEL_LENGTH_PX,
                BOTTLE_MODEL_WIDTH_PX,
            )
        if bottle_geometry is None and pick_localization_mode == "body-points" and pick_xy_mode != "fixed-z-plane":
            bottle_geometry = _bottle_geometry_from_body_points(
                bottle_body_points_px, depth_img, ds, base_T_cam, intr
            )

    if bottle_geometry is not None:
        pick_px, pick_py = bottle_geometry["center_px"]
        pick_d_mm = bottle_geometry["depth_mm"]
        pick_3d = np.array(bottle_geometry["center_base_xyz_m"], dtype=np.float64)
        pick_xy_source = bottle_geometry.get("mode", "vlm_body_points")
        depth_label = "none" if pick_d_mm is None else f"{pick_d_mm:.0f}mm"
        print(
            f"  Bottle geometry ({bottle_geometry.get('mode', 'vlm_body_points')}): "
            f"pixel ({pick_px},{pick_py}) depth={depth_label}"
        )
        print(
            f"  Bottle body axis: X={pick_3d[0]*1000:.1f} "
            f"Y={pick_3d[1]*1000:.1f} Z={pick_3d[2]*1000:.1f} mm "
            f"axis_base={bottle_geometry['axis_base']}"
        )
    else:
        pick_px, pick_py, pick_d_mm = refine_red_cap(color_img, depth_img, *vlm[PICK_OBJECT_NAME], ds=ds)
        if pick_d_mm is None and pick_xy_mode != "fixed-z-plane":
            raise RuntimeError("No valid depth for pick bottle")
        if pick_xy_mode == "fixed-z-plane":
            pick_3d = px_ray_to_base_z(
                pick_px,
                pick_py,
                successful_grasp_z_prior_mm / 1000.0,
                base_T_cam,
                intr,
            )
            pick_xy_source = "fixed_z_plane_projection"
            depth_label = "none" if pick_d_mm is None else f"{pick_d_mm:.0f}mm"
            print(
                f"  Bottle fallback cap: pixel ({pick_px},{pick_py}) "
                f"depth={depth_label} (depth ignored for XY; "
                f"projected to Z={successful_grasp_z_prior_mm:.1f}mm)"
            )
        else:
            pick_3d = px_to_base(pick_px, pick_py, pick_d_mm, base_T_cam, intr)
            pick_xy_source = "vision_depth"
            print(f"  Bottle fallback cap: pixel ({pick_px},{pick_py}) depth={pick_d_mm:.0f}mm")
    pick_3d[1] += grip_center_offset_y_mm / 1000.0
    print(
        f"  Bottle grasp 3D: X={pick_3d[0]*1000:.1f} "
        f"Y={pick_3d[1]*1000:.1f} Z={pick_3d[2]*1000:.1f} mm "
        f"(Y offset {grip_center_offset_y_mm}mm)"
    )
    pick_contact_3d = pick_3d.copy()
    pick_flange_3d, pick_contact_delta_base_m = _flange_target_from_contact(
        pick_contact_3d,
        grasp_template_rpy_rad,
        pick_contact_local_offset_mm,
    )
    print(
        "  SDK flange target from contact: "
        f"X={pick_flange_3d[0]*1000:.1f} Y={pick_flange_3d[1]*1000:.1f} "
        f"Z={pick_flange_3d[2]*1000:.1f} mm "
        f"(local contact offset {pick_contact_local_offset_mm}mm -> "
        f"base delta {[round(float(v) * 1000.0, 1) for v in pick_contact_delta_base_m]}mm)"
    )

    slot_3d = None
    slot_px = slot_py = slot_d_mm = None
    slot_geometry = None
    if slot_corners_px is not None:
        slot_geometry = _slot_geometry_from_corners(slot_corners_px, depth_img, ds, base_T_cam, intr)
        if slot_geometry is not None:
            slot_px, slot_py = slot_geometry["center_px"]
            slot_d_mm = slot_geometry["depth_mm"]
            slot_3d = np.array(slot_geometry["center_base_xyz_m"], dtype=np.float64)
            print(f"  Empty slot quad center: pixel ({slot_px},{slot_py}) depth={slot_d_mm:.0f}mm")
            print(
                f"  Empty slot centerline: X={slot_3d[0]*1000:.1f} "
                f"Y={slot_3d[1]*1000:.1f} Z={slot_3d[2]*1000:.1f} mm "
                f"axis_base={slot_geometry['axis_base']}"
            )
    if slot_3d is None and PLACE_OBJECT_NAME in vlm:
        slot_px, slot_py, slot_d_mm = refine_empty_slot(color_img, depth_img, *vlm[PLACE_OBJECT_NAME], ds=ds)
        if slot_d_mm is not None:
            slot_3d = px_to_base(slot_px, slot_py, slot_d_mm, base_T_cam, intr)
            print(f"  Empty slot fallback center: pixel ({slot_px},{slot_py}) depth={slot_d_mm:.0f}mm")
            print(
                f"  Empty slot 3D: X={slot_3d[0]*1000:.1f} "
                f"Y={slot_3d[1]*1000:.1f} Z={slot_3d[2]*1000:.1f} mm"
            )

    depth_based_pick_3d = (
        None
        if pick_d_mm is None
        else px_to_base(pick_px, pick_py, pick_d_mm, base_T_cam, intr)
    )
    depth_grasp_z_m = (
        float("nan")
        if depth_based_pick_3d is None
        else float(depth_based_pick_3d[2] - GRASP_BELOW_MM / 1000.0)
    )
    if pick_grasp_z_mode == "successful-prior":
        grasp_z = successful_grasp_z_prior_mm / 1000.0
        grasp_z_source = "successful_grasp_height_prior"
        print(
            "  NOTE for remote collaborators: pick grasp Z uses the "
            f"successful grasp height prior ({successful_grasp_z_prior_mm:.1f}mm), "
            f"not transparent-bottle depth "
            f"({'unavailable' if math.isnan(depth_grasp_z_m) else f'{depth_grasp_z_m * 1000.0:.1f}mm'})."
        )
    else:
        grasp_z = depth_grasp_z_m
        grasp_z_source = "vision_depth"
    place_z = None
    if slot_3d is not None and not skip_place:
        place_z = slot_3d[2] + (bottle_height_mm + place_clearance_mm) / 1000.0
    pick_xy_align_standoff_mm = max(STANDOFF_MM, PICK_XY_ALIGN_ABOVE_GRASP_MM + GRASP_BELOW_MM)

    print(
        f"\n  Grasp Z:  {grasp_z * 1000:.0f} mm ({grasp_z_source}) | "
        f"XY align height: {(pick_flange_3d[2] + pick_xy_align_standoff_mm / 1000.0) * 1000:.0f} mm | "
        f"Lift Z: {LIFT_Z_MM} mm",
        end="",
    )
    if place_z is not None:
        print(f" | Place Z: {place_z * 1000:.0f} mm")
    else:
        print()
    timer.lap("seeded refine + 3D")

    if dry_run:
        scan_color_path = out / "scan_color.jpg"
        scan_depth_path = out / "scan_depth.png"
        prompt_path = out / "vlm_prompt.txt"
        vlm_raw_path = out / "vlm_raw.txt"
        ann_path = out / "annotated.jpg"
        debug_path = out / "debug_result.json"

        scan_color_path.write_bytes(color_bytes)
        scan_depth_path.write_bytes(depth_bytes)
        _write_text(prompt_path, prompt)
        _write_text(vlm_raw_path, raw)

        img = Image.open(io.BytesIO(color_bytes))
        draw = ImageDraw.Draw(img)
        if bottle_body_points_px is not None:
            cap = bottle_body_points_px["cap_center"]
            bottom = bottle_body_points_px["bottom_center"]
            left_mid = bottle_body_points_px["left_mid"]
            right_mid = bottle_body_points_px["right_mid"]
            draw.line([cap, bottom], fill="orange", width=3)
            draw.line([left_mid, right_mid], fill="orange", width=3)
            for px in (cap, bottom, left_mid, right_mid):
                draw.ellipse([px[0] - 6, px[1] - 6, px[0] + 6, px[1] + 6], outline="orange", width=2)
        draw.ellipse([pick_px - 20, pick_py - 20, pick_px + 20, pick_py + 20], outline="lime", width=3)
        draw.text((pick_px + 25, pick_py - 15), "BOTTLE", fill="lime")
        if bottle_geometry is not None and bottle_geometry.get("model_box_px"):
            model_poly = [tuple(px) for px in bottle_geometry["model_box_px"]]
            draw.line(model_poly + [model_poly[0]], fill="deepskyblue", width=3)
            cap_px = bottle_geometry.get("cap_px")
            if cap_px is not None:
                draw.ellipse([cap_px[0] - 8, cap_px[1] - 8, cap_px[0] + 8, cap_px[1] + 8], outline="red", width=3)
                draw.text((cap_px[0] + 10, cap_px[1] - 12), "CAP_ANCHOR", fill="red")
        if slot_corners_px is not None:
            poly = [
                slot_corners_px["top_left"],
                slot_corners_px["top_right"],
                slot_corners_px["bottom_right"],
                slot_corners_px["bottom_left"],
                slot_corners_px["top_left"],
            ]
            draw.line(poly, fill="cyan", width=3)
            if slot_px is not None and slot_py is not None:
                draw.line([slot_px - 35, slot_py, slot_px + 35, slot_py], fill="cyan", width=2)
                draw.line([slot_px, slot_py - 35, slot_px, slot_py + 35], fill="cyan", width=2)
                draw.text((slot_px + 25, slot_py - 15), "SLOT QUAD", fill="cyan")
        if slot_3d is not None and slot_px is not None and slot_py is not None:
            draw.ellipse([slot_px - 25, slot_py - 25, slot_px + 25, slot_py + 25], outline="cyan", width=3)
            draw.text((slot_px + 30, slot_py - 15), "SLOT", fill="cyan")
        img.save(ann_path)

        _write_json(debug_path, {
            "config": {
                "pi_url": pi_url,
                "vlm_base": vlm_base,
                "vlm_model": vlm_model,
                "scan_pose_deg": scan_pose_deg,
                "standoff_mm": STANDOFF_MM,
                "pick_xy_align_above_grasp_mm": PICK_XY_ALIGN_ABOVE_GRASP_MM,
                "pick_xy_align_standoff_mm": pick_xy_align_standoff_mm,
                "pregrasp_only": pregrasp_only,
                "pick_localization_mode": pick_localization_mode,
                "pick_xy_mode": pick_xy_mode,
                "grasp_below_mm": GRASP_BELOW_MM,
                "lift_z_mm": LIFT_Z_MM,
                "pick_grasp_z_mode": pick_grasp_z_mode,
                "successful_grasp_z_prior_mm": successful_grasp_z_prior_mm,
                "bottle_height_mm": bottle_height_mm,
                "bottle_grasp_from_cap_mm": BOTTLE_GRASP_FROM_CAP_MM,
                "bottle_model_axis_uv": BOTTLE_MODEL_AXIS_UV,
                "bottle_model_axis_search_deg": BOTTLE_MODEL_AXIS_SEARCH_DEG,
                "bottle_model_length_px": BOTTLE_MODEL_LENGTH_PX,
                "bottle_model_width_px": BOTTLE_MODEL_WIDTH_PX,
                "place_clearance_mm": place_clearance_mm,
                "grip_center_offset_y_mm": grip_center_offset_y_mm,
                "pick_contact_local_offset_mm": pick_contact_local_offset_mm,
                "sdk_flange_frame": "official_urdf_gripper_joint_origin",
                "grasp_template_rpy_rad": grasp_template_rpy_rad,
                "slot_refinement_mode": "vlm_quad_centerline_with_seed_fallback",
            },
            "scan": {
                "flange_pose": fp6,
                "intrinsics": intr,
                "depth_scale": ds,
            },
            "vlm": {
                "raw": raw,
                "detections_px": {
                    name: {"u_px": int(px), "v_px": int(py)}
                    for name, (px, py) in vlm.items()
                },
                "empty_slot_quad": slot_vlm_meta,
                "bottle_body_points": bottle_vlm_meta,
            },
            "refined": {
                PICK_OBJECT_NAME: {
                    "u_px": pick_px,
                    "v_px": pick_py,
                    "depth_mm": pick_d_mm,
                    "xy_source": pick_xy_source,
                    "depth_based_base_xyz_m": None if depth_based_pick_3d is None else depth_based_pick_3d.tolist(),
                    "contact_base_xyz_m": pick_contact_3d.tolist(),
                    "sdk_flange_target_xyz_m": pick_flange_3d.tolist(),
                    "contact_offset_base_delta_m": pick_contact_delta_base_m.tolist(),
                    "base_xyz_m": pick_contact_3d.tolist(),
                    "geometry": bottle_geometry,
                },
                PLACE_OBJECT_NAME: None if slot_3d is None else {
                    "u_px": slot_px,
                    "v_px": slot_py,
                    "depth_mm": slot_d_mm,
                    "base_xyz_m": slot_3d.tolist(),
                    "geometry": slot_geometry,
                },
            },
            "motion_plan": {
                "pick_contact_xyz_m": pick_contact_3d.tolist(),
                "pick_sdk_flange_xyz_m": pick_flange_3d.tolist(),
                "pick_xy_source": pick_xy_source,
                "depth_based_grasp_z_mm": depth_grasp_z_m * 1000.0,
                "grasp_z_mm": grasp_z * 1000.0,
                "grasp_z_source": grasp_z_source,
                "remote_collaborator_note": (
                    "Transparent-bottle Z is using the successful grasp height prior, "
                    "not raw depth, when grasp_z_source is successful_grasp_height_prior."
                ),
                "place_z_mm": None if place_z is None else place_z * 1000.0,
            },
            "timing": [
                {"label": label, "seconds": round(dt, 4)}
                for label, dt in timer.laps
            ],
        })

        print(f"\n  Raw scan: {scan_color_path}")
        print(f"  Raw depth: {scan_depth_path}")
        print(f"  VLM prompt: {prompt_path}")
        print(f"  VLM raw: {vlm_raw_path}")
        print(f"  Annotated: {ann_path}")
        print(f"  Debug JSON: {debug_path}")
        print("  DRY-RUN complete. No motion commands sent.")
        print("\n" + timer.summary())
        return

    gate("Start pick?")
    print("\n" + "=" * 55)
    print("4/6  Move above -> Grasp -> Lift")
    print("=" * 55)
    res = api(
        "post",
        "/move_above",
        pi_url=pi_url,
        json={
            "xyz_m": pick_flange_3d.tolist(),
            "rpy_rad": grasp_template_rpy_rad,
            "standoff_mm": pick_xy_align_standoff_mm,
            "z_safe_mm": LIFT_Z_MM,
        },
        timeout=30,
    )
    print(f"  Above: flange={[f'{v:.3f}' for v in res['flange_xyz_m']]}")
    timer.lap("move_above")

    if pregrasp_only:
        print("  PREGRASP-ONLY complete. No descent, no gripper close, no lift.")
        print("\n" + timer.summary())
        return

    gate("Grasp?")
    res = api(
        "post",
        "/grasp",
        pi_url=pi_url,
        json={"grasp_z_mm": grasp_z * 1000, "lift_z_mm": LIFT_Z_MM},
        timeout=45,
    )
    grasped = res.get("grasped", False)
    print(f"  Grasp: grip_width={res.get('grip_width_m', 0) * 1000:.1f}mm  grasped={grasped}")
    timer.lap("grasp + lift")

    if not grasped:
        print("  x Grasp failed - aborting.")
        api("post", "/move_j", pi_url=pi_url, json={"angles_deg": scan_pose_deg}, timeout=30)
        print("\n" + timer.summary())
        return

    if skip_place or slot_3d is None:
        print("  Pick done (skip-place). Returning to scan pose.")
        api("post", "/move_j", pi_url=pi_url, json={"angles_deg": scan_pose_deg}, timeout=30)
        timer.lap("return scan pose")
        print("\n" + timer.summary())
        return

    gate("Place into empty slot?")
    print("\n" + "=" * 55)
    print("5/6  Place into empty slot")
    print("=" * 55)
    res = api(
        "post",
        "/place",
        pi_url=pi_url,
        json={"xyz_m": slot_3d.tolist(), "place_z_mm": place_z * 1000, "z_safe_mm": LIFT_Z_MM},
        timeout=60,
    )
    print(f"  Place: z={res.get('place_z_mm', 0):.0f}mm  status={res.get('status')}")
    timer.lap("place")

    print("\n" + "=" * 55)
    print("6/6  Return to scan pose")
    print("=" * 55)
    api("post", "/move_j", pi_url=pi_url, json={"angles_deg": scan_pose_deg}, timeout=60)
    timer.lap("return scan pose")

    print("\nOK Pick-and-place complete!")
    print("\n" + timer.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick-and-place orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Scan + detect only")
    parser.add_argument("--skip-place", action="store_true", help="Pick only, no place")
    parser.add_argument("--pregrasp-only", action="store_true", help="Move above pick target only; do not grasp")
    parser.add_argument("--confirm", action="store_true", help="Pause before each motion step")
    parser.add_argument("--save-dir", default="calibration/results/pick_place_debug")
    parser.add_argument("--pi-url", default=PI_URL, help="Pi robot_server base URL")
    parser.add_argument("--vlm-key", default=VLM_KEY, help="VLM API key")
    parser.add_argument("--vlm-base", default=VLM_BASE, help="VLM API base URL")
    parser.add_argument("--vlm-model", default=VLM_MODEL, help="VLM model name")
    parser.add_argument(
        "--scan-pose-deg",
        type=lambda raw: _parse_float_csv(raw, expected_len=7),
        default=SCAN_POSE_DEG,
        help="Comma-separated 7 joint angles in degrees",
    )
    parser.add_argument("--bottle-height-mm", type=float, default=BOTTLE_HEIGHT_MM)
    parser.add_argument("--place-clearance-mm", type=float, default=PLACE_CLEARANCE_MM)
    parser.add_argument("--grip-center-offset-y-mm", type=float, default=GRIP_CENTER_OFFSET_Y_MM)
    parser.add_argument(
        "--pick-contact-local-offset-mm",
        type=lambda raw: _parse_float_csv(raw, expected_len=3),
        default=PICK_CONTACT_LOCAL_OFFSET_MM,
        help=(
            "Comma-separated contact point offset in SDK flange local XYZ mm. "
            "Vision gives the desired contact point; the commanded SDK flange target is "
            "contact - R_grasp @ offset."
        ),
    )
    parser.add_argument(
        "--pick-localization-mode",
        default=PICK_LOCALIZATION_MODE,
        choices=["red-cap-single", "red-cap-model", "body-points"],
        help="Pick target localization strategy; default returns to the proven red-cap single-point path",
    )
    parser.add_argument(
        "--grasp-template-rpy-rad",
        type=lambda raw: _parse_float_csv(raw, expected_len=3),
        default=GRASP_TEMPLATE_RPY_RAD,
        help="Comma-separated grasp template roll,pitch,yaw in radians",
    )
    parser.add_argument(
        "--pick-grasp-z-mode",
        default=PICK_GRASP_Z_MODE,
        choices=["depth", "successful-prior", "success-prior", "prior"],
        help=(
            "Source for pick descent Z. 'successful-prior' uses the validated "
            "transparent-bottle grasp height and records this explicitly."
        ),
    )
    parser.add_argument(
        "--successful-grasp-z-prior-mm",
        type=float,
        default=SUCCESSFUL_GRASP_Z_PRIOR_MM,
        help="Validated transparent-bottle grasp Z height in base frame, in mm.",
    )
    parser.add_argument(
        "--pick-xy-mode",
        default=PICK_XY_MODE,
        choices=["depth", "vision-depth", "fixed-z-plane", "support-plane", "plane"],
        help=(
            "Source for pick XY. 'fixed-z-plane' projects the RGB pixel ray to "
            "the fixed grasp/support Z plane instead of using transparent-bottle depth."
        ),
    )
    args = parser.parse_args()
    run(
        dry_run=args.dry_run,
        skip_place=args.skip_place,
        pregrasp_only=args.pregrasp_only,
        confirm=args.confirm,
        save_dir=args.save_dir,
        pi_url=args.pi_url,
        vlm_key=args.vlm_key,
        vlm_base=args.vlm_base,
        vlm_model=args.vlm_model,
        scan_pose_deg=args.scan_pose_deg,
        bottle_height_mm=args.bottle_height_mm,
        place_clearance_mm=args.place_clearance_mm,
        grip_center_offset_y_mm=args.grip_center_offset_y_mm,
        pick_contact_local_offset_mm=args.pick_contact_local_offset_mm,
        grasp_template_rpy_rad=args.grasp_template_rpy_rad,
        pick_localization_mode=args.pick_localization_mode,
        pick_grasp_z_mode=args.pick_grasp_z_mode,
        successful_grasp_z_prior_mm=args.successful_grasp_z_prior_mm,
        pick_xy_mode=args.pick_xy_mode,
    )
