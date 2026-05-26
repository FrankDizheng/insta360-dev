#!/usr/bin/env python3
"""Detect the empty white foam slot (凹槽) from a fixed/top camera image.

Pipeline:
  fixed_camera.jpg
    -> red box ROI
    -> white foam slot connected component
    -> slot corners + mask-centroid centerline in pixels
    -> robot base XY via support-plane homography
    -> base XYZ with configured support-plane Z
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gripper_align_place import attach_p5_place_target  # noqa: E402
from top_camera_plane_project import (  # noqa: E402
    DEFAULT_HOMOGRAPHY,
    load_homography,
    pixel_to_base_xy,
)

DEFAULT_SUPPORT_Z_M = 0.248
DEFAULT_HOMOGRAPHY_PATH = DEFAULT_HOMOGRAPHY


def _connected_component_from_seed(
    mask: np.ndarray, seed_x: int, seed_y: int
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = mask.shape
    if not (0 <= seed_x < w and 0 <= seed_y < h) or not mask[seed_y, seed_x]:
        return None

    visited = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(seed_x, seed_y)])
    visited[seed_y, seed_x] = True
    xs: list[int] = []
    ys: list[int] = []

    while queue:
        x, y = queue.popleft()
        xs.append(x)
        ys.append(y)
        for ny in range(max(0, y - 1), min(h, y + 2)):
            for nx in range(max(0, x - 1), min(w, x + 2)):
                if mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))

    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)


def _largest_connected_component(mask: np.ndarray, min_area: int = 200) -> tuple[np.ndarray, np.ndarray] | None:
    visited = np.zeros_like(mask, dtype=bool)
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_area = 0

    for seed_y in range(mask.shape[0]):
        for seed_x in range(mask.shape[1]):
            if not mask[seed_y, seed_x] or visited[seed_y, seed_x]:
                continue
            component = _connected_component_from_seed(mask, seed_x, seed_y)
            if component is None:
                continue
            xs, ys = component
            visited[ys, xs] = True
            area = len(xs)
            if area >= min_area and area > best_area:
                best_area = area
                best = component

    return best


def _order_box_corners(box: np.ndarray) -> dict[str, list[int]]:
    """Order minAreaRect corners as top_left .. bottom_left (image y-down)."""
    pts = box.astype(np.float64)
    order = np.argsort(pts[:, 1])
    top = pts[order[:2]]
    bottom = pts[order[2:]]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    corners = {
        "top_left": top[0],
        "top_right": top[1],
        "bottom_right": bottom[1],
        "bottom_left": bottom[0],
    }
    return {name: [int(round(p[0])), int(round(p[1]))] for name, p in corners.items()}


def _slot_axis_from_corners(corners_px: dict[str, list[int]]) -> tuple[np.ndarray, float, float]:
    ordered = [
        np.array(corners_px["top_left"], dtype=np.float64),
        np.array(corners_px["top_right"], dtype=np.float64),
        np.array(corners_px["bottom_right"], dtype=np.float64),
        np.array(corners_px["bottom_left"], dtype=np.float64),
    ]
    top_axis = ordered[1] - ordered[0]
    bottom_axis = ordered[2] - ordered[3]
    left_axis = ordered[3] - ordered[0]
    right_axis = ordered[2] - ordered[1]
    horizontal = (top_axis + bottom_axis) * 0.5
    vertical = (left_axis + right_axis) * 0.5
    long_axis = horizontal if np.linalg.norm(horizontal) >= np.linalg.norm(vertical) else vertical
    axis_norm = float(np.linalg.norm(long_axis))
    if axis_norm < 1.0:
        raise ValueError("Slot axis too short to define orientation")
    axis_uv = long_axis / axis_norm
    length_px = float(max(np.linalg.norm(horizontal), np.linalg.norm(vertical)))
    width_px = float(min(np.linalg.norm(horizontal), np.linalg.norm(vertical)))
    return axis_uv, length_px, width_px


def detect_red_box_contour(color_bgr: np.ndarray) -> np.ndarray | None:
    b, g, r = cv2.split(color_bgr)
    mask_red = (r.astype(np.int16) > 70) & (r > g + 18) & (r > b + 12)
    mask_u8 = mask_red.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def detect_empty_slot(
    color_bgr: np.ndarray,
    *,
    red_erode_px: int = 25,
    white_brightness_min: float = 150.0,
    white_sat_max: float = 50.0,
    min_component_area: int = 200,
) -> dict:
    """Detect empty white foam slot inside the red box."""
    h, w = color_bgr.shape[:2]
    red_contour = detect_red_box_contour(color_bgr)
    if red_contour is None:
        raise RuntimeError("Red box contour not found")

    red_area = float(cv2.contourArea(red_contour))
    if red_area < 5000:
        raise RuntimeError(f"Red box contour too small (area={red_area:.0f})")

    mask_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_inner, [red_contour], contourIdx=-1, color=255, thickness=-1)
    kernel = np.ones((red_erode_px, red_erode_px), dtype=np.uint8)
    mask_inner = cv2.erode(mask_inner, kernel)

    x, y, roi_w, roi_h = cv2.boundingRect(red_contour)
    roi = color_bgr[y : y + roi_h, x : x + roi_w].astype(np.float64)
    brightness = roi.mean(axis=2)
    saturation = roi.max(axis=2) - roi.min(axis=2)
    mask_white = (
        (brightness > white_brightness_min)
        & (saturation < white_sat_max)
        & (mask_inner[y : y + roi_h, x : x + roi_w] > 0)
    )

    component = _largest_connected_component(mask_white, min_area=min_component_area)
    if component is None:
        raise RuntimeError("White foam slot component not found inside red box")

    xs_local, ys_local = component
    xs = xs_local + x
    ys = ys_local + y

    # Mask centroid aligns better with the visual center of the white foam slot than
    # the minAreaRect geometric center (often biased by uneven foam edges).
    center_arr = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
    center_px = [int(round(center_arr[0])), int(round(center_arr[1]))]

    points = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    corners_px = _order_box_corners(box)
    axis_uv, length_px, width_px = _slot_axis_from_corners(corners_px)
    rect_center_px = [int(round(rect[0][0])), int(round(rect[0][1]))]

    red_x, red_y, red_w, red_h = cv2.boundingRect(red_contour)
    return {
        "center_px": center_px,
        "center_mode": "mask_centroid",
        "rect_center_px": rect_center_px,
        "corners_px": corners_px,
        "axis_uv": [float(axis_uv[0]), float(axis_uv[1])],
        "length_px": length_px,
        "width_px": width_px,
        "red_box_bbox_px": [int(red_x), int(red_y), int(red_w), int(red_h)],
        "red_box_area_px": red_area,
        "slot_component_area_px": int(len(xs)),
    }


def slot_pixels_to_base(
    slot: dict,
    homography: np.ndarray,
    *,
    support_z_m: float = DEFAULT_SUPPORT_Z_M,
) -> dict:
    center_px = tuple(slot["center_px"])
    center_xy = pixel_to_base_xy(center_px, homography)
    center_xyz = [center_xy[0], center_xy[1], float(support_z_m)]

    corners_base: dict[str, list[float]] = {}
    for name, px in slot["corners_px"].items():
        xy = pixel_to_base_xy((float(px[0]), float(px[1])), homography)
        corners_base[name] = [xy[0], xy[1], float(support_z_m)]

    axis_uv = np.array(slot["axis_uv"], dtype=np.float64)
    half_len = float(slot["length_px"]) * 0.5
    center_arr = np.array(center_px, dtype=np.float64)
    p1_px = center_arr - axis_uv * half_len
    p2_px = center_arr + axis_uv * half_len
    p1_xy = pixel_to_base_xy((float(p1_px[0]), float(p1_px[1])), homography)
    p2_xy = pixel_to_base_xy((float(p2_px[0]), float(p2_px[1])), homography)
    axis_base_xy = np.array([p2_xy[0] - p1_xy[0], p2_xy[1] - p1_xy[1]], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis_base_xy))
    if axis_norm > 1e-9:
        axis_base_xy = axis_base_xy / axis_norm

    return {
        "center_base_xyz_m": center_xyz,
        "corners_base_xyz_m": corners_base,
        "axis_base_xy": [float(axis_base_xy[0]), float(axis_base_xy[1])],
        "support_z_m": float(support_z_m),
        "homography_note": "XY from support-plane homography; Z is a fixed prior, not measured.",
    }


def draw_overlay(color_bgr: np.ndarray, slot: dict) -> np.ndarray:
    out = color_bgr.copy()
    corners = slot["corners_px"]
    pts = np.array(
        [
            corners["top_left"],
            corners["top_right"],
            corners["bottom_right"],
            corners["bottom_left"],
        ],
        dtype=np.int32,
    )
    cv2.polylines(out, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

    cx, cy = slot["center_px"]
    axis_uv = np.array(slot["axis_uv"], dtype=np.float64)
    half = float(slot["length_px"]) * 0.5
    p1 = (int(round(cx - axis_uv[0] * half)), int(round(cy - axis_uv[1] * half)))
    p2 = (int(round(cx + axis_uv[0] * half)), int(round(cy + axis_uv[1] * half)))
    cv2.line(out, p1, p2, color=(0, 255, 255), thickness=2)
    cv2.circle(out, (cx, cy), 8, color=(0, 255, 255), thickness=-1)

    rx, ry, rw, rh = slot["red_box_bbox_px"]
    cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), color=(0, 0, 255), thickness=2)
    cv2.putText(out, "SLOT", (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def analyze_image(
    image_path: Path,
    *,
    homography_path: Path = DEFAULT_HOMOGRAPHY_PATH,
    support_z_m: float = DEFAULT_SUPPORT_Z_M,
    include_p5_place_target: bool = False,
) -> dict:
    color = cv2.imread(str(image_path))
    if color is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    homography = load_homography(homography_path)
    slot_px = detect_empty_slot(color)
    slot_base = slot_pixels_to_base(slot_px, homography, support_z_m=support_z_m)

    result = {
        "source_image": str(image_path).replace("\\", "/"),
        "image_shape_hw": [int(color.shape[0]), int(color.shape[1])],
        "homography": str(homography_path).replace("\\", "/"),
        "detection_mode": "fixed_camera_cv_white_foam_mask_centroid",
        "slot_pixels": slot_px,
        "slot_base": slot_base,
    }
    if include_p5_place_target:
        result = attach_p5_place_target(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect empty foam slot in fixed/top camera image and project to robot base."
    )
    parser.add_argument("image", type=Path, help="Input image (fixed_camera.jpg or top_camera.jpg)")
    parser.add_argument(
        "--homography",
        type=Path,
        default=DEFAULT_HOMOGRAPHY_PATH,
        help="Support-plane homography JSON",
    )
    parser.add_argument(
        "--support-z-m",
        type=float,
        default=DEFAULT_SUPPORT_Z_M,
        help="Robot base Z prior on the support plane (metres)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory to write overlay + result JSON",
    )
    parser.add_argument(
        "--include-p5-place-target",
        action="store_true",
        help="Attach Plan A P5 contact-aligned place standoff flange target",
    )
    args = parser.parse_args()

    result = analyze_image(
        args.image,
        homography_path=args.homography,
        support_z_m=args.support_z_m,
        include_p5_place_target=args.include_p5_place_target,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.image.stem
        if args.image.parent.name not in ("", ".", ".."):
            stem = f"{args.image.parent.name}_{stem}"
        json_path = args.out_dir / f"{stem}_slot_detect.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        color = cv2.imread(str(args.image))
        overlay = draw_overlay(color, result["slot_pixels"])
        overlay_path = args.out_dir / f"{stem}_slot_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        print(f"Wrote {json_path}", file=sys.stderr)
        print(f"Wrote {overlay_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
