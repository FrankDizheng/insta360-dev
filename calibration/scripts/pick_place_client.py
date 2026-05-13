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

PICK_OBJECT_NAME = "outer_bottle_cap"
PLACE_OBJECT_NAME = "empty_slot"

SCAN_POSE_DEG = _env_float_csv(
    "PICK_PLACE_SCAN_POSE_DEG",
    [-6.93, -22.44, -2.76, 67.8, -1.21, 1.37, 71.51],
    expected_len=7,
)

FLANGE_T_CAM = np.array([
    [0.01984594, 0.36872915, -0.92932500, 0.09973553],
    [-0.01531797, -0.92928683, -0.36904113, 0.06435725],
    [-0.99968570, 0.02155934, -0.01279439, -0.01483681],
    [0.0, 0.0, 0.0, 1.0],
])

STANDOFF_MM = 80
GRASP_BELOW_MM = 10
LIFT_Z_MM = 420
BOTTLE_HEIGHT_MM = _env_float("PICK_PLACE_BOTTLE_HEIGHT_MM", 95.0)
PLACE_CLEARANCE_MM = _env_float("PICK_PLACE_PLACE_CLEARANCE_MM", -3.0)
GRIPPER_OPEN_M = 0.06
GRIPPER_FORCE = 1.0
GRIP_CENTER_OFFSET_Y_MM = _env_float("PICK_PLACE_GRIP_CENTER_OFFSET_Y_MM", -15.0)
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


def px_to_base(u: int, v: int, d_mm: float, base_T_cam: np.ndarray, intr: dict) -> np.ndarray:
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    d = d_mm / 1000.0
    p_cam = np.array([(u - cx) * d / fx, (v - cy) * d / fy, d, 1.0])
    return (base_T_cam @ p_cam)[:3]


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
    grasp_template_rpy_rad: list[float] = GRASP_TEMPLATE_RPY_RAD,
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
    prompt = (
        f"This is a {w}x{h} image from a robot camera looking down.\n"
        "Find:\n"
        f"1. {PICK_OBJECT_NAME} - the center of the RED screw cap on the single glass bottle "
        "that is OUTSIDE the red box. Ignore bottles already inside the red box.\n"
        f"2. {PLACE_OBJECT_NAME} - the center of the empty white foam slot inside the red box "
        "where the next bottle should be inserted.\n"
        f'Return ONLY JSON: {{"objects":[{{"name":"{PICK_OBJECT_NAME}","u":0.XX,"v":0.XX}},'
        f'{{"name":"{PLACE_OBJECT_NAME}","u":0.XX,"v":0.XX}}]}}\n'
        "u,v normalized 0-1, top-left=(0,0)."
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
        max_tokens=200,
    )
    raw = resp.choices[0].message.content
    print(f"  VLM raw: {raw}")

    match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
    if not match:
        raise RuntimeError(f"Failed to parse VLM JSON from response: {raw}")
    parsed = json.loads(match.group())
    detected = parsed if isinstance(parsed, list) else parsed.get("objects", [parsed])

    vlm = {}
    for obj in detected:
        name = obj["name"]
        vlm[name] = (int(obj["u"] * w), int(obj["v"] * h))
        print(f"  {name}: pixel {vlm[name]}")

    if PICK_OBJECT_NAME not in vlm:
        raise RuntimeError(f"VLM did not detect {PICK_OBJECT_NAME}")
    if PLACE_OBJECT_NAME not in vlm and not skip_place:
        raise RuntimeError(f"VLM did not detect {PLACE_OBJECT_NAME}")
    timer.lap("VLM inference")

    print("\n" + "=" * 55)
    print("3/6  Seeded refinement + 3D localization")
    print("=" * 55)

    pick_px, pick_py, pick_d_mm = refine_red_cap(color_img, depth_img, *vlm[PICK_OBJECT_NAME], ds=ds)
    if pick_d_mm is None:
        raise RuntimeError("No valid depth for pick bottle cap")
    pick_3d = px_to_base(pick_px, pick_py, pick_d_mm, base_T_cam, intr)
    pick_3d[1] += grip_center_offset_y_mm / 1000.0
    print(f"  Bottle cap: pixel ({pick_px},{pick_py}) depth={pick_d_mm:.0f}mm")
    print(
        f"  Bottle cap 3D: X={pick_3d[0]*1000:.1f} "
        f"Y={pick_3d[1]*1000:.1f} Z={pick_3d[2]*1000:.1f} mm "
        f"(Y offset {grip_center_offset_y_mm}mm)"
    )

    slot_3d = None
    slot_px = slot_py = slot_d_mm = None
    if PLACE_OBJECT_NAME in vlm:
        slot_px, slot_py, slot_d_mm = refine_empty_slot(color_img, depth_img, *vlm[PLACE_OBJECT_NAME], ds=ds)
        if slot_d_mm is not None:
            slot_3d = px_to_base(slot_px, slot_py, slot_d_mm, base_T_cam, intr)
            print(f"  Empty slot: pixel ({slot_px},{slot_py}) depth={slot_d_mm:.0f}mm")
            print(
                f"  Empty slot 3D: X={slot_3d[0]*1000:.1f} "
                f"Y={slot_3d[1]*1000:.1f} Z={slot_3d[2]*1000:.1f} mm"
            )

    grasp_z = pick_3d[2] - GRASP_BELOW_MM / 1000.0
    place_z = None
    if slot_3d is not None and not skip_place:
        place_z = slot_3d[2] + (bottle_height_mm + place_clearance_mm) / 1000.0

    print(f"\n  Grasp Z:  {grasp_z * 1000:.0f} mm | Lift Z: {LIFT_Z_MM} mm", end="")
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
        draw.ellipse([pick_px - 20, pick_py - 20, pick_px + 20, pick_py + 20], outline="lime", width=3)
        draw.text((pick_px + 25, pick_py - 15), "BOTTLE", fill="lime")
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
                "grasp_below_mm": GRASP_BELOW_MM,
                "lift_z_mm": LIFT_Z_MM,
                "bottle_height_mm": bottle_height_mm,
                "place_clearance_mm": place_clearance_mm,
                "grip_center_offset_y_mm": grip_center_offset_y_mm,
                "grasp_template_rpy_rad": grasp_template_rpy_rad,
                "slot_refinement_mode": "seed_connected_component",
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
            },
            "refined": {
                PICK_OBJECT_NAME: {
                    "u_px": pick_px,
                    "v_px": pick_py,
                    "depth_mm": pick_d_mm,
                    "base_xyz_m": pick_3d.tolist(),
                },
                PLACE_OBJECT_NAME: None if slot_3d is None else {
                    "u_px": slot_px,
                    "v_px": slot_py,
                    "depth_mm": slot_d_mm,
                    "base_xyz_m": slot_3d.tolist(),
                },
            },
            "motion_plan": {
                "grasp_z_mm": grasp_z * 1000.0,
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
            "xyz_m": pick_3d.tolist(),
            "rpy_rad": grasp_template_rpy_rad,
            "standoff_mm": STANDOFF_MM,
            "z_safe_mm": LIFT_Z_MM,
        },
        timeout=30,
    )
    print(f"  Above: flange={[f'{v:.3f}' for v in res['flange_xyz_m']]}")
    timer.lap("move_above")

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
        "--grasp-template-rpy-rad",
        type=lambda raw: _parse_float_csv(raw, expected_len=3),
        default=GRASP_TEMPLATE_RPY_RAD,
        help="Comma-separated grasp template roll,pitch,yaw in radians",
    )
    args = parser.parse_args()
    run(
        dry_run=args.dry_run,
        skip_place=args.skip_place,
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
        grasp_template_rpy_rad=args.grasp_template_rpy_rad,
    )
