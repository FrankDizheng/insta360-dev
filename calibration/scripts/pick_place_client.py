"""
pick_place_client.py — Mac-side pick-and-place orchestrator (v4).

Usage:
    python3 pick_place_client.py                # full auto run
    python3 pick_place_client.py --dry-run      # scan + detect + plan only
    python3 pick_place_client.py --skip-place   # pick only, no placement
    python3 pick_place_client.py --confirm      # pause before each motion step
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
from openai import OpenAI
from PIL import Image, ImageDraw

# ─────────────────────────── CONFIG ───────────────────────────

PI_URL      = "http://100.81.13.58:8765"
VLM_KEY     = "sk-ce34c80757f4459096adf73d53323c68"
VLM_BASE    = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
VLM_MODEL   = "qwen3-vl-flash"

SCAN_POSE_DEG = [-6.93, -22.44, -2.76, 67.8, -1.21, 1.37, 71.51]

FLANGE_T_CAM = np.array([
    [ 0.01984594,  0.36872915, -0.92932500,  0.09973553],
    [-0.01531797, -0.92928683, -0.36904113,  0.06435725],
    [-0.99968570,  0.02155934, -0.01279439, -0.01483681],
    [ 0.0,         0.0,         0.0,         1.0       ],
])

STANDOFF_MM        = 80
GRASP_BELOW_MM     = 10
LIFT_Z_MM          = 420
OBJECT_HEIGHT_MM   = 150
PLACE_CLEARANCE_MM = -5
GRIPPER_OPEN_M     = 0.06
GRIPPER_FORCE      = 1.0

GRIP_CENTER_OFFSET_Y_MM = -15

# ───────────────────────── MATH HELPERS ──────────────────────

def rpy_to_mat(roll, pitch, yaw) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
          @ np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
          @ np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]))


def pose_to_T(pose6) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_mat(*pose6[3:])
    T[:3, 3] = pose6[:3]
    return T


def px_to_base(u, v, d_mm, base_T_cam, intr):
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    d = d_mm / 1000.0
    p_cam = np.array([(u - cx) * d / fx, (v - cy) * d / fy, d, 1.0])
    return (base_T_cam @ p_cam)[:3]

# ───────────────────────── HTTP HELPERS ──────────────────────

_session = requests.Session()

def api(method, path, **kwargs):
    r = getattr(_session, method)(f"{PI_URL}{path}", **kwargs)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {path}: {r.text[:300]}")
    return r.json()

# ───────────────────── COLOR REFINEMENT ─────────────────────

def refine_white(color_img, depth_img, vlm_px, vlm_py, ds=1.0, radius=40):
    h, w = color_img.shape[:2]
    y1, y2 = max(0, vlm_py - radius), min(h, vlm_py + radius)
    x1, x2 = max(0, vlm_px - radius), min(w, vlm_px + radius)
    crop = color_img[y1:y2, x1:x2].astype(float)

    brightness = crop.mean(axis=2)
    sat = crop.max(axis=2) - crop.min(axis=2)
    mask = (brightness > 160) & (sat < 50)
    pts = np.argwhere(mask)

    if len(pts) < 10:
        return vlm_px, vlm_py, None

    cy, cx = pts.mean(axis=0)
    ref_px, ref_py = int(x1 + cx), int(y1 + cy)
    d_roi = depth_img[y1:y2, x1:x2].astype(float) * ds
    d_vals = d_roi[mask]
    d_valid = d_vals[(d_vals > 100) & (d_vals < 2000)]
    depth_mm = float(np.median(d_valid)) if len(d_valid) > 0 else None
    return ref_px, ref_py, depth_mm


def refine_blue(color_img, depth_img, vlm_px, vlm_py, ds=1.0, radius=120):
    h, w = color_img.shape[:2]
    y1, y2 = max(0, vlm_py - radius), min(h, vlm_py + radius)
    x1, x2 = max(0, vlm_px - radius), min(w, vlm_px + radius)
    crop = color_img[y1:y2, x1:x2].astype(float)

    mask = ((crop[:,:,2] > 120)
          & (crop[:,:,2] > crop[:,:,0] + 30)
          & (crop[:,:,2] > crop[:,:,1] + 20)
          & (crop[:,:,0] < 150))
    pts = np.argwhere(mask)

    if len(pts) < 50:
        return vlm_px, vlm_py, None

    cy, cx = pts.mean(axis=0)
    ref_px, ref_py = int(x1 + cx), int(y1 + cy)
    d_roi = depth_img[y1:y2, x1:x2].astype(float) * ds
    d_vals = d_roi[mask]
    d_valid = d_vals[(d_vals > 100) & (d_vals < 2000)]
    depth_mm = float(np.median(d_valid)) if len(d_valid) > 0 else None
    return ref_px, ref_py, depth_mm

# ───────────────────────── TIMING ────────────────────────────

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
        lines.append("  " + "─" * 38)
        for label, dt in self.laps:
            lines.append(f"  {label:<30s} {dt:>5.1f}s")
        lines.append("  " + "─" * 38)
        lines.append(f"  {'TOTAL':<30s} {total:>5.1f}s")
        return "\n".join(lines)

# ───────────────────────── MAIN PIPELINE ─────────────────────

def run(dry_run=False, skip_place=False, confirm=False, save_dir="/tmp/pick_place"):
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer()

    def gate(msg):
        if confirm:
            input(f"\n⚠  {msg} Press ENTER to continue, Ctrl+C to abort: ")

    # ── 1. Scan ──
    print("═" * 55)
    print("1/6  Move to scan pose + capture RGBD")
    print("═" * 55)
    api("post", "/move_j", json={"angles_deg": SCAN_POSE_DEG}, timeout=30)
    timer.lap("move_j → scan pose")

    scan = api("post", "/scan", timeout=30)
    fp6   = scan["flange_pose"]
    intr  = scan["intrinsics"]
    ds    = scan["depth_scale"]
    w, h  = intr["width"], intr["height"]
    print(f"  Flange Z = {fp6[2]*1000:.0f} mm")

    color_bytes = base64.b64decode(scan["image_base64"])
    depth_bytes = base64.b64decode(scan["depth_base64"])
    color_img = np.array(Image.open(io.BytesIO(color_bytes)))
    depth_img = np.array(Image.open(io.BytesIO(depth_bytes)))
    img_b64 = scan["image_base64"]

    base_T_cam = pose_to_T(fp6) @ FLANGE_T_CAM
    timer.lap("scan + decode")

    # ── 2. VLM detection ──
    print("\n" + "═" * 55)
    print("2/6  VLM object detection")
    print("═" * 55)
    client = OpenAI(api_key=VLM_KEY, base_url=VLM_BASE)
    prompt = (
        f"This is a {w}x{h} image from a robot camera looking down.\n"
        "Find:\n"
        "1. bottle_cap - the white screw cap on the TRANSPARENT/CLEAR plastic water bottle "
        "(NOT the small opaque white cylindrical bottle)\n"
        "2. blue_board - center of the blue rectangular tray\n"
        'Return ONLY JSON: {"objects":[{"name":"bottle_cap","u":0.XX,"v":0.XX},'
        '{"name":"blue_board","u":0.XX,"v":0.XX}]}\n'
        "u,v normalized 0-1, top-left=(0,0)."
    )
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        extra_body={"enable_thinking": False},
        max_tokens=200,
    )
    raw = resp.choices[0].message.content
    print(f"  VLM raw: {raw}")
    txt = re.search(r'[\[{].*[\]}]', raw, re.DOTALL).group()
    txt = txt.replace('"v="', '"v":').replace('"u="', '"u":')
    parsed = json.loads(txt)
    detected = parsed if isinstance(parsed, list) else parsed.get("objects", [parsed])

    vlm = {}
    for o in detected:
        vlm[o["name"]] = (int(o["u"] * w), int(o["v"] * h))
        print(f"  {o['name']}: pixel {vlm[o['name']]}")

    if "bottle_cap" not in vlm:
        raise RuntimeError("VLM did not detect bottle_cap")
    if "blue_board" not in vlm and not skip_place:
        raise RuntimeError("VLM did not detect blue_board")
    timer.lap("VLM inference")

    # ── 3. Color refinement + 3D ──
    print("\n" + "═" * 55)
    print("3/6  Color refinement + 3D localization")
    print("═" * 55)

    cap_px, cap_py, cap_d_mm = refine_white(color_img, depth_img, *vlm["bottle_cap"], ds=ds)
    if cap_d_mm is None:
        raise RuntimeError("No valid depth for bottle cap")
    cap_3d = px_to_base(cap_px, cap_py, cap_d_mm, base_T_cam, intr)
    cap_3d[1] += GRIP_CENTER_OFFSET_Y_MM / 1000.0
    print(f"  Cap: pixel ({cap_px},{cap_py}) depth={cap_d_mm:.0f}mm")
    print(f"  Cap 3D: X={cap_3d[0]*1000:.1f} Y={cap_3d[1]*1000:.1f} Z={cap_3d[2]*1000:.1f} mm (Y offset {GRIP_CENTER_OFFSET_Y_MM}mm)")

    board_3d = None
    bd_px = bd_py = 0
    if "blue_board" in vlm:
        bd_px, bd_py, bd_d_mm = refine_blue(color_img, depth_img, *vlm["blue_board"], ds=ds)
        if bd_d_mm is not None:
            board_3d = px_to_base(bd_px, bd_py, bd_d_mm, base_T_cam, intr)
            print(f"  Board: pixel ({bd_px},{bd_py}) depth={bd_d_mm:.0f}mm")
            print(f"  Board 3D: X={board_3d[0]*1000:.1f} Y={board_3d[1]*1000:.1f} Z={board_3d[2]*1000:.1f} mm")

    grasp_z = cap_3d[2] - GRASP_BELOW_MM / 1000.0
    place_z = None
    if board_3d is not None and not skip_place:
        place_z = board_3d[2] + (OBJECT_HEIGHT_MM + PLACE_CLEARANCE_MM) / 1000.0

    print(f"\n  Grasp Z:  {grasp_z*1000:.0f} mm | Lift Z: {LIFT_Z_MM} mm", end="")
    if place_z is not None:
        print(f" | Place Z: {place_z*1000:.0f} mm")
    else:
        print()
    timer.lap("CV refine + 3D")

    if dry_run:
        img = Image.open(io.BytesIO(color_bytes))
        draw = ImageDraw.Draw(img)
        draw.ellipse([cap_px-20, cap_py-20, cap_px+20, cap_py+20], outline='lime', width=3)
        draw.text((cap_px+25, cap_py-15), 'CAP', fill='lime')
        if board_3d is not None:
            draw.ellipse([bd_px-25, bd_py-25, bd_px+25, bd_py+25], outline='cyan', width=3)
            draw.text((bd_px+30, bd_py-15), 'BOARD', fill='cyan')
        ann_path = out / "annotated.jpg"
        img.save(ann_path)
        print(f"\n  Annotated: {ann_path}")
        print("  DRY-RUN complete. No motion commands sent.")
        print("\n" + timer.summary())
        return

    # ── 4. Move above + Grasp ──
    gate("Start pick?")
    print("\n" + "═" * 55)
    print("4/6  Move above → Grasp → Lift")
    print("═" * 55)
    res = api("post", "/move_above", json={
        "xyz_m": cap_3d.tolist(),
        "standoff_mm": STANDOFF_MM,
        "z_safe_mm": LIFT_Z_MM,
    }, timeout=30)
    print(f"  Above: flange={[f'{v:.3f}' for v in res['flange_xyz_m']]}")
    timer.lap("move_above")

    gate("Grasp?")
    res = api("post", "/grasp", json={
        "grasp_z_mm": grasp_z * 1000,
        "lift_z_mm": LIFT_Z_MM,
    }, timeout=45)
    grasped = res.get("grasped", False)
    print(f"  Grasp: grip_width={res.get('grip_width_m',0)*1000:.1f}mm  grasped={grasped}")
    timer.lap("grasp + lift")

    if not grasped:
        print("  ✗ Grasp failed — aborting.")
        api("post", "/move_j", json={"angles_deg": SCAN_POSE_DEG}, timeout=30)
        print("\n" + timer.summary())
        return

    if skip_place or board_3d is None:
        print("  Pick done (skip-place). Returning to scan pose.")
        api("post", "/move_j", json={"angles_deg": SCAN_POSE_DEG}, timeout=30)
        timer.lap("return scan pose")
        print("\n" + timer.summary())
        return

    # ── 5. Place ──
    gate("Place on blue board?")
    print("\n" + "═" * 55)
    print("5/6  Place on blue board")
    print("═" * 55)
    res = api("post", "/place", json={
        "xyz_m": board_3d.tolist(),
        "place_z_mm": place_z * 1000,
        "z_safe_mm": LIFT_Z_MM,
    }, timeout=60)
    print(f"  Place: z={res.get('place_z_mm',0):.0f}mm  status={res.get('status')}")
    timer.lap("place")

    # ── 6. Home ──
    print("\n" + "═" * 55)
    print("6/6  Return to scan pose")
    print("═" * 55)
    api("post", "/move_j", json={"angles_deg": SCAN_POSE_DEG}, timeout=60)
    timer.lap("return scan pose")

    print("\n✓ Pick-and-place complete!")
    print("\n" + timer.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick-and-place orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Scan + detect only")
    parser.add_argument("--skip-place", action="store_true", help="Pick only, no place")
    parser.add_argument("--confirm", action="store_true", help="Pause before each motion step")
    parser.add_argument("--save-dir", default="/tmp/pick_place")
    args = parser.parse_args()
    run(dry_run=args.dry_run, skip_place=args.skip_place,
        confirm=args.confirm, save_dir=args.save_dir)
