#!/usr/bin/env python3
"""Local pick-place session client.

Runs on the laptop / local workstation and talks to a lightweight
Raspberry Pi bridge over HTTP.

Design boundary:
- Local machine: session flow, pixel selection, coordinate math, inference
- Raspberry Pi: camera I/O, CAN / SDK, motion execution

VLM inference, when enabled, also runs off-Pi through an OpenAI-compatible
endpoint such as vLLM.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

_TAG = "[session]"
DEFAULT_BRIDGE_URL = os.getenv("PI_BRIDGE_URL", "http://10.13.167.212:8765").rstrip("/")


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


def _parse_scan_pose(raw: str) -> list[float]:
    parts = [s.strip() for s in raw.split(",")]
    if len(parts) != 7:
        raise ValueError(f"scan-pose must have 7 comma-separated values, got {len(parts)}")
    return [float(p) for p in parts]


def _parse_pixel_coords(raw: str) -> dict[str, tuple[int, int]]:
    result = {}
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected 'name:u:v', got '{entry.strip()}'")
        result[parts[0].strip()] = (int(parts[1]), int(parts[2]))
    return result


def _load_handeye(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["flange_T_camera_np"] = np.array(data["handeye"]["flange_T_camera"], dtype=np.float64)
    return data


def _load_tcp_offset(path: str | Path) -> list[float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data["tcp_offset_xyzrpy_m_rad"])


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _pose_to_transform(pose: list[float] | np.ndarray) -> np.ndarray:
    x, y, z, roll, pitch, yaw = np.asarray(pose, dtype=np.float64).tolist()
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


def _depth_at_pixel(depth_u16: np.ndarray, u: int, v: int, depth_scale: float, window: int = 7) -> float | None:
    h, w = depth_u16.shape
    half = window // 2
    roi = depth_u16[max(0, v - half):min(h, v + half + 1), max(0, u - half):min(w, u + half + 1)]
    valid = roi[roi > 0]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) * depth_scale / 1000.0


def _pixel_to_base(u: int, v: int, depth_m: float, intrinsics: dict, base_T_camera: np.ndarray) -> np.ndarray:
    x = (u - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    p_cam_h = np.array([x, y, depth_m, 1.0])
    return (base_T_camera @ p_cam_h)[:3]


def _depth_colormap(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    depth_mm = depth_u16.astype(np.float32) * depth_scale
    valid = depth_mm > 0
    if not np.any(valid):
        return np.zeros((*depth_u16.shape, 3), dtype=np.uint8)
    max_mm = float(np.percentile(depth_mm[valid], 98))
    gray = (np.clip(depth_mm / max(max_mm, 1.0), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _decode_scan_payload(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    color_bytes = base64.b64decode(payload["color_jpeg_b64"])
    color_np = np.frombuffer(color_bytes, dtype=np.uint8)
    color_bgr = cv2.imdecode(color_np, cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise RuntimeError("Failed to decode color image from bridge")

    depth_bytes = base64.b64decode(payload["depth_u16_npy_b64"])
    depth_u16 = np.load(io.BytesIO(depth_bytes))
    return color_bgr, depth_u16


def _next_scan_dir(root: Path, start_idx: int) -> tuple[int, Path]:
    scan_idx = start_idx
    while True:
        scan_dir = root / f"scan_{scan_idx:03d}"
        if not scan_dir.exists():
            scan_dir.mkdir(parents=True, exist_ok=False)
            return scan_idx, scan_dir
        scan_idx += 1


def _extract_components(mask: np.ndarray, depth_u16: np.ndarray, depth_scale: float) -> list[dict]:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    components: list[dict] = []
    for label in range(1, num):
        x, y, w, h, area = stats[label]
        if area <= 0:
            continue
        region = labels == label
        valid = region & (depth_u16 > 0)
        valid_depth = depth_u16[valid]
        depth_m = None
        if valid_depth.size:
            depth_m = float(np.median(valid_depth)) * depth_scale / 1000.0
        components.append({
            "label": label,
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(area),
            "centroid": (float(centroids[label][0]), float(centroids[label][1])),
            "mask": region,
            "valid_count": int(valid_depth.size),
            "depth_m": depth_m,
        })
    return components


def _nearest_valid_component_pixel(component: dict, depth_u16: np.ndarray,
                                   preferred_uv: tuple[float, float]) -> tuple[int, int] | None:
    ys, xs = np.where(component["mask"] & (depth_u16 > 0))
    if len(xs) == 0:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    target = np.array(preferred_uv, dtype=np.float64)
    distances = np.linalg.norm(points - target, axis=1)
    best = points[int(np.argmin(distances))]
    return int(best[0]), int(best[1])


def _suggest_blue_board_pixel(scan: dict) -> tuple[tuple[int, int], str] | None:
    hsv = cv2.cvtColor(scan["color_bgr"], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (85, 60, 40), (125, 255, 255))
    components = _extract_components(mask, scan["depth_u16"], scan["depth_scale"])
    candidates = [
        comp for comp in components
        if comp["area"] > 5000 and comp["valid_count"] > 100 and comp["depth_m"] is not None
    ]
    if not candidates:
        return None
    board = max(candidates, key=lambda comp: comp["area"])
    preferred = board["centroid"]
    uv = _nearest_valid_component_pixel(board, scan["depth_u16"], preferred)
    if uv is None:
        return None
    return uv, (
        f"blue board centroid {board['centroid'][0]:.1f},{board['centroid'][1]:.1f} "
        f"(area={board['area']}, depth~{board['depth_m']*1000:.0f} mm)"
    )


def _suggest_bottle_pixel(scan: dict) -> tuple[tuple[int, int], str] | None:
    color = scan["color_bgr"]
    depth_u16 = scan["depth_u16"]
    depth_scale = scan["depth_scale"]
    h, w = depth_u16.shape
    depth_m = depth_u16.astype(np.float32) * depth_scale / 1000.0

    bg_roi = depth_m[int(h * 0.25):, int(w * 0.35):]
    bg_valid = bg_roi[bg_roi > 0]
    if bg_valid.size:
        background_depth_m = float(np.median(bg_valid))
        fg_mask = (
            (depth_m > 0.10)
            & (depth_m < background_depth_m - 0.025)
        ).astype(np.uint8) * 255
        fg_mask[:, int(w * 0.45):] = 0
        fg_mask[:int(h * 0.35), :] = 0
        fg_components = _extract_components(fg_mask, depth_u16, depth_scale)
        fg_candidates = [
            comp for comp in fg_components
            if comp["area"] > 500 and comp["valid_count"] > 200 and comp["depth_m"] is not None
        ]
        if fg_candidates:
            bottle_fg = max(fg_candidates, key=lambda comp: comp["area"])
            uv = _nearest_valid_component_pixel(bottle_fg, depth_u16, bottle_fg["centroid"])
            if uv is not None:
                return uv, (
                    f"depth foreground centroid {bottle_fg['centroid'][0]:.1f},{bottle_fg['centroid'][1]:.1f} "
                    f"(area={bottle_fg['area']}, object depth~{bottle_fg['depth_m']*1000:.0f} mm, "
                    f"background~{background_depth_m*1000:.0f} mm)"
                )

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    mask[:, int(w * 0.45):] = 0

    components = _extract_components(mask, depth_u16, depth_scale)
    candidates = [
        comp for comp in components
        if comp["area"] > 1000
        and comp["valid_count"] > 100
        and comp["depth_m"] is not None
        and 0.10 <= comp["depth_m"] <= 0.60
        and comp["centroid"][1] > h * 0.45
    ]
    if not candidates:
        return None

    body = max(
        [comp for comp in candidates if comp["centroid"][1] > h * 0.70] or candidates,
        key=lambda comp: comp["area"],
    )
    body_x, body_y, body_w, body_h = body["bbox"]

    upper_candidates = [
        comp for comp in candidates
        if comp is not body
        and comp["centroid"][1] < body["centroid"][1] - 40
        and comp["area"] < body["area"]
    ]

    preferred = body["centroid"]
    reason = (
        f"body centroid {body['centroid'][0]:.1f},{body['centroid'][1]:.1f} "
        f"(area={body['area']}, depth~{body['depth_m']*1000:.0f} mm)"
    )

    if upper_candidates:
        upper = min(
            upper_candidates,
            key=lambda comp: np.linalg.norm(np.array(comp["centroid"]) - np.array(body["centroid"])),
        )
        target_v = float(body_y + max(8, min(20, int(body_h * 0.10))))
        t = (target_v - upper["centroid"][1]) / max(body["centroid"][1] - upper["centroid"][1], 1e-6)
        target_u = upper["centroid"][0] + t * (body["centroid"][0] - upper["centroid"][0])
        preferred = (float(target_u), float(target_v))
        reason = (
            f"axis from upper {upper['centroid'][0]:.1f},{upper['centroid'][1]:.1f} "
            f"to body {body['centroid'][0]:.1f},{body['centroid'][1]:.1f}"
        )

    uv = _nearest_valid_component_pixel(body, depth_u16, preferred)
    if uv is None:
        return None

    return uv, f"{reason} -> using nearest valid pixel {uv[0]},{uv[1]}"


def suggest_pixels(scan: dict) -> dict[str, dict]:
    suggestions: dict[str, dict] = {}

    bottle = _suggest_bottle_pixel(scan)
    if bottle is not None:
        (u, v), reason = bottle
        suggestions["bottle"] = {"pixel_uv": [u, v], "reason": reason}

    board = _suggest_blue_board_pixel(scan)
    if board is not None:
        (u, v), reason = board
        suggestions["blue_board"] = {"pixel_uv": [u, v], "reason": reason}

    return suggestions


class BridgeClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _extract_error(self, resp: requests.Response) -> str:
        try:
            payload = resp.json()
        except Exception:
            return resp.text.strip() or f"HTTP {resp.status_code}"
        return payload.get("error", resp.text.strip() or f"HTTP {resp.status_code}")

    def get(self, path: str) -> dict:
        resp = requests.get(f"{self.base_url}{path}", timeout=60)
        if not resp.ok:
            raise RuntimeError(self._extract_error(resp))
        return resp.json()

    def post(self, path: str, payload: dict | None = None) -> dict:
        resp = requests.post(f"{self.base_url}{path}", json=payload or {}, timeout=120)
        if not resp.ok:
            raise RuntimeError(self._extract_error(resp))
        data = resp.json()
        if not data.get("ok", True):
            raise RuntimeError(data.get("error", f"Bridge call failed: {path}"))
        return data


def print_help() -> None:
    print("""
[session] Commands:
  scan                   Move Pi to scan pose and capture RGBD
  capture                Capture RGBD without moving (arm already at scan pose)
  detect [obj1 obj2...]  Scan + auto-detect objects via VLM
                         Default objects: bottle blue_board
  pixels name:u:v,...    Locate objects from last scan (reuse frame)
  suggest                Suggest bottle and blue_board pixels from last scan
  auto                   Use suggested pixels and write objects.json
  rescan name:u:v,...    Re-capture and locate in one step
  load [path]            Load objects.json directly; default is output-dir/objects.json
  objects                Show currently loaded object positions
  pick  <object>         Move above -> grasp -> lift  (<object> key in objects)
  place <destination>    Transit -> lower -> release -> lift away
  home                   Return robot to all-zero joint configuration
  status                 Query Raspberry Pi bridge status
  help                   Show this help
  quit                   Exit session and close the Pi-side camera session
""")


def cmd_status(client: BridgeClient) -> None:
    try:
        data = client.get("/status")
        if not data.get("session_open"):
            print("  session_open: false")
            return
        if "flange_pose" in data:
            p = data["flange_pose"]
            print(f"  Flange: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m  "
                  f"rpy=[{p[3]:.3f}, {p[4]:.3f}, {p[5]:.3f}] rad")
        if "tcp_pose" in data:
            p = data["tcp_pose"]
            print(f"  TCP:    [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m  "
                  f"rpy=[{p[3]:.3f}, {p[4]:.3f}, {p[5]:.3f}] rad")
        if "arm_status_code" in data:
            print(f"  arm_status: {data['arm_status_code']}")
        if "joint_angles_deg" in data:
            print(f"  joints(deg): {[round(v, 1) for v in data['joint_angles_deg']]}")
        if "gripper_width" in data:
            print(f"  gripper width: {data['gripper_width']:.4f} m")
    except Exception as exc:
        print(f"  status error: {exc}")


def _capture_scan_local(client: BridgeClient, handeye: dict, out_root: Path,
                        next_idx: int, flush_frames: int) -> tuple[int, dict, Path]:
    payload = client.post("/scan", {"flush_frames": flush_frames})
    color_bgr, depth_u16 = _decode_scan_payload(payload)
    scan_idx, scan_dir = _next_scan_dir(out_root, next_idx)

    flange_pose = np.array(payload["flange_pose"], dtype=np.float64)
    base_T_camera = _pose_to_transform(flange_pose) @ handeye["flange_T_camera_np"]
    camera_height_mm = float(base_T_camera[2, 3]) * 1000.0

    color_path = scan_dir / "scan_color.jpg"
    depth_path = scan_dir / "scan_depth.jpg"
    depth_raw_path = scan_dir / "scan_depth_u16.npy"
    cv2.imwrite(str(color_path), color_bgr)
    cv2.imwrite(str(depth_path), _depth_colormap(depth_u16, float(payload["depth_scale"])))
    np.save(depth_raw_path, depth_u16)

    print(f"{_TAG} Saved {color_path}")
    print(f"{_TAG} Saved {depth_path}")
    print(f"{_TAG} Saved {depth_raw_path}")
    print(f"{_TAG} Camera height: {camera_height_mm:.1f} mm above base")

    scan = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "flange_pose": flange_pose,
        "base_T_camera": base_T_camera,
        "camera_height_mm": round(camera_height_mm, 1),
        "color_bgr": color_bgr,
        "depth_u16": depth_u16,
        "depth_scale": float(payload["depth_scale"]),
        "intrinsics": payload["intrinsics"],
        "color_path": color_path,
    }
    return scan_idx + 1, scan, scan_dir


def phase_scan_local(client: BridgeClient, handeye: dict, scan_pose_deg: list[float],
                     out_root: Path, next_idx: int, speed_pct: int,
                     flush_frames: int) -> tuple[int, dict, Path]:
    client.post("/scan_pose", {"scan_pose_deg": scan_pose_deg, "speed_pct": speed_pct})
    return _capture_scan_local(client, handeye, out_root, next_idx, flush_frames)


def _write_objects_payload(scan: dict, objects: dict[str, dict], out_dir: Path) -> dict[str, dict]:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scan_timestamp": scan["timestamp"],
        "scan_pose_flange": scan["flange_pose"].tolist(),
        "camera_height_mm": scan["camera_height_mm"],
        "objects": objects,
    }
    objects_path = out_dir / "objects.json"
    objects_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{_TAG} Saved {objects_path}")
    latest_objects_path = out_dir.parent / "objects.json"
    latest_objects_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if latest_objects_path != objects_path:
        print(f"{_TAG} Saved {latest_objects_path}")
    return objects


def phase_locate_from_coords(scan: dict, coords: dict[str, tuple[int, int]], out_dir: Path) -> dict[str, dict]:
    objects: dict[str, dict] = {}

    for name, (u, v) in coords.items():
        depth_m = _depth_at_pixel(scan["depth_u16"], u, v, scan["depth_scale"])
        if depth_m is None or depth_m <= 0:
            print(f"{_TAG}   {name}: pixel ({u},{v}) - no valid depth, skipping")
            continue
        p_base = _pixel_to_base(u, v, depth_m, scan["intrinsics"], scan["base_T_camera"])
        depth_mm = round(depth_m * 1000.0, 1)
        objects[name] = {
            "pixel_uv": [u, v],
            "depth_mm": depth_mm,
            "base_xyz_m": [round(float(c), 5) for c in p_base],
        }
        print(f"{_TAG}   {name}: depth {depth_mm:.1f} mm  "
              f"base [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}] m")

    return _write_objects_payload(scan, objects, out_dir)


def phase_locate_local(scan: dict, pixel_coords_raw: str, out_dir: Path) -> dict[str, dict]:
    return phase_locate_from_coords(scan, _parse_pixel_coords(pixel_coords_raw), out_dir)


def _apply_local_pick_offset(target_xyz: list[float], flange_pose: list[float] | np.ndarray,
                             offset_local_xyz_m: tuple[float, float, float]) -> list[float]:
    base_T_flange = _pose_to_transform(flange_pose)
    delta_base = base_T_flange[:3, :3] @ np.array(offset_local_xyz_m, dtype=np.float64)
    return (np.array(target_xyz, dtype=np.float64) + delta_base).astype(np.float64).tolist()


_VLM_PROMPT = """\
This image is from a robot arm camera looking down at a workspace.
Identify the pixel coordinates (u=column, v=row) of each object listed.

Objects to find: {object_list}

Rules:
- Return ONLY a JSON object, no explanation, no markdown fences.
- Each key is the object name, each value is [u, v] integers.
- Use the CENTER pixel of the object.
- If an object is not visible, omit its key.

Example output: {{"bottle": [320, 240], "blue_board": [580, 310]}}
"""


def vlm_detect_objects(
    color_bgr: np.ndarray,
    object_names: list[str],
    endpoint: str,
    model: str,
    timeout_s: float = 20.0,
) -> dict[str, tuple[int, int]]:
    """Send image to a VLM endpoint and return {name: (u, v)} detections."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    ok, buf = cv2.imencode(".jpg", color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode scan image for VLM request")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    prompt = _VLM_PROMPT.format(object_list=", ".join(f'"{name}"' for name in object_names))
    client = OpenAI(base_url=endpoint, api_key="none", timeout=timeout_s)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=256,
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content.strip()
    print(f"{_TAG} VLM raw response: {raw_text}")

    if raw_text.startswith("```"):
        raw_text = "\n".join(
            line for line in raw_text.splitlines()
            if not line.startswith("```")
        ).strip()

    parsed = json.loads(raw_text)
    result: dict[str, tuple[int, int]] = {}
    for name, coords in parsed.items():
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            result[name] = (int(coords[0]), int(coords[1]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Local pick-place session over Pi bridge")
    parser.add_argument("--bridge-url", default=DEFAULT_BRIDGE_URL,
                        help="Base URL for the Raspberry Pi bridge")
    parser.add_argument("--handeye", required=True, help="Local path to handeye_result.json")
    parser.add_argument("--tcp", default=None, help="Local path to TCP offset JSON")
    parser.add_argument("--output-dir", default="session",
                        help="Local directory for scan images and objects.json")
    parser.add_argument("--scan-pose", default="-19.4,10.7,4.6,63.0,7.1,1.4,56.6",
                        help="7 joint angles (deg) for the fixed scan position")
    parser.add_argument("--standoff-mm", type=float, default=80,
                        help="Height above object for standoff (mm)")
    parser.add_argument("--grasp-z-mm", type=float, default=300,
                        help="Starting grasp Z in base frame (mm); Pi bridge auto-retries upward on IK fail")
    parser.add_argument("--grasp-retry-step-mm", type=float, default=15.0,
                        help="Z step (mm) for each IK retry attempt on Pi")
    parser.add_argument("--grasp-retry-count", type=int, default=2,
                        help="Max IK retry attempts on Pi")
    parser.add_argument("--lift-z-mm", type=float, default=400,
                        help="Z height to lift to after grasp (mm)")
    parser.add_argument("--place-z-mm", type=float, default=295,
                        help="Z height for placing object (mm)")
    parser.add_argument("--z-safe-mm", type=float, default=400,
                        help="Transit height for horizontal moves (mm)")
    parser.add_argument("--gripper-width", type=float, default=0.06,
                        help="Gripper open width before grasp (m)")
    parser.add_argument("--gripper-force", type=float, default=1.0,
                        help="Gripper close force")
    parser.add_argument("--pick-local-x-offset-mm", type=float, default=0.0,
                        help="Offset pick target along current flange local +X before move_above (mm)")
    parser.add_argument("--pick-local-y-offset-mm", type=float, default=0.0,
                        help="Offset pick target along current flange local +Y before move_above (mm)")
    parser.add_argument("--pick-local-z-offset-mm", type=float, default=0.0,
                        help="Offset pick target along current flange local +Z before move_above (mm)")
    parser.add_argument("--speed", type=int, default=10,
                        help="Robot speed percent (1-100)")
    parser.add_argument("--flush-frames", type=int, default=2,
                        help="Extra RGBD frames to discard before each capture on Pi")
    parser.add_argument("--vlm-endpoint", default=None,
                        help="OpenAI-compatible VLM endpoint URL, e.g. http://192.168.1.10:8000/v1")
    parser.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model name exposed by the VLM server")
    args = parser.parse_args()

    scan_pose_deg = _parse_scan_pose(args.scan_pose)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"{_TAG} Loading local calibration ...")
    handeye = _load_handeye(args.handeye)
    tcp_offset = _load_tcp_offset(args.tcp) if args.tcp else None

    client = BridgeClient(args.bridge_url)
    print(f"{_TAG} Opening Pi bridge session at {args.bridge_url} ...")
    client.post("/session/open", {
        "tcp_offset": tcp_offset,
        "speed_pct": args.speed,
        "flush_frames": args.flush_frames,
    })

    next_scan_idx = 0
    last_scan: dict | None = None
    last_scan_dir: Path | None = None
    objects: dict[str, dict] = {}

    print(f"\n{_TAG} Local session ready. Type 'help' to see commands.\n")
    print_help()

    try:
        while True:
            try:
                raw = input("\n[session]> ").strip()
            except EOFError:
                print(f"\n{_TAG} EOF - exiting.")
                break

            if not raw:
                continue
            if raw in {"quit", "q", "exit"}:
                print(f"{_TAG} Exiting session.")
                break
            if raw in {"help", "?"}:
                print_help()
                continue

            try:
                if raw == "scan":
                    next_scan_idx, last_scan, last_scan_dir = phase_scan_local(
                        client, handeye, scan_pose_deg, out_root, next_scan_idx, args.speed, args.flush_frames,
                    )
                    print(f"{_TAG} Inspect {last_scan['color_path']} then run:\n"
                          f"  suggest\n"
                          f"  auto\n"
                          f"  pixels bottle:U:V,blue_board:U2:V2")

                elif raw == "capture":
                    next_scan_idx, last_scan, last_scan_dir = _capture_scan_local(
                        client, handeye, out_root, next_scan_idx, args.flush_frames,
                    )
                    print(f"{_TAG} Captured (arm not moved). Inspect {last_scan['color_path']} then run:\n"
                          f"  suggest\n"
                          f"  auto\n"
                          f"  pixels bottle:U:V,blue_board:U2:V2")

                elif raw.startswith("detect"):
                    if not args.vlm_endpoint:
                        print(f"{_TAG} VLM not configured. Start session with --vlm-endpoint.")
                        continue
                    detect_names = raw.split()[1:] or ["bottle", "blue_board"]
                    next_scan_idx, last_scan, last_scan_dir = phase_scan_local(
                        client, handeye, scan_pose_deg, out_root, next_scan_idx, args.speed, args.flush_frames,
                    )
                    print(f"{_TAG} Sending image to VLM ({args.vlm_model}) for detection ...")
                    detected = vlm_detect_objects(
                        last_scan["color_bgr"],
                        detect_names,
                        endpoint=args.vlm_endpoint,
                        model=args.vlm_model,
                    )
                    if not detected:
                        print(f"{_TAG} VLM returned no objects. Try 'pixels ...' manually.")
                        continue

                    annotated = last_scan["color_bgr"].copy()
                    for name, (u, v) in detected.items():
                        cv2.circle(annotated, (u, v), 8, (0, 255, 0), 2)
                        cv2.putText(annotated, name, (u + 10, v - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        print(f"{_TAG}   VLM detected '{name}' at pixel ({u}, {v})")
                    detect_path = last_scan_dir / "scan_detected.jpg"
                    cv2.imwrite(str(detect_path), annotated)
                    print(f"{_TAG} Annotated image saved: {detect_path}")

                    objects = phase_locate_from_coords(last_scan, detected, last_scan_dir)
                    print(f"{_TAG} Auto-detection complete. {len(objects)} object(s) located.")

                elif raw.startswith("pixels "):
                    if last_scan is None or last_scan_dir is None:
                        print(f"{_TAG} No scan yet. Run 'scan' first.")
                        continue
                    pixel_raw = raw[len("pixels "):].strip()
                    objects = phase_locate_local(last_scan, pixel_raw, last_scan_dir)

                elif raw == "suggest":
                    if last_scan is None:
                        print(f"{_TAG} No scan yet. Run 'scan' first.")
                        continue
                    suggestions = suggest_pixels(last_scan)
                    if not suggestions:
                        print(f"{_TAG} No reliable suggestions found. Use manual 'pixels ...'.")
                        continue
                    print(f"{_TAG} Suggested pixels:")
                    for name, info in suggestions.items():
                        uv = info["pixel_uv"]
                        print(f"  {name}: {uv[0]}:{uv[1]}  ({info['reason']})")

                elif raw == "auto":
                    if last_scan is None or last_scan_dir is None:
                        print(f"{_TAG} No scan yet. Run 'scan' first.")
                        continue
                    suggestions = suggest_pixels(last_scan)
                    if not suggestions:
                        print(f"{_TAG} No reliable suggestions found. Use manual 'pixels ...'.")
                        continue
                    coords = {name: tuple(info["pixel_uv"]) for name, info in suggestions.items()}
                    print(f"{_TAG} Using suggested pixels: "
                          f"{', '.join(f'{name}:{uv[0]}:{uv[1]}' for name, uv in coords.items())}")
                    objects = phase_locate_from_coords(last_scan, coords, last_scan_dir)

                elif raw.startswith("rescan "):
                    pixel_raw = raw[len("rescan "):].strip()
                    next_scan_idx, last_scan, last_scan_dir = phase_scan_local(
                        client, handeye, scan_pose_deg, out_root, next_scan_idx, args.speed, args.flush_frames,
                    )
                    objects = phase_locate_local(last_scan, pixel_raw, last_scan_dir)

                elif raw.startswith("load"):
                    parts = raw.split(maxsplit=1)
                    obj_path = Path(parts[1].strip()) if len(parts) > 1 else out_root / "objects.json"
                    if not obj_path.exists():
                        print(f"{_TAG} File not found: {obj_path}")
                        continue
                    data = json.loads(obj_path.read_text(encoding="utf-8"))
                    objects = data.get("objects", {})
                    ts = data.get("timestamp") or data.get("scan_timestamp") or "unknown"
                    print(f"{_TAG} Loaded {len(objects)} object(s) from {obj_path} (timestamp: {ts})")
                    for name, obj in objects.items():
                        xyz = obj["base_xyz_m"]
                        print(f"  {name}: [{xyz[0] * 1000:.1f}, {xyz[1] * 1000:.1f}, {xyz[2] * 1000:.1f}] mm")

                elif raw == "objects":
                    if not objects:
                        print(f"{_TAG} No objects loaded. Run 'scan'/'pixels', 'auto', or 'load'.")
                        continue
                    for name, obj in objects.items():
                        xyz = obj["base_xyz_m"]
                        print(f"  {name}: [{xyz[0] * 1000:.1f}, {xyz[1] * 1000:.1f}, {xyz[2] * 1000:.1f}] mm")

                elif raw.startswith("pick"):
                    parts = raw.split()
                    if len(parts) < 2:
                        print(f"{_TAG} Usage: pick <object-name>")
                        continue
                    obj_name = parts[1]
                    if obj_name not in objects:
                        print(f"{_TAG} '{obj_name}' not in objects. Available: {list(objects)}")
                        continue
                    obj_xyz = objects[obj_name]["base_xyz_m"]
                    pick_xyz = list(obj_xyz)
                    local_pick_offset_mm = [
                        float(args.pick_local_x_offset_mm),
                        float(args.pick_local_y_offset_mm),
                        float(args.pick_local_z_offset_mm),
                    ]
                    if any(abs(v) > 1e-6 for v in local_pick_offset_mm):
                        status = client.get("/status")
                        flange_pose = status.get("flange_pose")
                        if flange_pose is None:
                            raise RuntimeError("Bridge status did not include flange_pose for pick offset")
                        pick_xyz = _apply_local_pick_offset(
                            obj_xyz,
                            flange_pose,
                            tuple(v / 1000.0 for v in local_pick_offset_mm),
                        )
                        delta_mm = [
                            round((pick_xyz[i] - obj_xyz[i]) * 1000.0, 1)
                            for i in range(3)
                        ]
                        print(f"{_TAG} Applying local pick offset "
                              f"[{local_pick_offset_mm[0]:.1f}, {local_pick_offset_mm[1]:.1f}, "
                              f"{local_pick_offset_mm[2]:.1f}] mm "
                              f"(X/Y/Z) -> delta base {delta_mm} mm")
                    print(f"{_TAG} Will pick '{obj_name}' at {[round(v * 1000, 1) for v in pick_xyz]} mm")
                    if not _confirm(f"{_TAG} Move above '{obj_name}' and grasp?"):
                        continue
                    client.post("/move_above", {
                        "target_xyz": pick_xyz,
                        "standoff_mm": args.standoff_mm,
                        "z_safe_mm": args.z_safe_mm,
                        "speed_pct": args.speed,
                        "tol_mm": 15.0,
                    })
                    result = client.post("/grasp", {
                        "grasp_z_mm": args.grasp_z_mm,
                        "lift_z_mm": args.lift_z_mm,
                        "gripper_width": args.gripper_width,
                        "gripper_force": args.gripper_force,
                        "speed_pct": args.speed,
                        "retry_step_mm": args.grasp_retry_step_mm,
                        "retry_count": args.grasp_retry_count,
                    })
                    if "actual_grasp_z_mm" in result:
                        print(f"{_TAG} Pick complete at Z={result['actual_grasp_z_mm']:.0f} mm")
                    else:
                        print(f"{_TAG} Pick complete.")

                elif raw.startswith("place"):
                    parts = raw.split()
                    if len(parts) < 2:
                        print(f"{_TAG} Usage: place <destination-name>")
                        continue
                    dest_name = parts[1]
                    if dest_name not in objects:
                        print(f"{_TAG} '{dest_name}' not in objects. Available: {list(objects)}")
                        continue
                    dest_xyz = objects[dest_name]["base_xyz_m"]
                    print(f"{_TAG} Will place at '{dest_name}' at {[round(v * 1000, 1) for v in dest_xyz]} mm")
                    if not _confirm(f"{_TAG} Transit to '{dest_name}' and release?"):
                        continue
                    client.post("/place", {
                        "dest_xyz": dest_xyz,
                        "place_z_mm": args.place_z_mm,
                        "z_safe_mm": args.z_safe_mm,
                        "speed_pct": args.speed,
                    })
                    print(f"{_TAG} Place complete.")

                elif raw == "home":
                    if not _confirm(f"{_TAG} Move to home (all joints zero)?"):
                        continue
                    client.post("/home", {"speed_pct": args.speed})

                elif raw == "status":
                    cmd_status(client)

                else:
                    print(f"{_TAG} Unknown command '{raw}'. Type 'help'.")

            except Exception as exc:
                print(f"{_TAG} ERROR: {exc}")
                print(f"{_TAG} Session remains open. Check 'status' before continuing.")

    finally:
        try:
            print(f"{_TAG} Closing Pi bridge session ...")
            client.post("/session/close", {})
        except Exception as exc:
            print(f"{_TAG} WARNING: failed to close bridge session cleanly: {exc}")
        print(f"{_TAG} Session ended.")


if __name__ == "__main__":
    main()
