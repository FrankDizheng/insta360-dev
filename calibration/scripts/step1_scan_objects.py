#!/usr/bin/env python3
"""Step 1: Capture RGBD scan and locate objects by pixel coordinates.

Captures an aligned color+depth image from the gripper-mounted Orbbec camera,
saves the images, and optionally converts user-supplied pixel coordinates to
3D positions in the robot's base frame using the calibrated hand-eye transform.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/pi")
from handeye_board_runtime import (
    capture_aligned_rgbd,
    connect_robot,
    load_handeye,
    pose_to_transform,
    PersistentAlignedRGBDCapture,
)


def parse_pixel_coords(raw: str) -> dict[str, tuple[int, int]]:
    result = {}
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected 'name:u:v', got '{entry.strip()}'")
        name, u, v = parts[0].strip(), int(parts[1]), int(parts[2])
        result[name] = (u, v)
    return result


def depth_at_pixel(depth_u16: np.ndarray, u: int, v: int, depth_scale: float, window: int = 7) -> float | None:
    """Return depth in metres at (u, v) using a median window, or None if invalid."""
    h, w = depth_u16.shape
    half = window // 2
    v0, v1 = max(0, v - half), min(h, v + half + 1)
    u0, u1 = max(0, u - half), min(w, u + half + 1)
    roi = depth_u16[v0:v1, u0:u1]
    valid = roi[roi > 0]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) * depth_scale / 1000.0


def pixel_to_camera_point(u: int, v: int, depth_m: float, intrinsics: dict) -> np.ndarray:
    x = (u - intrinsics["cx"]) * depth_m / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * depth_m / intrinsics["fy"]
    return np.array([x, y, depth_m], dtype=np.float64)


def depth_to_colormap(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    depth_mm = depth_u16.astype(np.float32) * depth_scale
    valid_mask = depth_mm > 0
    if not np.any(valid_mask):
        return np.zeros((*depth_u16.shape, 3), dtype=np.uint8)
    max_mm = float(np.percentile(depth_mm[valid_mask], 98))
    normalized = np.clip(depth_mm / max(max_mm, 1.0), 0, 1)
    gray = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~valid_mask] = 0
    return colored


def capture_scan(robot, handeye: dict, capture_rgbd) -> dict:
    flange_msg = robot.get_flange_pose()
    if flange_msg is None or flange_msg.msg is None:
        raise RuntimeError("Flange pose unavailable")
    flange_pose = np.array(flange_msg.msg[:6], dtype=np.float64)
    base_T_flange = pose_to_transform(flange_pose)
    base_T_camera = base_T_flange @ handeye["flange_T_camera_np"]

    print("Capturing aligned RGBD ...")
    color_bgr, depth_u16, depth_scale, intrinsics = capture_rgbd()

    camera_z_mm = float(base_T_camera[2, 3]) * 1000.0
    print(f"Camera height above base: {camera_z_mm:.1f} mm")
    return {
        "flange_pose": flange_pose,
        "base_T_camera": base_T_camera,
        "camera_height_mm": round(camera_z_mm, 1),
        "color_bgr": color_bgr,
        "depth_u16": depth_u16,
        "depth_scale": depth_scale,
        "intrinsics": intrinsics,
    }


def write_scan_outputs(scan: dict, out_dir: Path, pixel_coords_raw: str | None, save_images: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    color_path = out_dir / "scan_color.jpg"
    depth_path = out_dir / "scan_depth.jpg"
    if save_images:
        cv2.imwrite(str(color_path), scan["color_bgr"])
        cv2.imwrite(str(depth_path), depth_to_colormap(scan["depth_u16"], scan["depth_scale"]))
        print(f"Saved {color_path}")
        print(f"Saved {depth_path}")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scan_pose_flange": scan["flange_pose"].tolist(),
        "camera_height_mm": scan["camera_height_mm"],
        "objects": {},
    }

    if pixel_coords_raw:
        coords = parse_pixel_coords(pixel_coords_raw)
        for name, (u, v) in coords.items():
            depth_m = depth_at_pixel(scan["depth_u16"], u, v, scan["depth_scale"])
            if depth_m is None or depth_m <= 0:
                print(f"  {name}: pixel ({u},{v}) — no valid depth, skipping")
                continue

            p_cam = pixel_to_camera_point(u, v, depth_m, scan["intrinsics"])
            p_cam_h = np.array([*p_cam, 1.0])
            p_base = (scan["base_T_camera"] @ p_cam_h)[:3]

            depth_mm = round(depth_m * 1000.0, 1)
            output["objects"][name] = {
                "pixel_uv": [u, v],
                "depth_mm": depth_mm,
                "base_xyz_m": [round(float(c), 5) for c in p_base],
            }
            print(f"  {name}: pixel ({u},{v}), depth {depth_mm:.1f} mm, "
                  f"base [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}] m")

    objects_path = out_dir / "objects.json"
    objects_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {objects_path}")

    if not pixel_coords_raw:
        print(
            f"\nNo --pixel-coords provided. Inspect {color_path} to identify objects,\n"
            f"then re-run with --pixel-coords 'name1:u1:v1,name2:u2:v2' "
            f"or use 'pixels name1:u1:v1,name2:u2:v2' in persistent mode."
        )
    else:
        print(f"\n{len(output['objects'])} object(s) located in base frame.")
    return output


def run_single_scan(args, handeye: dict) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = connect_robot()
    if args.scan_height > 0:
        from safe_motion import safe_lift
        print(f"Lifting arm to Z={args.scan_height:.3f} m ...")
        safe_lift(robot, height_m=args.scan_height)

    scan = capture_scan(robot, handeye, capture_rgbd=capture_aligned_rgbd)
    write_scan_outputs(scan, out_dir, args.pixel_coords)


def next_scan_dir(root: Path, start_idx: int) -> tuple[int, Path]:
    scan_idx = start_idx
    while True:
        scan_dir = root / f"scan_{scan_idx:03d}"
        if not scan_dir.exists():
            scan_dir.mkdir(parents=True, exist_ok=False)
            return scan_idx, scan_dir
        scan_idx += 1


def capture_persistent_scan(
    robot,
    handeye: dict,
    camera: PersistentAlignedRGBDCapture,
    output_root: Path,
    next_idx: int,
    pixel_coords_raw: str | None,
    flush_frames: int,
) -> tuple[int, dict, Path]:
    scan_idx, scan_dir = next_scan_dir(output_root, next_idx)
    print(f"\n[persistent] Capturing scan_{scan_idx:03d} ...")
    scan = capture_scan(
        robot,
        handeye,
        capture_rgbd=lambda: camera.capture(flush_frames=flush_frames),
    )
    write_scan_outputs(scan, scan_dir, pixel_coords_raw)
    return scan_idx + 1, scan, scan_dir


def print_persistent_help() -> None:
    print(
        "\n[persistent] Commands:\n"
        "  <enter> or capture          Capture a new scan\n"
        "  capture name:u:v,...        Capture a new scan and locate objects\n"
        "  pixels name:u:v,...         Reuse the last captured frame and update objects.json\n"
        "  help                        Show this help\n"
        "  quit                        Exit persistent mode"
    )


def run_persistent_session(args, handeye: dict) -> None:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    robot = connect_robot()
    if args.scan_height > 0:
        from safe_motion import safe_lift
        print(f"Lifting arm to Z={args.scan_height:.3f} m ...")
        safe_lift(robot, height_m=args.scan_height)

    next_idx = 0
    last_scan = None
    last_scan_dir = None

    print("[persistent] Connecting RGBD camera once and keeping it open ...")
    with PersistentAlignedRGBDCapture(flush_frames=args.flush_frames) as camera:
        next_idx, last_scan, last_scan_dir = capture_persistent_scan(
            robot,
            handeye,
            camera,
            output_root,
            next_idx,
            args.pixel_coords,
            args.flush_frames,
        )
        print_persistent_help()

        while True:
            try:
                raw = input("\n[persistent] command> ").strip()
            except EOFError:
                print("\n[persistent] EOF received, exiting.")
                break

            if raw in {"quit", "q", "exit"}:
                print("[persistent] Exiting.")
                break
            if raw in {"help", "?"}:
                print_persistent_help()
                continue

            try:
                if raw == "" or raw == "capture":
                    next_idx, last_scan, last_scan_dir = capture_persistent_scan(
                        robot, handeye, camera, output_root, next_idx, None, args.flush_frames,
                    )
                    continue

                if raw.startswith("capture "):
                    pixel_coords_raw = raw[len("capture "):].strip()
                    next_idx, last_scan, last_scan_dir = capture_persistent_scan(
                        robot, handeye, camera, output_root, next_idx, pixel_coords_raw, args.flush_frames,
                    )
                    continue

                if raw.startswith("pixels "):
                    pixel_coords_raw = raw[len("pixels "):].strip()
                    if last_scan is None or last_scan_dir is None:
                        print("[persistent] No previous scan available to reuse.")
                        continue
                    print(f"[persistent] Reusing last frame in {last_scan_dir} ...")
                    write_scan_outputs(last_scan, last_scan_dir, pixel_coords_raw, save_images=False)
                    continue

                print("[persistent] Unknown command. Type 'help' for usage.")
            except Exception as exc:
                print(f"[persistent] ERROR: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Capture RGBD scan and locate objects")
    parser.add_argument("--handeye", required=True, help="Path to handeye_result.json")
    parser.add_argument("--output-dir", default=".", help="Directory to save outputs")
    parser.add_argument("--scan-height", type=float, default=0.0,
                        help="If > 0, lift arm to this Z height (metres) before scanning")
    parser.add_argument("--pixel-coords", type=str, default=None,
                        help='Comma-separated "name:u:v" entries, e.g. "bottle:500:300,board:200:400"')
    parser.add_argument("--persistent", action="store_true",
                        help="Keep robot and RGBD camera alive for repeated scans")
    parser.add_argument("--flush-frames", type=int, default=2,
                        help="Extra RGBD frames to discard before each persistent capture")
    args = parser.parse_args()

    handeye = load_handeye(args.handeye)
    if args.persistent:
        run_persistent_session(args, handeye)
    else:
        run_single_scan(args, handeye)


if __name__ == "__main__":
    main()
