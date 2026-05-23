#!/usr/bin/env python3
"""Project top-camera image pixels onto the robot base support plane.

The homography is produced by the ChArUco support-plane calibration workflow
and maps image pixel coordinates ``[u, v]`` directly to robot base ``[x, y]``
coordinates in metres on the calibrated plane.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_HOMOGRAPHY = Path("calibration/results/top_camera_plane_homography_current.json")


def load_homography(path: Path = DEFAULT_HOMOGRAPHY) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    matrix = payload.get("homography_pixel_to_base_xy_m")
    if not matrix:
        raise ValueError(f"homography_pixel_to_base_xy_m missing in {path}")
    h = np.asarray(matrix, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(f"Expected 3x3 homography in {path}, got {h.shape}")
    return h


def pixel_to_base_xy(pixel_uv: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    uv1 = np.array([float(pixel_uv[0]), float(pixel_uv[1]), 1.0], dtype=np.float64)
    xyw = homography @ uv1
    if abs(float(xyw[2])) < 1e-12:
        raise ZeroDivisionError(f"Degenerate homography projection for pixel {pixel_uv}")
    xy = xyw[:2] / xyw[2]
    return float(xy[0]), float(xy[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Project top-camera pixel UV to robot base XY.")
    parser.add_argument("u", type=float, help="Top-camera pixel u coordinate")
    parser.add_argument("v", type=float, help="Top-camera pixel v coordinate")
    parser.add_argument(
        "--homography",
        type=Path,
        default=DEFAULT_HOMOGRAPHY,
        help="Path to top-camera plane homography JSON",
    )
    args = parser.parse_args()

    h = load_homography(args.homography)
    x_m, y_m = pixel_to_base_xy((args.u, args.v), h)
    print(
        json.dumps(
            {
                "pixel_uv": [args.u, args.v],
                "base_xy_m": [x_m, y_m],
                "homography": str(args.homography),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
