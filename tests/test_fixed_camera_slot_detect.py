"""Tests for fixed-camera empty slot detection."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "calibration" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fixed_camera_slot_detect import (  # noqa: E402
    analyze_image,
    detect_empty_slot,
    slot_pixels_to_base,
)
from top_camera_plane_project import load_homography  # noqa: E402

CASE_IMAGE = (
    ROOT
    / "data"
    / "cases"
    / "dual_camera_bottle_moves_2026-05-23_170330"
    / "move_01"
    / "fixed_camera.jpg"
)
CASE_DIR = CASE_IMAGE.parent.parent


@pytest.mark.skipif(not CASE_IMAGE.is_file(), reason="dual-camera case dataset missing")
def test_detect_slot_on_move_01():
    img = cv2.imread(str(CASE_IMAGE))
    slot = detect_empty_slot(img)
    cx, cy = slot["center_px"]
    assert slot.get("center_mode") == "mask_centroid"
    assert 650 <= cx <= 730
    assert 560 <= cy <= 630
    assert slot["length_px"] > 30
    assert slot["width_px"] > 20


@pytest.mark.skipif(not CASE_DIR.is_dir(), reason="dual-camera case dataset missing")
def test_slot_base_xy_stable_when_bottle_moves():
    """move_01..03 move the bottle; fixed-camera slot base XY should stay stable."""
    centers = []
    for i in range(1, 4):
        image = CASE_DIR / f"move_{i:02d}" / "fixed_camera.jpg"
        result = analyze_image(image)
        xy = result["slot_base"]["center_base_xyz_m"][:2]
        centers.append(xy)

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    assert max(xs) - min(xs) < 0.003
    assert max(ys) - min(ys) < 0.003


@pytest.mark.skipif(not CASE_IMAGE.is_file(), reason="dual-camera case dataset missing")
def test_homography_projection_shape():
    img = cv2.imread(str(CASE_IMAGE))
    slot = detect_empty_slot(img)
    homography = load_homography()
    base = slot_pixels_to_base(slot, homography)
    assert len(base["center_base_xyz_m"]) == 3
    assert set(base["corners_base_xyz_m"]) == {
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    }


@pytest.mark.skipif(not CASE_IMAGE.is_file(), reason="dual-camera case dataset missing")
def test_center_is_mask_centroid_not_rect_center():
    img = cv2.imread(str(CASE_IMAGE))
    slot = detect_empty_slot(img)
    center = np.array(slot["center_px"], dtype=float)
    rect_center = np.array(slot["rect_center_px"], dtype=float)
    assert float(np.linalg.norm(center - rect_center)) > 5.0
