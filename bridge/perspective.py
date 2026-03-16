"""Extract flat perspective views from equirectangular 360 images.

Converts the warped equirectangular projection into normal-looking
rectilinear images, as if looking through a regular camera pointed
in a specific direction.
"""

import numpy as np
import cv2
from pathlib import Path


def _unsharp_mask(image: np.ndarray, strength: float = 0.7, blur_size: int = 3) -> np.ndarray:
    """Apply unsharp mask sharpening."""
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 1.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def equirect_to_perspective(
    equirect: np.ndarray,
    yaw: float = 0.0,
    pitch: float = 0.0,
    hfov: float = 90.0,
    out_width: int | None = None,
    out_height: int | None = None,
    sharpen: bool = True,
) -> np.ndarray:
    """Extract a perspective view from an equirectangular image.

    Args:
        equirect: Source equirectangular image (BGR numpy array).
        yaw: Horizontal look direction in degrees. 0=center, +90=right, -90=left, 180=behind.
        pitch: Vertical look direction in degrees. 0=horizon, +90=up, -90=down.
        hfov: Horizontal field of view in degrees (default 90).
        out_width: Output width. If None, auto-computed to match source pixel density.
        out_height: Output height. If None, uses 16:9 ratio from out_width.
        sharpen: Apply unsharp mask to compensate for resampling softness.

    Returns:
        Perspective-projected BGR image.
    """
    eq_h, eq_w = equirect.shape[:2]

    if out_width is None:
        out_width = int(eq_w * hfov / 360.0)
    if out_height is None:
        out_height = int(out_width * 9 / 16)

    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)

    f = (out_width / 2.0) / np.tan(np.radians(hfov) / 2.0)

    u = np.arange(out_width, dtype=np.float64) - out_width / 2.0
    v = np.arange(out_height, dtype=np.float64) - out_height / 2.0
    u, v = np.meshgrid(u, v)

    x = f
    y = u
    z = -v

    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)

    x1 = cos_yaw * x - sin_yaw * y
    y1 = sin_yaw * x + cos_yaw * y
    z1 = z

    x2 = cos_pitch * x1 + sin_pitch * z1
    y2 = y1
    z2 = -sin_pitch * x1 + cos_pitch * z1

    norm = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
    x2 /= norm
    y2 /= norm
    z2 /= norm

    lon = np.arctan2(y2, x2)
    lat = np.arcsin(np.clip(z2, -1.0, 1.0))

    map_x = ((lon / np.pi + 1.0) / 2.0 * eq_w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * eq_h).astype(np.float32)

    map_x = np.mod(map_x, eq_w).astype(np.float32)
    map_y = np.clip(map_y, 0, eq_h - 1).astype(np.float32)

    result = cv2.remap(equirect, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_WRAP)

    if sharpen:
        result = _unsharp_mask(result)

    return result


PRESET_VIEWS = {
    "front":  {"yaw": 0,    "pitch": 0,  "hfov": 90},
    "right":  {"yaw": 90,   "pitch": 0,  "hfov": 90},
    "back":   {"yaw": 180,  "pitch": 0,  "hfov": 90},
    "left":   {"yaw": -90,  "pitch": 0,  "hfov": 90},
    "up":     {"yaw": 0,    "pitch": 45, "hfov": 90},
    "down":   {"yaw": 0,    "pitch": -45,"hfov": 90},
    "wide":   {"yaw": 0,    "pitch": 0,  "hfov": 120},
}


def extract_view(equirect: np.ndarray, view_name: str = "front", **overrides) -> np.ndarray:
    """Extract a named preset view from equirectangular image."""
    if view_name not in PRESET_VIEWS:
        raise ValueError(f"Unknown view '{view_name}'. Choose from: {list(PRESET_VIEWS.keys())}")
    params = {**PRESET_VIEWS[view_name], **overrides}
    return equirect_to_perspective(equirect, **params)


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "../captures/x5_360_capture.jpg"
    equirect = cv2.imread(src)
    if equirect is None:
        print(f"Cannot read: {src}")
        sys.exit(1)

    print(f"Source: {src} ({equirect.shape[1]}x{equirect.shape[0]})")

    out_dir = Path(src).parent
    for name in PRESET_VIEWS:
        out = extract_view(equirect, name)
        out_path = out_dir / f"view_{name}.jpg"
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Saved: {out_path.name} ({out.shape[1]}x{out.shape[0]})")

    print("Done -- check the captures folder for flat perspective views.")
