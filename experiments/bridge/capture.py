"""Insta360 X5 webcam capture module.

Captures equirectangular 360 frames from the X5 in UVC webcam mode.
The X5 outputs pre-stitched equirectangular video at 2880x1440.
"""

import cv2
import time
import numpy as np
from pathlib import Path

X5_WIDTH = 2880
X5_HEIGHT = 1440
WARMUP_SECONDS = 8
MAX_DEVICE_SCAN = 5


def find_x5_device() -> int | None:
    """Scan video devices and find the Insta360 X5 by checking 2880x1440 support."""
    for i in range(MAX_DEVICE_SCAN):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, X5_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, X5_HEIGHT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w == X5_WIDTH and h == X5_HEIGHT:
            return i
    return None


class Insta360Capture:
    def __init__(self, device_index: int | None = None):
        if device_index is None:
            device_index = find_x5_device()
            if device_index is None:
                raise RuntimeError(
                    "Insta360 X5 not found. Make sure it's connected in USB Camera (webcam) mode."
                )
        self.device_index = device_index
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the webcam stream and wait for it to initialize."""
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video device {self.device_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, X5_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, X5_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"X5 stream opened: {w}x{h} @ device {self.device_index}")

        print(f"Warming up stream ({WARMUP_SECONDS}s)...")
        time.sleep(WARMUP_SECONDS)
        for _ in range(10):
            self._cap.read()

    def capture_frame(self) -> np.ndarray:
        """Capture a single equirectangular frame. Returns BGR numpy array."""
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Stream not open. Call open() first.")

        for attempt in range(60):
            ret, frame = self._cap.read()
            if ret and frame is not None and np.mean(frame) > 1.0:
                return frame
            if attempt % 15 == 14:
                time.sleep(1)

        raise RuntimeError("Failed to get a non-black frame after 60 attempts")

    def capture_to_file(self, path: str | Path, quality: int = 95) -> Path:
        """Capture a frame and save as JPEG. Returns the file path."""
        frame = self.capture_frame()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return path

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    print("Looking for Insta360 X5...")
    idx = find_x5_device()
    if idx is None:
        print("X5 not found!")
        exit(1)
    print(f"Found X5 at device {idx}")

    with Insta360Capture(idx) as cam:
        path = cam.capture_to_file("../captures/bridge_test.jpg")
        print(f"Captured: {path} ({path.stat().st_size} bytes)")
