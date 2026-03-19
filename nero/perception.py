import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import pyorbbecsdk as ob
    _HAS_ORBBEC = True
except ImportError:
    _HAS_ORBBEC = False


class CameraInterface(ABC):
    @abstractmethod
    def connect(self): ...

    @abstractmethod
    def disconnect(self): ...

    @abstractmethod
    def capture_rgb(self) -> np.ndarray: ...

    @abstractmethod
    def capture_depth(self) -> np.ndarray: ...

    @abstractmethod
    def capture_rgbd(self) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def pixel_to_3d(
        self, u: int, v: int, depth_mm: float | None = None
    ) -> tuple[float, float, float]: ...

    @abstractmethod
    def save_rgb(self, path: str) -> str: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...


class MockCamera(CameraInterface):
    WIDTH = 640
    HEIGHT = 480
    MOCK_DEPTH_MM = 500

    def __init__(self):
        self._connected = False

    def connect(self):
        print("[Camera] Connecting (mock mode)...")
        self._connected = True
        print("[Camera] Connected (mock mode)")

    def disconnect(self):
        print("[Camera] Disconnecting (mock mode)")
        self._connected = False

    def capture_rgb(self) -> np.ndarray:
        print("[Camera] Capturing RGB (mock)")
        return np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

    def capture_depth(self) -> np.ndarray:
        print("[Camera] Capturing depth (mock)")
        return np.full(
            (self.HEIGHT, self.WIDTH), self.MOCK_DEPTH_MM, dtype=np.uint16
        )

    def capture_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        return self.capture_rgb(), self.capture_depth()

    def pixel_to_3d(
        self, u: int, v: int, depth_mm: float | None = None
    ) -> tuple[float, float, float]:
        print(f"[Camera] pixel_to_3d({u}, {v}) (mock)")
        return (0.0, 0.0, 0.5)

    def save_rgb(self, path: str) -> str:
        img = self.capture_rgb()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_CV2:
            cv2.imwrite(str(out), img)
        else:
            # Raw fallback: write as .npy
            out = out.with_suffix(".npy")
            np.save(str(out), img)
        print(f"[Camera] Saved RGB to {out}")
        return str(out)

    @property
    def is_connected(self) -> bool:
        return self._connected


class OrbbecCamera(CameraInterface):
    def __init__(self):
        if not _HAS_ORBBEC:
            raise ImportError(
                "pyorbbecsdk is not installed. "
                "Install it to use OrbbecCamera, or use mock mode."
            )
        self._pipeline: "ob.Pipeline | None" = None
        self._config: "ob.Config | None" = None
        self._connected = False
        self._last_depth: np.ndarray | None = None
        self._depth_intrinsics = None
        self._depth_extrinsics = None
        self._depth_scale: float = 1.0

    def connect(self):
        print("[Camera] Connecting to Orbbec Gemini 335...")
        self._pipeline = ob.Pipeline()
        self._config = ob.Config()
        self._config.enable_stream(ob.OBSensorType.DEPTH_SENSOR)
        self._config.enable_stream(ob.OBSensorType.COLOR_SENSOR)
        self._pipeline.start(self._config)

        # Allow auto-exposure to settle
        time.sleep(0.5)

        self._store_intrinsics()
        self._connected = True
        print("[Camera] Orbbec camera connected")

    def _store_intrinsics(self):
        profile_list = self._pipeline.get_stream_profile_list(
            ob.OBSensorType.DEPTH_SENSOR
        )
        depth_profile = profile_list.get_default_video_stream_profile()
        self._depth_intrinsics = depth_profile.get_intrinsic()
        self._depth_extrinsics = depth_profile.get_extrinsic_to(depth_profile)

    def disconnect(self):
        print("[Camera] Disconnecting Orbbec camera...")
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        self._connected = False
        self._last_depth = None
        print("[Camera] Orbbec camera disconnected")

    def _wait_for_frameset(self):
        frameset = self._pipeline.wait_for_frames(1000)
        if frameset is None:
            raise RuntimeError("Timed out waiting for frames from Orbbec camera")
        return frameset

    def capture_rgb(self) -> np.ndarray:
        frameset = self._wait_for_frameset()
        color_frame = frameset.get_color_frame()
        if color_frame is None:
            raise RuntimeError("No color frame received")
        w = color_frame.get_width()
        h = color_frame.get_height()
        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        fmt = color_frame.get_format()
        if fmt == ob.OBFormat.RGB:
            img = data.reshape((h, w, 3))
            if _HAS_CV2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif fmt == ob.OBFormat.BGRA:
            img = data.reshape((h, w, 4))[:, :, :3]
        elif fmt == ob.OBFormat.RGBA:
            img = data.reshape((h, w, 4))[:, :, :3]
            if _HAS_CV2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = data.reshape((h, w, 3))
        return img

    def capture_depth(self) -> np.ndarray:
        frameset = self._wait_for_frameset()
        depth_frame = frameset.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError("No depth frame received")
        w = depth_frame.get_width()
        h = depth_frame.get_height()
        self._depth_scale = depth_frame.get_depth_scale()
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
        self._last_depth = depth
        return depth

    def capture_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        frameset = self._wait_for_frameset()

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        if color_frame is None or depth_frame is None:
            raise RuntimeError("Incomplete frameset (missing color or depth)")

        cw, ch = color_frame.get_width(), color_frame.get_height()
        color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        fmt = color_frame.get_format()
        if fmt == ob.OBFormat.RGB:
            rgb = color_data.reshape((ch, cw, 3))
            if _HAS_CV2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif fmt in (ob.OBFormat.BGRA, ob.OBFormat.RGBA):
            rgb = color_data.reshape((ch, cw, 4))[:, :, :3]
            if fmt == ob.OBFormat.RGBA and _HAS_CV2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            rgb = color_data.reshape((ch, cw, 3))

        dw, dh = depth_frame.get_width(), depth_frame.get_height()
        self._depth_scale = depth_frame.get_depth_scale()
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((dh, dw))
        self._last_depth = depth

        return rgb, depth

    def pixel_to_3d(
        self, u: int, v: int, depth_mm: float | None = None
    ) -> tuple[float, float, float]:
        if depth_mm is None:
            if self._last_depth is None:
                raise RuntimeError(
                    "No depth frame available. Call capture_depth() or capture_rgbd() first."
                )
            depth_mm = float(self._last_depth[v, u]) * self._depth_scale
        if self._depth_intrinsics is None:
            raise RuntimeError("Camera intrinsics not available. Call connect() first.")
        point = ob.transformation2dto3d(
            ob.OBPoint2f(float(u), float(v)),
            depth_mm,
            self._depth_intrinsics,
            self._depth_extrinsics,
        )
        return (point.x, point.y, point.z)

    def save_rgb(self, path: str) -> str:
        img = self.capture_rgb()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_CV2:
            cv2.imwrite(str(out), img)
        else:
            out = out.with_suffix(".npy")
            np.save(str(out), img)
        print(f"[Camera] Saved RGB to {out}")
        return str(out)

    @property
    def is_connected(self) -> bool:
        return self._connected


def get_camera(mode: str | None = None) -> CameraInterface:
    selected = (mode or os.getenv("CAMERA_MODE", "mock")).lower()
    if selected == "mock":
        return MockCamera()
    if selected == "orbbec":
        return OrbbecCamera()
    raise ValueError(f"Unknown camera mode: {selected!r}. Use 'mock' or 'orbbec'.")
