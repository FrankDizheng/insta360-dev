import os
import tempfile
import unittest

import numpy as np

from nero.perception import MockCamera, get_camera


class MockCameraTest(unittest.TestCase):
    def test_mock_camera_connect_disconnect(self):
        cam = MockCamera()
        self.assertFalse(cam.is_connected)
        cam.connect()
        self.assertTrue(cam.is_connected)
        cam.disconnect()
        self.assertFalse(cam.is_connected)

    def test_mock_camera_capture_rgb_shape(self):
        cam = MockCamera()
        cam.connect()
        rgb = cam.capture_rgb()
        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_mock_camera_capture_depth_shape(self):
        cam = MockCamera()
        cam.connect()
        depth = cam.capture_depth()
        self.assertEqual(depth.shape, (480, 640))
        self.assertEqual(depth.dtype, np.uint16)
        self.assertTrue(np.all(depth == 500))

    def test_mock_camera_capture_rgbd(self):
        cam = MockCamera()
        cam.connect()
        rgb, depth = cam.capture_rgbd()
        self.assertEqual(rgb.shape, (480, 640, 3))
        self.assertEqual(depth.shape, (480, 640))

    def test_mock_camera_pixel_to_3d(self):
        cam = MockCamera()
        cam.connect()
        x, y, z = cam.pixel_to_3d(320, 240)
        self.assertEqual((x, y, z), (0.0, 0.0, 0.5))

    def test_mock_camera_save_rgb(self):
        cam = MockCamera()
        cam.connect()
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            result = cam.save_rgb(path)
            self.assertTrue(os.path.exists(result))
        finally:
            npy_path = os.path.splitext(path)[0] + ".npy"
            for p in (path, npy_path):
                if os.path.exists(p):
                    os.unlink(p)


class FactoryTest(unittest.TestCase):
    def test_factory_mock(self):
        cam = get_camera("mock")
        self.assertIsInstance(cam, MockCamera)

    def test_factory_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_camera("xyz")


if __name__ == "__main__":
    unittest.main()
