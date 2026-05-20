import math
import unittest

import numpy as np

from nero.kinematics import flange_position, forward_kinematics


class NeroKinematicsTest(unittest.TestCase):
    def test_sdk_flange_frame_matches_official_gripper_joint_origin(self):
        # At the all-zero pose, SDK flange is the official gripper joint origin,
        # not link7.  This value is from the URDF/xacro chain.
        q = [0.0] * 7
        flange_xyz = flange_position(q, clamp=False)
        np.testing.assert_allclose(
            flange_xyz,
            np.array([0.0, 0.0, 0.89131], dtype=np.float64),
            atol=2e-5,
        )

    def test_j7_sweep_moves_sdk_flange_on_expected_radius(self):
        q = [0.0] * 7
        p0 = flange_position(q, clamp=False)

        q[6] = math.radians(90.0)
        p90 = flange_position(q, clamp=False)

        # The J7-only sweep showed the SDK flange point rotates about 173.3 mm
        # from the J7 axis under the official xacro gripper joint frame.
        np.testing.assert_allclose(p90 - p0, np.array([-0.1733, 0.0, -0.1733]), atol=2e-5)

    def test_forward_kinematics_exposes_debug_frames(self):
        fk = forward_kinematics([0.0] * 7, clamp=False)
        self.assertIn("link7_T", fk)
        self.assertIn("gripper_flange_T", fk)
        self.assertIn("gripper_base_T", fk)
        self.assertIn("flange_T", fk)
        self.assertEqual(fk["link_positions"].shape, (8, 3))


if __name__ == "__main__":
    unittest.main()
