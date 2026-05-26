import math
import unittest

import numpy as np

from nero.kinematics import forward_kinematics
from nero.geometry import WorkspaceSafetyEnvelope, envelope_penalty
from nero.planning import plan_pose_motion, plan_relaxed_pose_motion
from nero.types import JOINT_LIMITS_DEG, clamp_joints


def _matrix_to_rpy(rot: np.ndarray) -> list[float]:
    pitch = math.asin(max(-1.0, min(1.0, -float(rot[2, 0]))))
    roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
    yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    return [roll, pitch, yaw]


class NeroLimitsAndWorkspaceTest(unittest.TestCase):
    def test_j2_site_operational_limit_includes_scan_pose(self):
        self.assertEqual(JOINT_LIMITS_DEG[1], (-100.0, 100.0))
        self.assertEqual(clamp_joints([0.0, -51.497, 0.0, 0.0, 0.0, 0.0, 0.0])[1], -51.497)
        self.assertEqual(clamp_joints([0.0, -120.0, 0.0, 0.0, 0.0, 0.0, 0.0])[1], -100.0)

    def test_right_wall_penalty_is_reported(self):
        _penalty, details = envelope_penalty(
            [0.0, math.radians(-90.0), 0.0, 0.0, 0.0, 0.0, 0.0],
            workspace=WorkspaceSafetyEnvelope(wall_x_max_m=0.41),
        )
        self.assertIn("wall_penalty", details)
        self.assertIn("min_wall_clearance_m", details)
        self.assertEqual(details["wall_x_max_m"], 0.41)

    def test_pose_ik_solves_known_zero_flange_pose(self):
        fk = forward_kinematics([0.0] * 7, clamp=False)["flange_T"]
        result = plan_pose_motion(
            fk[:3, 3].tolist(),
            _matrix_to_rpy(fk[:3, :3]),
            [0.0] * 7,
            position_tolerance_m=0.006,
            rotation_tolerance_rad=0.10,
        )
        self.assertTrue(result.ok, result)
        self.assertLess(result.position_error_m, 0.006)
        self.assertLess(result.rotation_error_rad, 0.10)

    def test_relaxed_pose_ik_reports_selected_axis_offset(self):
        fk = forward_kinematics([0.0] * 7, clamp=False)["flange_T"]
        result = plan_relaxed_pose_motion(
            fk[:3, 3].tolist(),
            _matrix_to_rpy(fk[:3, :3]),
            [0.0] * 7,
            free_axis="tool_z",
            sweep_rad=math.radians(30.0),
            step_rad=math.radians(15.0),
            position_tolerance_m=0.006,
            rotation_tolerance_rad=0.10,
        )
        self.assertTrue(result.ok, result)
        self.assertEqual(result.geometry_details["relaxed_free_axis"], "tool_z")
        self.assertIn("relaxed_axis_offset_deg", result.geometry_details)


if __name__ == "__main__":
    unittest.main()
