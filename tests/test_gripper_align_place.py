import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "calibration/scripts"))

from gripper_align_place import (  # noqa: E402
    attach_p5_place_target,
    load_p5_place_standoff_target,
    patch_plan_place_standoff,
    plan_p5_place_standoff_from_q,
)


class GripperAlignPlaceTest(unittest.TestCase):
    def test_load_p5_place_standoff_target(self):
        target = load_p5_place_standoff_target()
        self.assertEqual(len(target["flange_xyz_m"]), 3)
        self.assertAlmostEqual(target["flange_xyz_m"][0], -0.38435, places=4)
        self.assertEqual(target["method"], "plan_a_direct_p5_measurement")

    def test_plan_from_pick_standoff_q(self):
        pick_q = [4.857, -16.502, -19.179, 91.860, -0.141, -4.009, 41.474]
        plan = plan_p5_place_standoff_from_q(pick_q, fixed_rpy=True)
        self.assertTrue(plan["ok"], plan.get("reason"))
        self.assertEqual(len(plan["goal_q_deg"]), 7)
        self.assertLess(plan["position_error_mm"], 5.0)

    def test_patch_plan_place_standoff(self):
        plan_path = (
            REPO
            / "calibration/results/live_rescan_relaxed_plan_2026-05-26_123215"
            / "relaxed_centerline_offline_plan_fixed_grasp.json"
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        p5_plan = plan_p5_place_standoff_from_q(plan["stages"][0]["goal_q_deg"], fixed_rpy=True)
        patched = patch_plan_place_standoff(plan, p5_plan)
        place = next(s for s in patched["stages"] if s["name"] == "place_standoff_p5_calib")
        self.assertEqual(place["method"], "plan_a_direct_p5_measurement")
        self.assertNotEqual(place["target_xyz_m"], plan["stages"][1]["target_xyz_m"])

    def test_attach_p5_place_target(self):
        slot = {
            "slot_base": {
                "center_base_xyz_m": [-0.35262, 0.22200, 0.248],
            }
        }
        out = attach_p5_place_target(slot)
        self.assertIn("place_standoff", out)
        self.assertIn("delta_mm_vs_homography_slot_center", out["place_standoff"])


if __name__ == "__main__":
    unittest.main()
