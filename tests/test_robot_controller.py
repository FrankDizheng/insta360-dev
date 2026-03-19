import unittest

from nero.controllers import MockRobotController, dispatch_action, get_robot_controller


class RobotControllerTest(unittest.TestCase):
    def test_mock_controller_dispatch_sequence(self):
        robot = MockRobotController(sleep_s=0)
        robot.connect()

        self.assertTrue(dispatch_action(robot, "move_above", "target_a"))
        self.assertTrue(dispatch_action(robot, "lower", "target_a"))
        self.assertTrue(dispatch_action(robot, "grasp"))
        self.assertTrue(dispatch_action(robot, "lift"))
        self.assertTrue(dispatch_action(robot, "release"))
        self.assertEqual(robot.get_status()["last_action"], "release")

    def test_factory_defaults_to_mock(self):
        controller = get_robot_controller("mock")
        self.assertIsInstance(controller, MockRobotController)


if __name__ == "__main__":
    unittest.main()