"""Minimal NERO planning demo for the early digital-twin stage.

Usage example:
    python experiments/nero_sim/plan_to_point.py --x 0.22 --y -0.08 --z 0.18
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nero import Position3D, get_robot_controller  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a NERO move to a Cartesian target")
    parser.add_argument("--x", type=float, required=True, help="Target x (m) in base frame")
    parser.add_argument("--y", type=float, required=True, help="Target y (m) in base frame")
    parser.add_argument("--z", type=float, required=True, help="Target z (m) in base frame")
    parser.add_argument(
        "--mode",
        default="move_above",
        choices=["move_above", "lower"],
        help="High-level action to test",
    )
    args = parser.parse_args()

    robot = get_robot_controller("sim-nero")
    robot.connect()

    target = Position3D(args.x, args.y, args.z)
    before = robot.get_state()
    print("Before:", before.to_dict())

    if args.mode == "move_above":
        ok = robot.move_above("demo_target", target)
    else:
        ok = robot.lower("demo_target", target)

    after = robot.get_state()
    print("Success:", ok)
    print("After:", after.to_dict())

    robot.disconnect()


if __name__ == "__main__":
    main()
