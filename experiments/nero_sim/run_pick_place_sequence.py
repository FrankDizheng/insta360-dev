"""Run a minimal pick-place style sequence with the SimNERO controller."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nero import Position3D, get_robot_controller  # noqa: E402


def _parse_xyz(values: list[float]) -> Position3D:
    return Position3D(values[0], values[1], values[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SimNERO A-to-B pick-place sequence")
    parser.add_argument(
        "--pick",
        nargs=3,
        type=float,
        default=[0.0, 0.45, 0.18],
        metavar=("X", "Y", "Z"),
        help="Pick point in base frame metres",
    )
    parser.add_argument(
        "--place",
        nargs=3,
        type=float,
        default=[0.1, 0.45, 0.18],
        metavar=("X", "Y", "Z"),
        help="Place point in base frame metres",
    )
    args = parser.parse_args()

    pick = _parse_xyz(args.pick)
    place = _parse_xyz(args.place)

    robot = get_robot_controller("sim-nero")
    robot.connect()

    steps = [
        ("move_above pick", lambda: robot.move_above("pick", pick)),
        ("lower pick", lambda: robot.lower("pick", pick)),
        ("grasp", robot.grasp),
        ("lift", robot.lift),
        ("move_above place", lambda: robot.move_above("place", place)),
        ("lower place", lambda: robot.lower("place", place)),
        ("release", robot.release),
    ]

    print("Initial:", robot.get_state().to_dict())
    for name, fn in steps:
        ok = fn()
        print(f"{name}: {'OK' if ok else 'FAILED'}")
        print(robot.get_state().to_dict())
        if not ok:
            break

    robot.disconnect()


if __name__ == "__main__":
    main()
