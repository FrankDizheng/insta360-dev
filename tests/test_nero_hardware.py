#!/usr/bin/env python3
"""NERO hardware validation script — run on Raspberry Pi with arm connected.

Prerequisites:
  1. CAN interface is up:  sudo ip link set can0 up type can bitrate 1000000
  2. Arm is powered on
  3. pyAgxArm is installed:  pip install pyAgxArm

Usage:
  python tests/test_nero_hardware.py              # run all tests
  python tests/test_nero_hardware.py --test connect   # only connection test
  python tests/test_nero_hardware.py --test joints    # only joint movement
  python tests/test_nero_hardware.py --test gripper   # only gripper test
  python tests/test_nero_hardware.py --skip-move      # read-only, no motion

The script is interactive — it pauses before any movement and asks for
confirmation, so it's safe to run even if you're unsure about clearance.
"""

import argparse
import math
import sys
import time

sys.path.insert(0, ".")

from nero.types import ArmState, clamp_joints, NUM_JOINTS


def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def confirm(msg: str) -> bool:
    resp = input(f"\n  >>> {msg} [y/N] ").strip().lower()
    return resp in ("y", "yes")


def test_can_interface():
    """Check that can0 is up."""
    banner("TEST: CAN Interface")
    import subprocess
    result = subprocess.run(["ip", "link", "show", "can0"], capture_output=True, text=True)
    if result.returncode != 0:
        print("  FAIL: can0 not found")
        print("  Fix:  sudo ip link set can0 up type can bitrate 1000000")
        return False
    if "UP" in result.stdout:
        print("  OK: can0 is UP")
        return True
    print("  FAIL: can0 exists but is DOWN")
    print("  Fix:  sudo ip link set can0 up type can bitrate 1000000")
    return False


def test_connect():
    """Connect to arm, read state, disconnect."""
    banner("TEST: Connect & Read State")
    from nero import get_robot_controller

    robot = get_robot_controller("nero")
    print("  Connecting...")
    robot.connect()

    state = robot.get_state()
    print(f"\n  Connected:    {state.connected}")
    print(f"  Enabled:      {state.enabled}")
    print(f"  Mode:         {state.mode}")
    print(f"  Joints (deg): {[f'{a:.2f}' for a in state.joint_angles_deg]}")
    if state.flange_pose:
        fp = state.flange_pose
        print(f"  Flange pose:  x={fp.x:.3f} y={fp.y:.3f} z={fp.z:.3f}")
        print(f"                roll={fp.roll:.3f} pitch={fp.pitch:.3f} yaw={fp.yaw:.3f}")
    print(f"  Gripper avail: {state.gripper.available}")
    if state.gripper.available:
        print(f"  Gripper width: {state.gripper.width}")
        print(f"  Gripper force: {state.gripper.force}")

    if not state.joint_angles_deg:
        print("\n  WARNING: joint angles are empty — set_normal_mode may have failed")

    robot.disconnect()
    print("\n  Disconnected OK")
    return state.connected and len(state.joint_angles_deg) == NUM_JOINTS


def test_joints(skip_move: bool = False):
    """Test each joint with a small ±3° movement."""
    banner("TEST: Joint Movement (±3°)")
    from nero import get_robot_controller

    robot = get_robot_controller("nero")
    robot.connect()

    state = robot.get_state()
    if not state.joint_angles_deg:
        print("  FAIL: cannot read joint angles")
        robot.disconnect()
        return False

    print(f"  Current angles: {[f'{a:.2f}' for a in state.joint_angles_deg]}")

    if skip_move:
        print("  --skip-move: skipping actual motion")
        robot.disconnect()
        return True

    results = {}
    for j in range(NUM_JOINTS):
        joint_name = f"J{j+1}"
        if not confirm(f"Test {joint_name}? (will move ±3°)"):
            results[joint_name] = "skipped"
            continue

        print(f"\n  Testing {joint_name}...")

        # Read fresh angles
        fresh = robot.get_state()
        before = fresh.joint_angles_deg[j]

        # Move +3°
        ok1 = robot.move_joint_relative(j, 3.0)
        time.sleep(0.3)
        after_plus = robot.get_state().joint_angles_deg[j]

        # Move back -3°
        ok2 = robot.move_joint_relative(j, -3.0)
        time.sleep(0.3)
        after_back = robot.get_state().joint_angles_deg[j]

        delta = abs(after_plus - before)
        moved = delta > 0.5

        status = "OK" if moved else "NO RESPONSE"
        results[joint_name] = status
        print(f"  {joint_name}: before={before:.2f} → +3°={after_plus:.2f} → back={after_back:.2f}  [{status}]")

    robot.disconnect()

    banner("Joint Test Summary")
    all_ok = True
    for jname, status in results.items():
        icon = "pass" if status == "OK" else ("skip" if status == "skipped" else "FAIL")
        print(f"  {jname}: {icon}")
        if status == "NO RESPONSE":
            all_ok = False

    return all_ok


def test_gripper(skip_move: bool = False):
    """Test gripper open/close."""
    banner("TEST: Gripper")
    from nero import get_robot_controller

    robot = get_robot_controller("nero")
    robot.connect()

    state = robot.get_state()
    print(f"  Gripper available: {state.gripper.available}")
    print(f"  Gripper width:     {state.gripper.width}")
    print(f"  Gripper enabled:   {state.gripper.enabled}")

    if not state.gripper.available:
        print("\n  Gripper not initialized — SDK could not bind to end effector")
        print("  This may be expected if gripper CAN frames are on internal bus only")
        robot.disconnect()
        return False

    if skip_move:
        print("  --skip-move: skipping actual motion")
        robot.disconnect()
        return True

    if not confirm("Test gripper open/close?"):
        robot.disconnect()
        return True

    print("  Opening gripper (width=0.05)...")
    ok1 = robot.open_gripper(width=0.05, force=1.0)
    print(f"  Result: {'OK' if ok1 else 'FAILED'}")
    time.sleep(1.0)

    print("  Closing gripper...")
    ok2 = robot.close_gripper(force=1.0)
    print(f"  Result: {'OK' if ok2 else 'FAILED'}")

    after = robot.get_state()
    print(f"  Gripper width after close: {after.gripper.width}")

    robot.disconnect()
    return ok1 and ok2


def test_home(skip_move: bool = False):
    """Test homing to [0,0,0,0,0,0,0]."""
    banner("TEST: Home Position")
    from nero import get_robot_controller

    robot = get_robot_controller("nero")
    robot.connect()

    state = robot.get_state()
    print(f"  Current angles: {[f'{a:.2f}' for a in state.joint_angles_deg]}")

    if skip_move:
        print("  --skip-move: skipping actual motion")
        robot.disconnect()
        return True

    if not confirm("Move to HOME [0,0,0,0,0,0,0]? Make sure arm has clearance!"):
        robot.disconnect()
        return True

    print("  Moving to home...")
    ok = robot.home()
    time.sleep(1.0)

    after = robot.get_state()
    print(f"  After home: {[f'{a:.2f}' for a in after.joint_angles_deg]}")

    robot.disconnect()
    return ok


def main():
    parser = argparse.ArgumentParser(description="NERO hardware validation")
    parser.add_argument("--test", choices=["can", "connect", "joints", "gripper", "home"],
                        help="Run a specific test only")
    parser.add_argument("--skip-move", action="store_true",
                        help="Read-only mode — no motion commands")
    args = parser.parse_args()

    results = {}

    if args.test:
        tests = [args.test]
    else:
        tests = ["can", "connect", "joints", "gripper"]

    for t in tests:
        if t == "can":
            results["CAN Interface"] = test_can_interface()
        elif t == "connect":
            results["Connect & State"] = test_connect()
        elif t == "joints":
            results["Joint Movement"] = test_joints(args.skip_move)
        elif t == "gripper":
            results["Gripper"] = test_gripper(args.skip_move)
        elif t == "home":
            results["Home"] = test_home(args.skip_move)

    banner("FINAL RESULTS")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print()

    if all(results.values()):
        print("  All tests passed!")
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"  Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
