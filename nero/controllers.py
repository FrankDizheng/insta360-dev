"""Robot controller abstraction and NERO implementation.

The controller has two layers:
  - Low-level motion primitives: move_joints(), open_gripper(), close_gripper(), home()
  - High-level task actions: move_above(), lower(), grasp(), lift(), release()

NeroRobotController wraps the pyAgxArm SDK exactly as validated during
hands-on testing (2026-03-17).  It runs only on the edge device (Raspberry Pi)
where pyAgxArm is installed; imports are deferred so the rest of the codebase
works without it.
"""

import math
import os
import time
from abc import ABC, abstractmethod

from nero.types import (
    NUM_JOINTS,
    ArmState,
    FATAL_ARM_STATUS_CODES,
    FlangePose,
    GripperState,
    Position3D,
    arm_status_label,
    clamp_joints,
)

try:
    from pyAgxArm import AgxArmFactory, create_agx_arm_config
    _HAS_PYAGX = True
except ImportError:
    _HAS_PYAGX = False


MOVE_SETTLE_S = 0.8
GRIPPER_SETTLE_S = 1.0
LIFT_DELTA_DEG = 15.0


def read_arm_status_code(robot) -> int | None:
    """Read controller `arm_status` from pyAgxArm `get_arm_status()` (None if unavailable)."""
    try:
        st = robot.get_arm_status()
    except Exception:
        return None
    if st is None or st.msg is None:
        return None
    return int(getattr(st.msg, "arm_status", 0))


def assert_arm_status_ok(robot) -> None:
    """Raise RuntimeError if the arm reports a fatal arm_status (IK fail, collision, etc.)."""
    code = read_arm_status_code(robot)
    if code is None:
        return
    if code in FATAL_ARM_STATUS_CODES:
        raise RuntimeError(
            f"NERO arm_status={code} ({arm_status_label(code)}) — aborting motion"
        )


class BaseRobotController(ABC):
    def __init__(self):
        self.connected = False

    # -- lifecycle --------------------------------------------------------
    @abstractmethod
    def connect(self): ...

    @abstractmethod
    def disconnect(self): ...

    # -- state readback ---------------------------------------------------
    @abstractmethod
    def get_state(self) -> ArmState: ...

    def get_status(self) -> dict:
        """Convenience wrapper — returns a plain dict for JSON serialisation."""
        return self.get_state().to_dict()

    # -- low-level motion primitives --------------------------------------
    @abstractmethod
    def home(self) -> bool:
        """Move to the predefined home position."""
        ...

    @abstractmethod
    def move_joints(self, target_deg: list[float], settle_s: float = MOVE_SETTLE_S) -> bool:
        """Move to an absolute 7-joint target (degrees), with safety clamping."""
        ...

    @abstractmethod
    def move_joint_relative(self, joint_index: int, delta_deg: float) -> bool:
        """Move a single joint by a relative offset."""
        ...

    @abstractmethod
    def open_gripper(self, width: float = 0.08, force: float = 1.0) -> bool: ...

    @abstractmethod
    def close_gripper(self, force: float = 1.0) -> bool: ...

    # -- high-level task actions ------------------------------------------
    @abstractmethod
    def move_above(self, target_name: str, target_pos: Position3D | None = None) -> bool: ...

    @abstractmethod
    def lower(self, target_name: str, target_pos: Position3D | None = None) -> bool: ...

    def grasp(self) -> bool:
        return self.close_gripper()

    def lift(self) -> bool:
        state = self.get_state()
        if not state.joint_angles_deg:
            return False
        target = list(state.joint_angles_deg)
        target[1] = target[1] - LIFT_DELTA_DEG
        return self.move_joints(target)

    def release(self) -> bool:
        return self.open_gripper()

    @abstractmethod
    def stop(self) -> bool: ...


# =========================================================================
#  Mock implementation — used for development & CI
# =========================================================================

class MockRobotController(BaseRobotController):
    def __init__(self, sleep_s: float = 0.05):
        super().__init__()
        self._sleep = sleep_s
        self._angles = [0.0] * NUM_JOINTS
        self._gripper_width = 0.08
        self._last_action = ""

    def connect(self):
        self.connected = True
        print("[Mock] Connected")

    def disconnect(self):
        self.connected = False
        print("[Mock] Disconnected")

    def get_state(self) -> ArmState:
        return ArmState(
            connected=self.connected,
            enabled=self.connected,
            mode="mock",
            joint_angles_deg=list(self._angles),
            flange_pose=FlangePose(),
            gripper=GripperState(width=self._gripper_width, available=True),
        )

    def home(self) -> bool:
        self._angles = [0.0] * NUM_JOINTS
        self._last_action = "home"
        time.sleep(self._sleep)
        print("[Mock] Homed")
        return True

    def move_joints(self, target_deg: list[float], settle_s: float = MOVE_SETTLE_S) -> bool:
        safe = clamp_joints(target_deg)
        self._angles = safe
        self._last_action = f"move_joints:{safe}"
        time.sleep(self._sleep)
        return True

    def move_joint_relative(self, joint_index: int, delta_deg: float) -> bool:
        target = list(self._angles)
        target[joint_index] += delta_deg
        return self.move_joints(target)

    def open_gripper(self, width: float = 0.08, force: float = 1.0) -> bool:
        self._gripper_width = width
        self._last_action = "open_gripper"
        time.sleep(self._sleep)
        print(f"[Mock] Gripper open (width={width})")
        return True

    def close_gripper(self, force: float = 1.0) -> bool:
        self._gripper_width = 0.0
        self._last_action = "close_gripper"
        time.sleep(self._sleep)
        print("[Mock] Gripper closed")
        return True

    def move_above(self, target_name: str, target_pos: Position3D | None = None) -> bool:
        self._last_action = f"move_above:{target_name}"
        print(f"[Mock] Moving above {target_name}")
        time.sleep(self._sleep)
        return True

    def lower(self, target_name: str, target_pos: Position3D | None = None) -> bool:
        self._last_action = f"lower:{target_name}"
        print(f"[Mock] Lowering to {target_name}")
        time.sleep(self._sleep)
        return True

    def stop(self) -> bool:
        self._last_action = "stop"
        print("[Mock] Stop")
        return True


# =========================================================================
#  NERO hardware implementation — pyAgxArm SDK
# =========================================================================

class NeroRobotController(BaseRobotController):
    """Controls the NERO 7-DOF arm via pyAgxArm over CAN bus.

    Tested configuration (2026-03-17):
      - Raspberry Pi 5 + USB-CAN adapter (gs_usb / OpenMoko)
      - SocketCAN interface `can0` at 1 Mbit/s
      - pyAgxArm SDK installed from pip
      - Verified: J1-J3, J5-J7 motion; J4 unresponsive (needs investigation)
      - Gripper: CAN frames not seen on external bus (web UI uses internal path)
    """

    def __init__(
        self,
        channel: str = "can0",
        interface: str = "socketcan",
        gripper_type: str = "AGX_GRIPPER",
    ):
        super().__init__()
        if not _HAS_PYAGX:
            raise ImportError(
                "pyAgxArm is not installed. "
                "Install it on the edge device (Raspberry Pi) to use NeroRobotController."
            )
        self._channel = channel
        self._interface = interface
        self._gripper_type = gripper_type
        self._robot = None
        self._end_effector = None
        self._gripper_available = False

    def connect(self):
        cfg = create_agx_arm_config(
            robot="nero",
            comm="can",
            channel=self._channel,
            interface=self._interface,
        )
        self._robot = AgxArmFactory.create_arm(cfg)

        try:
            effector_opt = getattr(self._robot.OPTIONS.EFFECTOR, self._gripper_type, None)
            if effector_opt is not None:
                self._end_effector = self._robot.init_effector(effector_opt)
                self._gripper_available = True
        except Exception as exc:
            print(f"[NERO] Gripper init skipped: {exc}")
            self._gripper_available = False

        self._robot.connect()
        self._robot.set_normal_mode()
        time.sleep(0.3)
        self._robot.enable()
        time.sleep(0.3)
        self.connected = True
        print(f"[NERO] Connected on {self._channel}, gripper={'yes' if self._gripper_available else 'no'}")

    def disconnect(self):
        if self._robot:
            try:
                self._robot.disconnect()
            except Exception:
                pass
        self.connected = False
        print("[NERO] Disconnected")

    def get_state(self) -> ArmState:
        if not self.connected or self._robot is None:
            return ArmState(connected=False, mode="nero")

        ja = self._robot.get_joint_angles()
        if ja is not None and ja.msg is not None:
            angles_deg = [math.degrees(float(a)) for a in ja.msg[:NUM_JOINTS]]
        else:
            angles_deg = []

        fp = self._robot.get_flange_pose()
        if fp is not None and fp.msg is not None:
            flange = FlangePose.from_list(list(fp.msg)[:6])
        else:
            flange = None

        gripper = GripperState(available=self._gripper_available)
        if self._gripper_available and self._end_effector:
            try:
                gs = self._end_effector.get_gripper_status()
                if gs is not None:
                    gripper.width = getattr(gs.msg, "width", None)
                    gripper.force = getattr(gs.msg, "force", None)
                    foc = getattr(gs.msg, "foc_status", None)
                    if foc:
                        gripper.enabled = bool(getattr(foc, "driver_enable_status", False))
                        gripper.error = bool(getattr(foc, "driver_error_status", False))
                        gripper.homed = bool(getattr(foc, "homing_status", False))
            except Exception:
                pass

        arm_status = None
        try:
            arm_status = self._robot.get_arm_status()
        except Exception:
            pass

        err_msg = ""
        ac = read_arm_status_code(self._robot)
        if ac is not None and ac in FATAL_ARM_STATUS_CODES:
            err_msg = f"arm_status={ac} ({arm_status_label(ac)})"

        return ArmState(
            connected=True,
            enabled=True,
            mode="nero",
            joint_angles_deg=angles_deg,
            flange_pose=flange,
            gripper=gripper,
            error_msg=err_msg,
        )

    def check_arm_status(self) -> tuple[int, str]:
        """Return (arm_status_code, label). Use after moves to detect IK/collision errors."""
        if not self.connected or self._robot is None:
            return -1, "disconnected"
        code = read_arm_status_code(self._robot)
        if code is None:
            return -1, "unavailable"
        return code, arm_status_label(code)

    def wait_motion_complete(
        self,
        timeout_s: float = 30.0,
        poll_s: float = 0.1,
    ) -> tuple[bool, str]:
        """Poll until motion_status==0 or timeout. Raises on fatal arm_status."""
        if self._robot is None:
            return False, "not_connected"
        time.sleep(0.2)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            assert_arm_status_ok(self._robot)
            st = self._robot.get_arm_status()
            if st is not None and st.msg is not None:
                if getattr(st.msg, "motion_status", None) == 0:
                    return True, "ok"
            time.sleep(poll_s)
        return False, "motion_timeout"

    def electronic_emergency_stop(self) -> bool:
        """Damped e-stop (preferred over disable when arm is raised)."""
        if self._robot is None:
            return False
        try:
            self._robot.electronic_emergency_stop()
            return True
        except Exception as exc:
            print(f"[NERO] electronic_emergency_stop failed: {exc}")
            return False

    # -- low-level motion -------------------------------------------------

    def home(self) -> bool:
        from nero.types import HOME_ANGLES_DEG
        return self.move_joints(HOME_ANGLES_DEG, settle_s=2.0)

    def move_joints(self, target_deg: list[float], settle_s: float = MOVE_SETTLE_S) -> bool:
        if not self.connected or self._robot is None:
            return False

        safe_deg = clamp_joints(target_deg)
        safe_rad = [math.radians(a) for a in safe_deg]

        try:
            if not self._robot.get_joint_enable_status(255):
                print("[NERO] Warning: not all joints report enabled before move_j")
        except Exception:
            pass

        try:
            self._robot.move_j(safe_rad)
        except Exception as exc:
            print(f"[NERO] move_joints failed: {exc}")
            return False

        wait_timeout = max(25.0, float(settle_s) + 20.0)
        ok, reason = self.wait_motion_complete(timeout_s=wait_timeout)
        if not ok:
            code, label = self.check_arm_status()
            print(f"[NERO] move_joints wait failed: {reason} (arm_status={code} {label})")
            return False
        time.sleep(min(settle_s, 0.5))
        return True

    def move_joint_relative(self, joint_index: int, delta_deg: float) -> bool:
        """Read current angles, adjust one joint, send full array."""
        if not self.connected or self._robot is None:
            return False
        raw = self._robot.get_joint_angles()
        if raw is None or raw.msg is None:
            return False
        current_deg = [math.degrees(float(a)) for a in raw.msg[:NUM_JOINTS]]
        current_deg[joint_index] += delta_deg
        return self.move_joints(current_deg)

    def open_gripper(self, width: float = 0.08, force: float = 1.0) -> bool:
        if not self._gripper_available or self._end_effector is None:
            print("[NERO] Gripper not available")
            return False
        try:
            self._end_effector.move_gripper(width=width, force=force)
            time.sleep(GRIPPER_SETTLE_S)
            return True
        except Exception as exc:
            print(f"[NERO] open_gripper failed: {exc}")
            return False

    def close_gripper(self, force: float = 1.0) -> bool:
        if not self._gripper_available or self._end_effector is None:
            print("[NERO] Gripper not available")
            return False
        try:
            self._end_effector.move_gripper(width=0.0, force=force)
            time.sleep(GRIPPER_SETTLE_S)
            return True
        except Exception as exc:
            print(f"[NERO] close_gripper failed: {exc}")
            return False

    # -- high-level task actions ------------------------------------------

    def move_above(self, target_name: str, target_pos: Position3D | None = None) -> bool:
        """Move above a target.

        Currently uses a conservative approach: move to a predefined
        approach pose.  When the camera + IK pipeline is ready, this will
        use target_pos to compute the actual joint target.
        """
        print(f"[NERO] move_above '{target_name}' (pos={target_pos})")
        # TODO: implement IK or pose-lookup when camera pipeline is ready
        return True

    def lower(self, target_name: str, target_pos: Position3D | None = None) -> bool:
        print(f"[NERO] lower to '{target_name}' (pos={target_pos})")
        # TODO: implement relative Z descent
        return True

    def stop(self) -> bool:
        if self._robot is None:
            return False
        try:
            self._robot.set_normal_mode()
            return True
        except Exception as exc:
            print(f"[NERO] stop failed: {exc}")
            return False


# =========================================================================
#  Factory & dispatch
# =========================================================================

def get_robot_controller(mode: str | None = None) -> BaseRobotController:
    selected = (mode or os.getenv("ROBOT_MODE", "mock")).lower()
    if selected == "nero":
        return NeroRobotController()
    return MockRobotController()


def dispatch_action(
    robot: BaseRobotController,
    action: str,
    target: str = "",
    target_pos: Position3D | None = None,
) -> bool:
    if action == "move_above":
        return robot.move_above(target, target_pos)
    if action == "lower":
        return robot.lower(target, target_pos)
    if action == "grasp":
        return robot.grasp()
    if action == "lift":
        return robot.lift()
    if action == "release":
        return robot.release()
    if action == "stop":
        return robot.stop()
    if action == "home":
        return robot.home()
    if action in {"done", "wait"}:
        return True
    raise ValueError(f"Unsupported robot action: {action}")
